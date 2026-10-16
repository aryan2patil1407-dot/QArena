"""
QArena Orchestrator — wires the 8-stage / 36-step master pipeline:
Stage 2 sandbox -> 3b explore -> 4 plan -> 5a/5b execute+judge (per test)
-> 6a replay-verify (per suspect) -> 7a report -> 7b export+self-verify
-> 8 benchmark score. Emits SSE events the whole way. Zero AI outside the
4 sanctioned call sites; everything else is Playwright + plain Python.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .contracts import (FiledBug, FlakySuspect, PossibleBug, RunResult,
                        TestCase)
from .explorer import Explorer
from .exporter import Exporter
from .judge import Judge
from .llm import LLMPolicy
from .planner import Planner
from .reporter import Reporter
from .sandbox import Sandbox
from .tester import Tester
from .verifier import Verifier


@dataclass
class RunHandle:
    run_id: str
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    history: list[dict] = field(default_factory=list)
    result: RunResult | None = None
    run_dir: Path | None = None
    thread: threading.Thread | None = None
    current_stage: str = ""      # for diagnostics: names the stage in a traceback
    seq: int = 0                 # monotonic event id: makes UI replays idempotent


class Orchestrator:
    def __init__(self) -> None:
        self.runs: dict[str, RunHandle] = {}

    # ------------------------------------------------------------ lifecycle
    def start_run(self, target_url: str, description: str,
                  email: str = "", password: str = "",
                  whitebox: dict | None = None) -> str:
        run_id = uuid.uuid4().hex[:10]
        handle = RunHandle(run_id=run_id)
        run_dir = config.ARTIFACTS / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        handle.run_dir = run_dir
        handle.result = RunResult(run_id=run_id, target_url=target_url,
                                  description=description)
        t = threading.Thread(
            target=self._safe_pipeline,
            args=(handle, target_url, description, email, password, whitebox),
            daemon=True)
        handle.thread = t
        self.runs[run_id] = handle
        t.start()
        return run_id

    def emit_of(self, run_id: str):
        h = self.runs[run_id]

        def emit(e: dict) -> None:
            e["ts"] = round(time.time(), 2)
            # every event carries a monotonic id. A live SSE client sees events
            # streamed once, then gets the whole history replayed on any
            # reconnect — without an id the UI showed the same event twice (two
            # identical stage rows after the EventSource reconnected).
            h.seq += 1
            e["seq"] = h.seq
            h.history.append(e)
            h.events.put(e)
            try:
                # explicit UTF-8: event payloads carry "→" from signal evidence
                # and would abort on the platform default code page (Windows).
                with open(h.run_dir / "run.log", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(e) + "\n")
            except Exception:
                pass
        return emit

    @staticmethod
    def _push(h: RunHandle, e: dict) -> None:
        """Queue-only event (terminal notices): streamed live, not replayed."""
        e["ts"] = round(time.time(), 2)
        h.seq += 1
        e["seq"] = h.seq
        h.events.put(e)

    def _guarded(self, h: RunHandle, res: RunResult, key: str, fn, emit):
        """Run one *non-fatal* stage. Late-stage failures must never erase
        findings that earlier stages already verified: record the error, log the
        traceback, tell the UI, and carry on with a ``None`` result."""
        try:
            return fn()
        except Exception as exc:
            ctx = f"{h.current_stage}: " if h.current_stage else ""
            res.stage_errors[key] = f"{ctx}{exc.__class__.__name__}: {exc}"
            self._log_traceback(h, key)
            emit({"type": "warning", "stage": key,
                  "msg": f"{key} stage failed ({exc.__class__.__name__}: {exc})"
                         f" — verified findings from earlier stages are kept"})
            return None

    def _log_traceback(self, h: RunHandle, stage_key: str) -> None:
        """Full stack trace into the run's own log — bounded to the run dir and
        never echoed to a console that may not encode it."""
        try:
            with open(h.run_dir / "run.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "traceback", "stage": stage_key,
                                     "ts": round(time.time(), 2),
                                     "msg": traceback.format_exc()}) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------- pipeline
    def _safe_pipeline(self, h: RunHandle, url: str, desc: str,
                       email: str, password: str, whitebox: dict | None):
        try:
            self._pipeline(h, url, desc, email, password, whitebox)
        except Exception as exc:
            # fatal: name the stage, keep the stack trace, and do NOT discard
            # whatever the earlier stages already verified and persisted.
            self._log_traceback(h, h.current_stage or "pipeline")
            msg = f"{exc.__class__.__name__}: {exc}"
            if h.result:
                h.result.status = "error"
                h.result.error = (f"{h.current_stage}: {msg}"
                                  if h.current_stage else msg)
            self._push(h, {"type": "error", "msg": msg,
                           "stage": h.current_stage})
        finally:
            self._save(h)
            self._push(h, {"type": "done",
                           "status": h.result.status if h.result else "error",
                           "result": self._result_json(h)})

    def _pipeline(self, h: RunHandle, url: str, desc: str,
                  email: str, password: str, whitebox: dict | None):
        llm = LLMPolicy()
        sbx = Sandbox(url, h.run_dir)
        emit = self.emit_of(h.run_id)
        res = h.result
        res.llm_mode = llm.status

        def stage(n: int, name: str):
            h.current_stage = f"stage {n} ({name.split(' — ')[0].strip()})"
            emit({"type": "stage", "stage": n, "name": name})

        def guarded(key: str, fn):
            return self._guarded(h, res, key, fn, emit)

        try:
            # ------------------------------- STAGE 2: sandbox + safety + login
            stage(2, "Sandbox / Safety Guard / Login-once")
            if whitebox and whitebox.get("folder"):
                sbx.boot_local_app(whitebox["folder"], whitebox["command"],
                                   int(whitebox["port"]))
            sbx.wait_until_responsive(url, timeout=20)
            seeded = sbx.seed_target()
            emit({"type": "log", "msg": f"target responsive; seed={'yes' if seeded else 'n/a'}"})
            sbx.start_browser()
            if email:
                ok = sbx.login_once(email, password)
                emit({"type": "log", "msg": f"login-once: {'ok' if ok else 'skipped/failed'}"})
            else:
                if config.AUTH_STATE.exists():
                    config.AUTH_STATE.unlink()

            # ------------------------------- STAGE 3b: exploration
            stage(3, "Explorer — mapping the territory")
            t0 = time.time()
            explorer = Explorer(sbx, h.run_dir, emit)
            appmap = explorer.explore(desc)
            res.states_found = len(appmap.states)
            self._save(h)
            emit({"type": "map", "states": res.states_found,
                  "seconds": round(time.time() - t0, 1)})

            # ------------------------------- STAGE 4: planning + guardrails
            stage(4, "Planner — journeys & adversarial cases")
            planner = Planner(appmap, llm)
            cases, rejected = planner.plan()
            res.tests_planned = len(cases)
            res.rejected_steps = rejected
            emit({"type": "plan", "tests": len(cases), "rejected": len(rejected),
                  "titles": [c.title for c in cases]})
            self._save(h)

            # ------------------------------- STAGE 5a+5b: execute + judge
            tester = Tester(sbx, appmap, h.run_dir, emit)
            judge = Judge(llm)
            verifier = Verifier(tester, judge, emit)
            passed_cases: list[TestCase] = []
            suspects: list[PossibleBug] = []

            for tc in cases:
                stage(5, f"Tester × Judge — {tc.test_id}: {tc.title[:48]}")
                sbx.seed_target()  # per-test world reset when hook exists
                verdict, evidences = tester.run_test(tc, tracing=True)
                res.steps_executed += len(evidences)
                emit({"type": "verdict", "test": tc.test_id, "verdict": verdict})
                test_suspects: list[PossibleBug] = []
                for i, ev in enumerate(evidences):
                    for pb in judge.evaluate(tc.title, tc.test_id, i, ev,
                                             tc.steps[: i + 1]):
                        pb.oracle = [tc.expected.model_dump()]  # replay contract
                        test_suspects.append(pb)
                # cross-step merge: S2 on a dead-link navigation absorbs the
                # S3 'dead link' spotted on the page that hosted it
                merged_here = self._merge_deadlink_pairs(test_suspects)
                suspects.extend(merged_here)
                for pb in merged_here:
                    emit({"type": "suspect", "journey": tc.test_id,
                          "score": pb.score, "summary": pb.summary[:140]})
                if verdict == "passed":
                    passed_cases.append(tc)
                self._save(h)

            # pre-verify dedupe: one verdict per (route, group); content bugs
            # (intent/console families) dedupe globally regardless of route
            unique: dict[tuple, PossibleBug] = {}
            for pb in suspects:
                # content defects (console noise, price/money semantics) fire
                # on every page they appear — dedupe them globally by group so
                # "-$64.95 on /cart, /checkout, /order" is ONE filing
                route_insens = pb.group.startswith(("intent:", "console:",
                                                    "form:money", "flow:badge",
                                                    "axe:"))
                key = ((pb.group,) if route_insens
                       else (self._route_of(pb), pb.group))
                # final-gate contract groups ("silently accepted",
                # "forbidden text") collapse DISTINCT defects that share a
                # route+group label but have different root causes per
                # journey (empty-cart vs empty-address vs bad-qty checkout)
                # — keep one per test instead, else BUG-12-style recall leak
                if pb.group.startswith(("form:invalid input was silently",
                                        "form:forbidden text present")):
                    key = key + (pb.test_id,)
                if key not in unique or pb.score > unique[key].score:
                    unique[key] = pb
            suspects = list(unique.values())
            emit({"type": "log",
                  "msg": f"{len(suspects)} unique suspects after dedupe"})

            # ------------------------------- STAGE 6a: replay verification
            stage(6, "Replay Verifier — proving suspects in clean browsers")
            confirmed: list[tuple[PossibleBug, str]] = []
            for pb in suspects:
                ok, count = verifier.verify(pb)
                if ok:
                    confirmed.append((pb, count))
                    emit({"type": "confirmed", "journey": f"{pb.test_id} {pb.group}",
                          "count": count})
                else:
                    res.flaky.append(FlakySuspect(
                        journey=f"{pb.journey} [{pb.group}]", summary=pb.summary[:200],
                        reproduced=count, signals=[s.signal for s in pb.signals]))
                    emit({"type": "flaky", "journey": pb.journey, "count": count})
                # incremental safekeeping: each proof lands on disk as it is made,
                # so even a crash inside the replay loop cannot lose it
                self._persist_confirmed(h, confirmed)
                self._save(h)

            # verified findings hit disk BEFORE any late stage can fail, so a
            # crash in reporting/export/benchmark can never erase them
            self._persist_confirmed(h, confirmed)

            # ------------------------------- STAGE 7a: dedup + reports
            stage(7, "Reporter — dedup, severity, human-quality reports")
            reporter = Reporter(h.run_dir, llm)
            filed = guarded("reporter", lambda: reporter.file_bugs(confirmed))
            if filed is None:
                # reporting died as a whole: still surface the proven findings
                # through the fallback filer (facts unchanged, prose canonical)
                filed = reporter.fallback_file_bugs(confirmed)
            res.filed_bugs = filed
            self._save(h)

            # ------------------------------- STAGE 7b: export + self-verify
            stage(7, "Exporter — pytest suite, self-verified green")
            exporter = Exporter(h.run_dir, url)

            def _export() -> list[str]:
                files = exporter.export(passed_cases)
                keep, pytest_log = exporter.self_verify(files)
                (h.run_dir / "pytest_self_run.log").write_text(
                    pytest_log, encoding="utf-8")
                return keep

            keep = guarded("exporter", _export)
            res.exported_tests = keep if keep is not None else []
            res.tests_passed = len(passed_cases)
            res.coverage = self._coverage(appmap, cases, passed_cases)

            # ------------------------------- STAGE 8: benchmark
            stage(8, "Benchmark — precision / recall / F1 vs ground truth")
            from scorer.evaluate_benchmark import evaluate
            bench = guarded("benchmark", lambda: evaluate(h.run_dir))
            res.benchmark = (bench if bench is not None
                             else {"available": False,
                                   "error": res.stage_errors.get("benchmark", "")})
            res.tokens_used = llm.total_tokens
            res.llm_mode = llm.status      # failures during the run are admitted here
            # honest terminal status: everything ran => "done"; a late stage
            # failed but the core pipeline (and its verified findings) survived
            # => "partial"
            res.status = "partial" if res.stage_errors else "done"
        finally:
            sbx.shutdown()

    @staticmethod
    def _persist_confirmed(h: RunHandle,
                           confirmed: list[tuple[PossibleBug, str]]) -> None:
        """Write replay-verified findings to their own artifact immediately."""
        try:
            payload = [{"possible_bug": pb.model_dump(), "reproduced_count": count}
                       for pb, count in confirmed]
            (h.run_dir / "confirmed.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # best-effort safekeeping; the authoritative filing path is Stage 7a
            pass

    # ---------------------------------------------------------------- misc
    @staticmethod
    def _merge_deadlink_pairs(pbs: list[PossibleBug]) -> list[PossibleBug]:
        """If an S3 'dead link' and an S2 '404 document' in the same test point
        at the same URL, they're one defect — merge (S2 suspect absorbs S3)."""
        s3_dead = {}
        for pb in pbs:
            if pb.group.startswith("broken:"):
                for b in pb.evidence.broken_items:
                    if b.get("kind") == "link":
                        s3_dead[(b.get("detail") or "").split(" ")[0]] = pb
        absorbed: set[int] = set()
        for i, pb in enumerate(pbs):
            if not pb.group.startswith("http:"):
                continue
            path = pb.group[5:]
            for dead_url, s3pb in list(s3_dead.items()):
                if dead_url and path.endswith(dead_url):
                    sig = {h.signal for h in pb.signals}
                    if 3 not in sig:
                        pb.signals = s3pb.signals + pb.signals
                        pb.score = min(1.0, round(pb.score + 0.3, 3))
                        pb.summary += " | " + s3pb.summary
                    absorbed.add(id(s3pb))
        return [pb for pb in pbs if id(pb) not in absorbed]

    @staticmethod
    def _route_of(pb: PossibleBug) -> str:
        import re
        from urllib.parse import urlparse
        url = pb.evidence.url_after or pb.evidence.url_before or ""
        return (re.sub(r"/\d+(?=/|$)", "/*", urlparse(url).path) or "/").rstrip("/") or "/"

    @staticmethod
    def _coverage(appmap, cases, passed) -> dict:
        routes_hit = set()
        for tc in cases:
            for st in tc.steps:
                if st.action == "goto":
                    routes_hit.add(st.target)
        total_routes = len({s.route for s in appmap.states if s.http_status == 200})
        elements_total = sum(len(s.elements) for s in appmap.states
                             if s.http_status == 200)
        exercised = set()
        for tc in cases:
            for st in tc.steps:
                if st.role:
                    exercised.add((st.role, st.name))
        return {
            "route_coverage": round(min(1.0, len(routes_hit) / max(total_routes, 1)) * 100, 1),
            "routes_tested": len(routes_hit), "routes_total": total_routes,
            "element_coverage": round(min(1.0, len(exercised) / max(elements_total, 1)) * 100, 1),
            "elements_exercised": len(exercised), "elements_total": elements_total,
        }

    @staticmethod
    def _result_json(h: RunHandle) -> dict:
        return h.result.model_dump() if h.result else {}

    def _save(self, h: RunHandle) -> None:
        if h.result and h.run_dir:
            # UTF-8 explicitly: pydantic does not escape non-ASCII, so signal
            # evidence ("→ 404") reaches this file verbatim
            (h.run_dir / "results.json").write_text(
                h.result.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------------------- utilities
    def latest_run(self) -> RunHandle | None:
        return self.runs[sorted(self.runs)[-1]] if self.runs else None

    def zip_run(self, run_id: str) -> Path | None:
        h = self.runs.get(run_id)
        if not h:
            return None
        archive = shutil.make_archive(str(h.run_dir), "zip", h.run_dir)
        return Path(archive)
