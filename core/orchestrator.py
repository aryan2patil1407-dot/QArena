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
from .provenance import RoutePolicy, canonical_route as route_of_url
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
            executed_cases: list[tuple[TestCase, str, list]] = []
            errored_cases: list[TestCase] = []
            suspects: list[PossibleBug] = []
            routes_executed: set[str] = set()
            controls_exercised: set[tuple[str, str]] = set()

            for tc in cases:
                stage(5, f"Tester × Judge — {tc.test_id}: {tc.title[:48]}")
                sbx.seed_target()  # per-test world reset when hook exists
                verdict, evidences = tester.run_test(tc, tracing=True)
                res.steps_executed += len(evidences)
                emit({"type": "verdict", "test": tc.test_id, "verdict": verdict})
                # execution facts, taken from what the browser actually did — never
                # from the plan (a planned journey that never ran is not coverage).
                # verdict == "error" means the journey could not be driven at all,
                # so its steps are not evidence of coverage either.
                if verdict == "error":
                    errored_cases.append(tc)
                if verdict in ("passed", "failed"):
                    executed_cases.append((tc, verdict, evidences))
                    for i, ev in enumerate(evidences):
                        for u in (ev.url_before, ev.url_after):
                            if u:
                                routes_executed.add(route_of_url(u))
                        spec = tc.steps[i] if i < len(tc.steps) else None
                        if spec is not None and spec.role:
                            controls_exercised.add((spec.role, spec.name))
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
            # Evidence admissibility gate: a finding whose evidence was taken on
            # a route QArena invented is not a finding about the target. Drop it
            # BEFORE replay — repeating an invented navigation cannot make it
            # true, and "it happened 3/3 times" would be pure theatre.
            admiss: list[PossibleBug] = []
            policy = RoutePolicy(appmap)
            for pb in suspects:
                reason = policy.blocked_reason(pb.evidence.url_after
                                               or pb.evidence.url_before)
                if reason:
                    res.suppressed.append({
                        "journey": pb.journey, "test_id": pb.test_id,
                        "group": pb.group, "summary": pb.summary[:200],
                        "reason": reason,
                        "signals": [s.signal for s in pb.signals]})
                    emit({"type": "suppressed", "journey": pb.journey,
                          "reason": reason})
                else:
                    admiss.append(pb)
            if res.suppressed:
                emit({"type": "log",
                      "msg": f"{len(res.suppressed)} suspect(s) withheld: their "
                             f"evidence came from a route QArena probed itself"})

            confirmed: list[tuple[PossibleBug, str]] = []
            for pb in admiss:
                ok, count, outcome = verifier.verify(pb)
                if ok:
                    confirmed.append((pb, count))
                    emit({"type": "confirmed", "journey": f"{pb.test_id} {pb.group}",
                          "count": count})
                else:
                    # a clean non-reproduction (flaky) is evidence about the app;
                    # an inconclusive or infrastructure-broken replay is evidence
                    # about the test environment. Keep them apart and never call
                    # a crashed replay a "flaky" defect.
                    res.flaky.append(FlakySuspect(
                        journey=f"{pb.journey} [{pb.group}]", summary=pb.summary[:200],
                        reproduced=count, signals=[s.signal for s in pb.signals],
                        kind=outcome.kind, detail=outcome.detail,
                        attempts_matched=outcome.matched,
                        attempts_completed=outcome.completed,
                        attempts_total=outcome.attempts))
                    emit({"type": "flaky", "journey": pb.journey, "count": count,
                          "kind": outcome.kind, "detail": outcome.detail})
                # incremental safekeeping: each proof lands on disk as it is made,
                # so even a crash inside the replay loop cannot lose it
                self._persist_confirmed(h, confirmed)
                self._save(h)

            # verified findings hit disk BEFORE any late stage can fail, so a
            # crash in reporting/export/benchmark can never erase them
            self._persist_confirmed(h, confirmed)

            # ------------------------------- STAGE 7a: dedup + reports
            stage(7, "Reporter — dedup, severity, human-quality reports")
            reporter = Reporter(h.run_dir, llm, appmap)
            filed = guarded("reporter", lambda: reporter.file_bugs(confirmed))
            if filed is None:
                # reporting died as a whole: still surface the proven findings
                # through the fallback filer (facts unchanged, prose canonical)
                filed = reporter.fallback_file_bugs(confirmed)
            res.filed_bugs = filed
            # the reporter's own synthetic-route safety net may have withheld
            # more suspects: keep the run's accounts complete
            if reporter.suppressed:
                res.suppressed.extend(
                    s for s in reporter.suppressed
                    if not any(x.get("test_id") == s.get("test_id")
                               and x.get("group") == s.get("group")
                               for x in res.suppressed))
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
            # a planned journey with no verdict never ran to completion (a stage
            # failed) — counted separately so coverage cannot overstate itself
            res.tests_unexecuted = max(0, len(cases) - len(executed_cases))
            res.coverage = self._coverage(
                appmap, cases, executed_cases, passed_cases,
                routes_executed, controls_exercised,
                suppressed=len(res.suppressed), rejected=len(rejected),
                errored=len(errored_cases))

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
        return route_of_url(pb.evidence.url_after or pb.evidence.url_before or "")

    @staticmethod
    def _coverage(appmap, cases, executed_cases, passed, routes_executed,
                  controls_exercised, suppressed: int = 0, rejected: int = 0,
                  errored: int = 0) -> dict:
        """Honest coverage: measured from execution, never from the plan.

        Definitions (these are the labels the dashboard shows):

        * ``routes_discovered``   every canonical route the explorer saw
        * ``routes_eligible``     routes the target established (linked/served) —
                                  the only routes anyone may test
        * ``routes_live``         eligible routes that are actually serving a page
        * ``routes_planned``      routes named in the plan, executed or not
        * ``routes_executed``     eligible routes the browser really visited
        * ``probe_only_routes``   URLs QArena guessed that the target never
                                  confirmed — excluded from testing by design
        * ``elements_total``      controls on eligible pages (the testable surface)
        * ``elements_exercised``  mapped controls an executed step really used

        A route or control that merely appeared in a plan is NOT counted as
        coverage. Zero denominators are reported as ``*_pct = None`` (unavailable)
        rather than as a fake 0% or 100%.
        """
        policy = RoutePolicy(appmap)
        eligible = policy.eligible_routes()
        live = policy.live_routes()
        discovered = set(policy._by_route)
        # compare like with like: a journey targets a concrete URL (/product/2)
        # while execution is recorded per canonical route (/product/*), so both
        # sides are canonicalised or a visited page looks unvisited
        planned = {route_of_url(st.target) for tc in cases for st in tc.steps
                   if st.action == "goto"}
        executed_ok = {r for r in routes_executed if r in eligible}

        elements_total = sum(len(s.elements) for s in appmap.states
                             if policy.confirmed(s.route))
        mapped_controls = {(e.role, e.name) for s in appmap.states
                           if policy.confirmed(s.route) for e in s.elements}
        exercised = {c for c in controls_exercised if c in mapped_controls}
        planned_controls = {(st.role, st.name) for tc in cases for st in tc.steps
                            if st.role and st.action != "goto"}

        def pct(num: int, den: int):
            return round(min(1.0, num / den) * 100, 1) if den else None

        return {
            # --- execution-based (the honest headline numbers) --------------
            # denominator = routes the target established (the testable surface),
            # so the numerator can never exceed it: a linked page that answers
            # 500 is a real route and counts on both sides of the fraction.
            "route_coverage": pct(len(executed_ok), len(eligible)),
            # `routes_tested`/`routes_total` are the long-standing keys the API
            # and dashboard read; `routes_executed`/`routes_eligible_count` say
            # the same thing in the honesty vocabulary. Both describe EXECUTED
            # routes over the TESTABLE surface.
            "routes_tested": len(executed_ok), "routes_total": len(eligible),
            "routes_executed": len(executed_ok),
            "routes_eligible_count": len(eligible),
            "routes_live_count": len(live),
            "element_coverage": pct(len(exercised), elements_total),
            "elements_exercised": len(exercised), "elements_total": elements_total,
            # --- route bookkeeping ------------------------------------------
            "routes_discovered": len(discovered),
            "routes_eligible": len(eligible),
            "routes_live": len(live),
            "routes_planned": len(planned),
            "routes_planned_not_executed": sorted(planned - executed_ok),
            "probe_only_routes": sorted(r for r in discovered if not policy.confirmed(r)),
            # --- control bookkeeping ----------------------------------------
            "controls_planned": len(planned_controls),
            "controls_planned_never_exercised": sorted(
                f"{r} {n}".strip() for r, n in (planned_controls - exercised)),
            # --- journey bookkeeping ----------------------------------------
            "journeys_planned": len(cases),
            "journeys_executed": len(executed_cases),
            "journeys_passed": sum(1 for _, v, _ in executed_cases if v == "passed"),
            "journeys_failed": sum(1 for _, v, _ in executed_cases if v == "failed"),
            # a journey that produced no verdict at all (a stage died / the
            # browser could not drive it) is NOT executed coverage
            "journeys_errored": errored,
            "journeys_unexecuted": max(0, len(cases) - len(executed_cases)),
            "journeys_rejected_ungrounded": rejected,
            "findings_withheld_synthetic": suppressed,
            "measured_from": "browser execution evidence (not the test plan)",
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
