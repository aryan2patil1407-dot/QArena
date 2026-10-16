"""Test-local configuration and shared pipeline harness.

`core.contracts.TestCase` is a domain model (a *test case* under test), not a
pytest test class — tell pytest not to try to collect it.

The `ctx` fixture below runs the REAL Orchestrator, the REAL Planner, the REAL
gate/Verifier/Reporter/persistence chain, with only the browser-facing pieces
(Sandbox / Explorer / Tester) replaced by stand-ins. Tests that assert on
progress the persisted artifacts read those files; they never assert on a
helper's return value alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config
from core import orchestrator as orch_mod
from core.contracts import (AppMap, AppState, PossibleBug, RunResult,
                            StateElement, StepEvidence, TestCase)
from core.llm import LLMPolicy

TestCase.__test__ = False


# --------------------------------------------------------------------- builders
def make_state(state_id: str, route: str, status: int = 200, elements=None,
               probed: bool = False, discovered_via: str = "",
               link_name: str = "", discovered_from: str = "") -> AppState:
    """One AppMap state. ``discovered_via`` states HOW QArena found it."""
    return AppState(state_id=state_id, route=route,
                    url=f"http://site.test{route if route != '/' else '/'}",
                    state_hash=f"h{state_id}", http_status=status,
                    probed=probed, discovered_via=discovered_via,
                    link_name=link_name, discovered_from=discovered_from,
                    elements=[e for e in (elements or []) if e is not None])


def make_map(states, edges=None, app_url: str = "http://site.test",
             catalog: dict | None = None) -> AppMap:
    am = AppMap(app_url=app_url, description="test app")
    am.states = list(states)
    if edges:
        am.edges = list(edges)
    if catalog:
        # names the explorer actually scraped off the target, with their prices
        am.catalog_prices = dict(catalog)
    return am


class FakeLLM(LLMPolicy):
    """A key-free policy: no model, so the deterministic grammar is exercised."""

    def __init__(self, available: bool = False) -> None:
        super().__init__()
        self._forced_available = available

    @property
    def available(self) -> bool:              # type: ignore[override]
        return self._forced_available


# ------------------------------------------------------------------ browser fakes
class _FakeSandbox:
    def __init__(self, url: str, run_dir: Path) -> None:
        self.url, self.run_dir = url, run_dir
        self.base_host = url.replace("http://", "").replace("https://", "").split("/")[0]
        self.shut_down = False

    def wait_until_responsive(self, *a, **k) -> None:
        return None

    def seed_target(self) -> bool:
        return True

    def start_browser(self):
        return object()

    def login_once(self, *a, **k) -> bool:
        return True

    def shutdown(self) -> None:
        self.shut_down = True


class _FakeExplorer:
    """Returns a caller-supplied AppMap verbatim — provenance is the input data."""

    def __init__(self, sbx, run_dir, emit) -> None:
        self.emit = emit

    def explore(self, description: str) -> AppMap:
        am = self._map
        for st in am.states:
            self.emit({"type": "state", "state_id": st.state_id,
                       "route": st.route, "status": st.http_status,
                       "elements": len(st.elements)})
        return am


class _FakeTester:
    """Browser stand-in that reports a failure on the route the journey reached.

    Stage-5 execution and Stage-6 replay both come through here, so the harness
    can tell them apart by test_id ("REPRO" is always a replay attempt).
    """

    def __init__(self, sbx, appmap, run_dir, emit) -> None:
        self.sbx, self.run_dir, self.emit = sbx, run_dir, emit
        self.ran: list[TestCase] = []
        self.verdict = "failed"
        self.status_map: dict[str, int] = {}
        self.default_status = 404
        # journeys the browser could not drive at all (verdict "error")
        self.error_titles: set[str] = set()

    def run_test(self, tc: TestCase, tracing: bool = True):
        self.ran.append(tc)
        if tc.title in self.error_titles:
            return "error", [StepEvidence(step=1, action="goto", target="/",
                                          status="error", note="browser died")]
        last = next((s.target for s in reversed(tc.steps)
                     if s.action == "goto"), "/")
        status = self.status_map.get(last, self.default_status)
        url = f"http://site.test{last if last != '/' else '/'}"
        ev = StepEvidence(
            step=len(tc.steps), action="goto", target=last, status="failed",
            url_before="http://site.test/", url_after=url,
            http_failures=[{"method": "GET", "url": url, "status": status}],
            broken_items=[{"kind": "link", "name": last.strip('/') or "home",
                           "detail": f"{last} → {status}"}])
        return self.verdict, [ev]


class _FakeJudge:
    """Injects the suspects a caller wants the pipeline to see.

    Used to prove DEFENCE IN DEPTH: even if some upstream stage produced a
    suspect about a route QArena invented, the guard ahead of replay and the
    reporter must still keep it out of the report.
    """

    def __init__(self, llm) -> None:
        self.injected: list[PossibleBug] = []
        self.seen: list = []

    def evaluate(self, journey, test_id, step_index, ev, steps=None):
        self.seen.append((test_id, ev))
        return list(self.injected)


class _RecordingVerifier:
    """The REAL verifier, wrapped so tests can see whether it was consulted."""

    real_cls = None
    instances: list = []

    def __init__(self, tester, judge, emit) -> None:
        self._impl = _RecordingVerifier.real_cls(tester, judge, emit)
        self.calls: list[PossibleBug] = []
        _RecordingVerifier.instances.append(self)

    def verify(self, pb):
        self.calls.append(pb)
        return self._impl.verify(pb)


class _Run:
    """Everything a test needs to judge one pipeline run."""

    def __init__(self, h, tester, verifier, judge, events, run_dir) -> None:
        self.h, self.tester, self.verifier = h, tester, verifier
        self.judge, self.events, self.run_dir = judge, events, run_dir

    @property
    def res(self) -> RunResult:
        return self.h.result

    def filed(self):
        """The bugs a user would read — straight from the persisted artifact."""
        path = self.run_dir / "bugs_filed.json"
        assert path.exists(), "bugs_filed.json was never written"
        return json.loads(path.read_text(encoding="utf-8"))

    def results(self) -> dict:
        return json.loads((self.run_dir / "results.json").read_text(encoding="utf-8"))

    @property
    def replay_attempts(self) -> list[TestCase]:
        """Stage-6 replay runs only (the real Verifier uses test_id REPRO)."""
        return [t for t in self.tester.ran if t.test_id == "REPRO"]


class _Ctx:
    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.monkeypatch, self.tmp_path = monkeypatch, tmp_path
        self._n = 0

    def run(self, appmap: AppMap, verdict: str = "failed",
            evidence_url: str = "", suspect_group: str = "",
            suspects: list | None = None, tester: object | None = None,
            status_map: dict | None = None, default_status: int = 404) -> _Run:
        """Run the pipeline over a hand-written AppMap.

        ``suspect_group``/``evidence_url`` (or ``suspects``) inject the suspects
        Stage 5b is to hand over; everything downstream is the real code.
        """
        from core.verifier import Verifier as RealVerifier
        _RecordingVerifier.real_cls = RealVerifier

        self.monkeypatch.setattr(config, "ARTIFACTS", self.tmp_path / "artifacts")
        self.monkeypatch.setattr(config, "AUTH_STATE", self.tmp_path / "auth.json")
        _FakeExplorer._map = appmap
        self.monkeypatch.setattr(orch_mod, "Sandbox", _FakeSandbox)
        self.monkeypatch.setattr(orch_mod, "Explorer", _FakeExplorer)
        self.monkeypatch.setattr(orch_mod, "Verifier", _RecordingVerifier)
        _RecordingVerifier.instances = []
        tester_holder: dict = {}

        if tester is not None:
            def _tester_factory(sbx, appmap_, run_dir, emit):
                tester.sbx, tester.run_dir = sbx, run_dir
                tester.emit = emit
                tester_holder["t"] = tester
                return tester
            self.monkeypatch.setattr(orch_mod, "Tester", _tester_factory)
        else:
            def _tester_factory(sbx, appmap_, run_dir, emit):
                t = _FakeTester(sbx, appmap_, run_dir, emit)
                t.verdict = verdict
                if status_map:
                    t.status_map = status_map
                t.default_status = default_status
                tester_holder["t"] = t
                return t
            self.monkeypatch.setattr(orch_mod, "Tester", _tester_factory)

        self._n += 1
        run_id = f"ptest{self._n:05d}"
        run_dir = config.ARTIFACTS / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        h = orch_mod.RunHandle(run_id=run_id)
        h.run_dir = run_dir
        h.result = RunResult(run_id=run_id, target_url="http://site.test",
                             description="test app")
        orch = orch_mod.Orchestrator()
        orch.runs[run_id] = h

        judge_holder = {}
        if suspects is not None or suspect_group:

            def _judge_factory(llm):
                j = _FakeJudge(llm)
                wanted = (list(suspects) if suspects is not None
                          else [(suspect_group, evidence_url, 1.0)])
                for group, url, score in wanted:
                    status = 404
                    m = __import__("re").search(r"->\s*(\d+)", group)
                    if m:
                        status = int(m.group(1))
                    from urllib.parse import urlparse
                    from core.contracts import StepSpec
                    path = urlparse(url).path or "/"
                    j.injected.append(PossibleBug(
                        test_id="TC_001", journey="injected journey",
                        group=group, summary=group, score=score,
                        step_index=0, signals=[],
                        # a real judge always carries the steps it executed
                        repro_steps=[StepSpec(step=1, action="goto", target="/"),
                                     StepSpec(step=2, action="goto", target=path)],
                        evidence=StepEvidence(
                            step=1, action="goto", target="/", status="failed",
                            url_before="http://site.test/", url_after=url,
                            http_failures=[{"method": "GET", "url": url,
                                            "status": status}])))
                judge_holder["j"] = j
                return j
            self.monkeypatch.setattr(orch_mod, "Judge", _judge_factory)

        orch._safe_pipeline(h, "http://site.test", "test app", "", "", None)
        verifier = _RecordingVerifier.instances[-1] if _RecordingVerifier.instances else None
        return _Run(h, tester_holder.get("t"), verifier,
                    judge_holder.get("j"), h.history, run_dir)


@pytest.fixture()
def ctx(monkeypatch, tmp_path):
    return _Ctx(monkeypatch, tmp_path)
