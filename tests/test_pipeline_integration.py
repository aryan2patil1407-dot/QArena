"""Pipeline-level integration tests: Stage 2 -> 8 through the real orchestrator.

The reported failure was an *integration* defect — the judge produced verified
findings, the reporter died, and the orchestrator discarded them. Unit tests on
the reporter alone would not have caught it.

A real browser cannot run in CI or in this sandbox, so the browser-facing stages
are replaced by fakes. Everything *between* them stays real: the real Judge
scores the evidence, the real Reporter files the bugs, the real Exporter runs its
verification, the real scorer evaluates, and the real Orchestrator decides the
terminal status. Only `Explorer`/`Tester`/`Verifier`/`Sandbox` are stand-ins.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config, orchestrator as orch_mod
from core.contracts import (AppMap, AppState, ExpectedContract, RunResult,  # noqa: E402
                            StateElement, StepEvidence, StepSpec, TestCase)
from core.orchestrator import Orchestrator, RunHandle
from core.reporter import Reporter

TARGET = "https://sweetshop.test"


# --------------------------------------------------------------------- fakes
class FakeSandbox:
    """Stands in for Playwright only. Records the lifecycle the pipeline drives."""

    def __init__(self, url: str, run_dir: Path) -> None:
        self.url, self.run_dir = url, run_dir
        self.base_host = url.replace("http://", "").replace("https://", "").split("/")[0]
        self.calls: list[str] = []
        self.shut_down = False

    def wait_until_responsive(self, url: str, timeout: int = 30) -> None:
        self.calls.append("probe")

    def seed_target(self) -> bool:
        self.calls.append("seed")
        return True

    def start_browser(self):
        self.calls.append("start_browser")
        return object()

    def login_once(self, *a, **k) -> bool:
        self.calls.append("login")
        return True

    def shutdown(self) -> None:
        self.shut_down = True


class FakeExplorer:
    def __init__(self, sbx, run_dir, emit) -> None:
        self.run_dir, self.emit = run_dir, emit

    def explore(self, description: str) -> AppMap:
        am = AppMap(app_url=TARGET, description=description)
        am.states = [
            AppState(state_id="S00", route="/", url=TARGET + "/",
                     state_hash="h0", http_status=200, discovered_via="seed",
                     elements=[StateElement(id="S00e0", role="link", name="Basket")]),
            # /cart was reached by following the home page's OWN "Basket" link —
            # that link is what makes a failure there reportable (a URL QArena
            # merely guessed would not be). The crawl graph records the link.
            AppState(state_id="S01", route="/cart", url=TARGET + "/cart",
                     state_hash="h1", http_status=200, discovered_via="link",
                     link_name="Basket", discovered_from="/"),
        ]
        am.edges = [{"from": "S00", "to": "/cart", "via": 'link "Basket"'}]
        for st in am.states:                       # the real explorer reports each
            self.emit({"type": "state", "state_id": st.state_id,
                       "route": st.route, "status": st.http_status,
                       "elements": len(st.elements)})
        return am


class FakePlanner:
    def __init__(self, appmap, llm) -> None:
        self.map = appmap

    def plan(self):
        tc = TestCase(test_id="TC_001", journey_type="happy_path",
                      title="Basket link serves a real page",
                      steps=[StepSpec(step=1, action="goto", target="/"),
                             StepSpec(step=2, action="goto", target="/cart")],
                      expected=ExpectedContract())
        # same shape the real `Planner.reject_invented` returns: the reason is
        # part of the contract so the dashboard can say WHY a step was refused
        rejected = [{"test_id": "TC_002", "title": "Invented journey",
                     "reason": "missing_control",
                     "invented_steps": ['click button "Pay Now"']}]
        return [tc], rejected


class FakeTester:
    """Emits browser-shaped evidence; the REAL judge scores it."""

    def __init__(self, sbx, appmap, run_dir, emit) -> None:
        self.sbx, self.run_dir, self.emit = sbx, run_dir, emit

    def run_test(self, tc: TestCase, tracing: bool = True):
        ev = StepEvidence(
            step=2, action="goto", target="/cart", status="failed",
            url_before=TARGET + "/", url_after=TARGET + "/cart",
            http_failures=[{"method": "GET", "url": TARGET + "/cart",
                            "status": 404}],
            console_errors=["Failed to load resource: 404 {/cart}"],
            broken_items=[{"kind": "link", "name": "Basket",
                           "detail": "/cart → 404"}],
            screenshot="steps/TC_001_step2.png")
        return "failed", [ev]


class FakeVerifier:
    def __init__(self, tester, judge, emit) -> None:
        self.tester, self.emit = tester, emit

    def verify(self, pb):
        """Same contract as the real pass^3 gate: fresh context per attempt,
        one `replay` event per attempt, a count of clean-browser reproductions,
        and a classified outcome (matched / not_reproduced / inconclusive /
        infrastructure_error)."""
        from core.verifier import MATCHED, NOT_REPRODUCED, ReplayOutcome
        hits = 0
        for attempt in (1, 2, 3):
            self.tester.sbx.seed_target()
            hits += 1
            self.emit({"type": "replay", "journey": f"{pb.test_id} {pb.group}",
                       "attempt": attempt, "matched": True})
        outcome = ReplayOutcome(
            attempts=3, completed=3, matched=hits,
            kind=MATCHED if hits >= 2 else NOT_REPRODUCED,
            detail=f"reproduced {hits}/3 completed clean-browser replays")
        return outcome.confirmed, outcome.summary(), outcome


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    """A browserless but otherwise real Orchestrator run."""
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(config, "AUTH_STATE", tmp_path / "auth_state.json")
    monkeypatch.setattr(orch_mod, "Sandbox", FakeSandbox)
    monkeypatch.setattr(orch_mod, "Explorer", FakeExplorer)
    monkeypatch.setattr(orch_mod, "Planner", FakePlanner)
    monkeypatch.setattr(orch_mod, "Tester", FakeTester)
    monkeypatch.setattr(orch_mod, "Verifier", FakeVerifier)

    def build(run_id: str = "itest00001"):
        orch = Orchestrator()
        h = RunHandle(run_id=run_id)
        h.run_dir = config.ARTIFACTS / f"run_{run_id}"
        h.run_dir.mkdir(parents=True, exist_ok=True)
        h.result = RunResult(run_id=run_id, target_url=TARGET, description="shop")
        orch.runs[run_id] = h
        h._test_orch = orch            # type: ignore[attr-defined]
        return h

    return build


def run_pipeline(h: RunHandle) -> None:
    h._test_orch._safe_pipeline(h, TARGET, "shop", "", "", None)  # type: ignore[attr-defined]


# ------------------------------------------------------------------- the paths
def test_stages_2_to_8_complete_and_artifacts_agree(pipeline):
    h = pipeline()
    run_pipeline(h)
    res = h.result
    events = h.history                      # emit_of() records here

    assert res.status == "done", res.stage_errors
    assert not res.stage_errors
    assert res.states_found == 2
    assert res.tests_planned == 1
    assert res.stage_errors == {}
    # the title now names the target's OWN broken link and the request that
    # failed, because that is the actionable fact (previously "Basket: Network
    # Failures — …" never said who published the link)
    assert [b.title for b in res.filed_bugs] == [
        'Broken link "Basket" on / — GET https://sweetshop.test/cart → 404']

    # the whole pipeline reported itself: stages, log lines, states, plan,
    # verdict, suspect, replay proof and the confirmation
    assert {e["type"] for e in events} >= {"stage", "log", "state", "plan",
                                           "verdict", "suspect", "replay",
                                           "confirmed"}
    # every stage announces itself (7 twice: reporter, then exporter)
    assert sorted({e["stage"] for e in events if e["type"] == "stage"}) == [2, 3, 4, 5, 6, 7, 8]

    # every artifact agrees with the run result
    results = json.loads((h.run_dir / "results.json").read_text(encoding="utf-8"))
    filed = json.loads((h.run_dir / "bugs_filed.json").read_text(encoding="utf-8"))
    confirmed = json.loads((h.run_dir / "confirmed.json").read_text(encoding="utf-8"))
    assert len(results["filed_bugs"]) == len(filed) == len(confirmed) == 1
    assert results["status"] == "done"
    assert (h.run_dir / "bug_reports" / "QA-01.md").exists()
    # the report carries the raw console/network facts a reader needs
    md = (h.run_dir / "bug_reports" / "QA-01.md").read_text(encoding="utf-8")
    assert "Network: GET https://sweetshop.test/cart → 404" in md
    assert "Console: Failed to load resource: 404 {/cart}" in md


def test_reporting_failure_keeps_verified_findings_and_marks_partial(pipeline, monkeypatch):
    """The reported incident, at pipeline level: reporting dies, findings survive."""
    h = pipeline("itest00002")

    def exploding_file_bugs(self, confirmed):
        raise UnicodeEncodeError("charmap", "\u2192", 75, 76,
                                 "character maps to <undefined>")

    monkeypatch.setattr(Reporter, "file_bugs", exploding_file_bugs)
    run_pipeline(h)
    res = h.result
    events = h.history

    # honest status: not "done", not a wiped "error"
    assert res.status == "partial"
    assert "reporter" in res.stage_errors
    assert "UnicodeEncodeError" in res.stage_errors["reporter"]
    assert "stage 7" in res.stage_errors["reporter"]

    # the verified finding still reaches the dashboard, the disk and the pack
    assert len(res.filed_bugs) == 1
    assert res.filed_bugs[0].reproduced_count == "3/3"
    assert res.filed_bugs[0].evidence.get("warnings")
    filed = json.loads((h.run_dir / "bugs_filed.json").read_text(encoding="utf-8"))
    confirmed = json.loads((h.run_dir / "confirmed.json").read_text(encoding="utf-8"))
    assert len(filed) == len(confirmed) == 1
    assert (h.run_dir / "bug_reports" / "QA-01.md").exists()

    # the UI was told, in plain language, and the traceback is on disk
    warnings = [e for e in events if e["type"] == "warning"]
    assert warnings and warnings[0]["stage"] == "reporter"
    assert "kept" in warnings[0]["msg"]
    log = (h.run_dir / "run.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "UnicodeEncodeError" in log
    # the later stages still ran on valid partial results
    assert res.coverage and res.coverage["routes_total"] == 2
    assert res.benchmark is not None


def test_export_failure_does_not_touch_the_filings(pipeline, monkeypatch):
    h = pipeline("itest00003")
    from core.exporter import Exporter

    def exploding_export(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Exporter, "export", exploding_export)
    run_pipeline(h)
    res = h.result
    assert res.status == "partial"
    assert "exporter" in res.stage_errors
    assert len(res.filed_bugs) == 1, "filing must survive an export failure"
    assert res.benchmark.get("available") is False


def test_benchmark_failure_does_not_touch_the_filings(pipeline, monkeypatch):
    h = pipeline("itest00004")
    import scorer.evaluate_benchmark as bench

    def exploding_eval(*a, **k):
        raise FileNotFoundError("bugs.json")

    monkeypatch.setattr(bench, "evaluate", exploding_eval)
    run_pipeline(h)
    res = h.result
    assert res.status == "partial"
    assert "benchmark" in res.stage_errors
    assert res.benchmark.get("available") is False
    assert len(res.filed_bugs) == 1


def test_flaky_and_confirmed_are_kept_distinct(pipeline, monkeypatch):
    """A suspect that fails the replay gate must never be counted as a bug."""
    h = pipeline("itest00005")

    def one_of_three(self, pb):
        from core.verifier import NOT_REPRODUCED, ReplayOutcome
        out = ReplayOutcome(attempts=3, completed=3, matched=1,
                            kind=NOT_REPRODUCED,
                            detail="reproduced 1/3 completed replays")
        return False, out.summary(), out

    monkeypatch.setattr(orch_mod.Verifier, "verify", one_of_three)
    run_pipeline(h)
    res = h.result
    assert res.status == "done"
    assert res.filed_bugs == []
    assert len(res.flaky) == 1
    assert res.flaky[0].reproduced == "1/3"
    confirmed = json.loads((h.run_dir / "confirmed.json").read_text(encoding="utf-8"))
    assert confirmed == [], "nothing unproven may be persisted as confirmed"


def test_rejected_steps_are_reported_not_hidden(pipeline):
    h = pipeline("itest00006")
    run_pipeline(h)
    assert [r["test_id"] for r in h.result.rejected_steps] == ["TC_002"]
    assert h.result.rejected_steps[0]["invented_steps"] == ['click button "Pay Now"']
    # the rejection says WHY, so the dashboard can explain it
    assert h.result.rejected_steps[0]["reason"]
    plan = [e for e in h.history if e["type"] == "plan"][0]
    assert plan["rejected"] == 1


def test_run_log_is_utf8_and_replayable(pipeline):
    """run.log is the artifact `load_handle` replays after a restart."""
    h = pipeline("itest00007")
    run_pipeline(h)
    raw = (h.run_dir / "run.log").read_text(encoding="utf-8")
    lines = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    assert all(isinstance(ln, dict) for ln in lines)
    # the arrow survives the JSON round-trip (the log escapes it as \u2192, so
    # the file itself stays ASCII-safe on any console/code page)
    assert any("\u2192" in json.dumps(ln, ensure_ascii=False) for ln in lines)
    from app.main import load_handle
    h2 = load_handle("itest00007")
    assert h2 is not None and h2.history == h.history
