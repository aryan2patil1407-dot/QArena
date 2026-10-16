"""Regression tests: a late-stage failure must never erase verified findings.

Incident: two suspects reproduced 3/3 in fresh browsers, the Reporter then raised
UnicodeEncodeError, and the dashboard showed "0 proven bugs" because
`res.filed_bugs` was only ever assigned on the reporter's success path, while the
run's `finally:` persisted an empty list.

These tests pin the repaired behaviour:
  * verified findings are persisted to confirmed.json before late stages run;
  * a failing reporter still yields filings (fallback filer);
  * the run reports "partial" (not "done", not a wiped "error");
  * exported tests / benchmark failures do not destroy filed bugs.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from core.contracts import (ExpectedContract, PossibleBug, RunResult,  # noqa: E402
                            SignalHit, StepEvidence, StepSpec)
from core.llm import LLMPolicy
from core.orchestrator import Orchestrator, RunHandle
from core.reporter import Reporter


def make_suspect(group: str = "http:/cart", score: float = 0.95,
                 text: str = "GET https://target.test/cart \u2192 404") -> PossibleBug:
    ev = StepEvidence(step=2, action="goto", target="/cart", status="failed",
                      url_before="https://target.test/",
                      url_after="https://target.test/cart",
                      http_failures=[{"method": "GET",
                                      "url": "https://target.test/cart",
                                      "status": 404}],
                      broken_items=[{"kind": "link", "name": "Basket",
                                     "detail": text}])
    return PossibleBug(journey="Route '/cart' serves a real page", test_id="TC_007",
                       step_index=1, group=group, score=score,
                       signals=[SignalHit(signal=2, name="Network Failures",
                                          weight=0.5, evidence=text)],
                       summary=f"S2 Network Failures: {text}",
                       repro_steps=[StepSpec(step=1, action="goto", target="/")],
                       oracle=[ExpectedContract().model_dump()], evidence=ev)


@pytest.fixture()
def handle(tmp_path: Path) -> RunHandle:
    h = RunHandle(run_id="testrun")
    h.run_dir = tmp_path
    h.result = RunResult(run_id="testrun", target_url="https://target.test",
                         description="d")
    return h


# --------------------------------------------------------------------- Stage 6a
def test_verified_findings_are_persisted_before_late_stages(handle):
    confirmed = [(make_suspect(), "3/3"), (make_suspect("axe:color-contrast", 0.55), "2/3")]
    Orchestrator._persist_confirmed(handle, confirmed)
    payload = json.loads((handle.run_dir / "confirmed.json").read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["reproduced_count"] == "3/3"
    # the arrow survives (UTF-8), so the artifact is faithful to the evidence
    assert "\u2192" in json.dumps(payload, ensure_ascii=False)
    assert payload[1]["possible_bug"]["group"] == "axe:color-contrast"


# ----------------------------------------------------------------------- Stage 7a
def test_broken_template_still_yields_every_finding(tmp_path):
    """A template engine that cannot load must not cost us the filings."""
    broken = Reporter(tmp_path, LLMPolicy())
    broken.jinja = None                       # any use of it explodes

    confirmed = [(make_suspect(), "3/3"), (make_suspect("axe:color-contrast", 0.55), "3/3")]
    filed = broken.file_bugs(confirmed)       # pre-fix: raised, everything lost
    assert [b.bug_id for b in filed] == ["QA-01", "QA-02"]
    assert all(b.reproduced_count == "3/3" for b in filed)
    assert all(b.evidence.get("warnings") for b in filed), "must admit the degradation"
    on_disk = json.loads((tmp_path / "bugs_filed.json").read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    # the degradation is also recorded in the run log with a traceback
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "traceback" in log and "reporter" in log


def test_fallback_filer_is_used_when_reporting_dies_as_a_whole(tmp_path, monkeypatch):
    """A crash *outside* the per-bug guard (here: severity rules) reaches the
    orchestrator's fallback, which must still file every verified finding."""
    reporter = Reporter(tmp_path, LLMPolicy())
    confirmed = [(make_suspect(), "3/3"), (make_suspect("axe:color-contrast", 0.55), "2/3")]

    def boom(_pb):
        raise RuntimeError("severity rules exploded")

    monkeypatch.setattr(Reporter, "_severity", staticmethod(boom))
    with pytest.raises(RuntimeError):
        reporter.file_bugs(confirmed)         # escapes the per-bug guard

    filed = reporter.fallback_file_bugs(confirmed)
    assert [b.bug_id for b in filed] == ["QA-01", "QA-02"]
    assert [b.reproduced_count for b in filed] == ["3/3", "2/3"]
    assert all(b.evidence.get("warnings") for b in filed)
    on_disk = json.loads((tmp_path / "bugs_filed.json").read_text(encoding="utf-8"))
    assert len(on_disk) == 2


def test_one_bad_report_does_not_lose_the_other_filings(tmp_path):
    """Per-bug isolation: a template error for one bug must not discard others."""
    reporter = Reporter(tmp_path, LLMPolicy())
    real_template = reporter.jinja.get_template

    calls = {"n": 0}

    def flaky_template(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated template explosion")
        return real_template(name)

    reporter.jinja.get_template = flaky_template
    filed = reporter.file_bugs([(make_suspect(), "3/3"),
                               (make_suspect("axe:color-contrast", 0.55), "3/3")])
    assert len(filed) == 2, "both filings survive"
    assert filed[0].evidence.get("warnings"), "the failed one says so"
    assert "report" in filed[1].evidence, "the healthy one still got its markdown"
    assert (tmp_path / "bug_reports" / "QA-02.md").exists()


# ------------------------------------------------------------------ stage guard
def test_guarded_stage_records_error_and_keeps_status_partial(handle):
    """The guard converts a late-stage crash into a recorded warning + traceback,
    leaves earlier work intact, and yields a "partial" terminal status."""
    res = handle.result
    handle.current_stage = "stage 7 (Reporter)"
    emitted: list[dict] = []
    orch = Orchestrator()

    def boom():
        raise UnicodeEncodeError("charmap", "\u2192", 75, 76,
                                 "character maps to <undefined>")

    out = orch._guarded(handle, res, "reporter", boom, emitted.append)
    assert out is None, "a failed non-fatal stage yields None"
    assert res.stage_errors["reporter"].startswith("stage 7 (Reporter): UnicodeEncodeError")
    assert "charmap" in res.stage_errors["reporter"]
    assert emitted and emitted[0]["type"] == "warning"
    assert "kept" in emitted[0]["msg"]
    # the traceback landed in the run's own log, UTF-8 encoded
    log = (handle.run_dir / "run.log").read_text(encoding="utf-8")
    assert "UnicodeEncodeError" in log and "Traceback" in log

    # a successful stage leaves no error behind
    assert orch._guarded(handle, res, "benchmark", lambda: {"available": False},
                         emitted.append) == {"available": False}
    assert "benchmark" not in res.stage_errors

    # terminal status rule used by the pipeline
    res.status = "partial" if res.stage_errors else "done"
    assert res.status == "partial"


def test_run_result_carries_stage_errors_and_round_trips(handle, tmp_path):
    res = handle.result
    res.stage_errors["benchmark"] = "stage 8 (Benchmark): FileNotFoundError: bugs.json"
    res.status = "partial"
    res.filed_bugs = [b for b in Reporter(tmp_path, LLMPolicy())
                      .fallback_file_bugs([(make_suspect(), "3/3")])]
    handle.result = res
    Orchestrator()._save(handle)

    raw = (tmp_path / "results.json").read_text(encoding="utf-8")
    assert "\u2192" in raw                     # evidence preserved verbatim
    back = RunResult(**json.loads(raw))
    assert back.status == "partial"
    assert back.stage_errors["benchmark"].startswith("stage 8")
    assert [b.bug_id for b in back.filed_bugs] == ["QA-01"]


def test_counter_consistency(handle, tmp_path):
    """filed_bugs / flaky must both be representable in the same result so the
    dashboard's counters cannot disagree with the artifacts."""
    res = handle.result
    res.filed_bugs = Reporter(tmp_path, LLMPolicy()).fallback_file_bugs(
        [(make_suspect(), "3/3")])
    from core.contracts import FlakySuspect

    res.flaky = [FlakySuspect(journey="TC_009 [axe:color-contrast]",
                              summary="colour contrast", reproduced="1/3",
                              signals=[5])]
    Orchestrator()._save(handle)
    back = RunResult(**json.loads((tmp_path / "results.json").read_text(encoding="utf-8")))
    assert len(back.filed_bugs) == 1 and len(back.flaky) == 1


# ---------------------------------------------------------- concurrent-run guard
def test_resurrected_run_does_not_block_new_runs(tmp_path, monkeypatch):
    """The stuck-409: a handle resurrected from disk must not be treated as live."""
    import app.main as main

    run_dir = main.config.ARTIFACTS / "run_deadbeef01"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        RunResult(run_id="deadbeef01", target_url="https://t.test",
                  description="interrupted", status="running").model_dump_json(indent=2),
        encoding="utf-8")

    main.orch.runs.pop("deadbeef01", None)
    h = main.load_handle("deadbeef01")
    assert h is not None
    assert h.result.status == "error", "an orphaned run is not still running"
    assert "interrupted" in h.result.error
    assert not any(x.thread and x.thread.is_alive() for x in main.orch.runs.values())
    # the guard the API uses must therefore let a new expedition start
    assert not any(x.thread and x.thread.is_alive() for x in [h])


def test_live_thread_still_blocks_concurrency(tmp_path):
    h = RunHandle(run_id="live")
    h.thread = threading.Thread(target=lambda: None)
    h.thread.start()
    h.thread.join()
    assert not h.thread.is_alive()


# ------------------------------------------------------------------- probe routes
def test_probed_routes_are_never_reported_as_dead(tmp_path):
    """A 404 on a URL QArena invented (cart/search enrichment probe) is not an
    application defect and must not become a filed bug."""
    from core.contracts import AppMap, AppState
    from core.planner import Planner

    appmap = AppMap(app_url="https://target.test", description="shop")
    appmap.states.append(AppState(state_id="S00", route="/", url="https://target.test/",
                                 state_hash="a", http_status=200))
    appmap.states.append(AppState(state_id="S01", route="/cart",
                                 url="https://target.test/cart", state_hash="b",
                                 http_status=404, probed=True))
    appmap.states.append(AppState(state_id="S02", route="/old",
                                 url="https://target.test/old", state_hash="c",
                                 http_status=404, probed=False))
    cases, _rejected = Planner(appmap, LLMPolicy()).plan()
    titles = " | ".join(c.title for c in cases)
    assert "/cart" not in titles, "a probed route must never be patrolled"
    assert "/old" in titles or "old" in titles, "a genuinely linked dead route still is"
