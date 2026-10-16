"""Regression tests for the real-browser expedition fix set.

Each test pins one defect that a real run against a seeded fixture exposed:
  1. the planner fabricated target data ("Quantum Mug") and badge oracles
  2. the login journey demanded an invented "Log out" control
  3. steps were validated globally, not against the page they run on
  4. per-step evidence was sliced before the browser events were dispatched
  5. the reporter merged/fused unrelated signal families and mislabelled titles
  6. exported suites could not use an external browser
  7. the landing page linked to a hard-coded, long-gone run
  8. GROQ might be configured but was not recognised as an LLM key
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.contracts import (AppMap, AppState, ExpectedContract, PossibleBug,
                            SignalHit, StateElement, StepEvidence, StepSpec,
                            TestCase)
from core.exporter import Exporter
from core.llm import KEY_ENV_NAMES, LLMPolicy
from core.planner import Planner
from core.reporter import Reporter
from core.tester import Tester

Tester.__test__ = False  # imported class, not a test container

TARGET = "http://t.test"


# --------------------------------------------------------------------- helpers
def _map(states=None, prices=None) -> AppMap:
    am = AppMap(app_url=TARGET, description="shop")
    am.states = states or []
    am.catalog_prices = prices or {}
    return am


def _state(sid, route, elements, *, status=200, probed=False, badge=False) -> AppState:
    return AppState(state_id=sid, route=route, url=f"{TARGET}{route}".replace("/*", "/1"),
                    state_hash=f"h{sid}", http_status=status, probed=probed,
                    has_cart_badge=badge,
                    elements=[StateElement(id=f"{sid}e{i}", role=r, name=n)
                              for i, (r, n) in enumerate(elements)])


def _case(tid, steps, expected=None, title="case", jtype="happy_path") -> TestCase:
    return TestCase(test_id=tid, journey_type=jtype, title=title, steps=steps,
                    expected=expected or ExpectedContract())


_SIG_NAME = {1: "Console Errors", 2: "Network Failures", 3: "Broken Page Items",
             4: "Form Failures", 5: "Accessibility", 6: "Intent Check"}


def _bug(group, signal, weight, evidence, *, axe=None, url_after=TARGET + "/") -> PossibleBug:
    return PossibleBug(
        journey="J", test_id="TC_003", group=group, step_index=0,
        signals=[SignalHit(signal=signal, name=_SIG_NAME.get(signal, "signal"),
                           weight=weight, evidence=evidence)],
        score=weight, summary=f"S{signal} {evidence}",
        repro_steps=[StepSpec(step=1, action="goto", target="/")],
        evidence=StepEvidence(step=1, action="goto", target="/",
                              url_before=TARGET + "/", url_after=url_after,
                              axe_violations=axe or []))


# -------------------------------------------------------- 1) planner grounding
def test_planner_never_invents_a_product_name():
    am = _map([_state("S00", "/", [("button", "Add to Cart")])])
    planner = Planner(am, LLMPolicy())
    assert planner.products == []          # pre-fix: ["Quantum Mug"]
    cases, _ = planner.plan()
    assert "Quantum Mug" not in " ".join(t.title for t in cases)


def test_badge_oracles_only_when_the_app_has_a_badge():
    without = Planner(_map([_state("S00", "/", [("button", "Add to Cart")])]), LLMPolicy())
    assert [t for t in without.grammar_plan()
            if t.expected.badge_equals is not None] == []

    with_badge = Planner(_map([_state("S00", "/", [("button", "Add to Cart")],
                                      badge=True)]), LLMPolicy())
    badge_cases = [t for t in with_badge.grammar_plan()
                   if t.expected.badge_equals is not None]
    assert badge_cases, "apps that really have a badge must still get badge journeys"


# ------------------------------------------------------------ 2) login oracle
def _login_map(*extra):
    return _map([
        _state("S00", "/", [("link", "Login")]),
        _state("S01", "/login", [("textbox", "Email"), ("textbox", "Password"),
                                 ("button", "Log in"), *extra]),
    ])


def _login_case(am):
    cases = Planner(am, LLMPolicy()).grammar_plan()
    return next(t for t in cases if t.title == "Log in with valid credentials")


def test_login_oracle_is_grounded_not_hard_coded():
    case = _login_case(_login_map())
    assert "Log out" not in case.expected.text_required     # pre-fix: hard-coded
    assert case.expected.disallowed_url == "/login"


def test_login_oracle_uses_the_apps_own_logout_control():
    case = _login_case(_login_map(("button", "Log out")))
    assert case.expected.text_required == ["Log out"]


# --------------------------------------------------- 3) route-scoped validation
def test_invented_route_element_pair_is_rejected():
    am = _map([_state("S00", "/product/*", [("button", "Add to Cart")]),
               _state("S01", "/", [("heading", "Welcome")])])
    planner = Planner(am, LLMPolicy())
    good = _case("T1", [StepSpec(step=1, action="goto", target="/product/1"),
                        StepSpec(step=2, action="click", role="button", name="Add to Cart")])
    bad = _case("T2", [StepSpec(step=1, action="goto", target="/"),
                       StepSpec(step=2, action="click", role="button", name="Add to Cart")])
    valid, rejected = planner.reject_invented([good, bad])
    assert [t.test_id for t in valid] == ["T1"]
    assert rejected and "is not on /" in rejected[0]["invented_steps"][0]


# ------------------------------------------------------ 4) evidence capture
class _Resp:
    def __init__(self, url, status, method="GET"):
        self.url, self.status = url, status
        self.request = SimpleNamespace(method=method)


class _PumpPage:
    """Fake page whose wait_for_timeout() dispatches queued browser events —
    exactly like Playwright's sync API, which only pumps during calls."""

    def __init__(self):
        self.url = TARGET + "/"
        self._handlers: dict[str, list] = {}
        self._late: list = []

    def on(self, event, handler):
        self._handlers.setdefault(event, []).append(handler)

    def wait_for_timeout(self, ms):
        for item in list(self._late):
            self._late.remove(item)
            for h in self._handlers.get(item["event"], []):
                h(item["payload"])

    def screenshot(self, path=None):
        Path(path).write_bytes(b"")


class _FakeSandbox:
    """Just enough sandbox for run_test(): a context + the seed hook."""

    target_url = TARGET
    base_host = "t.test"

    def __init__(self, page):
        self._page = page

    def new_context(self, **kw):
        ctx = SimpleNamespace()
        ctx.new_page = lambda: self._page
        ctx.close = lambda: None
        ctx.tracing = SimpleNamespace(start=lambda **k: None, stop=lambda **k: None)
        return ctx

    def seed_target(self):
        return False


def _seed_events(page, *, console_text, url, status):
    page._late += [
        {"event": "response", "payload": _Resp(url, status)},
        {"event": "console", "payload": SimpleNamespace(
            type="error", text=console_text, location={"url": url})},
    ]


@pytest.fixture()
def tester(tmp_path, monkeypatch):
    page = _PumpPage()
    t = Tester(_FakeSandbox(page), _map([]), tmp_path, emit=lambda e: None)
    t._fake_page = page
    monkeypatch.setattr(t, "_dom_scan", lambda page: [])
    monkeypatch.setattr(t, "_axe_scan", lambda page: [])
    monkeypatch.setattr(t, "_intent_scan", lambda page, ex: [])
    monkeypatch.setattr(t, "_eval_contract", lambda page, ex, is_final=False: [])
    return t


def test_late_browser_events_are_attributed_to_the_step_that_caused_them(
        tester, monkeypatch, tmp_path):
    """The browser reports the image 404 just after the step's action."""
    page = tester._fake_page

    def fake_do_step(page_, spec, tc):
        _seed_events(page_, console_text="Failed to load resource",
                     url=TARGET + "/static/missing.png", status=404)
        return True, ""

    monkeypatch.setattr(tester, "_do_step", fake_do_step)
    tc = _case("T", [StepSpec(step=1, action="goto", target="/")])
    _, evidences = tester.run_test(tc, tracing=False)

    assert evidences[0].http_failures and \
        evidences[0].http_failures[0]["status"] == 404     # pre-fix: []
    assert evidences[0].console_errors == \
        ["Failed to load resource {/static/missing.png}"]


def test_events_arriving_after_the_last_step_are_not_dropped(tester, monkeypatch, tmp_path):
    page = tester._fake_page
    # consume nothing in _capture: the events can only land via the end-of-test flush
    monkeypatch.setattr(tester, "_capture",
                        lambda *a, **k: setattr(a[1], "url_after", page.url))

    def fake_do_step(page_, spec, tc):
        page_._late.append({"event": "response",
                            "payload": _Resp(TARGET + "/late", 500)})
        return True, ""

    monkeypatch.setattr(tester, "_do_step", fake_do_step)
    tc = _case("T", [StepSpec(step=1, action="goto", target="/")])
    _, evidences = tester.run_test(tc, tracing=False)
    assert evidences[-1].http_failures, "a failure reported after the last step must survive"
    assert evidences[-1].http_failures[0]["status"] == 500


# ---------------------------------------------------------- 5) reporter precision
def test_distinct_signal_families_are_filed_separately(tmp_path):
    console = _bug("console:[uncaught] boom", 1, 0.6, "[uncaught] boom")
    contrast = _bug("axe:color-contrast", 5, 0.55, "color-contrast on footer > p",
                    axe=[{"id": "color-contrast", "impact": "serious",
                          "target": "footer > p"}])
    filed = Reporter(tmp_path, LLMPolicy()).file_bugs([(console, "3/3"), (contrast, "3/3")])
    assert len(filed) == 2          # pre-fix: one merged, mislabelled report
    titles = " | ".join(b.title for b in filed)
    assert "Console Errors" in titles and "color-contrast" in titles


def test_console_error_is_not_labelled_with_an_axe_selector(tmp_path):
    bug = _bug("console:[uncaught] boom", 1, 0.6, "[uncaught] boom",
               axe=[{"id": "color-contrast", "impact": "serious",
                     "target": "footer > p"}])
    filed = Reporter(tmp_path, LLMPolicy()).file_bugs([(bug, "3/3")])
    assert filed[0].element_name != "@footer > p"
    assert "@footer > p" not in filed[0].title


# ------------------------------------------------------------- 6) export support
def test_exported_suite_ships_the_external_browser_override(tmp_path):
    ex = Exporter(tmp_path, TARGET)
    files = ex.export([_case("TC_010", [StepSpec(step=1, action="goto", target="/")])])
    assert files
    conftest = (tmp_path / "exported_tests" / "conftest.py").read_text(encoding="utf-8")
    assert "QARENA_CHROMIUM_PATH" in conftest
    assert "browser_type_launch_args" in conftest


# pytest's own JUnit output: classname + parametrised name, NO file attribute
_JUNIT_NO_FILE_ATTR = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0"
 skipped="0" tests="2" time="0.8"><testcase classname="artifacts.run_x.exported_tests.test_tc_010_route_ok"
 name="test_tc_010_route_ok[chromium]" time="0.5" /><testcase classname="artifacts.run_x.exported_tests.test_tc_011_route_bad"
 name="test_tc_011_route_bad[chromium]" time="0.3"><failure message="boom">traceback</failure></testcase>
</testsuite></testsuites>"""


def test_junit_without_a_file_attribute_is_matched_by_classname(tmp_path):
    ex = Exporter(tmp_path, TARGET)
    report = tmp_path / "junit.xml"
    report.write_text(_JUNIT_NO_FILE_ATTR, encoding="utf-8")
    passed, note = ex._read_junit(report)
    # pre-fix: empty set ("0 file(s) executed") -> every green test dropped
    assert passed == {"test_tc_010_route_ok.py"}
    assert "2 file" in note


def test_verified_files_are_kept_runnable_and_failures_are_dropped(tmp_path, monkeypatch):
    ex = Exporter(tmp_path, TARGET)
    names = ["test_tc_010_route_ok.py", "test_tc_011_route_bad.py"]
    for n in names:
        (ex.out_dir / n).write_text("# exported\n", encoding="utf-8")
    monkeypatch.setattr("core.exporter.subprocess.run",
                        lambda *a, **k: SimpleNamespace(
                            returncode=1, stdout="", stderr=""))

    def fake_read(report):
        return {"test_tc_010_route_ok.py"}, "JUnit report parsed: 2 file(s) executed"

    monkeypatch.setattr(ex, "_read_junit", fake_read)
    keep, _note = ex.self_verify(names)
    assert keep == ["test_tc_010_route_ok.py"]
    assert (ex.out_dir / "test_tc_010_route_ok.py").exists()          # pre-fix: renamed
    assert not (ex.out_dir / "test_tc_010_route_ok.py.dropped").exists()
    assert (ex.out_dir / "test_tc_011_route_bad.py.dropped").exists()


# ------------------------------------------------------------------ 7) landing
def test_landing_page_links_only_to_real_runs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setattr(main.config, "ARTIFACTS", tmp_path)
    client = TestClient(main.app)

    html = client.get("/").text
    assert "cli1791734448" not in html        # pre-fix: hard-coded dead run id
    assert 'href="/runs/' not in html         # no completed run -> no link at all

    # a run that errored (e.g. a test fixture stub) is not a "real report"
    stub = tmp_path / "run_deadbeef01"
    stub.mkdir()
    (stub / "results.json").write_text(
        '{"run_id": "deadbeef01", "status": "error"}', encoding="utf-8")
    assert 'href="/runs/' not in client.get("/").text

    done = tmp_path / "run_cliTEST1234"
    done.mkdir()
    (done / "results.json").write_text(
        '{"run_id": "cliTEST1234", "status": "done"}', encoding="utf-8")
    assert "/runs/cliTEST1234" in client.get("/").text


# --------------------------------------------------------------------- 8) LLM
def test_groq_api_key_is_recognised(monkeypatch):
    assert "GROQ_API_KEY" in KEY_ENV_NAMES
    for name in KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert LLMPolicy().available is False
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used-here")
    assert LLMPolicy().available is True
