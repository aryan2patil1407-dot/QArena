"""
P4 — a filed finding must be complete, concrete and honestly grounded.

The defect: ``Reporter._expect_actual`` returned ONE hard-coded sentence for
every finding in the report — "valid inputs are accepted, invalid inputs are
rejected…" — even for a broken link, a console error or a contrast violation.
That is the report inventing a requirement the target never stated, which is
exactly the "professional-looking report describing something QArena made up"
failure mode.

The rule now: an expectation is quoted from (1) the target's own behaviour, or
(2) the explicit contract the journey was executed against, or (3) the report
says plainly that no requirement was supplied. A raw 404 is evidence — it is not
automatically a defect, and a third-party request failing is not the target's
fault.
"""
from __future__ import annotations

import json

import pytest

from core.contracts import (ExpectedContract, PossibleBug, SignalHit,
                            StepEvidence, StepSpec)
from core.llm import LLMPolicy
from core.reporter import Reporter
from tests.conftest import make_map, make_state


# ------------------------------------------------------------------ builders
def _bug(group: str, url: str, *, oracle: list | None = None,
         summary: str = "observed failure") -> PossibleBug:
    return PossibleBug(
        journey="checkout journey", test_id="TC_001", group=group, step_index=1,
        score=2.5, summary=summary,
        signals=[SignalHit(signal=2, name="http", weight=2.0, evidence="404")],
        repro_steps=[StepSpec(step=1, action="goto", target="/")],
        oracle=oracle or [],
        evidence=StepEvidence(step=1, action="goto", target="/", status="failed",
                              url_before="http://site.test/",
                              url_after=url,
                              http_failures=[{"method": "GET", "url": url,
                                              "status": 404}]))


def _map_with_dead_link():
    home = make_state("S00", "/", 200, [], discovered_via="seed")
    dead = make_state("S01", "/old-page", 404, [], discovered_via="link",
                      link_name="Old page", discovered_from="/")
    return make_map([home, dead], edges=[{"from": "S00", "to": "/old-page",
                                          "via": 'link "Old page"'}],
                    app_url="http://site.test")


# ------------------------------------------------------------------- grounding
def test_broken_link_expectation_quotes_the_targets_own_link(tmp_path):
    rep = Reporter(tmp_path, LLMPolicy(), _map_with_dead_link())
    filed = rep.file_bugs([(_bug("http:GET /old-page -> 404",
                                 "http://site.test/old-page"), "3/3")])
    assert len(filed) == 1
    bug = filed[0]
    assert "Old page" in bug.expected_behavior
    assert "link" in bug.expected_behavior.lower()
    assert "target" in bug.evidence["expected_source"]
    assert bug.evidence["network_analysis"][0]["origin"] == \
        "dead link published by the target"


def test_third_party_failure_is_not_reported_as_the_targets_defect(tmp_path):
    rep = Reporter(tmp_path, LLMPolicy(), _map_with_dead_link())
    filed = rep.file_bugs([(_bug("console:TypeError: x is undefined",
                                 "http://site.test/"), "3/3")])
    bug = filed[0]
    assert "no explicit requirement was supplied" in \
        bug.evidence["expected_source"]
    assert bug.evidence["expected_source"].endswith("flow") or \
        "requirement" in bug.evidence["expected_source"]


def test_explicit_contract_is_the_stated_expectation(tmp_path):
    contract = ExpectedContract(should_reject=True, error_text="between 1 and 10")
    rep = Reporter(tmp_path, LLMPolicy())
    filed = rep.file_bugs([(_bug("form:invalid input was silently accepted",
                                 "http://site.test/cart",
                                 oracle=[contract.model_dump()]), "3/3")])
    bug = filed[0]
    assert "between 1 and 10" in bug.expected_behavior
    assert "contract" in bug.evidence["expected_source"]


def test_money_contract_expectation_is_the_invariant(tmp_path):
    contract = ExpectedContract(money_non_negative=True)
    filed = Reporter(tmp_path, LLMPolicy()).file_bugs(
        [(_bug("form:money went negative", "http://site.test/cart",
               oracle=[contract.model_dump()]), "3/3")])
    assert "negative" in filed[0].expected_behavior.lower()


def test_no_requirement_is_never_invented(tmp_path):
    """With no contract and no link, the report says so instead of guessing."""
    filed = Reporter(tmp_path, LLMPolicy()).file_bugs(
        [(_bug("intent:price looks wrong", "http://site.test/cart"), "3/3")])
    src = filed[0].evidence["expected_source"]
    assert "no explicit requirement" in src
    assert "valid inputs are accepted" not in filed[0].expected_behavior


# ------------------------------------------------------------------ completeness
def test_every_filed_bug_carries_the_full_report_standard(tmp_path):
    rep = Reporter(tmp_path, LLMPolicy(), _map_with_dead_link())
    bug = rep.file_bugs([(_bug("http:GET /old-page -> 404",
                               "http://site.test/old-page"), "3/3")])[0]
    doc = json.loads(bug.model_dump_json())

    assert doc["title"], "factual title"
    assert doc["severity"] in ("Critical", "Major", "Minor"), "severity"
    assert doc["confidence_basis"], "justified confidence"
    assert doc["route"], "affected page/URL"
    assert doc["evidence"]["url"], "the URL at failure"
    assert doc["steps_to_reproduce"], "numbered reproduction"
    assert doc["expected_behavior"] and doc["actual_behavior"]
    assert doc["evidence"]["signals"], "evidence"
    assert doc["evidence"]["http_failures"], "console/HTTP failures"
    assert doc["evidence"]["screenshot"] is not None, "screenshot ref"
    assert doc["evidence"]["trace"] or doc["evidence"].get("trace") is None
    assert doc["reproduced_count"], "successful reproductions"
    assert doc["evidence"]["why_it_matters"], "why it matters"


def test_confidence_basis_names_the_signals_and_reproductions(tmp_path):
    filed = Reporter(tmp_path, LLMPolicy()).file_bugs(
        [(_bug("http:GET /x -> 500", "http://site.test/"), "2/3")])
    basis = filed[0].confidence_basis
    assert "2/3" in basis and "signal" in basis.lower()
    assert filed[0].evidence["reproductions"] == "2/3"


def test_report_markdown_includes_provenance_and_network_origin(tmp_path):
    rep = Reporter(tmp_path, LLMPolicy(), _map_with_dead_link())
    rep.file_bugs([(_bug("http:GET /old-page -> 404",
                         "http://site.test/old-page"), "3/3")])
    md = (tmp_path / "bug_reports" / "QA-01.md").read_text(encoding="utf-8")
    assert "Where this expectation comes from" in md
    assert "Why this is a problem" in md
    assert "broken navigation" in md or "dead end" in md
    # the raw fact is still present verbatim
    assert "Network: GET http://site.test/old-page → 404" in md


def test_third_party_network_failure_is_labelled_in_the_report(tmp_path):
    """A failing analytics beacon is reported as third-party, not as a defect."""
    home = make_state("S00", "/", 200, [], discovered_via="seed")
    appmap = make_map([home])
    ev = StepEvidence(step=1, action="goto", target="/", status="failed",
                      url_before="http://site.test/", url_after="http://site.test/",
                      http_failures=[{"method": "GET",
                                      "url": "https://www.googletagmanager.com/x.js",
                                      "status": 404}])
    pb = PossibleBug(journey="home", test_id="TC_9", group="http:GET / -> 404",
                     step_index=1, score=1.0, summary="third party 404",
                     signals=[SignalHit(signal=2, name="http", weight=1.0,
                                        evidence="404")],
                     repro_steps=[], evidence=ev)
    bug = Reporter(tmp_path, LLMPolicy(), appmap).file_bugs([(pb, "2/3")])[0]
    analysis = bug.evidence["network_analysis"][0]
    assert analysis["origin"] == "third_party"
    assert "not owned by the target" in analysis["note"]


# --------------------------------------------------- attribution & severity
def test_element_names_the_link_that_actually_failed(tmp_path):
    """A page can carry two broken links; the finding must name the right one."""
    home = make_state("S00", "/", 200, [], discovered_via="seed")
    dead = make_state("S01", "/old-page", 404, [], discovered_via="link",
                      link_name="Old Page", discovered_from="/")
    appmap = make_map([home, dead], edges=[{"from": "S00", "to": "/old-page",
                                            "via": 'link "Old Page"'}],
                      app_url="http://site.test")
    pb = _bug("http:GET /old-page -> 404", "http://site.test/old-page")
    # the page also hosts an unrelated broken link, listed first
    pb.evidence.broken_items = [
        {"kind": "link", "name": "Reports", "detail": "/reports → 500"},
        {"kind": "link", "name": "Old Page", "detail": "/old-page → 404"}]
    bug = Reporter(tmp_path, LLMPolicy(), appmap).file_bugs([(pb, "3/3")])[0]
    assert bug.element_name == "Old Page", bug.element_name
    assert "Old Page" in bug.title


def test_server_error_on_a_linked_page_is_major_not_minor(tmp_path):
    home = make_state("S00", "/", 200, [], discovered_via="seed")
    rep = Reporter(tmp_path, LLMPolicy(), make_map([home]))
    ev = StepEvidence(step=1, action="goto", target="/", status="failed",
                      url_before="http://site.test/", url_after="http://site.test/reports",
                      http_failures=[{"method": "GET",
                                      "url": "http://site.test/reports",
                                      "status": 500}])
    pb = PossibleBug(journey="link journey", test_id="TC_7",
                     group="http:GET /reports -> 500", step_index=0, score=2.5,
                     summary="500 on /reports",
                     signals=[SignalHit(signal=2, name="http", weight=2.0,
                                        evidence="500")],
                     repro_steps=[StepSpec(step=1, action="goto", target="/")],
                     evidence=ev)
    bug = rep.file_bugs([(pb, "3/3")])[0]
    assert bug.severity == "Major", bug.severity
    assert "failure page" in bug.evidence["why_it_matters"]
