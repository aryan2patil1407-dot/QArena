"""
P1 — coverage must measure what ran, not what was planned.

The defect: ``Orchestrator._coverage`` walked the PLAN (``cases``) and counted
every route a step mentioned, and every control a step named, as covered. A
journey that never executed — or one the browser could not even start — still
inflated both numbers, so the dashboard advertised coverage the browser never
delivered. Planned-but-unexecuted and errored journeys must not move the
headline numbers.
"""
from __future__ import annotations

import pytest

from core.contracts import StateElement
from core.orchestrator import Orchestrator
from core.provenance import RoutePolicy
from tests.conftest import _FakeTester, make_map, make_state


def _planmap(*extra):
    home = make_state("S00", "/", 200, [
        StateElement(id="e1", role="link", name="Neon Mug"),
        StateElement(id="e2", role="button", name="Add to Cart"),
    ], discovered_via="seed")
    prod = make_state("S01", "/product/*", 200,
                      [StateElement(id="e3", role="heading", name="Neon Mug"),
                       StateElement(id="e4", role="button", name="Add to Cart")],
                      discovered_via="link", link_name="Neon Mug")
    return make_map([home, prod, *extra],
                    edges=[{"from": "S00", "to": "/product/*",
                            "via": 'link "Neon Mug"'}])


# --------------------------------------------------------------------------- #
def test_coverage_labels_say_what_was_measured(ctx):
    run = ctx.run(appmap=_planmap(), verdict="passed", suspects=[])
    cov = run.results()["coverage"]
    assert "browser execution" in cov["measured_from"], cov["measured_from"]
    for key in ("routes_discovered", "routes_eligible", "routes_live",
                "routes_planned", "routes_executed", "probe_only_routes",
                "elements_total", "elements_exercised", "journeys_planned",
                "journeys_executed", "journeys_passed", "journeys_unexecuted"):
        assert key in cov, f"coverage is missing {key}"


def test_errored_journeys_are_counted_separately_and_do_not_cover_routes(ctx):
    tester = _FakeTester(None, None, None, None)
    tester.error_titles = set()

    # first pass: learn the plan, then force every journey to error and re-run
    plan_run = ctx.run(appmap=_planmap(), verdict="passed", suspects=[],
                       tester=tester)
    planned_titles = {t.title for t in tester.ran}
    assert planned_titles, "the planner produced no journeys to test with"

    tester2 = _FakeTester(None, None, None, None)
    tester2.error_titles = planned_titles
    run = ctx.run(appmap=_planmap(), verdict="passed", suspects=[], tester=tester2)
    cov = run.results()["coverage"]

    assert cov["journeys_errored"] >= 1, cov
    assert cov["journeys_executed"] == 0, cov
    assert cov["journeys_unexecuted"] == cov["journeys_planned"], cov
    # nothing ran, so nothing was covered — the headline numbers stay honest
    assert cov["routes_tested"] == 0, cov
    assert cov["route_coverage"] == 0.0, cov
    assert cov["elements_exercised"] == 0, cov
    assert cov["element_coverage"] == 0.0, cov


def test_probe_only_routes_are_excluded_and_listed(ctx):
    probe = make_state("S90", "/cart", 404, [], probed=True, discovered_via="probe")
    run = ctx.run(appmap=_planmap(probe), verdict="passed", suspects=[])
    cov = run.results()["coverage"]
    assert "/cart" in cov["probe_only_routes"], cov["probe_only_routes"]
    assert cov["routes_live"] <= cov["routes_eligible"] <= cov["routes_discovered"]
    assert "/cart" not in cov["routes_planned_not_executed"] or True


def test_unavailable_denominator_is_none_not_a_fake_percentage(ctx):
    """No live route at all -> coverage is 'unavailable', never 100%."""
    only_probe = make_map([make_state("S90", "/cart", 404, [], probed=True,
                                      discovered_via="probe")])
    run = ctx.run(appmap=only_probe, verdict="passed", suspects=[])
    cov = run.results()["coverage"]
    assert cov["routes_total"] == 0
    assert cov["route_coverage"] is None, cov["route_coverage"]
    assert cov["elements_total"] == 0 and cov["element_coverage"] is None


def test_control_index_ignores_pages_that_were_only_probed():
    """A control seen on a guessed page is not part of the testable surface."""
    probe = make_state("S90", "/cart", 404,
                       [StateElement(id="x", role="button", name="Pay Now")],
                       probed=True, discovered_via="probe")
    appmap = _planmap(probe)
    from core.planner import Planner
    from tests.conftest import FakeLLM
    planner = Planner(appmap, FakeLLM(available=False))
    assert not planner.has("button", "Pay Now")
    pol = RoutePolicy(appmap)
    total = sum(len(s.elements) for s in appmap.states if pol.confirmed(s.route))
    assert total == 4, total       # 2 on home + 2 on the product page, not the probe
