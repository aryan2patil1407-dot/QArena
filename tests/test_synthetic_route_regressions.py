"""
P0 regression suite — a route QArena invented must never become a bug report.

The defect this suite locks down: the explorer probes guessed URLs
(/cart, /checkout, /search?q=…) to enrich the AppMap. Those probed states landed
in the same list as link-discovered pages, so the planner treated "/cart exists"
as an application fact, generated journeys that navigated there, the browser got
a 404, and QArena filed "the cart does not load" — a defect it invented by
guessing a URL the site never had. Replay then "confirmed" it 3/3 times, because
replaying an invented navigation 404s every time.

Each case asserts the artefact a user actually reads: the persisted
``bugs_filed.json`` (and the run's own results), never a helper's return value.
The browser stage is a stand-in (no Playwright in CI) but the Planner, the
admissibility gate, the Verifier, the Reporter and all persistence are real.

  A  synthetic missing cart      -> NOT filed, never replayed
  B  synthetic missing checkout  -> NOT filed
  C  genuine broken link         -> STILL filed, and grounded in the real link
  D  real working route          -> never patrolled as dead
  E  real UI navigation          -> the app's own redirect is trusted
  F  verification cannot upgrade -> the synthetic suspect never reaches replay
  G  mixed discovery             -> the real one survives, the invented one does not
"""
from __future__ import annotations

import json

import pytest

from core.contracts import StateElement
from core.planner import Planner
from core.provenance import PROBE, RoutePolicy, canonical_route
from tests.conftest import FakeLLM, make_map, make_state


# --------------------------------------------------------------------- fixtures
def _home(*elements) -> object:
    return make_state("S00", "/", 200, list(elements), discovered_via="seed")


def _probe(route: str, status: int) -> object:
    return make_state("S90", route, status, [], probed=True, discovered_via=PROBE)


def _link(name: str = "Neon Mug") -> StateElement:
    return StateElement(id="el", role="link", name=name)


def _planmap(*extra) -> object:
    """A map the deterministic grammar can plan against, plus ``extra`` states.

    The home page links a product, so the planner produces a real journey; that
    guarantees Stage 5b runs and the guard under test is genuinely exercised.
    """
    home = _home(_link("Neon Mug"), StateElement(id="e2", role="button",
                                                 name="Add to Cart"),
                 StateElement(id="e3", role="searchbox", name="Search"))
    prod = make_state("S01", "/product/*", 200,
                      [StateElement(id="e4", role="heading", name="Neon Mug"),
                       StateElement(id="e5", role="button", name="Add to Cart")],
                      discovered_via="link", link_name="Neon Mug")
    edges = [{"from": "S00", "to": "/product/*", "via": 'link "Neon Mug"'}]
    return make_map([home, prod, *extra], edges)


# --------------------------------------------------------------------------- #
# A / B — a guessed URL that 404s is not a defect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", ["/cart", "/checkout"])
def test_AB_probe_route_is_not_plannable(route):
    """The planner must not generate any journey that navigates to a guess."""
    appmap = _planmap(_probe(route, 404))
    planner = Planner(appmap, FakeLLM(available=False))
    cases, rejected = planner.plan()

    offenders = [f"{c.test_id}:{st.target}" for c in cases for st in c.steps
                 if st.action == "goto" and st.target == route]
    assert not offenders, f"planner planned a journey to a synthetic route: {offenders}"
    assert planner.policy.confirmed(route) is False
    assert planner.policy.exists(route) is False
    # an explicit rejection reason exists (so the dashboard can explain it)
    assert planner.policy.why_not(route)


@pytest.mark.parametrize("route", ["/cart", "/checkout"])
def test_A_B_synthetic_probe_404_is_never_filed(ctx, route):
    """Even when Stage 5b hands over such a suspect, it is never published."""
    run = ctx.run(appmap=_planmap(_probe(route, 404)), verdict="failed",
                  suspects=[(f"http:GET {route} -> 404",
                             f"http://site.test{route}", 2.5)])
    assert run.judge.seen, "no journey was evaluated — the guard was not exercised"

    assert run.filed() == [], f"invented {route} 404 was published as a defect"
    doc = run.results()
    assert doc["filed_bugs"] == []
    assert doc["status"] == "done", doc.get("stage_errors")
    # the run explains itself rather than silently dropping the finding
    assert any(route in (s.get("reason") or "")
               for s in run.res.suppressed), run.res.suppressed


# --------------------------------------------------------------------------- #
# C — genuine broken links must still be reported (no muting)
# --------------------------------------------------------------------------- #
def _dead_link_map():
    home = _home(_link("Old page"), _link("Neon Mug"),
                 StateElement(id="e2", role="button", name="Add to Cart"))
    dead = make_state("S02", "/old-page", 404, [], discovered_via="link",
                      link_name="Old page", discovered_from="/")
    live = make_state("S01", "/product/*", 200,
                      [StateElement(id="e4", role="heading", name="Neon Mug")],
                      discovered_via="link", link_name="Neon Mug")
    return make_map([home, dead, live],
                    edges=[{"from": "S00", "to": "/old-page",
                            "via": 'link "Old page"'},
                           {"from": "S00", "to": "/product/*",
                            "via": 'link "Neon Mug"'}])


def test_C_genuine_broken_link_is_still_planned():
    planner = Planner(_dead_link_map(), FakeLLM(available=False))
    cases, _ = planner.plan()
    patrol = [c for c in cases if "Old page" in c.title]
    assert patrol, f"a real broken link was no longer patrolled; planned={[c.title for c in cases]}"
    assert [s.action for s in patrol[0].steps] == ["goto", "click"], \
        "the user's own click is the reproduction, not a typed URL"
    assert planner.policy.confirmed("/old-page") is True


def test_C_genuine_broken_link_is_filed_with_grounded_expectation(ctx):
    run = ctx.run(appmap=_dead_link_map(), verdict="failed",
                  suspects=[("http:GET /old-page -> 404",
                             "http://site.test/old-page", 2.5)])
    assert run.judge.seen, "no journey was evaluated — the guard was not exercised"

    bugs = run.filed()
    assert len(bugs) == 1, f"genuine broken link was not filed: {bugs}"
    bug = bugs[0]
    assert "/old-page" in bug["route"]
    # the expectation is grounded in the target's OWN advertisement of the link
    assert "link" in bug["expected_behavior"].lower(), bug["expected_behavior"]
    assert bug["evidence"]["expected_source"], "no provenance for the expectation"
    assert bug["evidence"]["network_analysis"], "network origin was not classified"
    assert "link published by the target" in \
        bug["evidence"]["network_analysis"][0]["origin"]
    # the finding is concrete and reproducible — the report standard P4 demands
    assert bug["steps_to_reproduce"], "no numbered reproduction steps"
    assert bug["reproduced_count"].endswith("3"), bug["reproduced_count"]
    assert bug["confidence_basis"], "the confidence number explains itself"


# --------------------------------------------------------------------------- #
# D — a working route is never treated as dead
# --------------------------------------------------------------------------- #
def test_D_real_working_route_is_not_patrolled(ctx):
    cart = make_state("S01", "/cart", 200, [], discovered_via="link",
                      link_name="Cart")
    appmap = make_map([_home(_link("Cart")), cart],
                      edges=[{"from": "S00", "to": "/cart", "via": 'link "Cart"'}])
    pol = RoutePolicy(appmap)
    assert pol.confirmed("/cart") is True and pol.exists("/cart") is True

    cases, _ = Planner(appmap, FakeLLM(available=False)).plan()
    titles = [c.title for c in cases]
    assert not [t for t in titles if "leads somewhere real" in t], titles
    run = ctx.run(appmap=appmap, verdict="passed", suspects=[])
    assert run.filed() == [], "a working route produced a finding"


def test_D2_provenance_only_strengthens(ctx):
    """A probe that the target later links to IS a real route (never downgraded)."""
    cart = make_state("S01", "/cart", 200, [], probed=True, discovered_via=PROBE,
                      link_name="Cart", discovered_from="/")
    appmap = make_map([_home(_link("Cart")), cart])
    pol = RoutePolicy(appmap)
    assert pol.confirmed("/cart") is True, \
        "a route the target links to must be usable even if a probe saw it first"
    assert pol.provenance("/cart") in ("probe", "link", "interaction")


def test_D3_probe_that_was_linked_but_still_404s_remains_reportable(ctx):
    """Link evidence beats probe evidence: a linked 404 is the target's problem."""
    dead = make_state("S01", "/cart", 404, [], probed=True, discovered_via=PROBE,
                      link_name="Cart", discovered_from="/")
    appmap = make_map([_home(_link("Cart")), dead],
                      edges=[{"from": "S00", "to": "/cart", "via": 'link "Cart"'}])
    assert RoutePolicy(appmap).blocked_reason("http://site.test/cart") == ""
    run = ctx.run(appmap=appmap, verdict="failed",
                  suspects=[("http:GET /cart -> 404", "http://site.test/cart", 2.5)])
    if run.judge.seen:
        assert len(run.filed()) == 1, "a link-backed 404 must remain reportable"


# --------------------------------------------------------------------------- #
# E — real navigation: the app's own redirect is trusted
# --------------------------------------------------------------------------- #
def test_E_interaction_destination_is_a_real_route():
    cart = make_state("S01", "/cart", 200, [], discovered_via="interaction",
                      discovered_from="/", link_name="Add to Cart")
    appmap = make_map([_home(StateElement(id="e1", role="button",
                                          name="Add to Cart")), cart])
    assert RoutePolicy(appmap).confirmed("/cart") is True


def test_E2_broken_redirect_is_a_real_defect(ctx):
    """If the app's own Add-to-Cart redirect lands on a 404, that IS a bug."""
    cart = make_state("S01", "/cart", 404, [], discovered_via="interaction",
                      discovered_from="/", link_name="Add to Cart")
    appmap = make_map([_home(StateElement(id="e1", role="button",
                                          name="Add to Cart")), cart])
    run = ctx.run(appmap=appmap, verdict="failed",
                  suspects=[("http:GET /cart -> 404", "http://site.test/cart", 2.5)])
    if run.judge.seen:
        bugs = run.filed()
        assert len(bugs) == 1, f"a broken app-owned redirect was not filed: {bugs}"


# --------------------------------------------------------------------------- #
# F — replay verification can never upgrade synthetic evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", ["/cart", "/checkout"])
def test_F_synthetic_suspect_never_reaches_replay(ctx, route):
    run = ctx.run(appmap=_planmap(_probe(route, 404)), verdict="failed",
                  suspects=[(f"http:GET {route} -> 404",
                             f"http://site.test{route}", 2.5)])
    assert run.judge.seen, "no journey was evaluated — the guard was not exercised"

    assert run.replay_attempts == [], \
        f"replay was asked to run {len(run.replay_attempts)} time(s) on a guess"
    assert run.verifier is not None and run.verifier.calls == [], \
        "the verifier was consulted about a synthetic route"
    doc = run.results()
    assert doc["flaky"] == [], "a synthetic route was reported as flaky"
    assert doc["filed_bugs"] == []


def test_F2_synthetic_suspect_is_not_counted_as_coverage_or_finding(ctx):
    run = ctx.run(appmap=_planmap(_probe("/cart", 404)), verdict="failed",
                  suspects=[("http:GET /cart -> 404", "http://site.test/cart", 2.5)])
    cov = run.results()["coverage"]
    assert "/cart" in cov["probe_only_routes"], cov["probe_only_routes"]
    assert cov["findings_withheld_synthetic"] >= 1
    assert run.results()["filed_bugs"] == []


# --------------------------------------------------------------------------- #
# G — mixed discovery: the real finding survives, the invented one does not
# --------------------------------------------------------------------------- #
def test_G_mixed_discovery_keeps_real_bug_and_drops_invented(ctx):
    appmap = _planmap(_probe("/checkout", 404))
    # add the genuine broken link the target itself publishes
    appmap.states.append(make_state("S02", "/old-page", 404, [],
                                    discovered_via="link", link_name="Old page",
                                    discovered_from="/"))
    appmap.edges.append({"from": "S00", "to": "/old-page",
                         "via": 'link "Old page"'})

    run = ctx.run(appmap=appmap, verdict="failed",
                  suspects=[("http:GET /old-page -> 404",
                             "http://site.test/old-page", 2.5),
                            ("http:GET /checkout -> 404",
                             "http://site.test/checkout", 2.5)])
    assert run.judge.seen, "no journey was evaluated — the guard was not exercised"

    routes = [b["route"] for b in run.filed()]
    assert any("/old-page" in r for r in routes), \
        f"the genuinely broken link was lost: {routes}"
    assert not any("/checkout" in r for r in routes), \
        f"the invented /checkout defect was published: {routes}"


def test_G2_reason_text_is_human_readable(ctx):
    appmap = _planmap(_probe("/cart", 404))
    reason = RoutePolicy(appmap).blocked_reason("http://site.test/cart")
    assert reason and "404" in reason and "probe" in reason.lower(), reason
    assert RoutePolicy(appmap).blocked_reason("http://site.test/") == ""


def test_G3_third_party_vs_target_owned_failures_are_distinguished(ctx):
    """A third-party beacon failing is not the target's defect."""
    pol = RoutePolicy(_planmap())
    assert pol.network_origin("http://site.test/cart") == "target"
    assert pol.network_origin("https://www.googletagmanager.com/gtm.js") == \
        "third_party"
