"""
P3 — QArena must not assume every target is an online shop.

The failure mode: the deterministic grammar was written around an e-commerce
demo, so a documentation site, a dashboard or an internal tool got journeys like
"Checkout blocks empty cart", "Promo SAVE10 applies a 10% discount" and
"Route '/checkout' serves a real page" — tests for features the site never
claimed to have. Their failures were then filed as defects.

The rule: a journey may only exercise functionality the target itself exposes
(a real page it serves, a real control it shows). Negative and boundary testing
stays — but only against features that exist.
"""
from __future__ import annotations

from core.contracts import StateElement
from core.planner import Planner
from tests.conftest import FakeLLM, make_map, make_state


def _docs_site():
    """"A documentation site: search, an index, articles — no shop at all."""
    home = make_state("S00", "/", 200, [
        StateElement(id="e1", role="searchbox", name="Search"),
        StateElement(id="e2", role="link", name="Getting Started"),
        StateElement(id="e3", role="link", name="API Reference"),
        StateElement(id="e4", role="link", name="Changelog"),
    ], discovered_via="seed")
    doc = make_state("S01", "/docs/getting-started", 200, [
        StateElement(id="e5", role="heading", name="Getting Started"),
        StateElement(id="e6", role="link", name="Install"),
    ], discovered_via="link", link_name="Getting Started")
    return make_map([home, doc], edges=[
        {"from": "S00", "to": "/docs/getting-started",
         "via": 'link "Getting Started"'},
        {"from": "S00", "to": "/docs/api", "via": 'link "API Reference"'},
    ])


def test_no_cart_or_checkout_journeys_on_a_non_shop_site():
    cases, _ = Planner(_docs_site(), FakeLLM(available=False)).plan()
    blob = " ".join(f"{c.title} {' '.join(s.target or s.name or '' for s in c.steps)}"
                    for c in cases).lower()
    for forbidden in ("cart", "checkout", "promo", "pay now", "add to cart"):
        assert forbidden not in blob, f"assumed a shop: '{forbidden}' appears in {blob}"


def test_no_negative_cart_test_without_a_cart():
    cases, _ = Planner(_docs_site(), FakeLLM(available=False)).plan()
    assert not [c for c in cases if "cart" in c.title.lower()]
    assert not [c for c in cases if "quantity" in c.title.lower()]


def test_boundary_testing_still_happens_when_the_feature_exists():
    """"Search exists, so searching for nothing is a fair boundary test."""
    home = make_state("S00", "/", 200, [
        StateElement(id="e1", role="searchbox", name="Search"),
        StateElement(id="e2", role="link", name="Widget One"),
    ], discovered_via="seed")
    prod = make_state("S01", "/product/*", 200, [
        StateElement(id="e3", role="heading", name="Widget One"),
        StateElement(id="e4", role="textbox", name="Promo code"),
        StateElement(id="e5", role="button", name="Apply"),
    ], discovered_via="link", link_name="Widget One")
    # the catalog name is scraped from the target, so it is legitimate to use
    cases, _ = Planner(make_map([home, prod], catalog={"Widget One": 1299}),
                       FakeLLM(available=False)).plan()
    # the promo control only exists on a page that is NOT the cart: the promo
    # journey must not fire, because there is no cart to apply it to
    blob = " ".join(c.title for c in cases).lower()
    assert "promo save10" not in blob, blob
    # ...but the search journey (a feature that does exist) is planned
    assert any("search" in c.title.lower() for c in cases), [c.title for c in cases]


def test_route_scoped_controls_are_not_used_off_route():
    """A control that lives on page B cannot be clicked on page A."""
    home = make_state("S00", "/", 200,
                      [StateElement(id="e1", role="link", name="Cart")],
                      discovered_via="seed")
    cart = make_state("S01", "/cart", 200, [
        StateElement(id="e2", role="spinbutton", name="Quantity"),
        StateElement(id="e3", role="button", name="Update"),
    ], discovered_via="link", link_name="Cart")
    appmap = make_map([home, cart],
                      edges=[{"from": "S00", "to": "/cart",
                              "via": 'link "Cart"'}])
    planner = Planner(appmap, FakeLLM(available=False))
    cases, rejected = planner.plan()
    for c in cases:
        for i, st in enumerate(c.steps):
            if st.action == "click" and st.name == "Update":
                assert c.steps[i - 1].target == "/cart" or \
                    any(s.action == "goto" and s.target == "/cart"
                        for s in c.steps[:i]), \
                    f"{c.title} clicks a cart-only control from the wrong page"


def test_llm_proposals_are_gated_by_the_same_rules():
    """A model may be creative about scenarios, never about pages/controls."""
    from core.contracts import ExpectedContract, StepSpec, TestCase

    class InventingLLM(FakeLLM):
        def __init__(self):
            super().__init__(available=True)

    llm = InventingLLM()
    planner = Planner(_docs_site(), llm)
    invented = TestCase(
        test_id="LLM_001", journey_type="happy_path",
        title="Buy the widget with a promo code",
        steps=[StepSpec(step=1, action="goto", target="/"),
               StepSpec(step=2, action="goto", target="/cart"),
               StepSpec(step=3, action="goto", target="/checkout")],
        expected=ExpectedContract())
    grounded = TestCase(
        test_id="LLM_002", journey_type="happy_path",
        title="Read the getting-started guide",
        steps=[StepSpec(step=1, action="goto", target="/"),
               StepSpec(step=2, action="goto", target="/docs/getting-started")],
        expected=ExpectedContract())
    valid, rejected = planner.reject_invented([invented, grounded])
    assert [c.test_id for c in valid] == ["LLM_002"], \
        "the LLM's invented shop pages were not filtered"
    assert any(r["test_id"] == "LLM_001" for r in rejected)
    assert rejected[0]["reason"], "the refusal explains itself"
    assert any("/cart" in s for s in rejected[0]["invented_steps"])


def test_non_existent_control_is_reported_as_ungrounded_not_as_a_bug():
    """A missing control is a planning fact, never a filed defect."""
    cases, rejected = Planner(_docs_site(), FakeLLM(available=False)).plan()
    reasons = {r.get("reason") for r in rejected}
    assert reasons <= {"synthetic_route", "missing_control", "wrong_page", "invented"}
