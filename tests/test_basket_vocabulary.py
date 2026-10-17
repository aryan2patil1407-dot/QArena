"""
The shop that calls its cart a "basket".

Found by running QArena against a real sweet-shop style target: the AppMap was
perfect (it discovered /basket through the "Basket" link, and /checkout through
"Proceed to checkout"), but the planner generated no basket journeys, because it
looked for a route containing the string "cart". On such a site QArena reported
almost nothing — which reads to a user as "the tool doesn't work".

Rules these tests pin down:

* a feature is recognised by the TARGET's vocabulary (route path, link label) or
  by its structure (a quantity control plus an update control), never by
  QArena's own naming habit;
* a page-level console check does not require an "Add to Cart" button that the
  page never had (that coupling hid an uncaught TypeError);
* QArena never invents a valid promo code, nor a specific validation message.
"""
from __future__ import annotations

from core.contracts import StateElement
from core.planner import Planner
from tests.conftest import FakeLLM, make_map, make_state


def _shop(basket_path="/basket", basket_label="Basket", with_cta=False):
    home = make_state("S00", "/", 200, [
        StateElement(id="e1", role="searchbox", name="Search"),
        StateElement(id="e2", role="link", name=basket_label),
        StateElement(id="e3", role="link", name="Sherbet Lemon"),
    ], discovered_via="seed")
    basket = make_state("S01", basket_path, 200, [
        StateElement(id="b1", role="spinbutton", name="Quantity"),
        StateElement(id="b2", role="button", name="Update"),
        StateElement(id="b3", role="textbox", name="Promo code"),
        StateElement(id="b4", role="button", name="Apply"),
        StateElement(id="b5", role="link", name="Proceed to checkout"),
    ], discovered_via="link", link_name=basket_label)
    checkout = make_state("S02", "/checkout", 200, [
        StateElement(id="c1", role="textbox", name="Full name"),
        StateElement(id="c2", role="textbox", name="Email"),
        StateElement(id="c3", role="textbox", name="Street address"),
        StateElement(id="c4", role="button", name="Pay Now"),
    ], discovered_via="link", link_name="Proceed to checkout")
    prod = make_state("S03", "/product/*", 200,
                      [StateElement(id="p1", role="heading", name="Sherbet Lemon")],
                      discovered_via="link", link_name="Sherbet Lemon")
    prod.url = "http://site.test/product/1"
    elements = [StateElement(id="p2", role="button", name="Add to Cart")] if with_cta else []
    prod.elements += elements
    edges = [{"from": "S00", "to": basket_path, "via": f'link "{basket_label}"'},
             {"from": "S00", "to": "/checkout", "via": 'link "Proceed to checkout"'},
             {"from": "S00", "to": "/product/*", "via": 'link "Sherbet Lemon"'}]
    return make_map([home, basket, checkout, prod], edges=edges,
                    catalog={"Sherbet Lemon": 1299})


# --------------------------------------------------------------- discovery
def test_basket_route_is_found_by_the_targets_own_word():
    planner = Planner(_shop(), FakeLLM(available=False))
    assert planner.basket_route() == "/basket"


def test_basket_route_is_found_by_the_link_label_even_on_an_odd_url():
    """/my-trolley with a link named "Basket" is still a basket."""
    planner = Planner(_shop(basket_path="/my-trolley"), FakeLLM(available=False))
    assert planner.basket_route() == "/my-trolley"


def test_basket_route_is_found_by_structure_when_nothing_is_named():
    """A quantity + update control IS a basket, whatever the URL says."""
    planner = Planner(_shop(basket_path="/xyz", basket_label="Continue"),
                      FakeLLM(available=False))
    # label "Continue" carries no basket word; the quantity/update pair does
    assert planner.basket_route() == "/xyz"


def test_no_basket_is_invented_on_a_site_without_one():
    home = make_state("S00", "/", 200,
                      [StateElement(id="e1", role="link", name="Docs")],
                      discovered_via="seed")
    docs = make_state("S01", "/docs", 200,
                      [StateElement(id="e2", role="heading", name="Docs")],
                      discovered_via="link", link_name="Docs")
    planner = Planner(make_map([home, docs]), FakeLLM(available=False))
    assert planner.basket_route() == ""


# --------------------------------------------------------------- journeys
def test_basket_journeys_are_planned_without_an_add_to_cart_button():
    """The real target has no 'Add to Cart' on product pages — tests still run."""
    cases, _ = Planner(_shop(with_cta=False), FakeLLM(available=False)).plan()
    titles = " ".join(c.title for c in cases).lower()
    assert "negative quantity" in titles, [c.title for c in cases]
    steps = [s for c in cases if "negative quantity" in c.title.lower()
             for s in c.steps]
    assert not any(s.name == "Add to Cart" for s in steps), \
        "a journey must not click a control the shop does not have"
    assert any(s.target == "/basket" for s in steps)


def test_negative_quantity_does_not_demand_invented_wording():
    """A shop that phrases validation its own way must not be reported."""
    cases, _ = Planner(_shop(), FakeLLM(available=False)).plan()
    # the specific rejection journey, not the adversarial negative-total one
    neg = next(c for c in cases if c.title == "Cart rejects negative quantity (-5)")
    assert neg.expected.should_reject is True
    assert neg.expected.error_text == "", \
        "hard-coded 'between 1 and 10' would fail a correctly-behaving shop"
    assert neg.expected.money_non_negative is True


def test_no_journey_asserts_an_invented_promo_code():
    """QArena cannot know the shop's valid codes, so it must not pretend to."""
    cases, _ = Planner(_shop(), FakeLLM(available=False)).plan()
    blob = " ".join(
        f"{c.title} {c.expected.model_dump_json()} "
        + " ".join(s.value or "" for s in c.steps)
        for c in cases).lower()
    assert "save10" not in blob, "invented promo code SAVE10 still planned"
    assert "applies a 10% discount" not in blob
    # graceful rejection of an unknown code is grounded and stays
    assert any("promo" in c.title.lower() for c in cases)


def test_page_load_check_runs_without_a_cart_button():
    """The console check must not be gated on a control the page lacks."""
    cases, rejected = Planner(_shop(with_cta=False), FakeLLM(available=False)).plan()
    prod = [c for c in cases if c.title.lower().startswith("product page")]
    assert prod, f"page-load check missing; rejected={rejected}"
    assert len(prod[0].steps) == 1 and prod[0].steps[0].action == "goto"


def test_page_check_still_clicks_add_to_cart_when_the_page_has_it():
    cases, _ = Planner(_shop(with_cta=True), FakeLLM(available=False)).plan()
    prod = next(c for c in cases if c.title.lower().startswith("product page"))
    assert [s.action for s in prod.steps] == ["goto", "click"]
    assert prod.steps[1].name == "Add to Cart"
