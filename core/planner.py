"""
Stage 4 — Planner Agent: test case synthesis & guardrails.
[LLM #1: LiteLLM + Pydantic] when a key is configured; otherwise the
deterministic journey-grammar generator (works key-free).
[Python Check vs AppMap] REJECT INVENTED STEPS  |  [Python Priority Score]
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from . import config, provenance
from .contracts import AppMap, ExpectedContract, StepSpec, TestCase
from .llm import LLMPolicy
from .provenance import RoutePolicy

DESCRIPTION_WORDS = ("sign up", "signup", "log in", "login", "search", "cart",
                     "promo", "coupon", "checkout", "order", "pay")


class Planner:
    def __init__(self, appmap: AppMap, llm: LLMPolicy) -> None:
        self.map = appmap
        self.llm = llm
        # route provenance: the single authority on which routes are real.
        # A URL QArena merely guessed is not an application route, so it may
        # neither be planned against nor reported on (see core/provenance.py).
        self.policy = RoutePolicy(appmap)
        self.element_index = self._index_elements()
        # real catalog names only: an invented product name here produced
        # reports about products the target does not have — never fabricate
        # target data. Journeys needing a product name are skipped instead.
        self.products = [n for n in appmap.catalog_prices.keys()
                         if len(n) > 2][:8]
        # may we assert a cart badge? only when the explorer actually saw one
        self.has_badge = any(s.has_cart_badge for s in appmap.states)

    # ------------------------------------------------------- element index
    def _index_elements(self) -> dict[str, set[str]]:
        idx: dict[str, set[str]] = {}
        for s in self.map.states:
            # controls QArena saw on a page it merely guessed at are not proof
            # that the application has that control
            if not self.policy.confirmed(s.route):
                continue
            for e in s.elements:
                idx.setdefault(e.role, set()).add(self._norm(e.name))
        return idx

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def has(self, role: str, name_frag: str) -> bool:
        if role == "route":
            # must be a route the target actually serves — a guessed URL that
            # only ever 404'd is not a feature of this application
            return any(self._norm(name_frag) in self._norm(s.route)
                       for s in self.map.states
                       if self.policy.exists(s.route))
        frag = self._norm(name_frag)
        return any(frag in n for n in self.element_index.get(role, set()))

    def route_real(self, route: str) -> bool:
        """A feature-level route: confirmed AND serving a page right now."""
        return self.policy.exists(route)

    def _concrete_urls(self, frag: str, limit: int = 3) -> list[str]:
        """Distinct real URLs (not canonical routes) whose route matches `frag`.

        The AppMap canonicalises ids away so that one journey covers a template,
        but the explorer stores the exact URL it visited — those are the pages a
        user can actually open, and each one is worth visiting.
        """
        want = self._norm(frag)
        out: list[str] = []
        for s in self.map.states:
            if not self.policy.exists(s.route) or want not in self._norm(s.route):
                continue
            url = urlparse(s.url or "").path or "/"
            if url not in out:
                out.append(url)
        return out[:limit]

    def feature_route(self, *frags: str, default: str = "") -> str:
        """First eligible (link-discovered) route matching any fragment.

        Applies the same provenance rule to the LLM path as to the grammar:
        a journey may only navigate to a route the target established.
        """
        for s in self.map.states:
            if not self.policy.exists(s.route):
                continue
            if all(self._norm(f) in self._norm(s.route) for f in frags):
                return s.route.replace("/*", "/1")
        return default

    # ---------------------------------------------------- feature discovery
    def route_named(self, *words: str) -> str:
        """A real route the target calls one of `words` — by URL or by its link.

        Shops name the same page differently: /cart, /basket, /bag, /trolley.
        Reading only the URL made QArena blind on any site that says "Basket"
        (the target's own vocabulary is the evidence, not QArena's assumption).
        The link label counts too, because the label is how the target presents
        the feature to its users.
        """
        wanted = [self._norm(w) for w in words]
        for s in self.map.states:
            if not self.policy.exists(s.route):
                continue
            hay = self._norm(f"{s.route} {s.link_name or ''}")
            if any(w in hay for w in wanted):
                return s.route.replace("/*", "/1")
        return ""

    def basket_route(self) -> str:
        """The page where a user reviews/updates what they are buying.

        Two independent kinds of evidence, strongest first:
          1. the route or its link label uses basket vocabulary;
          2. the page carries a quantity control plus an update/remove control —
             that structure IS a basket regardless of what it is called.
        """
        by_name = self.route_named("cart", "basket", "bag", "trolley")
        if by_name:
            return by_name
        for s in self.map.states:
            if not self.policy.exists(s.route):
                continue
            roles = {(e.role, self._norm(e.name)) for e in s.elements}
            has_qty = any(r == "spinbutton" or "quantity" in n for r, n in roles)
            has_update = any(
                r == "button" and any(w in n for w in
                                      ("update", "remove", "delete", "change"))
                for r, n in roles)
            if has_qty and has_update:
                return s.route.replace("/*", "/1")
        return ""

    def named(self, role: str, fragments) -> str:
        """First mapped element (original casing) whose name matches a fragment."""
        for s in self.map.states:
            if not self.policy.confirmed(s.route):
                continue
            for e in s.elements:
                if e.role == role and any(f in self._norm(e.name)
                                          for f in fragments):
                    return e.name
        return ""

    def route_of(self, frag: str, default: str) -> str:
        f = self._norm(frag)
        for s in self.map.states:
            if self._norm(f) in self._norm(s.route) and s.http_status == 200 \
                    and self.policy.confirmed(s.route):
                return s.route.replace("/*", "/1")
        return default

    # ---------------------------------------------------------------- main
    def plan(self) -> tuple[list[TestCase], list[dict]]:
        cases = self.llm_plan() if self.llm.available else []
        if not cases:
            cases = self.grammar_plan()
        else:
            # an LLM's proposal is a *hypothesis*: it must clear the same
            # grounding gate as the heuristic grammar before anything runs
            cases = self._ground_llm_cases(cases)
        valid, rejected = self.reject_invented(cases)
        for tc in valid:
            tc.priority_score = self.priority(tc)
        valid.sort(key=lambda t: t.priority_score, reverse=True)
        return valid[: config.MAX_TESTS], rejected

    def feature_inventory(self) -> dict:
        """What the target actually has — the planner's only source of truth."""
        return {
            "routes": sorted(self.policy.live_routes()),
            "eligible_routes": sorted(self.policy.eligible_routes()),
            "probe_only_routes": sorted(
                r for r in self.policy._by_route if not self.policy.confirmed(r)),
            "controls": {role: sorted(names)[:40]
                         for role, names in self.element_index.items()},
            "has_badge": self.has_badge,
            "products": self.products,
        }

    def _ground_llm_cases(self, cases: list[TestCase]) -> list[TestCase]:
        """Keep LLM journeys that reference real routes and real controls.

        The model is allowed to be creative about *scenarios*; it is not allowed
        to invent pages or buttons. Unusable journeys are dropped here and are
        reported by ``reject_invented`` like any other ungrounded proposal.
        """
        inv = self.feature_inventory()
        live = set(inv["routes"])
        out: list[TestCase] = []
        for tc in cases:
            ok = True
            for st in tc.steps:
                if st.action == "goto":
                    pat = re.sub(r"/\d+(?=/|$)", "/*", st.target)
                    if st.target not in live and pat not in live:
                        ok = False
                        break
            if ok:
                out.append(tc)
        return out

    # ------------------------------------------------- LLM #1 (optional)
    def llm_plan(self) -> list[TestCase]:
        states = [{"route": s.route, "status": s.http_status,
                   "elements": [{"role": e.role, "name": e.name}
                                for e in s.elements[:22]]}
                   for s in self.map.states]
        system = (
            "You are a senior QA engineer. Produce adversarial AND happy-path "
            "browser test cases as strict JSON: {\"cases\":[{test_id, "
            "journey_type, title, steps:[{step, action, target, role, name, "
            "value}], expected:{success_url_contains, disallowed_url, "
            "text_required, text_forbidden, should_reject, error_text, "
            "badge_equals, money_non_negative}}]}. Rules: actions in "
            "goto|click|dblclick|fill|press|select|check_text|assert_badge|wait. "
            "goto target = route path. click/fill use role+name exactly as "
            "listed. Cover boundary values (0, -1, 11), invalid emails, empty "
            "required fields, double-clicks, empty-cart checkout, promo abuse.")
        user = ("App description:\n" + self.map.description +
                "\n\nAppMap JSON:\n" + __import__("json").dumps(states)[:9000])
        data = self.llm.call_json("planner#1", system, user)
        if not data or "cases" not in data:
            return []
        out: list[TestCase] = []
        try:
            for raw in data["cases"]:
                raw["steps"] = [StepSpec(**{k: v for k, v in st.items()
                                            if k in StepSpec.model_fields})
                                for st in raw.get("steps", [])]
                raw["expected"] = ExpectedContract(
                    **{k: v for k, v in raw.get("expected", {}).items()
                       if k in ExpectedContract.model_fields})
                out.append(TestCase(**{k: v for k, v in raw.items()
                                       if k in ("test_id", "journey_type",
                                                "priority_score", "title",
                                                "steps", "expected")}))
        except Exception:
            return []
        return out

    # ------------------------------------- deterministic journey grammar
    def grammar_plan(self) -> list[TestCase]:
        T: list[TestCase] = []
        uid = [0]

        def nid() -> str:
            uid[0] += 1
            return f"TC_{uid[0]:03d}"

        prod = self.products[0] if self.products else ""

        # --- auth journeys ---------------------------------------------------
        if self.has("route", "signup"):
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path", title="Sign up new account",
                steps=[StepSpec(step=1, action="goto", target="/signup"),
                       StepSpec(step=2, action="fill", role="textbox", name="Full name", value="Trial User"),
                       StepSpec(step=3, action="fill", role="textbox", name="Email", value="trial@qarena.dev"),
                       StepSpec(step=4, action="fill", role="textbox", name="Password", value="passw0rd!"),
                       StepSpec(step=5, action="click", role="button", name="Create account")],
                expected=ExpectedContract(text_required=["Add to Cart"])))
            T.append(TestCase(
                test_id=nid(), journey_type="edge_case_negative",
                title="Sign-up rejects invalid email",
                steps=[StepSpec(step=1, action="goto", target="/signup"),
                       StepSpec(step=2, action="fill", role="textbox", name="Full name", value="Bad Mail"),
                       StepSpec(step=3, action="fill", role="textbox", name="Email", value="not-an-email"),
                       StepSpec(step=4, action="fill", role="textbox", name="Password", value="passw0rd!"),
                       StepSpec(step=5, action="click", role="button", name="Create account")],
                expected=ExpectedContract(should_reject=True, error_text="valid email",
                                          disallowed_url="/")))
        if self.has("route", "login"):
            # grounded oracle: demand the app's OWN logout control when the map
            # shows one; otherwise demand only that login LEFT the login page.
            # The hard-coded "Log out" expectation filed false bugs on apps that
            # never had such a control.
            logout = (self.named("button", ("log out", "logout", "sign out"))
                      or self.named("link", ("log out", "logout", "sign out")))
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path", title="Log in with valid credentials",
                steps=[StepSpec(step=1, action="goto", target="/login"),
                       StepSpec(step=2, action="fill", role="textbox", name="Email", value="demo@cloudcart.dev"),
                       StepSpec(step=3, action="fill", role="textbox", name="Password", value="demo12345"),
                       StepSpec(step=4, action="click", role="button", name="Log in")],
                expected=(ExpectedContract(text_required=[logout]) if logout
                          else ExpectedContract(disallowed_url="/login"))))

        # A feature journey may only navigate to a route the target itself
        # established (linked to it, or served it). A guessed /cart answering
        # 404 is not a broken cart: the site simply has no such route.
        # The feature is recognised by the target's OWN vocabulary: a shop that
        # calls the page "Basket" (/basket) has a basket, even though the URL
        # never says "cart".
        cart_route = self.basket_route()
        promo_route = cart_route if self.has("textbox", "promo") else ""
        checkout_route = self.route_named("checkout", "check-out", "payment",
                                          "pay") or self.feature_route("checkout")

        # --- search intent ---------------------------------------------------
        if self.has("searchbox", "") and prod:
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path",
                title=f"Search finds exact product name '{prod}'",
                steps=[StepSpec(step=1, action="goto", target="/"),
                       StepSpec(step=2, action="fill", role="searchbox", name="Search", value=prod),
                       StepSpec(step=3, action="press", role="searchbox", name="Search", target="Enter")],
                expected=ExpectedContract(text_required=[prod],
                                          text_forbidden=["No products found"])))

        # --- cart / promo / quantity adversarial -----------------------------
        can_cart = self.has("button", "add to cart")
        if can_cart and self.has_badge:
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path",
                title=f"Add '{prod}' to cart — badge shows 1" if prod
                      else "Add to Cart — badge shows 1",
                steps=[StepSpec(step=1, action="goto", target="/"),
                       StepSpec(step=2, action="click", role="button", name="Add to Cart")],
                expected=ExpectedContract(badge_equals=1)))
            T.append(TestCase(
                test_id=nid(), journey_type="adversarial",
                title="Rapid double-click adds exactly one unit",
                steps=[StepSpec(step=1, action="goto", target="/"),
                       StepSpec(step=2, action="dblclick", role="button", name="Add to Cart")],
                expected=ExpectedContract(badge_equals=1)))
        # The quantity control must exist ON the basket page — that control is
        # the evidence the feature exists. "Add to Cart" is only used to seed the
        # basket when the app actually has such a button; a shop that lists items
        # without one still deserves the quantity test.
        qty_on_basket = cart_route and self._on_route(cart_route, "spinbutton",
                                                      "Quantity")
        update_on_basket = cart_route and self._on_route(cart_route, "button",
                                                         "Update")
        if qty_on_basket and update_on_basket:
            seed = ([StepSpec(step=1, action="goto", target="/"),
                     StepSpec(step=2, action="click", role="button",
                              name="Add to Cart")]
                    if can_cart else [])
            neg_steps = [*seed,
                         StepSpec(step=len(seed) + 1, action="goto", target=cart_route),
                         StepSpec(step=len(seed) + 2, action="fill", role="spinbutton",
                                  name="Quantity", value="-5"),
                         StepSpec(step=len(seed) + 3, action="click", role="button",
                                  name="Update")]
            for i, st in enumerate(neg_steps, 1):
                st.step = i
            T.append(TestCase(
                test_id=nid(), journey_type="edge_case_negative",
                title="Cart rejects negative quantity (-5)",
                steps=neg_steps,
                # NO invented wording: we assert only that invalid input is
                # refused and that money never goes negative. Requiring a
                # specific message ("between 1 and 10") would fail a shop that
                # phrases its validation differently.
                expected=ExpectedContract(should_reject=True,
                                          money_non_negative=True)))
            if checkout_route and self.has("button", "pay"):
                T.append(TestCase(
                    test_id=nid(), journey_type="adversarial",
                    title="Negative quantity must never produce a negative money total",
                    steps=[*neg_steps,
                           StepSpec(step=len(neg_steps) + 1, action="goto",
                                    target=checkout_route),
                           StepSpec(step=len(neg_steps) + 2, action="fill", role="textbox",
                                    name="Full name", value="Neg Tester"),
                           StepSpec(step=len(neg_steps) + 3, action="fill", role="textbox",
                                    name="Email", value="neg@qarena.dev"),
                           StepSpec(step=len(neg_steps) + 4, action="fill", role="textbox",
                                    name="Street address", value="1 Bug Lane"),
                           StepSpec(step=len(neg_steps) + 5, action="click",
                                    role="button", name="Pay Now")],
                    # NOT should_reject: on a correct app the -5 is rejected at
                    # Update and the remaining positive cart may legitimately be
                    # paid. The true invariant: money must never go negative.
                    expected=ExpectedContract(money_non_negative=True)))
        # --- product detail pages: console probe -----------------------------
        # One journey per CONCRETE page the explorer actually visited, not just
        # the first: sibling pages share a canonical route (/product/1 and
        # /product/2 are both "/product/*") but only one of them may be broken.
        # Testing a single representative page hides exactly the defects a user
        # would hit, so every observed page gets its own check.
        for i, purl in enumerate(self._concrete_urls("product", limit=3)):
            # Does THIS page carry an add-to-cart button? A catalog page that
            # only links onward has no button, and demanding one would reject a
            # perfectly useful page-load check (that is how a page-throwing
            # script went unseen on a shop without "Add to Cart").
            has_cta = self._on_route(purl, "button", "Add to Cart")
            label = ("Product page" if i == 0 else f"Product page {purl}")
            steps = [StepSpec(step=1, action="goto", target=purl)]
            if has_cta:
                steps.append(StepSpec(step=2, action="click", role="button",
                                      name="Add to Cart"))
                label += ": Add to Cart with a clean console"
            else:
                label += ": loads with a clean console"
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path", title=label,
                steps=steps, expected=ExpectedContract(badge_equals=None)))

        if promo_route and self.has("textbox", "promo") \
                and self.has("button", "apply"):
            # Only the graceful-rejection journey is grounded: QArena cannot
            # know which promo codes the shop actually honours, so "SAVE10
            # applies 10%" was an invented requirement (and filing its failure
            # blamed the target for QArena's guess). An unknown code must be
            # refused politely — that every shop owes its users.
            seed = ([StepSpec(step=1, action="goto", target="/"),
                     StepSpec(step=2, action="click", role="button",
                              name="Add to Cart")]
                    if self.has("button", "add to cart")
                    else [StepSpec(step=1, action="goto", target="/")])
            steps = [*seed,
                     StepSpec(step=0, action="goto", target=promo_route),
                     StepSpec(step=0, action="fill", role="textbox",
                              name="Promo code", value="XXX-NOPE"),
                     StepSpec(step=0, action="click", role="button", name="Apply")]
            for i, st in enumerate(steps, 1):
                st.step = i
            T.append(TestCase(
                test_id=nid(), journey_type="edge_case_negative",
                title="Unknown promo code fails gracefully",
                steps=steps,
                expected=ExpectedContract(should_reject=True)))

        # --- checkout ---------------------------------------------------------
        # every step here is conditional on the target ACTUALLY having that
        # feature (route + control + pay button). "Cart is empty", "address is
        # required" and "Order confirmed" are the app's own vocabulary; on an
        # app that only has a cart (no checkout), demanding them filed bugs
        # against features that do not exist.
        pay_btn = self.named("button", ("pay", "place order", "submit order"))
        if checkout_route and pay_btn:
            cart_first_ok = self.has("button", "add to cart")
            if cart_first_ok:
                T.append(TestCase(
                    test_id=nid(), journey_type="edge_case_negative",
                    title="Checkout blocks empty cart",
                    steps=[StepSpec(step=1, action="goto", target=checkout_route),
                           StepSpec(step=2, action="click", role="button", name=pay_btn)],
                    expected=ExpectedContract(should_reject=True,
                                              text_forbidden=["Order confirmed"],
                                              error_text="cart is empty")))
                T.append(TestCase(
                    test_id=nid(), journey_type="edge_case_negative",
                    title="Checkout requires shipping address",
                    steps=[StepSpec(step=1, action="goto", target="/"),
                           StepSpec(step=2, action="click", role="button", name="Add to Cart"),
                           StepSpec(step=3, action="goto", target=checkout_route),
                           StepSpec(step=4, action="fill", role="textbox", name="Full name", value="No Addr"),
                           StepSpec(step=5, action="fill", role="textbox", name="Email", value="noaddr@qarena.dev"),
                           StepSpec(step=6, action="click", role="button", name=pay_btn)],
                    expected=ExpectedContract(should_reject=True,
                                              error_text="address is required",
                                              text_forbidden=["Order confirmed"])))
                T.append(TestCase(
                    test_id=nid(), journey_type="happy_path",
                    title=f"Full purchase journey: {prod}" if prod
                          else "Full purchase journey",
                    steps=[StepSpec(step=1, action="goto", target="/"),
                           StepSpec(step=2, action="click", role="button", name="Add to Cart"),
                           StepSpec(step=3, action="goto", target=checkout_route),
                           StepSpec(step=4, action="fill", role="textbox", name="Full name", value="Happy Buyer"),
                           StepSpec(step=5, action="fill", role="textbox", name="Email", value="happy@qarena.dev"),
                           StepSpec(step=6, action="fill", role="textbox", name="Street address", value="42 Green Rd"),
                           StepSpec(step=7, action="click", role="button", name=pay_btn)],
                    # never invent the app's confirmation URL or success copy:
                    # assert only the invariant that always holds (money stays
                    # positive) plus the app's own requirement, when the caller
                    # supplied one in the description
                    expected=ExpectedContract(money_non_negative=True)))

        # --- dead-link patrol from exploration facts ---------------------------
        # A failing destination is only a defect when the TARGET sent us there:
        # the page must have been reached by following a real link the target
        # published. A 404 on a URL QArena guessed (cart/checkout/search
        # enrichment probes) says nothing about the application and is never
        # patrolled — that was the synthetic-route false positive.
        dead = [s for s in self.map.states
                if s.http_status >= 400
                and self.policy.provenance_of(s) in (provenance.SEED,
                                                     provenance.LINK,
                                                     provenance.INTERACTION)]
        for ds in dead[:2]:
            src = self.policy.link_source(ds.route)
            from_route = (src or {}).get("from_route") or "/"
            label = (ds.link_name or (src or {}).get("link_name")
                     or ds.route.strip("/") or ds.route)
            if src and src.get("link_name"):
                # reproduce it the way a user does: be on the page that shows
                # the link, then follow it
                steps = [StepSpec(step=1, action="goto", target=from_route),
                         StepSpec(step=2, action="click", role="link",
                                  name=src["link_name"])]
            else:
                # link text unknown (icon-only/anchor without a name): the
                # destination is still evidenced by the link the crawler found
                steps = [StepSpec(step=1, action="goto", target=from_route),
                         StepSpec(step=2, action="goto", target=ds.route)]
            T.append(TestCase(
                test_id=nid(), journey_type="happy_path",
                title=(f"Link '{label}' leads somewhere real"
                       if steps[-1].action == "click"
                       else f"Linked route '{label}' serves a real page"),
                steps=steps,
                expected=ExpectedContract(no_new_errors_5xx=True)))
        return T

    # --------------------------------------------- invented-step rejection
    def reject_invented(self, cases: list[TestCase]) -> tuple[list[TestCase], list[dict]]:
        """Keep only journeys that run against routes the target really has.

        Two distinct failures, both fatal to a journey:

        * a route QArena only guessed (``policy.eligible`` is False) — this is
          the synthetic-route guard: navigating there and finding a 404 proves
          QArena's guess was wrong, not that the app is broken;
        * a step naming a control the page does not actually have.

        Every rejection records *why*, so the dashboard can show the reason
        instead of a bare "invented".
        """
        valid, rejected = [], []
        for tc in cases:
            ok, bad_steps, why = True, [], ""
            route = "/"          # page reached by the most recent goto step
            for st in tc.steps:
                if st.action == "goto":
                    if not self.policy.eligible(st.target):
                        ok = False
                        why = why or "synthetic_route"
                        bad_steps.append(
                            f"goto {st.target} — "
                            f"{self.policy.why_not(st.target)}")
                    route = st.target
                elif st.action in ("click", "fill", "select", "dblclick", "press"):
                    if not self.has(st.role, st.name):
                        ok = False
                        why = why or "missing_control"
                        bad_steps.append(f'{st.action} {st.role} "{st.name}"')
                    elif st.action in ("click", "dblclick") and \
                            not self._on_route(route, st.role, st.name):
                        # the control exists somewhere in the app but not on the
                        # page this step is actually on -> an invented route+
                        # element pair (clicking "Add to Cart" on a page that
                        # has no such button)
                        ok = False
                        why = why or "wrong_page"
                        bad_steps.append(f'{st.action} {st.role} "{st.name}" '
                                         f"is not on {route}")
            if ok:
                valid.append(tc)
            else:
                rejected.append({"test_id": tc.test_id, "title": tc.title,
                                 "reason": why or "invented",
                                 "invented_steps": bad_steps})
        return valid, rejected

    def _on_route(self, route: str, role: str, name: str) -> bool:
        """Does `role`/`name` exist on the page reached by `route`?

        Only answers when that page is actually mapped (otherwise the caller
        falls back to the global check) — a route we never mapped must not make
        a step look invented. Pages QArena merely guessed are never used as
        evidence that a control exists.
        """
        pat = re.sub(r"/\d+(?=/|$)", "/*", route or "")
        state = next((s for s in self.map.states
                      if (s.route == pat or s.route == route)
                      and self.policy.confirmed(s.route)), None)
        if state is None or not state.elements:
            return True
        frag = self._norm(name)
        return any(e.role == role and frag in self._norm(e.name)
                   for e in state.elements)

    # -------------------------------------------------------------- priority
    @staticmethod
    def priority(tc: TestCase) -> int:
        score = 50
        blob = (tc.title + " " + " ".join(s.target for s in tc.steps)).lower()
        if tc.journey_type.startswith("edge"):
            score += 15
        if tc.journey_type == "adversarial":
            score += 25
        if "checkout" in blob or "pay now" in blob:
            score += 30
        if "login" in blob or "sign" in blob:
            score += 10
        if "promo" in blob:
            score += 10
        return min(score, 99)
