"""
Route provenance & evidence admissibility (Stage 3b -> 7a).

A route is only as trustworthy as the way QArena learned about it:

``seed``        the app root handed to QArena by the user.
``link``        reached by following a real link the target itself published.
``interaction`` the target navigated there itself, in response to a real user
                action (clicking "Add to Cart" redirected to /cart).
``probe``       QArena *guessed* the URL (cart / checkout / search enrichment).

The rule that keeps invented routes out of bug reports:

* a route is **confirmed** when the target either linked to it or served it
  successfully (2xx). The target's own advertisement or its own successful
  response is evidence that the route exists.
* a route that exists **only** as a failed guess proves nothing about the
  application: a guessed ``/cart`` answering 404 says "QArena guessed wrong",
  not "the site is broken". It may never be planned against, and evidence
  taken only from it may never be filed as a defect.

Defence in depth: the planner consults this policy before generating anything,
the orchestrator drops inadmissible suspects before replay, and the verifier
refuses to replay them at all — so repetition cannot launder an invented route
into a "verified" finding.

This module is pure data + pure functions: no browser, no network.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .contracts import AppMap, AppState

# provenance values (kept as plain strings: they are serialised into artifacts)
SEED = "seed"
LINK = "link"
INTERACTION = "interaction"
PROBE = "probe"
UNKNOWN = "unknown"

_OK = range(200, 300)


def canonical_route(url: str) -> str:
    """/product/3 -> /product/* ; /search?q=neon -> /search ; "" -> /."""
    p = urlparse(url or "")
    path = re.sub(r"/\d+(?=/|$)", "/*", p.path) or "/"
    return path.rstrip("/") or "/"


def _is_ok(status: int) -> bool:
    return status in _OK


class RoutePolicy:
    """Answers, for any route: may we test it? may we report on it?"""

    def __init__(self, appmap: AppMap) -> None:
        self.map = appmap
        self.target_host = urlparse(appmap.app_url or "").netloc
        self._by_route: dict[str, list[AppState]] = {}
        for s in appmap.states:
            self._by_route.setdefault(canonical_route(s.route or s.url), []).append(s)
        self._edges: list[dict] = list(appmap.edges or [])

    # ------------------------------------------------------------ provenance
    @staticmethod
    def provenance_of(state: AppState) -> str:
        via = (state.discovered_via or "").strip().lower()
        if via in (SEED, LINK, INTERACTION, PROBE):
            return via
        return PROBE if state.probed else LINK

    def states_for(self, route: str) -> list[AppState]:
        return self._by_route.get(canonical_route(route), [])

    def provenance(self, route: str) -> str:
        """Best (most trustworthy) provenance observed for this route."""
        states = self.states_for(route)
        if not states:
            return UNKNOWN
        rank = {SEED: 0, LINK: 1, INTERACTION: 2, PROBE: 3}
        return min((self.provenance_of(s) for s in states),
                   key=lambda p: rank.get(p, 4))

    def confirmed(self, route: str) -> bool:
        """Is this route part of the application, as far as evidence goes?

        True when the target advertised it (a real link / the app navigated
        there) or when the target answered it successfully. A guessed URL that
        only ever failed is not a route.

        The link check reads the crawl GRAPH as well as the states: edges are
        only ever created from anchors the crawler really saw in the DOM, so a
        link edge is the target's own claim regardless of which page QArena
        happened to record first.
        """
        states = self.states_for(route)
        if not states:
            return False
        if any(self.provenance_of(s) != PROBE for s in states):
            return True
        if self.link_edges_to(route):
            return True
        return any(_is_ok(s.http_status) for s in states)

    def link_edges_to(self, route: str) -> list[dict]:
        """Edges that came from a real anchor/button the crawler clicked past.

        ``via`` is written by the explorer: ``link "<name>"`` for an anchor,
        ``searchbox "Search"`` for QArena's own search probe — only the former
        is the target advertising a destination.
        """
        want = canonical_route(route)
        out = []
        for e in self._edges:
            via = str(e.get("via") or "")
            if via.startswith(('link "', "button ")) and \
                    canonical_route(e.get("to") or "") == want:
                out.append(e)
        return out

    def exists(self, route: str) -> bool:
        """Confirmed AND actually serving a page (2xx) — the bar for planning a
        feature journey such as login / cart / checkout."""
        return any(_is_ok(s.http_status) for s in self.states_for(route))

    def eligible(self, route: str) -> bool:
        """May a generated test navigate here?"""
        return self.confirmed(route)

    def eligible_routes(self) -> set[str]:
        return {r for r in self._by_route if self.confirmed(r)}

    def live_routes(self) -> set[str]:
        return {r for r in self._by_route if self.exists(r)}

    def why_not(self, route: str) -> str:
        states = self.states_for(route)
        if not states:
            return "route was never observed on the target"
        if self.confirmed(route):
            return ""
        return (f"route was only ever QArena's own probe "
                f"({'; '.join(sorted({f'{s.http_status}' for s in states}))}) — "
                f"the target never linked to it and never served it")

    # -------------------------------------------------------------- evidence
    def admissible(self, url: str) -> bool:
        """May evidence taken at this URL support a filed defect?"""
        return not self.blocked_reason(url)

    def blocked_reason(self, url: str) -> str:
        """Why evidence at this URL must not be filed (``""`` when it may)."""
        route = canonical_route(url)
        if route not in self._by_route:
            # A URL the crawler never mapped: the app asked for it itself (a
            # subresource, or a page reached by a real interaction), so its
            # behaviour is the target's own. Never block these.
            return ""
        if self.confirmed(route):
            return ""
        seen = ", ".join(sorted({str(s.http_status) for s in self.states_for(route)}))
        return (f"{route} exists only as a QArena probe (observed status {seen}): "
                f"the target never linked to it and never served it, so a failure "
                f"there is not evidence of an application defect")

    def network_origin(self, url: str) -> str:
        """``target`` for the app under test, ``third_party`` for anything else."""
        host = urlparse(url or "").netloc
        if not host or not self.target_host:
            return "target"
        return "target" if host == self.target_host else "third_party"

    # ------------------------------------------------------------- link data
    def link_source(self, route: str) -> dict | None:
        """Where a real link to this route lives, when the crawler recorded it.

        Returns ``{"from_route", "link_name", "via"}`` or ``None``. This is what
        turns "GET /old-page -> 404" into "the footer link 'Old page' on / is
        broken" — a fact about the target rather than about a URL.
        """
        want = canonical_route(route)
        id_to_route = {s.state_id: canonical_route(s.route or s.url)
                       for s in self.map.states}
        for e in self._edges:
            via = str(e.get("via") or "")
            to = e.get("to") or ""
            if canonical_route(to) != want:
                continue
            m = re.search(r'(?:link|button)\s+"([^"]+)"', via)
            return {"from_route": id_to_route.get(e.get("from"), ""),
                    "link_name": (m.group(1) if m else ""), "via": via}
        return None
