"""
Stage 7a — Reporter Agent: dedup + severity + report generation.
[Python Signature Match] merge duplicate bugs
[Python Rules] deterministic severity      [LLM #4] title/summary prose ONLY
[Jinja2 Template] fill report from saved factual data — never from model memory.

Every file is written with an explicit ``encoding="utf-8"``. Relying on the
platform default (cp1252 / "charmap" on Windows) aborts the whole stage the
moment evidence contains a character outside the ANSI code page — signal
evidence routinely does (``GET /cart → 404``).
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import config
from .contracts import FiledBug, PossibleBug
from .llm import LLMPolicy
from .provenance import RoutePolicy

SEVERITY_RANK = {"Critical": 3, "Major": 2, "Minor": 1}


class Reporter:
    def __init__(self, run_dir: Path, llm: LLMPolicy, appmap=None) -> None:
        self.run_dir = run_dir
        self.llm = llm
        # the AppMap lets the report separate a target-owned failure from a
        # third-party request, a dead link the target itself published, and a
        # URL QArena merely probed. Optional: reports degrade gracefully to
        # host-level classification without it.
        self.appmap = appmap
        self.policy = RoutePolicy(appmap) if appmap is not None else None
        # findings withheld because their evidence came from a synthetic route
        self.suppressed: list[dict] = []
        self.jinja = Environment(
            loader=FileSystemLoader(str(config.JINJA_DIR)), autoescape=False)

    # ------------------------------------------------------- dedup + file
    def file_bugs(self, confirmed: list[tuple[PossibleBug, str]]) -> list[FiledBug]:
        # FINAL SAFETY NET (third layer, after the planner and the verifier):
        # never publish a finding whose evidence was taken on a route QArena
        # invented. Each layer checks independently, so a future code path that
        # skips one guard still cannot put an invented defect in the report.
        blocked = {id(pb) for pb, _ in self.block_inadmissible(confirmed)}
        confirmed = [c for c in confirmed if id(c[0]) not in blocked]
        merged: dict[tuple, dict] = {}
        for pb, repro in confirmed:
            ev = pb.evidence
            route = self._route_of(ev.url_after or ev.url_before)
            elem = self._element_of(pb)
            # merge key includes test_id AND the signal family: co-located
            # signals of the SAME family from one journey are one defect, while
            # distinct journeys landing on the same route+element stay distinct.
            # Cross-family merging used to fuse unrelated defects (a console
            # error + an axe contrast violation became one mislabelled bug and
            # the contrast finding vanished from the filed list).
            key = (route, re.sub(r"\s+", " ", elem.lower()).strip(),
                   pb.test_id, pb.group.split(":")[0])
            sev = self._severity(pb)
            if key not in merged:
                merged[key] = {"pb": pb, "repro": repro, "route": route,
                               "elem": elem, "sev": sev, "score": pb.score,
                               "signals": [h for h in pb.signals]}
            else:
                m = merged[key]
                have = {h.signal for h in m["signals"]}
                for h in pb.signals:
                    if h.signal not in have:
                        m["signals"].append(h)
                if repro.split("/")[0] > m["repro"].split("/")[0]:
                    m["repro"] = repro
                if SEVERITY_RANK[sev] > SEVERITY_RANK[m["sev"]]:
                    m["sev"] = sev
                if pb.score > m["score"]:
                    m["score"], m["pb"] = pb.score, pb

        filed: list[FiledBug] = []
        reports_dir = self.run_dir / "bug_reports"
        reports_dir.mkdir(exist_ok=True)
        # cross-step co-assembly INSIDE one journey: groups fired by the same
        # step / route are one defect event (BUG-01: silently-accepted qty +
        # negative money + price intent all born of the same Update click).
        coalesced: dict[tuple, dict] = {}
        for m in merged.values():
            pb = m["pb"]
            # route-insensitive groups already deduped globally — pin them to
            # the richest step's filing instead (keeps console/intent/etc at one)
            if pb.group.startswith(("intent:", "console:", "form:money",
                                    "flow:", "axe:")):
                ckey = ("content", pb.test_id, "money" if "money" in pb.group or
                        "money" in pb.summary.lower() or "-$" in pb.summary
                        else pb.group.split(":")[0])
            else:
                ckey = ("http", self._route_of(
                    pb.evidence.url_after or pb.evidence.url_before), pb.test_id)
            tgt = coalesced.setdefault(ckey, m)
            if tgt is not m:
                have = {h.signal for h in tgt["signals"]}
                for h in pb.signals:
                    if h.signal not in have:
                        tgt["signals"].append(h)
                        have.add(h.signal)
                if pb.score > tgt["score"]:
                    tgt["score"], tgt["pb"] = pb.score, pb
                    tgt["repro"] = m["repro"]

        for i, m in enumerate(sorted(
                coalesced.values(), key=lambda x: -x["score"]), 1):
            bug_id = f"QA-{i:02d}"
            pb, repro = m["pb"], m["repro"]
            fb = self._to_filed_bug(bug_id, pb, repro, m["elem"], m["route"],
                                    m["sev"], m["score"], m["signals"],
                                    title=self._title(pb, m["elem"], m["route"]))
            # render + write the human report. A failure here must not cost us
            # the filing itself: keep the FiledBug and say so in its evidence.
            try:
                md = self.jinja.get_template("bug_report.md.j2").render(bug=fb)
                (reports_dir / f"{bug_id}.md").write_text(md, encoding="utf-8")
                fb.evidence["report"] = f"bug_reports/{bug_id}.md"
            except Exception as exc:
                fb.evidence["warnings"] = list(fb.evidence.get("warnings", [])) + [
                    f"markdown report could not be written "
                    f"({exc.__class__.__name__}: {exc}) — filing kept, facts above "
                    f"are unchanged"]
                self._log_warning(bug_id, exc)
            filed.append(fb)
        (self.run_dir / "bugs_filed.json").write_text(
            json.dumps([b.model_dump() for b in filed], indent=2),
            encoding="utf-8")
        return filed

    # ------------------------------------------------------- filing builder
    def _to_filed_bug(self, bug_id: str, pb: PossibleBug, repro: str, elem: str,
                      route: str, sev: str, score: float,
                      signals: list, title: str = "") -> FiledBug:
        """Assemble one report record from already-verified facts. Pure data —
        only the prose *title* is ever model-authored, and it is optional."""
        url = pb.evidence.url_after or pb.evidence.url_before or ""
        expected, actual, basis = Reporter._expect_actual(pb, self.policy)
        http_lines = [f"{f.get('method', '?')} {f.get('url', '?')} "
                      f"→ {f.get('status', '?')}"
                      for f in pb.evidence.http_failures[:10]]
        return FiledBug(
            bug_id=bug_id, title=title or self._fallback_title(pb, elem, route),
            severity=sev, journey=pb.journey,
            route=route, element_name=elem,
            signals=[h.signal for h in signals], score=score,
            reproduced_count=repro,
            steps_to_reproduce=Reporter._human_steps(pb),
            expected_behavior=expected,
            actual_behavior=actual,
            confidence_basis=(
                f"weighted signal score {score} from signal(s) "
                f"{', '.join(str(h.signal) for h in signals)}; "
                f"reproduced {repro} in independent clean-browser replays"),
            evidence={"screenshot": pb.evidence.screenshot,
                      "trace": f"trace_{pb.test_id}.zip",
                      "url": url, "test_id": pb.test_id,
                      "confidence": score,
                      "reproductions": repro,
                      "affected_element": elem,
                      "expected_source": basis,
                      "why_it_matters": Reporter._why_it_matters(pb),
                      # each failing request is classified, because a raw 404 is
                      # evidence, not automatically a user-facing defect
                      "network_analysis": self._classify_network(pb),
                      # raw console/network facts, so a reader can check the
                      # finding without opening the trace. Bounded on purpose.
                      "http_failures": http_lines,
                      "console_errors": [str(c)[:200]
                                         for c in pb.evidence.console_errors[:10]],
                      "signals": [f"S{h.signal} {h.name}: {h.evidence[:160]}"
                                  for h in signals]})

    def fallback_file_bugs(self, confirmed: list[tuple[PossibleBug, str]]) -> list[FiledBug]:
        """Last-resort filing used when the reporting stage dies as a whole.

        Verified findings are already proved (>=2/3 clean-browser replays) — they
        must still reach the dashboard, with honest titles and no traceback.
        Reports are written when possible; failures are recorded, not raised.
        """
        # the synthetic-route guard applies here too: a crashed reporting stage
        # is not a licence to publish invented defects
        blocked = {id(pb) for pb, _ in self.block_inadmissible(confirmed)}
        confirmed = [c for c in confirmed if id(c[0]) not in blocked]
        filed: list[FiledBug] = []
        reports_dir = self.run_dir / "bug_reports"
        try:
            reports_dir.mkdir(exist_ok=True)
        except Exception:
            pass
        def safely(fn, default):
            """The fallback must not depend on the code that just failed."""
            try:
                return fn()
            except Exception:
                return default

        for i, (pb, repro) in enumerate(confirmed, 1):
            bug_id = f"QA-{i:02d}"
            route = safely(lambda: self._route_of(
                pb.evidence.url_after or pb.evidence.url_before), "")
            elem = safely(lambda: self._element_of(pb), "page")
            sev = safely(lambda: self._severity(pb), "Major")
            try:
                fb = self._to_filed_bug(bug_id, pb, repro, elem, route,
                                        sev, pb.score, pb.signals)
            except Exception:
                # absolute last resort: keep the verified finding, plainest form
                fb = FiledBug(
                    bug_id=bug_id, title=pb.journey or bug_id, severity=sev,
                    journey=pb.journey, route=route, element_name=elem,
                    signals=[h.signal for h in pb.signals], score=pb.score,
                    reproduced_count=repro,
                    steps_to_reproduce=[], expected_behavior="",
                    actual_behavior=pb.summary,
                    evidence={"screenshot": pb.evidence.screenshot,
                              "signals": [h.evidence[:160] for h in pb.signals]})
            fb.evidence["warnings"] = list(fb.evidence.get("warnings", [])) + [
                "reported through the fallback filer because the reporting stage "
                "failed — evidence above is unchanged, prose is not LLM-written"]
            try:
                md = self.jinja.get_template("bug_report.md.j2").render(bug=fb)
                (reports_dir / f"{bug_id}.md").write_text(md, encoding="utf-8")
                fb.evidence["report"] = f"bug_reports/{bug_id}.md"
            except Exception as exc:
                fb.evidence["warnings"].append(
                    f"markdown report could not be written "
                    f"({exc.__class__.__name__}: {exc})")
            filed.append(fb)
        try:
            (self.run_dir / "bugs_filed.json").write_text(
                json.dumps([b.model_dump() for b in filed], indent=2),
                encoding="utf-8")
        except Exception:
            pass
        return filed

    def _log_warning(self, bug_id: str, exc: Exception) -> None:
        """Diagnostics go to a UTF-8 file with a traceback — never to stdout with
        the platform default encoding."""
        try:
            with open(self.run_dir / "run.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "type": "warning", "stage": "reporter", "bug_id": bug_id,
                    "msg": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc()}) + "\n")
        except Exception:
            pass


    # ------------------------------------------------------------ severity
    @staticmethod
    def _route_of(url: str) -> str:
        from urllib.parse import urlparse
        p = urlparse(url or "")
        return (re.sub(r"/\d+(?=/|$)", "/*", p.path) or "/").rstrip("/") or "/"

    def _element_of(self, pb: PossibleBug) -> str:
        g = pb.group
        # a broken item names the defendant precisely — but ONLY when the bug
        # is actually about that broken thing (every page shares the same
        # broken footer link; it must not own unrelated form bugs)
        if g.startswith(("http:", "broken:")):
            # match the broken item to the URL that actually failed: a page can
            # carry several broken links, and the finding must name the one the
            # journey followed (a 404 report once named a different link)
            route = self._route_of(pb.evidence.url_after or pb.evidence.url_before)
            for b in pb.evidence.broken_items:
                if b.get("name") and route and route in str(b.get("detail") or ""):
                    return b["name"]
            src = self.policy.link_source(route) if self.policy is not None else None
            if src and src.get("link_name"):
                return src["link_name"]
            for b in pb.evidence.broken_items:
                if b.get("name"):
                    return b["name"]
        if g.startswith("intent:"):
            m = re.search(r"'([^']{3,40})'", g)
            if m:
                return m.group(1)
        # the interactive element acted upon at the failing step
        if 0 <= pb.step_index < len(pb.repro_steps):
            s = pb.repro_steps[pb.step_index]
            if s.action in ("click", "dblclick", "fill", "select", "press"):
                return s.name or s.role
        for s in pb.repro_steps:
            if s.action in ("click", "dblclick", "fill", "select", "press"):
                return s.name or s.role
        # an axe target describes an accessibility finding only. Using it as a
        # generic fallback mislabelled other findings (a console error got the
        # footer selector of an unrelated contrast violation in its title).
        if g.startswith(("axe:", "a11y:")):
            for v in pb.evidence.axe_violations:
                if v.get("target"):
                    return f"@{v['target'][:40]}"
        return pb.evidence.target or "page"
    @staticmethod
    def _severity(pb: PossibleBug) -> str:
        blob = (pb.summary + " " + pb.journey).lower()
        sig = {h.signal for h in pb.signals}
        # read the actual status code rather than the summary text, so the
        # severity always matches the recorded evidence
        statuses = {f.get("status") for f in pb.evidence.http_failures}
        server_error = any(isinstance(s, int) and s >= 500 for s in statuses)
        if "negative" in blob and "$" in blob or "price mismatch" in blob:
            return "Critical"
        if server_error and any(w in blob for w in
                               ("promo", "pay", "checkout", "order", "cart")):
            return "Critical"
        if server_error:
            # a page the target links to or serves returns a server error: the
            # user gets a failure page instead of content
            return "Major"
        if 4 in sig or "silently accepted" in blob or "$0.00" in blob \
                or "badge expected" in blob or "badge" in blob:
            return "Major"
        if 2 in sig and ("404" in blob or 404 in statuses):
            return "Minor"
        if sig & {1, 3, 5}:
            return "Minor"
        return "Major"

    # ------------------------------------------------------------ prose
    def _title(self, pb: PossibleBug, elem: str, route: str) -> str:
        fallback = self._fallback_title(pb, elem, route)
        if not self.llm.available:
            return fallback
        system = ("You write one-line bug titles. Facts only, <= 110 chars, "
                  "no emoji. Reply as JSON {\"title\": ..., \"summary\": ...}.")
        user = f"Signals: {pb.summary}\nRoute: {route}\nElement: {elem}\nJourney: {pb.journey}"
        data = self.llm.call_json("reporter#4", system, user)
        if data and data.get("title"):
            return re.sub(r"\s+", " ", data["title"]).strip()[:140]
        return fallback

    def _fallback_title(self, pb: PossibleBug, elem: str, route: str) -> str:
        """A factual one-line title, built from the evidence itself.

        HTTP findings name the real request and (when the crawl graph knows it)
        the link that leads there, because "GET /old-page → 404, linked as
        'Old page' from /" is a fact a developer can act on, while
        "route: http — 404" is not.
        """
        s0 = max(pb.signals, key=lambda h: h.weight) if pb.signals else None
        statuses = {f.get("status") for f in pb.evidence.http_failures}
        if (pb.group or "").startswith("http:") and pb.evidence.http_failures:
            f = pb.evidence.http_failures[0]
            verb = f"{f.get('method', 'GET')} {f.get('url', route)}"
            code = f.get("status", "?")
            src = self.policy.link_source(route) if self.policy is not None else None
            if src and src.get("link_name"):
                return (f"Broken link \"{src['link_name']}\" on "
                        f"{src.get('from_route') or '/'} — {verb} → {code}")[:140]
            who = elem if elem and elem not in ("page", "/") else route
            return f"{who}: {verb} → {code}"[:140]
        if not s0:
            return pb.journey
        ev = re.sub(r"\s+", " ", s0.evidence).strip()
        subject = elem if elem and elem != "page" else route
        return f"{subject}: {s0.name} — {ev[:88]}"[:140]

    @staticmethod
    def _human_steps(pb: PossibleBug) -> list[str]:
        out = []
        for s in pb.repro_steps:
            if s.action == "goto":
                out.append(f"Open `{s.target}`")
            elif s.action in ("click", "dblclick"):
                verb = "Click" if s.action == "click" else "Rapidly double-click"
                out.append(f'{verb} the {s.role} **"{s.name}"**')
            elif s.action == "fill":
                out.append(f'Type `{s.value}` into the {s.role} **"{s.name}"**')
            elif s.action == "press":
                out.append(f"Press **{s.target or 'Enter'}**")
            elif s.action == "check_text":
                out.append(f'Verify the text "{s.target}" is shown')
            else:
                out.append(f"{s.action} {s.target}")
        return [f"{i}. {t}" for i, t in enumerate(out, 1)]

    @staticmethod
    def _expect_actual(pb: PossibleBug, policy=None) -> tuple[str, str, str]:
        """Expected behaviour, actual behaviour, and WHERE the expectation comes from.

        QArena never invents a requirement. The expectation is built from, in
        order of strength:

        1. the target's own behaviour — a link the target publishes must lead
           somewhere that loads;
        2. the explicit test contract that was executed (``pb.oracle``);
        3. otherwise the report says plainly that no explicit requirement was
           supplied and judges only the observed behaviour.
        """
        actual = f"{pb.summary} — observed during journey: {pb.journey}"
        contract = pb.oracle[0] if pb.oracle else {}
        group = pb.group or ""
        url = pb.evidence.url_after or pb.evidence.url_before or ""
        route = Reporter._route_of(url)

        # 1) target behaviour: this page is reachable from the target's own link
        if group.startswith("http:"):
            src = policy.link_source(route) if policy is not None else None
            if src and (src.get("from_route") or src.get("link_name")):
                via = (f"the link \"{src.get('link_name') or route}\" on "
                       f"`{src.get('from_route') or '/'}`") if src.get("link_name") \
                    else f"a link on `{src.get('from_route') or '/'}`"
                return ("Following " + via + " should open a working page — the "
                        "target itself advertises this destination.",
                        actual,
                        "observed target behaviour (the target publishes this link)")
            return (f"`{route}` is part of the application and should load "
                    f"without an HTTP error.",
                    actual, "observed target behaviour (the page was reached through "
                            "the application itself)")

        # 2) the executed test contract
        if contract:
            if contract.get("should_reject"):
                wanted = contract.get("error_text") or "a clear validation message"
                return (f"Invalid input must be rejected with {wanted!r} — the test "
                        f"contract executed for this journey.",
                        actual, "the test contract for this journey (supplied rule)")
            if contract.get("money_non_negative"):
                return ("No monetary amount may ever be displayed as negative.",
                        actual, "the test contract for this journey (supplied rule)")
            if contract.get("badge_equals") is not None:
                return (f"The cart badge must show {contract['badge_equals']} after "
                        f"this action.",
                        actual, "the test contract for this journey (supplied rule)")
            req = contract.get("text_required") or []
            if req:
                return (f"The page must show {req} after this journey.",
                        actual, "the test contract for this journey (supplied rule)")
            if contract.get("text_forbidden"):
                return (f"The page must not show "
                        f"{contract.get('text_forbidden')} here.",
                        actual, "the test contract for this journey (supplied rule)")

        # 3) nothing to appeal to — say so instead of inventing a requirement
        return ("No explicit requirement was supplied for this flow; the finding "
                "is judged against the target's own observed behaviour and the "
                "browser evidence below.",
                actual, "no explicit requirement was supplied for this flow")

    @staticmethod
    def _why_it_matters(pb: PossibleBug) -> str:
        """Explain the user impact of the evidence — never just restate it.

        The HTTP statuses are read from the recorded failures rather than from
        the signal's prose, so the sentence always matches the actual evidence.
        """
        sig = {h.signal for h in pb.signals}
        g = pb.group or ""
        statuses = {f.get("status") for f in pb.evidence.http_failures}
        if g.startswith("http:") and any(
                isinstance(s, int) and s >= 500 for s in statuses):
            return ("A page the target serves or links to returns a server error, "
                    "so a user following that path sees a failure page instead of "
                    "content.")
        if g.startswith("http:") and 404 in statuses:
            return ("A page the target advertises is not served, so the user hits a "
                    "dead end (broken navigation) instead of the promised page.")
        if g.startswith("console:") or 1 in sig:
            return ("A JavaScript error is raised in the browser, so part of the "
                    "page's behaviour may silently not run for real users.")
        if g.startswith("axe:") or 5 in sig:
            return ("An accessibility violation means some users cannot perceive or "
                    "operate this control (assistive-technology impact).")
        if 4 in sig:
            return ("Invalid input is accepted (or valid input rejected) without a "
                    "clear message, so a user cannot tell what went wrong and bad "
                    "data can enter the application.")
        if 6 in sig:
            return ("The page does not deliver the outcome the user's action "
                    "promises, so the flow is broken from the user's point of view.")
        if 3 in sig:
            return ("The page renders broken content, so it does not present "
                    "correctly to the user.")
        return "The observed behaviour contradicts what the target itself promises."

    def block_inadmissible(self, confirmed: list[tuple[PossibleBug, str]]) -> list:
        """Findings whose evidence rests on a synthetic route — never published.

        They are not silently deleted: each one is recorded in ``self.suppressed``
        so the run can explain what was withheld and why.
        """
        if self.policy is None:
            return []
        out = []
        for pb, repro in confirmed:
            reason = self.policy.blocked_reason(pb.evidence.url_after
                                                or pb.evidence.url_before)
            if reason:
                out.append((pb, repro))
                self.suppressed.append({
                    "journey": pb.journey, "test_id": pb.test_id,
                    "group": pb.group, "summary": pb.summary[:200],
                    "reason": reason, "reproduced": repro,
                    "signals": [s.signal for s in pb.signals]})
        return out

    def _classify_network(self, pb: PossibleBug) -> list[dict]:
        """Where did each failing request come from?

        A 404 is evidence, not a verdict: a dead link the target published is the
        target's fault, a third-party analytics beacon failing is not, and a URL
        QArena guessed can never be filed at all.
        """
        out: list[dict] = []
        for f in pb.evidence.http_failures[:10]:
            url = str(f.get("url") or "")
            status = f.get("status")
            origin, note = "target", ""
            if self.policy is not None:
                origin = self.policy.network_origin(url)
                blocked = self.policy.blocked_reason(url)
                src = self.policy.link_source(url)
                if blocked:
                    origin, note = "synthetic probe", blocked
                elif src and src.get("link_name"):
                    origin = "dead link published by the target"
                    note = (f"linked as \"{src['link_name']}\" from "
                            f"`{src.get('from_route') or '/'}`")
                elif src:
                    origin = "link published by the target"
                    note = f"linked from `{src.get('from_route') or '/'}`"
                elif origin == "third_party":
                    note = "third-party host — not owned by the target"
                elif isinstance(status, int) and status >= 500:
                    note = "server error on a target-owned resource"
            elif self.appmap is None:
                note = "origin not classified (no app map available for this report)"
            out.append({"url": url, "status": status, "origin": origin, "note": note})
        return out
