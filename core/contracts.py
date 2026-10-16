"""
Data contracts — architecture section 4. Pydantic models shared by all stages:
AppMap.json (3b->4), Validated Test Case JSON (4->5a), Confirmed Bug Report (7a).
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

INTERACTIVE_ROLES = ("link", "button", "textbox", "searchbox", "spinbutton",
                     "combobox", "checkbox", "radio")


# ------------------------------------------------------------------ Stage 3b
class StateElement(BaseModel):
    id: str
    role: str
    name: str = ""
    required: bool = False
    enabled: bool = True


class AppState(BaseModel):
    state_id: str
    route: str                       # canonical route, e.g. /product/* or /search
    url: str                         # representative concrete URL
    state_hash: str
    http_status: int = 200
    elements: list[StateElement] = Field(default_factory=list)
    # provenance: True when QArena *guessed* the URL (enrichment/search probe)
    # instead of reaching it through a real link on the target. A probed route
    # 404ing is NOT an application defect — never report it as one.
    probed: bool = False
    # provenance: True when the explorer actually SAW a cart-badge style element
    # on this state. Oracles that assert a badge count may only be planned when
    # some state really has one — otherwise "badge not found" would be filed as
    # a defect on apps that never had a badge (false positive).
    has_cart_badge: bool = False
    # HOW QArena learned this route exists — the field every admissibility
    # decision reads (see core/provenance.py). One of:
    #   "seed"        the app root the user handed us
    #   "link"        reached by following a real link the target published
    #   "interaction" the target navigated here itself after a real user action
    #   "probe"       QArena GUESSED this URL (cart/checkout/search enrichment)
    # A probe-only state that never returned 2xx is not part of the application:
    # its failure may never be planned against or filed as a defect.
    discovered_via: str = ""
    # the page QArena was standing on when it found this state, plus the link's
    # own label — this is what makes a broken-link report concrete
    discovered_from: str = ""
    link_name: str = ""
    # set on states reached by following a link: what that link predicted
    # (the href). A 404 here is a broken link; a 404 on a probe is not.
    linked_href: str = ""

    def provenance_rank(self) -> int:
        """Lower is stronger evidence: seed(0) link(1) interaction(2) probe(3)."""
        order = {"seed": 0, "link": 1, "interaction": 2, "probe": 3}
        via = (self.discovered_via or "").strip().lower()
        if via in order:
            return order[via]
        return 3 if self.probed else 1


class AppMap(BaseModel):
    app_url: str
    description: str
    states: list[AppState] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)       # {from, to, via}
    catalog_prices: dict[str, int] = Field(default_factory=dict)  # name -> cents
    generated_by: str = "explorer"


# ------------------------------------------------------------------- Stage 4
class StepSpec(BaseModel):
    step: int
    action: Literal["goto", "click", "dblclick", "fill", "press", "select",
                    "check_text", "assert_badge", "wait"]
    target: str = ""                 # route for goto; text for check_text; value otherwise
    role: str = ""                   # element roles for click/fill/select/dblclick/press
    name: str = ""                   # accessible name
    value: str = ""                  # fill/select value


class ExpectedContract(BaseModel):
    """Deterministic stage-5a/5b oracle. All fields optional; only set ones check."""
    success_url_contains: str = ""
    disallowed_url: str = ""
    text_required: list[str] = Field(default_factory=list)
    text_forbidden: list[str] = Field(default_factory=list)
    should_reject: bool = False          # negative test: an error must appear
    error_text: str = ""                 # expected validation message substring
    no_new_errors_5xx: bool = True       # default: a correct step makes no 4xx/5xx
    badge_equals: Optional[int] = None   # cart badge exact value
    money_non_negative: bool = False     # no "-$" amount may be visible


class TestCase(BaseModel):
    test_id: str
    journey_type: str                  # happy_path | edge_case_negative | adversarial
    priority_score: int = 50
    title: str
    steps: list[StepSpec]
    expected: ExpectedContract = Field(default_factory=ExpectedContract)


# ---------------------------------------------------------------- Stage 5a/5b
class StepEvidence(BaseModel):
    step: int
    action: str
    target: str
    url_before: str = ""
    url_after: str = ""
    status: Literal["passed", "failed", "error"] = "passed"
    note: str = ""
    screenshot: str = ""
    console_errors: list[str] = Field(default_factory=list)
    http_failures: list[dict] = Field(default_factory=list)   # {method,url,status}
    broken_items: list[dict] = Field(default_factory=list)    # {kind,name,detail}
    axe_violations: list[dict] = Field(default_factory=list)  # {id,impact,target}
    contract_failures: list[str] = Field(default_factory=list)
    intent_findings: list[str] = Field(default_factory=list)


class SignalHit(BaseModel):
    signal: int
    name: str
    weight: float
    evidence: str


class PossibleBug(BaseModel):
    journey: str
    test_id: str = ""
    group: str = ""
    step_index: int
    signals: list[SignalHit]
    score: float
    summary: str
    repro_steps: list[StepSpec]
    oracle: list[dict] = Field(default_factory=list)  # deterministic re-check spec
    evidence: StepEvidence


# ---------------------------------------------------------------- Stage 6a/7a
class FiledBug(BaseModel):
    bug_id: str
    title: str
    severity: Literal["Critical", "Major", "Minor"]
    journey: str
    route: str = ""
    element_name: str = ""
    signals: list[int]
    score: float
    reproduced_count: str                # "matched/completed" — valid attempts only
    steps_to_reproduce: list[str]
    expected_behavior: str
    actual_behavior: str
    # what the confidence number is made of (signals + valid reproductions)
    confidence_basis: str = ""
    evidence: dict = Field(default_factory=dict)  # screenshot/trace/report paths


class FlakySuspect(BaseModel):
    journey: str
    summary: str
    reproduced: str                      # "1/3"
    signals: list[int]
    # "flaky"        replay ran and the defect only sometimes recurred
    # "inconclusive" replay ran but could not decide (a step errored)
    # "infrastructure" replay could not run at all (browser/target problem)
    # Only "flaky" is evidence about the application; the others are evidence
    # about the test environment and are reported separately.
    kind: str = "flaky"
    detail: str = ""                     # why it was inconclusive/infrastructure
    attempts_matched: int = 0            # completed attempts that reproduced it
    attempts_completed: int = 0          # attempts that ran to a verdict
    attempts_total: int = 0              # attempts we tried


class RunResult(BaseModel):
    run_id: str
    target_url: str
    description: str
    status: str = "running"
    llm_mode: str = "heuristic"
    tokens_used: int = 0
    states_found: int = 0
    tests_planned: int = 0
    tests_passed: int = 0
    steps_executed: int = 0
    # a planned journey that never produced a verdict (a later stage failed).
    # Kept separate from tests_passed so coverage cannot overstate what ran.
    tests_unexecuted: int = 0
    filed_bugs: list[FiledBug] = Field(default_factory=list)
    flaky: list[FlakySuspect] = Field(default_factory=list)
    # verified findings that may not be published because their evidence rests
    # on a route QArena invented (never dropped silently — always explained)
    suppressed: list[dict] = Field(default_factory=list)
    exported_tests: list[str] = Field(default_factory=list)
    rejected_steps: list[dict] = Field(default_factory=list)
    coverage: dict = Field(default_factory=dict)
    benchmark: dict = Field(default_factory=dict)
    # non-fatal stage failures, e.g. {"reporter": "UnicodeEncodeError: ..."}.
    # Populated when a late stage fails after verified findings already exist;
    # the run is then reported as "partial" rather than thrown away.
    stage_errors: dict[str, str] = Field(default_factory=dict)
    error: str = ""
