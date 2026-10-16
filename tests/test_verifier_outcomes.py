"""
P2 — the replay verifier must say exactly what happened.

The defect: ``Verifier.verify`` wrapped each replay in ``except Exception:
continue``. A browser that crashed, a target that went down mid-replay, or a step
that could not complete were all silently treated as "the defect did not
reproduce", and the suspect was then reported as *flaky* — a claim about the
application manufactured out of a failure of the test environment. Findings that
had already been proven could also lose their legs to an environment problem.

The taxonomy under test (``core.verifier``):
  matched              the defect recurred >= 2/3  -> confirmed and filed
  flaky                it recurred, but below the bar -> reported as flaky
  not_reproduced       every replay ran cleanly and it never recurred
  inconclusive         some replay could not decide (a step errored)
  infrastructure_error no replay reached the application at all
"""
from __future__ import annotations

import pytest

from core import config
from core.contracts import (ExpectedContract, PossibleBug, SignalHit,
                            StepEvidence, StepSpec)
from core.judge import Judge
from core.verifier import (FLAKY, INCONCLUSIVE, INFRASTRUCTURE, MATCHED,
                           NOT_REPRODUCED, ReplayOutcome, Verifier)


# ----------------------------------------------------------------- test doubles
class _Step:
    """What the tester returns for one replay step."""

    def __init__(self, kind: str) -> None:
        self.kind = kind


class _Tester:
    """Replays a scripted sequence of per-attempt behaviours."""

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.attempt = 0
        self.calls = 0
        self.sbx = self

    def seed_target(self) -> bool:
        return True

    def run_test(self, tc, tracing: bool = True):
        self.calls += 1
        kind = self.script[min(self.attempt, len(self.script) - 1)]
        self.attempt += 1
        if kind == "crash":
            raise RuntimeError("Target page, context or browser has been closed")
        if kind == "error_step":
            return "error", [StepEvidence(step=1, action="goto", target="/cart",
                                          status="error",
                                          note="locator timed out: button missing")]
        if kind == "clean":
            # the replay ran and the page behaved: nothing to judge
            return "passed", [StepEvidence(
                step=1, action="goto", target="/cart", status="passed",
                url_before="http://site.test/", url_after="http://site.test/cart")]
        return "failed", [StepEvidence(
            step=1, action="goto", target="/cart", status="failed",
            url_before="http://site.test/", url_after="http://site.test/cart",
            http_failures=[{"method": "GET", "url": "http://site.test/cart",
                            "status": 404}])]


class _Judge:
    """Reports the suspect's own group, but only when the evidence shows it."""

    def evaluate(self, journey, test_id, step_index, ev, steps=None):
        if not ev.http_failures:
            return []
        return [PossibleBug(journey=journey, test_id=test_id, group=GROUP,
                            step_index=step_index, score=2.0,
                            signals=[SignalHit(signal=2, name="http",
                                               weight=2.0, evidence="404")],
                            summary="GET /cart -> 404", repro_steps=[],
                            evidence=ev)]


GROUP = "http:GET /cart -> 404"


def _suspect() -> PossibleBug:
    return PossibleBug(journey="cart journey", test_id="TC_001", group=GROUP,
                       step_index=1, score=2.0,
                       signals=[SignalHit(signal=2, name="http", weight=2.0,
                                          evidence="404")],
                       summary="GET /cart -> 404", repro_steps=[],
                       evidence=StepEvidence(step=1, action="goto",
                                             target="/cart", status="failed",
                                             url_before="http://site.test/",
                                             url_after="http://site.test/cart",
                                             http_failures=[
                                                 {"method": "GET",
                                                  "url": "http://site.test/cart",
                                                  "status": 404}],
                                             ))


def _verify(script: list[str]):
    tester = _Tester(script)
    verifier = Verifier(tester, _Judge(), emit=lambda e: None)
    ok, count, outcome = verifier.verify(_suspect())
    return ok, count, outcome, tester


# ----------------------------------------------------------------------- tests
def test_matched_reproduced_three_times_is_confirmed():
    ok, count, outcome, _ = _verify(["match"] * 3)
    assert ok is True and count == "3/3"
    assert outcome.kind == MATCHED
    assert outcome.matched == 3 and outcome.completed == 3


def test_clean_non_reproduction_is_labelled_not_reproduced():
    ok, count, outcome, _ = _verify(["clean"] * 3)
    assert ok is False
    assert count == "0/3"
    assert outcome.kind == NOT_REPRODUCED
    assert "did not recur" in outcome.detail


def test_one_of_three_is_flaky_not_a_bug():
    """It really did recur once — say flaky, do not file it and do not deny it."""
    ok, count, outcome, tester = _verify(["match", "clean", "clean"])
    assert ok is False
    assert count == "1/3"
    assert outcome.kind == FLAKY
    assert "confirmation bar" in outcome.detail
    assert tester.calls == config.REPLAY_TIMES


def test_all_replays_crashing_is_an_infrastructure_error_not_flakiness():
    ok, count, outcome, tester = _verify(["crash"] * 3)
    assert ok is False
    assert outcome.kind == INFRASTRUCTURE
    assert outcome.completed == 0
    # the count never pretends attempts were valid
    assert count == "0/0 (of 3 attempted)", count
    assert "no replay attempt could run against the application" in outcome.detail
    assert "browser has been closed" in outcome.detail   # the real cause, admitted
    assert tester.calls == config.REPLAY_TIMES
    assert not any(a["result"] == NOT_REPRODUCED for a in outcome.per_attempt)


def test_one_crashing_replay_does_not_become_a_clean_non_reproduction():
    """Two clean non-reproductions + one crash is NOT a clean 0/3."""
    ok, _, outcome, _ = _verify(["clean", "crash", "clean"])
    assert ok is False
    assert outcome.kind == INCONCLUSIVE, outcome.kind
    assert "not every attempt completed" in outcome.detail


def test_a_step_that_cannot_complete_makes_the_replay_inconclusive():
    ok, count, outcome, _ = _verify(["error_step"] * 3)
    assert ok is False
    assert outcome.kind == INCONCLUSIVE
    assert "could not decide" in outcome.detail
    assert outcome.completed == 0
    assert count == "0/0 (of 3 attempted)"


def test_mixed_partial_failure_still_confirms_a_real_defect():
    """Preservation: an infrastructure hiccup must not erase a proven finding."""
    ok, count, outcome, _ = _verify(["match", "match", "crash"])
    assert ok is True, "2 real reproductions were thrown away by one crash"
    assert outcome.kind == MATCHED
    assert outcome.matched == 2 and outcome.completed == 2
    assert "could not run" in outcome.detail     # the hiccup is still admitted
    assert count == "2/2 (of 3 attempted)", count


def test_confirmation_threshold_is_not_weakened():
    """Two matches out of three attempts is the bar; one is not enough."""
    ok1, _, out1, _ = _verify(["match", "clean", "clean"])
    ok2, _, out2, _ = _verify(["match", "match", "clean"])
    assert out1.confirmed is False and ok1 is False, "1/3 must never confirm"
    assert out2.confirmed is True and ok2 is True, "2/3 must confirm"
    assert config.REPLAY_CONFIRM_MIN == 2


def test_outcome_reports_attempts_and_valid_attempts_separately():
    _, _, outcome, _ = _verify(["match", "error_step", "clean"])
    assert outcome.attempts == 3
    assert outcome.completed == 2
    assert outcome.matched == 1
    assert outcome.summary() == "1/2 (of 3 attempted)"
