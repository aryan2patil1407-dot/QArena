"""
Stage 6a — Deterministic 3x Replay Verifier (zero AI).
[Playwright: new_context] clean browser   [Playwright + JSON] replay exact steps
[Python Loop] repeat 3x (world re-seeded each attempt when the hook exists)
[Python Counter] >=2/3 valid reproductions => CONFIRMED

Every attempt is classified, because "the defect did not reappear" and "the
replay could not run" are different findings:

``matched``             the same evidence group re-triggered.
``flaky``               it re-triggered, but fewer than the confirmation minimum
                        times — a real but unreliable defect.
``not_reproduced``      every attempted replay ran to a clean verdict and the
                        defect never recurred.
``inconclusive``        some replay could not decide (a step errored, so the page
                        was not in the state under test).
``infrastructure_error`` no replay reached the application at all (browser
                        crash, navigation timeout, target outage).

Only ``matched`` counts toward confirmation. A crashed browser is evidence about
the test environment, never about the website, so it is never called flaky.
Attempts that could not run are reported as such instead of being silently
absorbed into a "1/3" that reads like the application misbehaved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .contracts import ExpectedContract, PossibleBug, TestCase
from .judge import Judge
from .tester import Tester

MATCHED = "matched"
FLAKY = "flaky"
NOT_REPRODUCED = "not_reproduced"
INCONCLUSIVE = "inconclusive"
INFRASTRUCTURE = "infrastructure_error"


@dataclass
class ReplayOutcome:
    """What actually happened across the replay attempts."""

    attempts: int = 0                     # attempts we tried
    completed: int = 0                    # attempts that ran to a verdict
    matched: int = 0                      # attempts that reproduced the defect
    kind: str = NOT_REPRODUCED            # outcome for the *suspect*
    detail: str = ""                      # why, in plain words
    per_attempt: list[dict] = field(default_factory=list)

    @property
    def count(self) -> str:
        """"matched/completed" — the honest fraction, never implied attempts."""
        return f"{self.matched}/{self.completed}"

    @property
    def confirmed(self) -> bool:
        return self.matched >= config.REPLAY_CONFIRM_MIN

    def summary(self) -> str:
        base = f"{self.matched}/{self.completed}"
        if self.completed < self.attempts:
            base += f" (of {self.attempts} attempted)"
        return base


class Verifier:
    def __init__(self, tester: Tester, judge: Judge, emit=lambda e: None) -> None:
        self.tester = tester
        self.judge = judge
        self.emit = emit

    # ---------------------------------------------------------------- replay
    def verify(self, pb: PossibleBug) -> tuple[bool, str, ReplayOutcome]:
        """Replay a suspect in clean browsers.

        Returns ``(confirmed, "matched/completed", outcome)``. The count string
        only ever describes attempts that actually ran: a suspect that failed
        replay because the browser crashed reports ``0/0 (of 3 attempted)``,
        not a misleading ``0/3``.
        """
        out = ReplayOutcome(attempts=config.REPLAY_TIMES)
        original = {h.signal for h in pb.signals}
        # replay the ORIGINAL contract: form/intent oracles are contract-driven
        expected = (ExpectedContract(**pb.oracle[0]) if pb.oracle
                    else ExpectedContract())
        repro_tc = TestCase(test_id="REPRO", journey_type="replay",
                            title=pb.journey, steps=pb.repro_steps,
                            expected=expected)
        problems: list[str] = []

        for attempt in range(1, config.REPLAY_TIMES + 1):
            self.tester.sbx.seed_target()
            try:
                _, evidences = self.tester.run_test(repro_tc, tracing=False)
            except Exception as exc:
                # the replay never reached the application
                problems.append(f"{exc.__class__.__name__}: {str(exc)[:120]}")
                out.per_attempt.append({"attempt": attempt,
                                        "result": INFRASTRUCTURE,
                                        "detail": str(exc)[:200]})
                self.emit({"type": "replay", "journey": f"{pb.test_id} {pb.group}",
                           "attempt": attempt, "matched": False,
                           "result": INFRASTRUCTURE})
                continue

            if any(e.status == "error" for e in evidences):
                # a step errored: the page was not in the state under test, so
                # this attempt cannot testify either way
                note = next((e.note for e in evidences if e.status == "error"), "")
                problems.append(note[:120] or "a replay step errored")
                out.per_attempt.append({"attempt": attempt, "result": INCONCLUSIVE,
                                        "detail": note[:200]})
                self.emit({"type": "replay", "journey": f"{pb.test_id} {pb.group}",
                           "attempt": attempt, "matched": False,
                           "result": INCONCLUSIVE})
                continue

            matched = False
            for i, ev in enumerate(evidences):
                for again in self.judge.evaluate(pb.journey, "REPRO", i, ev,
                                                 pb.repro_steps):
                    if again.group == pb.group or \
                            ({h.signal for h in again.signals} & original and
                             again.group.split(":")[0] == pb.group.split(":")[0]):
                        matched = True
            out.completed += 1
            if matched:
                out.matched += 1
            out.per_attempt.append(
                {"attempt": attempt,
                 "result": MATCHED if matched else NOT_REPRODUCED})
            self.emit({"type": "replay", "journey": f"{pb.test_id} {pb.group}",
                       "attempt": attempt, "matched": matched,
                       "result": MATCHED if matched else NOT_REPRODUCED})

        self._classify(out, problems)
        return out.confirmed, out.summary(), out

    # ------------------------------------------------------------- judgement
    @staticmethod
    def _classify(out: ReplayOutcome, problems: list[str]) -> None:
        """Decide what the attempts collectively mean for the *suspect*.

        The order matters: "nothing ran" outranks everything, a proven defect
        outranks a flaky one, and an attempt we could not complete weakens any
        negative conclusion — it never creates a positive one.
        """
        infra = [a for a in out.per_attempt if a["result"] == INFRASTRUCTURE]
        incon = [a for a in out.per_attempt if a["result"] == INCONCLUSIVE]
        bits = [f"reproduced {out.matched}/{out.completed} completed replay(s)"]
        if incon:
            bits.append(f"{len(incon)} replay(s) could not decide "
                        f"({incon[0].get('detail', '')[:80]})")
        if infra:
            bits.append(f"{len(infra)} replay(s) could not run at all "
                        f"({infra[0].get('detail', '')[:80]})")

        if out.completed == 0:
            # not one attempt produced a verdict: no statement about the app
            why = ("; ".join(problems[:2]) if problems
                   else "the replay never produced a verdict")
            if infra:
                out.kind = INFRASTRUCTURE
                out.detail = (f"no replay attempt could run against the "
                              f"application: {why}")
            else:
                out.kind = INCONCLUSIVE
                out.detail = (f"no replay reached a verdict — the replay could "
                              f"not decide: {why}")
            return
        if out.confirmed:
            out.kind = MATCHED
            out.detail = (f"reproduced {out.matched}/{out.completed} completed "
                          f"clean-browser replays"
                          + ("; " + "; ".join(bits[1:]) if len(bits) > 1 else ""))
            return
        if out.matched > 0:
            # it really did recur — just not reliably enough to file
            out.kind = FLAKY
            out.detail = (f"{'; '.join(bits)} — below the "
                          f"{config.REPLAY_CONFIRM_MIN}/{config.REPLAY_TIMES} "
                          f"confirmation bar")
            return
        if infra or incon:
            # a negative conclusion, but not a clean one
            out.kind = INCONCLUSIVE
            out.detail = "; ".join(bits) + (" — the defect did not recur in the "
                                            "attempts that completed, but not "
                                            "every attempt completed")
            return
        out.kind = NOT_REPRODUCED
        out.detail = (f"the replay ran cleanly {out.completed}/{out.attempts} "
                      f"times and the defect did not recur")
