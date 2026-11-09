"""Reward forms and the measurement contract (paper sections 2, 3, 5, 7.1).

All five arms of section 7.1 are computed from the same metered episode
signals (StepSignals from the environment): verified gives, verified takes,
and P_E. One code path, per the pre-registration.

Contract choices declared for this toy frame (C(f), paper section 3.1):
  * Training A per step is the verified take (harvest) -- the take-only
    proxy disclosed in section 6. Interval A for reporting is
    sum(gives) - sum(takes), signed, unfloored (section 2).
  * P_E is total food stock across both patches; the healthy baseline is
    total capacity (40 counts). Cropped E' is the healthiest patch only,
    mirroring the worked frame in section 5.
  * Y on an interval follows section 3.4: valid only if A + E > 0;
    A = E = 0 is undefined (None), never reported as 0.
"""

import math
from dataclasses import dataclass


def harmonic(a: float, e: float):
    """Y = 2AE/(A+E). None where undefined (A + E <= 0), per section 3.4."""
    if a + e <= 0:
        return None
    return 2.0 * a * e / (a + e)


def arithmetic(a: float, e: float) -> float:
    return (a + e) / 2.0


def minimum(a: float, e: float) -> float:
    return min(a, e)


def geometric(a: float, e: float) -> float:
    if a <= 0 or e <= 0:
        return 0.0
    return math.sqrt(a * e)


@dataclass
class ToyContract:
    """C(f) for frame 'two-patch-toy-v1'."""

    frame: str = "two-patch-toy-v1"
    resource: str = "food (counts)"
    baseline_e: float = 40.0
    health_line: float = 16.0  # constraint arm: 40% of baseline
    constraint_penalty: float = 1.0

    def arm_reward(self, arm: str, take: float, p_e: float) -> float:
        """Per-step training reward for one arm, from the same signals."""
        a = take  # declared take-only proxy (paper section 6)
        e = p_e
        if arm == "harmonic":
            y = harmonic(a, e)
            return 0.0 if y is None else y
        if arm == "arithmetic":
            return arithmetic(a, e)
        if arm == "minimum":
            return minimum(a, e)
        if arm == "geometric":
            return geometric(a, e)
        if arm == "constraint":
            penalty = self.constraint_penalty if e < self.health_line else 0.0
            return a - penalty
        if arm == "task_only":  # reference baseline of section 6, not a 7.1 arm
            return a
        raise ValueError(f"unknown arm: {arm}")

    @staticmethod
    def interval_y(gives: float, takes: float, p_e: float):
        """Interval Y from the section 2 ledger: A = gives - takes."""
        return harmonic(gives - takes, p_e)


ARMS = ("harmonic", "arithmetic", "minimum", "geometric", "constraint", "task_only")
