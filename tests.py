#!/usr/bin/env python3
"""Plain-assert checks. Run: python3 tests.py

Includes the worked frame from paper section 5, which the code must
reproduce exactly: A=40, P_E=120 -> Y=60; cropped E'=190 -> Y ~= 66.1.
"""
from golden_directive.env import TwoPatchEnv
from golden_directive.rewards import ToyContract, harmonic

# Section 5 worked frame.
assert harmonic(40, 120) == 60.0
cropped = harmonic(40, 190)
assert abs(cropped - 66.1) < 0.05, cropped
assert abs((cropped - 60.0) - 6.1) < 0.05

# Section 3.4 edge cases: 0/0 undefined, never 0; one term at 0 -> Y = 0.
assert harmonic(0, 0) is None
assert harmonic(0, 50) == 0.0
assert harmonic(50, 0) == 0.0
assert harmonic(-60, 40) is None  # A + E <= 0: closed deficit

# Environment: irreversible collapse is absorbing.
env = TwoPatchEnv()
env.reset()
for _ in range(200):
    env.step(1)  # harvest patch 0 only, relentlessly
assert not env.alive[0] and env.stock[0] == 0.0
for _ in range(10):
    _, _, info = env.step(3)  # tending a dead patch gives nothing
    assert info["signals"].give == 0.0

# Contract: every arm consumes the same signals and returns a float.
c = ToyContract()
for arm in ("harmonic", "arithmetic", "minimum", "geometric", "constraint", "task_only"):
    r = c.arm_reward(arm, take=3.0, p_e=25.0)
    assert isinstance(r, float), (arm, r)

print("all tests passed")
