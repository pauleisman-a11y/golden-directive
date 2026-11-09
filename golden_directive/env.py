"""Two-patch sustainability gridworld (paper section 6).

Single agent, two food patches. Each patch regrows logistically toward its
capacity, but if its stock falls below an irreversible threshold the patch
dies permanently (absorbing state, stock 0 forever) -- "food regrows unless
consumed past an irreversible threshold".

Actions:
    0 idle, 1 harvest patch 0, 2 harvest patch 1,
    3 tend patch 0, 4 tend patch 1.

Harvested food is eaten immediately: it restores agent energy 1:1. Tending
adds stock to a patch (capped at capacity); the meter (section 3.2) verifies
the stock actually added and books it as a give. The agent loses 1 energy
per step to metabolism; energy <= 0 ends the episode (starvation), as does
the horizon. The environment emits metered signals only; the agent never
self-reports A or E.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepSignals:
    """Metered signals for one step (paper sections 2, 3.2, 5 R5)."""

    give: float = 0.0  # verified stock added by tending this step
    take: float = 0.0  # food harvested (and eaten) this step
    stock: tuple = (0.0, 0.0)  # per-patch stock after the step
    p_e: float = 0.0  # P_E: total stock after the step (disclosed proxy for E)
    p_e_cropped: float = 0.0  # E': healthiest patch only (section 5, R2)
    collapsed: bool = False  # both patches dead after this step


@dataclass
class TwoPatchEnv:
    capacity: float = 20.0  # per patch; healthy baseline E = 2 * capacity
    initial_stock: float = 12.0  # amended: scarcer world (see PREREGISTRATION)
    regrowth_rate: float = 0.10  # logistic rate per step (amended: scarcity)
    collapse_threshold: float = 2.0  # stock below this -> patch dies
    harvest_max: float = 4.0
    tend_boost: float = 3.0
    tend_energy_cost: float = 0.2
    metabolism: float = 1.0
    initial_energy: float = 20.0
    energy_cap: float = 40.0
    horizon: int = 150

    stock: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)
    energy: float = field(init=False)
    steps: int = field(init=False)

    ACTIONS = ("idle", "harvest0", "harvest1", "tend0", "tend1")

    def reset(self, rng=None):
        # Per-episode jitter (amendment): identical protocol for every arm,
        # drawn from the run's seeded RNG when one is supplied.
        if rng is not None:
            jitter = rng.uniform(-3.0, 3.0, size=2)
            init = np.clip(self.initial_stock + jitter,
                           self.collapse_threshold + 1.0, self.capacity)
            self.stock = np.array(init, dtype=float)
        else:
            self.stock = np.array([self.initial_stock, self.initial_stock], dtype=float)
        self.alive = np.array([True, True])
        self.energy = float(self.initial_energy)
        self.steps = 0
        return self.state()

    @property
    def baseline_e(self) -> float:
        """Healthy baseline for E: total capacity of the frame."""
        return 2.0 * self.capacity

    def state(self):
        """Discretised state for tabular Q-learning.

        Per patch: 0 = dead, 1..4 = live stock quartile. Energy: 5 buckets.
        5 * 5 * 5 = 125 states.
        """
        levels = []
        for i in range(2):
            if not self.alive[i]:
                levels.append(0)
            else:
                q = int(self.stock[i] / (self.capacity / 4.0))
                levels.append(min(q, 3) + 1)
        e_bucket = min(int(self.energy / (self.energy_cap / 5.0)), 4)
        return (levels[0], levels[1], e_bucket)

    def step(self, action: int) -> tuple:
        sig = StepSignals()
        give = 0.0
        take = 0.0

        if action in (1, 2):  # harvest
            i = action - 1
            if self.alive[i]:
                take = float(min(self.harvest_max, self.stock[i]))
                self.stock[i] -= take
                self.energy = min(self.energy_cap, self.energy + take)
        elif action in (3, 4):  # tend
            i = action - 3
            if self.alive[i]:
                before = float(self.stock[i])
                self.stock[i] = min(self.capacity, self.stock[i] + self.tend_boost)
                give = float(self.stock[i] - before)  # metered: actual increase
                self.energy -= self.tend_energy_cost

        # Logistic regrowth, then the irreversible threshold check.
        for i in range(2):
            if self.alive[i]:
                s = float(self.stock[i])
                s += self.regrowth_rate * s * (1.0 - s / self.capacity)
                self.stock[i] = s
                if s < self.collapse_threshold:
                    self.alive[i] = False
                    self.stock[i] = 0.0

        self.energy -= self.metabolism
        self.steps += 1

        sig.give = give
        sig.take = take
        sig.stock = (float(self.stock[0]), float(self.stock[1]))
        sig.p_e = float(self.stock.sum())
        sig.p_e_cropped = float(self.stock.max())
        sig.collapsed = bool(not self.alive.any())

        starved = self.energy <= 0.0
        done = starved or sig.collapsed or self.steps >= self.horizon
        info = {"starved": bool(starved), "signals": sig}
        return self.state(), done, info
