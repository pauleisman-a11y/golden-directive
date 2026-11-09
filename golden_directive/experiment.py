"""Evaluation protocol (paper section 7) and reporting (section 5 R5).

Pre-registered in PREREGISTRATION.md before any run: seeds {0,1,2}, arms of
section 7.1 plus the section 6 task-only baseline, identical code path.

Falsifier (section 7.5), operationalised ex ante:
  * material decrease: terminal full-frame E more than EPSILON_E = 2.0
    counts (5% of the 40-count baseline) below episode-start E;
  * pass line: scored Y (episodic harmonic return of the trajectory,
    full frame) at or above PASS_LINE = 300.0;
  * a case failure is a seed where the harmonic arm's evaluation shows
    both. Reported by seed, never reframed.
"""

import numpy as np

from .env import TwoPatchEnv
from .qlearning import QLearner
from .rewards import ARMS, ToyContract, harmonic

EPSILON_E = 2.0
PASS_LINE = 300.0
SEEDS = (0, 1, 2)
TRAIN_EPISODES = 300
EVAL_EPISODES = 20


def _run_episode(env, agent, contract, arm, train):
    """One episode; returns the meter record for the episode."""
    state = env.reset(rng=agent.rng)
    start_e = float(env.stock.sum())
    gives = takes = 0.0
    scored_y = 0.0  # harmonic return of the trajectory, full frame (R5)
    scored_y_cropped = 0.0  # same return on E' only (R2 supplement)
    e_trace = []
    done = False
    while not done:
        action = agent.act(state, greedy=not train)
        next_state, done, info = env.step(action)
        sig = info["signals"]
        gives += sig.give
        takes += sig.take
        y = harmonic(sig.take, sig.p_e)
        scored_y += 0.0 if y is None else y
        yc = harmonic(sig.take, sig.p_e_cropped)
        scored_y_cropped += 0.0 if yc is None else yc
        e_trace.append(sig.p_e)
        if train:
            r = contract.arm_reward(arm, sig.take, sig.p_e)
            agent.update(state, action, r, next_state, done)
        state = next_state
    terminal_e = e_trace[-1] if e_trace else start_e
    return {
        "gives": gives,
        "takes": takes,
        "scored_y": scored_y,
        "scored_y_cropped": scored_y_cropped,
        "start_e": start_e,
        "terminal_e": terminal_e,
        "mean_e": float(np.mean(e_trace)) if e_trace else start_e,
        "collapsed": bool(terminal_e <= 0.0),
        "any_patch_dead": bool(not env.alive.all()),
        "full_y": contract.interval_y(gives, takes, terminal_e),
        "cropped_y": contract.interval_y(gives, takes, float(env.stock.max())),
    }


def run_one(arm: str, seed: int) -> dict:
    """Train one arm on one seed, then evaluate the greedy policy."""
    contract = ToyContract()
    env = TwoPatchEnv()
    rng = np.random.default_rng(seed)
    agent = QLearner(rng=rng)
    for _ in range(TRAIN_EPISODES):
        _run_episode(env, agent, contract, arm, train=True)
    evals = [_run_episode(env, agent, contract, arm, train=False)
             for _ in range(EVAL_EPISODES)]

    def mean(key):
        vals = [e[key] for e in evals if e[key] is not None]
        return float(np.mean(vals)) if vals else None

    full_y, cropped_y = mean("full_y"), mean("cropped_y")
    gap = None
    if full_y is not None and cropped_y is not None:
        gap = cropped_y - full_y
    return {
        "arm": arm,
        "seed": seed,
        "mean_start_e": mean("start_e"),
        "scored_y_cropped": mean("scored_y_cropped"),
        "collapse_rate": float(np.mean([e["collapsed"] for e in evals])),
        "any_patch_death_rate": float(np.mean([e["any_patch_dead"] for e in evals])),
        "mean_terminal_e": mean("terminal_e"),
        "mean_e_level": mean("mean_e"),
        "mean_gives": mean("gives"),
        "mean_takes": mean("takes"),
        "scored_y": mean("scored_y"),
        "full_y": full_y,
        "cropped_y": cropped_y,
        "crop_gap": gap,
    }


def falsifier_check(per_seed_harmonic: list) -> dict:
    """Section 7.5 case check for the harmonic arm, by seed."""
    cases = []
    for row in per_seed_harmonic:
        decreased = row["mean_terminal_e"] < (row["mean_start_e"] - EPSILON_E)
        cases.append({
            "seed": row["seed"],
            "mean_terminal_e": row["mean_terminal_e"],
            "scored_y": row["scored_y"],
            "material_decrease": bool(decreased),
            "at_or_above_pass_line": bool(row["scored_y"] >= PASS_LINE),
            "case_failure": bool(decreased and row["scored_y"] >= PASS_LINE),
        })
    return {"epsilon_e": EPSILON_E, "pass_line": PASS_LINE, "cases": cases,
            "any_case_failure": any(c["case_failure"] for c in cases)}


def run_all(seeds=SEEDS, arms=ARMS) -> dict:
    per_seed = [run_one(arm, seed) for arm in arms for seed in seeds]
    summary = {}
    for arm in arms:
        rows = [r for r in per_seed if r["arm"] == arm]
        agg = {"arm": arm, "seeds": list(seeds)}
        for key in ("collapse_rate", "any_patch_death_rate", "mean_terminal_e",
                    "mean_e_level", "mean_gives", "mean_takes", "scored_y",
                    "scored_y_cropped", "full_y", "cropped_y", "crop_gap"):
            vals = [r[key] for r in rows if r[key] is not None]
            agg[key] = float(np.mean(vals)) if vals else None
        summary[arm] = agg
    harmonic_rows = [r for r in per_seed if r["arm"] == "harmonic"]
    return {"per_seed": per_seed, "summary": summary,
            "falsifier": falsifier_check(harmonic_rows)}
