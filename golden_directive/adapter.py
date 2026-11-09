"""Adapter notes-in-code for Pihlakas's biological-alignment benchmarks.

His stack (github.com/biological-alignment-benchmarks):
  * ai-safety-gridworlds (branch biological-compatibility-benchmarks):
    extended DeepMind AI Safety Gridworlds framework; Gymnasium- and
    PettingZoo-compatible; environments expose multi-objective reward
    vectors instead of a single scalar.
  * biological-alignment-gridworlds-benchmarks: the concrete benchmark
    environments (homeostasis, sustainability, resource sharing, ...).

To train on Y there, you do not modify his environments. You wrap the
environment and replace the scalar reward with a form computed from the
objective vector, exactly as ToyContract.arm_reward does here from
StepSignals:

    class GoldenDirectiveRewardWrapper(gymnasium.Wrapper):
        def __init__(self, env, contract, arm="harmonic",
                     take_index=..., stock_index=...):
            ...
        def step(self, action):
            obs, reward_vec, terminated, truncated, info = self.env.step(action)
            take = reward_vec[take_index]        # verified take component
            p_e = info["stock_total"]            # or reward_vec[stock_index]
            return obs, self.contract.arm_reward(arm, take, p_e), \
                terminated, truncated, info

Mapping rules (declare them in your C(f) before running, paper section 3):
  * A: pick the objective component(s) that are verified gives and takes
    under your contract; interval A is gives - takes, never a stock.
  * E / P_E: a disclosed proxy for the state of everything not-A -- in his
    sustainability environments, total remaining resource stock is the
    natural proxy; in homeostasis environments, distance-to-band measures
    belong in the contract, not silently in E.
  * Keep his harness's own logging of the full objective vector: that is
    the meter record (section 3.2) and the source for full vs cropped Y.

This module is documentation only; gymnasium is not a dependency of the
standalone package.
"""
