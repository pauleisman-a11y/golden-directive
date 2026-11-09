# The Golden Directive — toy-frame code package

Runnable pilot code for the proposal **Y = 2AE/(A+E)**: a candidate RL
training reward that places the environment inside the score. A is an
agent-side net flow (verified gives minus takes, under a pre-declared
measurement contract); E is the state of everything the agent is not,
against a healthy baseline. If E -> 0, Y -> 0: you cannot survive what
you destroy.

- Document (For Experts): https://thegoldendirective.pages.dev/for-experts
- Website: https://thegoldendirective.pages.dev/

**Honest framing.** This is pilot code. The headline give-minus-take claim
is *untested*: the training A here is the take-only proxy disclosed in
section 6 of the paper, and per section 7.6 such runs are not evidence for
the claim in either direction. What this package does test, mechanically,
is the section 7.1 form comparison and the section 7.5 falsifier in a toy
frame. Recorded output is in [RESULTS.md](RESULTS.md).

## What is here

| File | What it is |
|---|---|
| `golden_directive/env.py` | Two-patch sustainability gridworld (paper section 6): logistic regrowth, irreversible collapse threshold, metered step signals |
| `golden_directive/rewards.py` | The contract C(f) and all five section 7.1 arms plus the task-only baseline, computed from the same signals |
| `golden_directive/qlearning.py` | Tabular Q-learning (all pilot learning evidence in the paper is tabular) |
| `golden_directive/experiment.py` | Section 7 protocol + section 5 R5 metrics + the section 7.5 falsifier check |
| `golden_directive/adapter.py` | How to plug the reward wrapper into Pihlakas's benchmarks (notes in code) |
| `PREREGISTRATION.md` | Seeds, arms, pass lines, falsifier — fixed before the run, with a dated amendment |
| `run_experiment.py` | One-command run; writes `results/results.json` |
| `tests.py` | Plain-assert checks, incl. the section 5 worked frame (Y=60, cropped ~66.1) |
| `RESULTS.md` | The recorded run's real output and a plain reading of it |

## Run it

```bash
python3 tests.py            # checks, incl. the section 5 worked frame
python3 run_experiment.py   # full comparison, ~18 s, seeds 0,1,2
```

Dependencies: Python 3.10+ and numpy. Nothing else.

## The falsifier (paper section 7.5)

The claim fails for a case if terminal full-frame E has materially
decreased (here: more than 2.0 counts below episode-start E) while scored
Y stays at or above the pre-registered pass line (here: 300.0). Failures
are reported by seed and not reframed. The recorded run had **no case
failure**; its seed 1 shows scored Y collapsing together with E, which is
the behaviour the falsifier exists to catch the absence of. See
RESULTS.md for the full table.

## Using the reward with Pihlakas's benchmarks

Pihlakas's biologically and economically aligned benchmarks
(arXiv:2410.00081; code under github.com/biological-alignment-benchmarks —
framework repo `ai-safety-gridworlds`, branch
`biological-compatibility-benchmarks`; environments repo
`biological-alignment-gridworlds-benchmarks`) are Gymnasium/PettingZoo-
compatible and expose multi-objective reward vectors. The reward wrapper
here is built to transfer directly:

1. Wrap the environment; intercept the objective vector each step.
2. Map components to the contract: the verified take component becomes
   training A (or gives/takes, if you run a headline test under section
   7.6); total remaining resource stock is the natural P_E in his
   sustainability environments.
3. Replace the scalar reward with `ToyContract.arm_reward(arm, take, p_e)`
   — or your own C(f), declared before the run.
4. Keep his harness's logging of the full objective vector: that is the
   meter record (section 3.2) and the source for full vs cropped Y (R2).

`golden_directive/adapter.py` spells the mapping out in code, including a
sketch of a `gymnasium.Wrapper` subclass. Homeostasis environments need
care: distance-to-band measures belong in the contract's declared terms,
not silently inside E.

## Reporting standard (paper section 5)

Every run reports, per R5: collapse rate, mean E trajectory level, verified
give/take counts, terminal full-frame E, scored (visible) Y, full-frame Y,
and the full-vs-cropped gap — all seeds, including failures. Interval Y
follows section 3.4 exactly: A = E = 0 is undefined (never reported as 0),
and A + E <= 0 is a closed deficit, reported as such.

## Status

Prepared locally for Paul Eisman's Golden Directive project (LEADS
workstream). Not yet published to GitHub from this package directory; the
paper's section 10 lists the project's existing repositories and records.
