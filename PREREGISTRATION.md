# Pre-registration — toy-frame form comparison (paper §7.1 / §7.5)

Written before the first run of this package, per §7 of the paper. Nothing in
this file was changed after seeing results; `RESULTS.md` reports what ran.

- **Frame f:** `two-patch-toy-v1` — single agent, two food patches, one
  resource (food, in counts). Contract C(f) is implemented in
  `golden_directive/rewards.py` (`ToyContract`) and `golden_directive/env.py`.
- **Arms (§7.1):** `harmonic` (Y = 2AE/(A+E)), `arithmetic` ((A+E)/2),
  `minimum` (min(A,E)), `geometric` (sqrt(A*E)), `constraint` (task reward
  minus a penalty while P_E is below the health line). Reference arm outside
  §7.1, matching the §6 pilot baseline: `task_only` (task reward alone).
- **Identical seeds and code path:** seeds {0, 1, 2} for every arm; one
  training/evaluation code path; per-run RNG seeded with the run seed.
- **Training:** tabular Q-learning, 300 episodes per run, horizon 150 steps,
  alpha 0.1, gamma 0.99, epsilon 0.1 (constant). Evaluation: greedy policy,
  20 episodes per run.
- **Per-step training quantities (declared contract choice):** A_t is the
  verified take (harvest) at step t — the take-only proxy of §6, disclosed.
  E_t is P_E, total food stock across both patches after the step, metered
  outside the agent. Gives (tending) enter the interval ledger and raise E.
  Per §7.6, take-only A means these runs are **not** evidence for the
  headline give-minus-take claim, in either direction.
- **Constraint arm health line:** penalty 1.0 per step while P_E < 16.0
  (40% of the 40-count healthy baseline).
- **Metrics (§5 R5), per run, from evaluation episodes:** collapse rate
  (episode ends with both patches dead), any-patch death rate, mean terminal
  full-frame E, mean E level over the episode, mean verified give and take
  counts, scored Y (mean episodic harmonic return of the trajectory,
  full frame), interval full-frame Y and cropped Y (E' = healthiest patch
  only) with the gap (R2).
- **Falsifier (§7.5), operationalised ex ante:** material decrease =
  terminal full-frame E more than epsilon_E = 2.0 counts (5% of baseline)
  below episode-start E. Pass line for scored Y = 300.0 (episodic harmonic
  return; per-step Y is at most ~7.3 in this frame, horizon 150).
  A **case failure** is a seed whose harmonic-arm evaluation shows mean
  terminal E materially decreased while mean scored Y is at or above the
  pass line. Case failures are reported by seed, not reframed.

## Amendment — after smoke run 1, before the recorded run (2026-11-09)

Smoke run 1 (same code, gentler parameters) finished in 18.4s and exposed
two defects in the *setting*, not the arms:

1. Evaluation was deterministic: fixed reset, greedy policy, no noise, so
   each run's 20 evaluation episodes were one episode repeated 20 times.
2. The setting was degenerate in the sense of section 7.1: regrowth 0.25
   and 16-count initial stock meant no arm, including the task-only
   baseline, ever approached collapse (collapse rate 0.00 everywhere,
   terminal E 33.7-37.9 of 40). A comparison in which the world cannot
   fail tests nothing.

Changes for the recorded run, fixed here before it runs:
- Reset jitter: per-episode initial stock per patch = initial_stock +/-
  up to 3 counts, drawn from the run's seeded RNG (identical protocol for
  every arm).
- Scarcity: regrowth_rate 0.25 -> 0.10; initial_stock 16 -> 12 (episode-
  start E = 24 nominal). Maximum total regrowth (~1.0 count/step) now
  roughly equals metabolic demand, so over-harvesting past the threshold
  is a live risk.
- Added metric: scored cropped Y (episodic harmonic return computed on
  E', the healthiest patch only) and the full-vs-cropped gap, as an
  R2 supplement alongside interval Y. Interval cropped Y will often be
  undefined under section 3.4(iii) (A + E' <= 0, closed deficit) because
  interval A is deeply negative in this frame; that is reported as-is.
- Falsifier unchanged: epsilon_E = 2.0, pass line 300.0 (not moved).
