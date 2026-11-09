# Results — recorded run, 2026-11-09

Exact output of the pre-registered run (PREREGISTRATION.md, including its
amendment). Nothing here is rounded beyond the table formatting, and the
full machine record is `results/results.json`.

- **Command:** `python3 run_experiment.py` (Python 3.12.3, numpy 1.26.4)
- **Runtime:** 18.0 s for all 18 runs (6 arms x seeds {0,1,2}; 300 training
  episodes + 20 greedy evaluation episodes each, horizon 150)
- **Checks:** `python3 tests.py` passes, including the paper section 5
  worked frame reproduced exactly (A=40, P_E=120 -> Y=60; cropped
  E'=190 -> Y ~= 66.1, gap ~= 6.1).

## Form comparison (means over seeds 0-2, evaluation episodes)

| Arm | Collapse rate | Terminal E | Mean E | Gives | Takes | Scored Y (full) | Scored Y (cropped) | Gap |
|---|---|---|---|---|---|---|---|---|
| harmonic | 0.00 | 24.12 | 24.15 | 98.75 | 175.94 | 304.7 | 284.1 | -20.6 |
| arithmetic | 0.00 | 36.91 | 35.98 | 119.41 | 153.67 | 274.1 | 252.5 | -21.6 |
| minimum | 0.00 | 24.07 | 23.35 | 116.00 | 211.82 | 358.3 | 331.0 | -27.3 |
| geometric | 0.00 | 27.34 | 26.91 | 132.40 | 204.58 | 350.7 | 328.7 | -22.0 |
| constraint | 0.17 | 22.18 | 23.67 | 148.21 | 241.29 | 411.9 | 381.2 | -30.7 |
| task_only (baseline) | 0.00 | 24.07 | 23.35 | 116.00 | 211.82 | 358.3 | 331.0 | -27.3 |

Interval Y (section 2 ledger, A = gives - takes) was undefined under
section 3.4(iii) (A + E <= 0, closed deficit) in nearly every evaluation
episode, because interval A is deeply negative in this frame (takes exceed
gives by 35-95 counts per episode). The one seed with defined values:
arithmetic seed 1, full Y = -31.28, cropped Y = -53.08, gap = -21.81 --
a negative interval Y, reported as-is per section 3.4(v) (net-destructive
interval). This is the section 7.6 point in concrete form: with a take-only
training A and a give-minus-take interval ledger, the interval score says
these agents are net takers, whatever the per-step score says.

## Falsifier (section 7.5) — harmonic arm, by seed

epsilon_E = 2.0 counts; pass line = scored Y 300.0; start E ~= 24 per episode.

| Seed | Terminal E | Scored Y | Material decrease? | At/above pass line? | Case failure |
|---|---|---|---|---|---|
| 0 | 28.21 | 334.3 | no | yes | **no** |
| 1 | 14.47 | 62.4 | yes | no | **no** |
| 2 | 29.67 | 517.4 | no | yes | **no** |

No case failure. Seed 1 is the informative one: one patch died in every
evaluation episode (any-patch death rate 1.0), terminal E fell materially,
and scored Y fell with it to 62.4 -- far below the pass line. In this frame,
on this seed, the score tracked the world down instead of staying high over
a degraded world. One seed is not evidence for the headline claim; it is
the falsifier machinery working as specified.

## Plain reading of the table (no claim beyond it)

- **No arm dominates.** The constraint arm earned the highest scored Y on
  average and was also the only arm with full collapses (seed 1: 50% of
  evaluation episodes ended with both patches dead). The harmonic arm sat
  mid-pack on scored Y and terminal E.
- **Arithmetic keeps the world fullest** (terminal E 36.91 of 40) because
  its reward pays for E's level directly; it scores lowest on Y.
- **minimum is identical to task_only on every seed.** In this frame
  min(A, E) = A almost everywhere (takes <= 4 per step, E usually > 4), so
  the two training signals coincide and the learned policies match exactly.
  That is a property of this frame's scales, reported not hidden.
- **Cropped scored Y is below full scored Y throughout** (gap -1.4 to
  -56.5 per seed). Here E' (healthiest patch) is a strict subset of E, and
  the harmonic form increases in E, so cropping mechanically lowers the
  score -- the opposite direction from the section 5 worked frame, where
  the cropped sub-frame overstated E. R2 asks for both numbers and the
  gap; the direction depends on what the crop hides.
- **Smoke run 1** (before the amendment: deterministic evaluation, gentle
  regrowth) had collapse rate 0.00 in every arm including the baseline and
  was discarded as a degenerate setting per section 7.1; see the amendment
  in PREREGISTRATION.md. Its output was not used anywhere above.

## Scope

Pilot code in a toy frame. Training A is the disclosed take-only proxy of
paper section 6, so per section 7.6 none of this is evidence for or against
the headline give-minus-take claim. The headline claim remains untested.
