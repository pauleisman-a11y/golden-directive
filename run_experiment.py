#!/usr/bin/env python3
"""Run the pre-registered toy-frame form comparison (paper section 7.1).

Usage: python3 run_experiment.py [--seeds 0,1,2]
Writes results/results.json and prints the section 5 R5 table.
"""

import argparse
import json
import time
from pathlib import Path

from golden_directive.experiment import run_all


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    t0 = time.time()
    out = run_all(seeds=seeds)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["seeds"] = list(seeds)

    path = Path(__file__).parent / "results" / "results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print(f"{'arm':<12}{'collapse':>9}{'terminalE':>10}{'meanE':>8}"
          f"{'gives':>8}{'takes':>8}{'scoredY':>9}{'fullY':>8}{'cropY':>8}{'gap':>7}")
    for arm, r in s.items():
        print(f"{arm:<12}{fmt(r['collapse_rate']) + '':>9}{fmt(r['mean_terminal_e']):>10}"
              f"{fmt(r['mean_e_level']):>8}{fmt(r['mean_gives']):>8}{fmt(r['mean_takes']):>8}"
              f"{fmt(r['scored_y'], 1):>9}{fmt(r['full_y']):>8}{fmt(r['cropped_y']):>8}"
              f"{fmt(r['crop_gap']):>7}")
    f = out["falsifier"]
    print(f"\nFalsifier (7.5): epsilon_E={f['epsilon_e']}, pass_line={f['pass_line']}, "
          f"case failure any seed: {f['any_case_failure']}")
    for c in f["cases"]:
        print(f"  seed {c['seed']}: terminalE={fmt(c['mean_terminal_e'])}, "
              f"scoredY={fmt(c['scored_y'], 1)}, decrease={c['material_decrease']}, "
              f"pass={c['at_or_above_pass_line']}, CASE FAILURE={c['case_failure']}")
    print(f"\nRuntime: {out['runtime_seconds']}s. Full record: {path}")


if __name__ == "__main__":
    main()
