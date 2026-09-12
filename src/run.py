"""Generate (and optionally simulate) a solution for one level.

    python src/run.py --level 1
    python src/run.py --level 1 --no-sim
    python src/run.py --all

Deterministic: no randomness anywhere, so the same input always produces the
same solution.json (Rule 6).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data import _ROOT, load_level                # noqa: E402
from engine.scoring import max_entropy_for, score_world  # noqa: E402
from engine.sim import Simulation                        # noqa: E402

# Resolved by walking up to the directory holding problem-statement/, so this
# works both in the repo (src/run.py) and in the flattened submission ZIP.
ROOT = _ROOT


def solve(level: int, run_sim: bool = True, alpha: float = 1.0,
          k: float = 1.0, verbose: bool = True) -> None:
    spec = load_level(level)
    module = importlib.import_module(f"solvers.level{level}")

    t0 = time.time()
    plan = module.build(spec)
    build_time = time.time() - t0

    out_dir = os.path.join(ROOT, "submissions", f"level{level}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "solution.json")
    plan.write(out_path)

    lines = [
        f"LEVEL {level}  {spec.rows}x{spec.cols}  ticks={spec.ticks}  "
        f"animals={spec.animals_enabled}",
        f"  plan: {plan.summary()}   (built in {build_time:.1f}s)",
        f"  wrote {os.path.relpath(out_path, ROOT)}",
    ]

    if run_sim:
        t0 = time.time()
        sim = Simulation(spec, plan)
        world = sim.run()
        breakdown = score_world(world, spec.ticks, alpha=alpha, k=k)
        lines += [
            f"  sim:  {breakdown}   ({time.time() - t0:.1f}s)",
            f"  H ceiling for {breakdown.species} species = "
            f"{max_entropy_for(breakdown.species):.4f}",
            f"  planted_ok={sim.planted_ok} rejected_locked={sim.rejected_locked} "
            f"rejected_soil={sim.rejected_soil}",
            "",
            world.describe(),
        ]

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    if verbose:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()

    levels = (1, 2, 3, 4) if args.all else (args.level,)
    if levels == (None,):
        parser.error("give --level N or --all")

    for level in levels:
        solve(level, run_sim=not args.no_sim, alpha=args.alpha, k=args.k)
        print()


if __name__ == "__main__":
    main()
