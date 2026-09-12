"""Package a self-contained submission ZIP for each level.

    python build.py            # all four levels
    python build.py 1          # just level 1

Each ZIP contains the shared engine, that level's solver, the runner and a
README, so it runs standalone with a stdlib Python 3 and reproduces the
solution.json byte for byte (Rule 6: deterministic and reproducible).
"""

from __future__ import annotations

import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
RESOURCES = os.path.join(ROOT, "problem-statement", "additional-resources")
LEVELS = os.path.join(ROOT, "problem-statement", "levels")

ENGINE_FILES = ("__init__.py", "data.py", "world.py", "sim.py", "scoring.py", "actions.py")
RESOURCE_FILES = ("plant_dataset.json", "plant_unlock_conditions.json",
                  "animals.json", "classifications.json")

README = """\
Root Cause Analysis - Level {level} submission
==============================================

Run:
    python run.py --level {level}

Writes submissions/level{level}/solution.json, which is the file uploaded
alongside this archive. The generator is fully deterministic - there is no
randomness anywhere - so the same input always produces the same output.

Requires only the Python 3 standard library (no third-party packages).

Layout
------
    run.py                  entry point
    engine/                 shared simulation + scoring
        data.py             loads the plant / animal / unlock datasets
        world.py            grid, terrain, soil, nutrients, dead matter
        sim.py              tick loop (maturity, spread, death, unlocks)
        scoring.py          entropy, coverage and longevity scoring
        actions.py          builds the submission JSON, enforces 20/tick
    solvers/level{level}.py        the strategy for this level
    problem-statement/      the provided datasets and level file

Strategy notes for this level are in the docstring at the top of
solvers/level{level}.py.
"""


def build(level: int) -> str:
    out_dir = os.path.join(ROOT, "submissions", f"level{level}")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"level{level}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(SRC, "run.py"), "run.py")
        for name in ENGINE_FILES:
            zf.write(os.path.join(SRC, "engine", name), f"engine/{name}")
        zf.write(os.path.join(SRC, "solvers", "__init__.py"), "solvers/__init__.py")
        zf.write(os.path.join(SRC, "solvers", f"level{level}.py"),
                 f"solvers/level{level}.py")
        for name in RESOURCE_FILES:
            zf.write(os.path.join(RESOURCES, name),
                     f"problem-statement/additional-resources/{name}")
        zf.write(os.path.join(LEVELS, f"{level}.json"),
                 f"problem-statement/levels/{level}.json")
        zf.writestr("README.txt", README.format(level=level))

    return zip_path


def main() -> None:
    levels = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    for level in levels:
        solver = os.path.join(SRC, "solvers", f"level{level}.py")
        if not os.path.exists(solver):
            print(f"level {level}: no solver yet, skipping")
            continue
        path = build(level)
        size = os.path.getsize(path) / 1024
        print(f"level {level}: {os.path.relpath(path, ROOT)}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
