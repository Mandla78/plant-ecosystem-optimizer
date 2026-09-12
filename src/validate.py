"""Independent validator for a solution.json.

Deliberately does NOT import the ActionPlan that produced the file - it re-reads
the JSON from disk and re-checks every constraint stated in the problem
statement from scratch. If the generator has a bug, a validator built from the
same code would happily agree with it; this one will not.

    python src/validate.py 1
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data import (                                # noqa: E402
    MAX_PLANTS_PER_TICK,
    STARTING_PLANTS,
    TERRAIN_GROUND,
    _ROOT,
    load_level,
    plants_by_index,
    plants_by_name,
)


def validate(level: int) -> bool:
    spec = load_level(level)
    path = os.path.join(_ROOT, "submissions", f"level{level}", "solution.json")

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)

    problems: list[str] = []
    warnings: list[str] = []

    # -- shape -------------------------------------------------------------
    if set(doc) != {"actions"}:
        problems.append(f"top-level keys are {sorted(doc)}, expected ['actions']")
    actions = doc.get("actions", [])
    if not isinstance(actions, list):
        problems.append("actions is not a list")
        return _report(level, path, problems, warnings, {})

    # -- terrain / soil lookup built independently -------------------------
    terrain = {}
    soil = {}
    for cell in spec.cells:
        terrain[(cell["row"], cell["col"])] = cell["terrain"]
        soil[(cell["row"], cell["col"])] = cell["soil"]

    by_index = plants_by_index()
    start_indices = {plants_by_name()[n].index for n in STARTING_PLANTS}

    seen_ticks: set[int] = set()
    per_tick_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
    counts = Counter()
    total = 0

    for entry in actions:
        if set(entry) - {"tick", "plants"}:
            problems.append(f"unexpected keys in tick entry: {sorted(entry)}")
        tick = entry.get("tick")
        if not isinstance(tick, int):
            problems.append(f"non-integer tick: {tick!r}")
            continue
        if not (0 <= tick <= spec.ticks - 1):
            problems.append(f"tick {tick} outside [0, {spec.ticks - 1}]")
        if tick in seen_ticks:
            problems.append(f"tick {tick} appears in more than one entry")
        seen_ticks.add(tick)

        plants = entry.get("plants", [])
        if len(plants) > MAX_PLANTS_PER_TICK:
            problems.append(
                f"tick {tick}: {len(plants)} plants, only the first "
                f"{MAX_PLANTS_PER_TICK} would be planted"
            )

        for p in plants:
            total += 1
            if "plant_index" not in p:
                problems.append(f"tick {tick}: entry missing 'plant_index': {p}")
                continue
            idx, row, col = p["plant_index"], p.get("row"), p.get("col")

            if idx not in by_index:
                problems.append(f"tick {tick}: unknown plant_index {idx}")
                continue
            if not (0 <= row < spec.rows):
                problems.append(f"tick {tick}: row {row} outside [0,{spec.rows - 1}]")
                continue
            if not (0 <= col < spec.cols):
                problems.append(f"tick {tick}: col {col} outside [0,{spec.cols - 1}]")
                continue

            if (row, col) in per_tick_cells[tick]:
                problems.append(f"tick {tick}: cell ({row},{col}) planted twice")
            per_tick_cells[tick].add((row, col))

            # terrain: cells absent from the level file default to plantable dirt
            t = terrain.get((row, col), TERRAIN_GROUND)
            if t != TERRAIN_GROUND:
                problems.append(
                    f"tick {tick}: ({row},{col}) is terrain {t}, not plantable"
                )
            s = soil.get((row, col), 0)
            plant = by_index[idx]
            if s not in plant.preferred_soil:
                problems.append(
                    f"tick {tick}: {plant.name} on soil {s}, prefers "
                    f"{sorted(plant.preferred_soil)}"
                )

            if not spec.animals_enabled and idx not in start_indices:
                warnings.append(
                    f"tick {tick}: {plant.name} needs an unlock, but this level "
                    "has animals disabled"
                )
            counts[plant.name] += 1

    return _report(level, path, problems, warnings, counts, total, spec)


def _report(level, path, problems, warnings, counts, total=0, spec=None) -> bool:
    print(f"VALIDATING level {level}: {os.path.relpath(path, _ROOT)}")
    if spec is not None:
        print(f"  grid {spec.rows}x{spec.cols}  ticks={spec.ticks}  "
              f"animals={spec.animals_enabled}")
    print(f"  total planting actions: {total}")
    if counts:
        print("  by species:")
        for name, n in counts.most_common():
            print(f"    {name:<20} {n:>6}")

    uniq = sorted(set(warnings))
    if uniq:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in uniq[:10]:
            print(f"    {w}")

    if problems:
        print(f"  FAILED - {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"    {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return False

    print("  PASSED - every documented constraint satisfied")
    return True


if __name__ == "__main__":
    levels = [int(a) for a in sys.argv[1:]] or [1]
    ok = all(validate(level) for level in levels)
    sys.exit(0 if ok else 1)
