"""Builds the submission JSON and enforces every documented constraint.

Constraints (from the problem statement):
  - ticks are integers in [0, T-1]
  - coordinates must be inside the grid
  - at most 20 plants per tick; ONLY THE FIRST 20 ARE PLANTED, the rest are
    silently dropped, so we refuse to emit more than 20 rather than let the
    engine throw work away
  - planting the same cell twice in one tick is wasted budget, so we dedupe
"""

from __future__ import annotations

import json
from collections import defaultdict

from .data import MAX_PLANTS_PER_TICK, LevelSpec


class ActionPlan:
    """Accumulates planting actions, then serialises them."""

    def __init__(self, spec: LevelSpec):
        self.spec = spec
        self._by_tick: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        self._seen: dict[int, set[tuple[int, int]]] = defaultdict(set)
        self.dropped_full = 0
        self.dropped_dupe = 0
        self.dropped_bounds = 0

    # -- adding ------------------------------------------------------------

    def capacity(self, tick: int) -> int:
        return MAX_PLANTS_PER_TICK - len(self._by_tick[tick])

    def add(self, tick: int, plant_index: int, row: int, col: int) -> bool:
        """Queue one planting. Returns False if it could not be scheduled."""
        if not (0 <= tick < self.spec.ticks):
            self.dropped_bounds += 1
            return False
        if not (0 <= row < self.spec.rows and 0 <= col < self.spec.cols):
            self.dropped_bounds += 1
            return False
        if (row, col) in self._seen[tick]:
            self.dropped_dupe += 1
            return False
        if len(self._by_tick[tick]) >= MAX_PLANTS_PER_TICK:
            self.dropped_full += 1
            return False

        self._by_tick[tick].append((plant_index, row, col))
        self._seen[tick].add((row, col))
        return True

    def add_flat(self, tick: int, plant_index: int, cell: int, cols: int) -> bool:
        row, col = divmod(cell, cols)
        return self.add(tick, plant_index, row, col)

    def add_spread_over_ticks(self, start_tick: int, plant_index: int,
                              cells: list[int], cols: int,
                              end_tick: int | None = None) -> int:
        """Schedule many placements from start_tick onward, 20 per tick.

        Returns how many were actually scheduled.
        """
        limit = self.spec.ticks if end_tick is None else min(end_tick, self.spec.ticks)
        tick = start_tick
        placed = 0
        for cell in cells:
            while tick < limit and self.capacity(tick) == 0:
                tick += 1
            if tick >= limit:
                break
            row, col = divmod(cell, cols)
            if self.add(tick, plant_index, row, col):
                placed += 1
        return placed

    # -- inspection --------------------------------------------------------

    def total_actions(self) -> int:
        return sum(len(v) for v in self._by_tick.values())

    def ticks_used(self) -> int:
        return sum(1 for v in self._by_tick.values() if v)

    def actions_for(self, tick: int) -> list[tuple[int, int, int]]:
        return self._by_tick.get(tick, [])

    # -- output ------------------------------------------------------------

    def to_dict(self) -> dict:
        actions = []
        for tick in sorted(self._by_tick):
            entries = self._by_tick[tick]
            if not entries:
                continue
            actions.append({
                "tick": tick,
                "plants": [
                    {"plant_index": idx, "row": row, "col": col}
                    for idx, row, col in entries
                ],
            })
        return {"actions": actions}

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    def summary(self) -> str:
        parts = [
            f"actions={self.total_actions()}",
            f"ticks_used={self.ticks_used()}/{self.spec.ticks}",
        ]
        if self.dropped_full:
            parts.append(f"dropped_full={self.dropped_full}")
        if self.dropped_dupe:
            parts.append(f"dropped_dupe={self.dropped_dupe}")
        if self.dropped_bounds:
            parts.append(f"dropped_bounds={self.dropped_bounds}")
        return "  ".join(parts)
