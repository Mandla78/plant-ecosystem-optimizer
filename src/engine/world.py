"""Grid state for a Photospheria garden.

Flat lists indexed by `row * cols + col` - no third-party deps, so the
submitted ZIP runs anywhere with a stdlib Python 3.
"""

from __future__ import annotations

from typing import Iterator

from .data import (
    LevelSpec,
    PASSABLE_TERRAIN,
    START_NUTRIENTS,
    TERRAIN_GROUND,
    TERRAIN_PATH,
    TERRAIN_STONE,
    SOIL_DIRT,
    SOIL_BURNT,
)

EMPTY = 0          # plant indices start at 1, so 0 is a safe "no plant" marker


class World:
    """Terrain, soil and per-cell plant state.

    Cells absent from the level file default to plantable ground + dirt soil -
    that default is the majority of every map (see docs/data_info.txt).
    """

    __slots__ = (
        "rows", "cols", "size", "spec",
        "terrain", "soil", "plant", "age", "nutrients", "dead_matter",
        "sub_plant", "sub_age", "shade",
    )

    def __init__(self, spec: LevelSpec):
        self.spec = spec
        self.rows = spec.rows
        self.cols = spec.cols
        self.size = spec.rows * spec.cols

        n = self.size
        self.terrain = bytearray([TERRAIN_GROUND]) * n
        self.soil = bytearray([SOIL_DIRT]) * n
        self.plant = [EMPTY] * n              # main occupant (plant index)
        self.age = [0] * n                    # ticks since that plant was placed
        self.nutrients = [float(START_NUTRIENTS)] * n
        self.dead_matter = bytearray(n)       # 1 = cell holds dead plant matter
        self.sub_plant = [EMPTY] * n          # subsurface_growth occupant
        self.sub_age = [0] * n
        self.shade = bytearray(n)             # recomputed each tick from mature casters

        for cell in spec.cells:
            i = cell["row"] * self.cols + cell["col"]
            self.terrain[i] = cell["terrain"]
            self.soil[i] = cell["soil"]

    # -- indexing ----------------------------------------------------------

    def idx(self, row: int, col: int) -> int:
        return row * self.cols + col

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.cols)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # -- terrain queries ---------------------------------------------------

    def is_passable(self, i: int) -> bool:
        """Can anything ever grow here? Water / stone / path are permanently out."""
        return self.terrain[i] in PASSABLE_TERRAIN

    def plantable_indices(self, soils: frozenset[int] | None = None) -> list[int]:
        """Cells a plant could occupy, optionally filtered to its preferred soil."""
        out = []
        terrain, soil = self.terrain, self.soil
        for i in range(self.size):
            if terrain[i] != TERRAIN_GROUND:
                continue
            if soils is not None and soil[i] not in soils:
                continue
            out.append(i)
        return out

    def neighbours(self, i: int) -> Iterator[int]:
        """4-neighbourhood, bounds-safe."""
        row, col = divmod(i, self.cols)
        if row > 0:
            yield i - self.cols
        if row + 1 < self.rows:
            yield i + self.cols
        if col > 0:
            yield i - 1
        if col + 1 < self.cols:
            yield i + 1

    def neighbours8(self, i: int) -> Iterator[int]:
        row, col = divmod(i, self.cols)
        for dr in (-1, 0, 1):
            r = row + dr
            if r < 0 or r >= self.rows:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                c = col + dc
                if 0 <= c < self.cols:
                    yield r * self.cols + c

    def adjacent_to_terrain(self, i: int, kinds: tuple[int, ...]) -> bool:
        return any(self.terrain[j] in kinds for j in self.neighbours8(i))

    def adjacent_to_water(self, i: int) -> bool:
        from .data import TERRAIN_WATER
        return self.adjacent_to_terrain(i, (TERRAIN_WATER,))

    def adjacent_to_rock_or_path(self, i: int) -> bool:
        return self.adjacent_to_terrain(i, (TERRAIN_STONE, TERRAIN_PATH))

    def occupied_neighbour_count(self, i: int) -> int:
        plant = self.plant
        return sum(1 for j in self.neighbours8(i) if plant[j] != EMPTY)

    # -- census ------------------------------------------------------------

    def counts(self) -> dict[int, int]:
        """Population per plant index, counting subsurface occupants too."""
        out: dict[int, int] = {}
        for p in self.plant:
            if p:
                out[p] = out.get(p, 0) + 1
        for p in self.sub_plant:
            if p:
                out[p] = out.get(p, 0) + 1
        return out

    def population(self) -> int:
        """C - total populated cells (a cell with a subsurface plant counts twice)."""
        return sum(1 for p in self.plant if p) + sum(1 for p in self.sub_plant if p)

    def dead_matter_count(self) -> int:
        return sum(self.dead_matter)

    def burnt_soil_count(self) -> int:
        return sum(1 for s in self.soil if s == SOIL_BURNT)

    def describe(self) -> str:
        from .data import plants_by_index
        by_index = plants_by_index()
        counts = self.counts()
        total = sum(counts.values())
        lines = [f"populated={total}  species={len(counts)}  dead_matter={self.dead_matter_count()}"]
        for idx in sorted(counts, key=lambda k: -counts[k]):
            share = counts[idx] / total if total else 0.0
            lines.append(f"  {by_index[idx].name:<20} {counts[idx]:>6}  {share:6.2%}")
        return "\n".join(lines)
