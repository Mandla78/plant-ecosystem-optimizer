"""Level 1 - Greenhouse.  50x50, 500 ticks, 1,800 plantable cells, 5 species.

THE KEY INSIGHT
---------------
Level 1 has animals disabled and no events, so nothing ever unlocks: we are
stuck with the 5 starting species and H is capped at log_31(5) = 0.4687.
That turns L1 from an ecology problem into a CONSTRUCTION problem.

Two facts combine:
  1. A plant survives ~100 ticks (100 nutrients, -1/tick), so the garden that
     gets scored at tick 500 is whatever was established after ~tick 400.
  2. We get 20 placements/tick. Ticks 410-499 is 90 ticks = 1,800 placements,
     which is EXACTLY the number of plantable cells on this map.

So we can hand-paint the entire final garden, cell by cell, in perfect 20%
proportions. We deliberately plant NOTHING early: an untouched cell still holds
its full 100 nutrients, whereas a cell used earlier would be partly drained and
flagged with dead matter (which also halves the refill rate), and might not
survive to the final tick.

ORDERING DETAIL
---------------
Oak Tree is painted LAST (ticks 482-499). Oak matures at age 20 and then casts
shade in radius 4, and Grass has no_shade_survival. By placing Oak in the final
18 ticks it never reaches maturity, so it never casts shade and never kills our
Grass - but it still counts as a living plant for both C and the species count.
"""

from __future__ import annotations

from engine.actions import ActionPlan
from engine.data import LevelSpec, plant
from engine.world import World

# Painted first (robust, no timing constraints), then Oak last.
FILLER_SPECIES = ("Grass", "Rose Bush", "Lavender", "Dwarf Sunflower")
LATE_SPECIES = "Oak Tree"

# The paint needs 1,800/20 = 90 ticks. Starting at 403 means the first-painted
# plants reach age 97 by tick 500 - just inside the ~100-tick nutrient limit,
# with 2 ticks of margin in case the official engine counts ticks differently.
# Earlier starts raise the longevity term; too early and the first plants starve
# before scoring. See the sweep in report.txt.
PAINT_START = 403
OAK_TAIL_TICKS = 18        # Oak must stay below its maturity of 20


def _territory_order(world: World, cells: list[int]) -> list[int]:
    """Order cells so each species gets a compact, contiguous block.

    Scoring only counts populations, not layout, but compact blocks keep each
    species' spread inside its own area during the paint, which stops the fast
    species (Grass) from stealing cells we have not painted yet.
    """
    return sorted(cells, key=lambda i: (i % world.cols, i // world.cols))


def build(spec: LevelSpec) -> ActionPlan:
    plan = ActionPlan(spec)
    world = World(spec)

    species = list(FILLER_SPECIES) + [LATE_SPECIES]
    soils = plant(species[0]).preferred_soil
    cells = _territory_order(world, world.plantable_indices(soils))

    total = len(cells)
    per_species = total // len(species)

    # --- carve territories -------------------------------------------------
    blocks: dict[str, list[int]] = {}
    cursor = 0
    for i, name in enumerate(species):
        take = per_species if i < len(species) - 1 else total - cursor
        blocks[name] = cells[cursor:cursor + take]
        cursor += take

    # --- Oak goes last so it never matures and never casts shade -----------
    oak_cells = blocks[LATE_SPECIES]
    oak_start = spec.ticks - OAK_TAIL_TICKS
    # Oak has more cells than its tail can hold at 20/tick; keep the remainder
    # and give it to the filler species instead of wasting the placements.
    oak_capacity = OAK_TAIL_TICKS * 20
    if len(oak_cells) > oak_capacity:
        overflow = oak_cells[oak_capacity:]
        oak_cells = oak_cells[:oak_capacity]
        # redistribute the overflow evenly across the filler species
        for n, cell in enumerate(overflow):
            blocks[FILLER_SPECIES[n % len(FILLER_SPECIES)]].append(cell)

    # --- paint the filler species -----------------------------------------
    tick = PAINT_START
    for name in FILLER_SPECIES:
        idx = plant(name).index
        tick = _paint(plan, blocks[name], idx, tick, oak_start, world.cols)

    # --- paint Oak in the tail --------------------------------------------
    _paint(plan, oak_cells, plant(LATE_SPECIES).index, oak_start, spec.ticks, world.cols)

    return plan


def _paint(plan: ActionPlan, cells: list[int], plant_index: int,
           start_tick: int, limit_tick: int, cols: int) -> int:
    """Place `cells` at 20/tick from start_tick. Returns the next free tick."""
    tick = start_tick
    for cell in cells:
        while tick < limit_tick and plan.capacity(tick) == 0:
            tick += 1
        if tick >= limit_tick:
            break
        row, col = divmod(cell, cols)
        plan.add(tick, plant_index, row, col)
    return tick
