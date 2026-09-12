"""Territory planner - written after the Level 3 evaluation log exposed how the
engine really behaves.

WHAT THE LOG TAUGHT US
----------------------
Level 3 scored 20,606,381. The log gave the engine's own statistics:

    entropy 0.0447   density 0.5522   main = entropy * density   (so ALPHA = 1)
    plant_counts: Oak Tree 12001, Dwarf Sunflower 413, Rose Bush 7, Lavender 4

Three corrections, all of which invalidate earlier assumptions:

 1. ALPHA = 1, not 2. main_score is exactly entropy * density.

 2. A MONOCULTURE FORMS. Oak took 96.6% of the garden. Entropy collapsed to
    0.045 against a ceiling near 1.0, and that single fact cost ~90% of the
    available score. Density was fine (0.55); diversity was the failure.

 3. PLACEMENT ONTO AN OCCUPIED CELL IS DENIED, NOT REPLACED. The problem
    statement says the existing plant "will be replaced by your new plant". The
    engine log says "Placement denied ...: plant already occupies cell". Every
    rebalancing-by-overwrite action we emitted did nothing.

 4. OAK IS POISON. Oak shades radius 4 once mature, Grass has
    no_shade_survival, so Oak wiped out every Grass plant. Loamcrawlers needs
    Grass coverage >= 4%, so no animal ever appeared and NOTHING unlocked -
    4 species in a 30-species level.

THE STRATEGY THIS IMPLIES
-------------------------
Score = 0.8 * entropy * density. Density largely takes care of itself, because
one fast species will fill roughly half the map whatever we do. Entropy is the
whole game, and hand-placing cannot buy it: only ~2,000 placements land inside
the ~100-tick survival window, which is worth about 50M at best.

So many species have to SPREAD IN PARALLEL and partition the map between them:

  * Split the map into a grid of BLOCKS and give each species its own blocks,
    far apart, so their spread fronts meet at block boundaries instead of one
    species sweeping the whole map.
  * Start at tick 0 so the unlock chain finishes early and newly unlocked
    species still have hundreds of ticks to claim territory.
  * NEVER PLANT OAK. One extra species is not worth losing Grass, the animals
    and the entire unlock tree.
  * Seed only into cells that are empty - anything else is discarded.
"""

from __future__ import annotations

from engine.actions import ActionPlan
from engine.analysis import reachable_for_level
from engine.data import LevelSpec, plants_by_name
from engine.sim import Simulation
from engine.world import EMPTY, World

# Oak Tree is deliberately excluded everywhere - see the module docstring.
BANNED = frozenset({"Oak Tree"})

# Order we hand out territory in. Starters first (always available), then the
# species each wave unlocks, cheapest gates first.
PREFERRED = [
    "Grass", "Lavender", "Dwarf Sunflower", "Rose Bush",
    "Razorgrass", "Crimson Vine", "Blue Moss", "Orange Blossom",
    "Stone Reed", "Ironthorn Shrub", "Crystal Cactus", "Glowcap Fungus",
    "Silver Fern", "Mire Bloom", "Purple Canopy Tree", "Moonpetal Lily",
    "Thornheart Bramble", "Whiteveil Mycelium", "Skyvine", "Amber Fern",
    "Ghost Orchid", "Sunshard Bloom", "Living Topiary", "Sporewood Tree",
    "Starcap Colony", "Bloodbloom",
]


class TerritoryPlanner:
    def __init__(self, spec: LevelSpec, *, blocks: int = 8,
                 seeds_per_block: int = 3, restock_every: int = 40,
                 banned: frozenset[str] = BANNED):
        self.spec = spec
        self.by_name = plants_by_name()
        reachable = reachable_for_level(spec)
        self.order = [n for n in PREFERRED if n in reachable and n not in banned]

        self.plan = ActionPlan(spec)
        self.seeds_per_block = seeds_per_block
        self.restock_every = restock_every

        probe = World(spec)
        self.probe = probe
        rows, cols = spec.rows, spec.cols
        self.block_rows = blocks
        self.block_cols = blocks

        # bucket every plantable cell into its block
        self.block_cells: dict[int, list[int]] = {}
        br = max(1, rows // blocks)
        bc = max(1, cols // blocks)
        for cell in probe.plantable_indices(frozenset({0, 1, 2})):
            r, c = divmod(cell, cols)
            b = min(blocks - 1, r // br) * blocks + min(blocks - 1, c // bc)
            self.block_cells.setdefault(b, []).append(cell)

        # scatter within each block so seeds are not adjacent
        for b, cells in self.block_cells.items():
            cells.sort(key=lambda i: (i * 2654435761) % 4294967296)

        self.free_blocks = sorted(self.block_cells)
        self.owned: dict[str, list[int]] = {}
        self._cursor: dict[str, int] = {}

    # -- territory allocation ---------------------------------------------

    def _claim(self, name: str, count: int) -> None:
        """Give `name` some blocks, spaced apart so rivals are not adjacent."""
        taken = []
        for _ in range(count):
            if not self.free_blocks:
                break
            # take from alternating ends so each species is spread around
            idx = 0 if len(self.owned) % 2 == 0 else len(self.free_blocks) - 1
            taken.append(self.free_blocks.pop(idx))
        if taken:
            self.owned.setdefault(name, []).extend(taken)

    def _cells_for(self, name: str) -> list[int]:
        out = []
        for b in self.owned.get(name, ()):
            out.extend(self.block_cells[b])
        return out

    # -- policy -------------------------------------------------------------

    def policy(self, tick, sim):
        world = sim.world
        budget = 20
        picks = []

        live = [n for n in self.order if n in sim.unlocked]

        # hand territory to anything newly unlocked
        per = max(1, (self.block_rows * self.block_cols) // max(1, len(self.order)))
        for name in live:
            if name not in self.owned:
                self._claim(name, per)

        # every restock_every ticks, top each species up inside its own blocks
        if tick % self.restock_every and tick > 0:
            return []

        for name in live:
            if len(picks) >= budget:
                break
            p = self.by_name[name]
            cells = self._cells_for(name)
            if not cells:
                continue
            start = self._cursor.get(name, 0)
            n = len(cells)
            placed = 0
            i = 0
            while i < n and placed < self.seeds_per_block and len(picks) < budget:
                cell = cells[(start + i) % n]
                i += 1
                if world.plant[cell] != EMPTY:
                    continue
                if world.soil[cell] not in p.preferred_soil:
                    continue
                if world.nutrients[cell] < 30:
                    continue
                picks.append((p.index, cell))
                placed += 1
            self._cursor[name] = (start + i) % max(1, n)

        cols = self.spec.cols
        return [(i, c // cols, c % cols) for i, c in picks[:budget]]

    def build(self) -> ActionPlan:
        sim = Simulation(self.spec, self.plan)
        sim.run(policy=self.policy)
        self.simulation = sim
        return self.plan
