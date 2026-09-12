"""Gate-securing planner - built after the Level 2 submission came back at 79M.

WHAT WENT WRONG THE FIRST TIME
-------------------------------
Our first L2 solution scored 79,390,810 against a predicted 533,576,542. Replaying
the exact submitted plan against models of the real engine showed the cause: the
plan spent 64% of its actions on species that NEVER UNLOCKED. It planted only 28
Grass, expecting spread to carry Grass past the 280 cells that the Loamcrawlers
animal needs. Real spread is far weaker than we modelled, Grass never got there,
no animals appeared, and two thirds of every action was silently discarded.

THE PRINCIPLE THIS PLANNER IS BUILT ON
--------------------------------------
Never rely on spread to clear a gate. Every unlock threshold is met by HAND, with
counted quotas, before a single action is spent on the species it unlocks.

Spread is then pure upside: if it is stronger than we think, coverage is higher
than planned; if it does nothing at all, the plan still stands up, because every
cell it depends on was placed directly.

PHASES
------
  SECURE   Plant counted quotas of the four starting species that drive the six
           easy animals. These are unlocked from tick 0, so they can never be
           rejected. Quotas are set above each threshold with margin.

  HARVEST  The animals are now present, so the first tier has unlocked. Spend
           every remaining action on those species, balanced, to raise entropy.

  All planting happens inside the final ~100 ticks so that everything placed is
  still alive on the only tick that is scored.
"""

from __future__ import annotations

from engine.actions import ActionPlan
from engine.data import LevelSpec, plants_by_name
from engine.sim import Simulation
from engine.world import EMPTY, World

# Hand-planted quotas, as a fraction of Cmax, for the species that drive the
# animal block. Each sits above the real threshold with margin:
#   Grass    0.05 Verdelopes, 0.04 Loamcrawlers + Grazeleths
#   Rose     0.04 Ironthorn, 0.02 Orange Blossom + Solwings, count>=10 Loamcrawlers
#   Lavender 0.04 Moonpetal, 0.02 Nectaris, count>=10 Virexids
#   Sunflower 0.03 Solwings
SECURE_QUOTAS: dict[str, float] = {
    "Grass": 0.055,
    "Rose Bush": 0.045,
    "Lavender": 0.045,
    "Dwarf Sunflower": 0.035,
}

# Tier-1 species, unlocked purely by the animals above plus starter coverage.
# Nothing deeper is attempted - those gates need coverage we cannot hand-place.
TIER1 = [
    "Razorgrass",        # Verdelopes + Grass > 0.05
    "Crimson Vine",      # Nectaris + Rose > 0 + Lavender > 0
    "Orange Blossom",    # Nectaris/Solwings + Rose > 0.02
    "Stone Reed",        # Virexids
    "Ironthorn Shrub",   # Grazeleths + Rose > 0.04
    "Blue Moss",         # Loamcrawlers + Grass > 0.03 + Rose > 0.01
]


def _scatter(cells: list[int]) -> list[int]:
    return sorted(cells, key=lambda i: (i * 2654435761) % 4294967296)


class GatedPlanner:
    def __init__(self, spec: LevelSpec, *, start: int,
                 tier1: list[str] | None = None,
                 quotas: dict[str, float] | None = None,
                 include_oak: bool = True):
        self.spec = spec
        self.start = start
        self.tier1 = list(TIER1 if tier1 is None else tier1)
        self.quotas = dict(SECURE_QUOTAS if quotas is None else quotas)
        self.include_oak = include_oak

        self.by_name = plants_by_name()
        self.plan = ActionPlan(spec)

        probe = World(spec)
        self._probe = probe
        self._cells = _scatter(probe.plantable_indices(frozenset({0, 1})))
        self._cursor = 0
        self._taken: set[int] = set()

        # remaining hand-planted quota per species, in cells
        self._todo: dict[str, int] = {
            name: int(frac * spec.cmax) + 1 for name, frac in self.quotas.items()
        }
        if include_oak:
            # Oak is a starter and free diversity; it never matures inside the
            # window so it cannot shade out our Grass.
            self._todo["Oak Tree"] = int(0.02 * spec.cmax)

    def _pick(self, world, name: str, wanted: int) -> list[int]:
        p = self.by_name[name]
        out: list[int] = []
        n = len(self._cells)
        i = 0
        while i < n and len(out) < wanted:
            cell = self._cells[(self._cursor + i) % n]
            i += 1
            if cell in self._taken:
                continue
            if world.soil[cell] not in p.preferred_soil:
                continue
            if world.plant[cell] != EMPTY or world.nutrients[cell] < 90:
                continue
            out.append(cell)
        self._cursor = (self._cursor + i) % max(1, n)
        self._taken.update(out)
        return out

    def policy(self, tick, sim):
        if tick < self.start:
            return []
        self._taken = set()
        world = sim.world
        budget = 20
        picks: list[tuple[int, int]] = []

        # -- SECURE: counted quotas first, nothing else until they are done ---
        for name, remaining in list(self._todo.items()):
            if remaining <= 0 or len(picks) >= budget:
                continue
            take = min(remaining, budget - len(picks))
            cells = self._pick(world, name, take)
            for cell in cells:
                picks.append((self.by_name[name].index, cell))
            self._todo[name] -= len(cells)

        if picks:
            cols = self.spec.cols
            return [(i, c // cols, c % cols) for i, c in picks[:budget]]

        # -- HARVEST: quotas done, spend everything on whatever has unlocked --
        counts = world.counts()
        live = [n for n in self.tier1 if n in sim.unlocked]
        if not live:
            # nothing unlocked yet - keep topping up starters rather than idle
            live = [n for n in self.quotas]
        total = max(1, sum(counts.values()))
        target = 1.0 / max(1, len(live))
        live.sort(key=lambda n: counts.get(self.by_name[n].index, 0) / total - target)

        for name in live:
            if len(picks) >= budget:
                break
            for cell in self._pick(world, name, min(4, budget - len(picks))):
                picks.append((self.by_name[name].index, cell))

        cols = self.spec.cols
        return [(i, c // cols, c % cols) for i, c in picks[:budget]]

    def build(self) -> ActionPlan:
        sim = Simulation(self.spec, self.plan)
        sim.run(policy=self.policy)
        self.simulation = sim
        return self.plan
