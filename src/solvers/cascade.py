"""Shared unlock-cascade planner for levels 2, 3 and 4.

The three big levels differ only in size and which events fire - the unlock
graph is identical - so they share one controller with different parameters.

THE TWO-ZONE DESIGN
-------------------
Level 1 taught us that a cell is worth far more if it is left untouched: it
keeps all 100 of its nutrients, so a plant put there late survives to the final
tick. A cell used early is drained, flagged with dead matter (which halves its
refill rate), and then thrashes - plants recolonise it at near-zero nutrients
and die within a few ticks. Our first L2 attempt did exactly that and ended at
46% coverage with 98% Grass.

That churn is expensive, but note the correction below: we originally believed
alpha was ~2 (calibrated from the Level 1 score alone) and over-weighted
coverage as a result. The engine's own evaluation log later proved ALPHA = 1 -
main_score is exactly entropy * density - and that density was already near its
ceiling on every level. Diversity, not coverage, was where the score was.
See docs/results.txt.

So the map is split in two:

  WORK ZONE (a minority of the map)
      Where the unlock cascade is run. These cells get burned through - carpet
      the starters, trigger the six easy animals, push each gating species past
      its coverage threshold, walk the waves. We do not care what state this
      zone ends in, only which species it unlocked.

  RESERVE ZONE (the majority)
      Kept completely untouched - full nutrients, no dead matter - until
      `paint_start`. Then it is seeded with every unlocked species in balanced
      proportions. Those plants are young, healthy, and still alive at the final
      tick, which is the only tick that is scored.

Coverage gates are fractions of the WHOLE grid, so the work zone has to be big
enough to host them: Starcap alone needs Whiteveil >5%, Glowcap >3% and Ghost
Orchid >1% of the whole map simultaneously. `work_fraction` is sized for that.
"""

from __future__ import annotations

from engine.actions import ActionPlan
from engine.analysis import reachable_for_level
from engine.data import LevelSpec, plants_by_index, plants_by_name
from engine.sim import Simulation
from engine.world import EMPTY

# Coverage each gating species must exceed, with margin over the real threshold.
GATE_TARGETS: dict[str, float] = {
    "Grass": 0.053,      # >0.05 Verdelopes, with margin             # >0.05 Razorgrass, >0.04 Loamcrawlers/Grazeleths
    "Rose Bush": 0.045,  # >0.04 Ironthorn, with margin         # >0.04 Ironthorn, >0.02 Orange Blossom
    "Lavender": 0.026,   # >=0.02 Nectaris. Moonpetal wants 0.04 but also needs
                         # Blue Moss 0.02 - too expensive, skipped deliberately.          # >0.04 Moonpetal, >=0.02 Nectaris
    "Dwarf Sunflower": 0.036,  # >=0.03 Solwings, with margin   # >=0.03 Solwings
    "Blue Moss": 0.060,         # >0.05 Mire Bloom, >0.03 Silver Fern
    "Crimson Vine": 0.060,      # >0.05 Bloodbloom, >0.04 Purple Canopy / Skyvine
    "Glowcap Fungus": 0.060,    # >0.05 Whiteveil, >0.03 Ghost Orchid / Starcap
    "Orange Blossom": 0.030,    # >0.02 Sunshard
    "Ironthorn Shrub": 0.060,   # >0.05 Thornheart
    "Whiteveil Mycelium": 0.060,   # >0.05 Emberroot / Starcap
    "Silver Fern": 0.070,          # >0.06 Amber Fern
    "Moonpetal Lily": 0.050,       # >0.04 Sunshard, >0.03 Ghost Orchid
    "Ghost Orchid": 0.015,         # >0.01 Starcap
    "Ashroot Bramble": 0.045,      # >0.04 Phoenix Bloom
}

COUNT_TARGETS: dict[str, int] = {
    "Oak Tree": 10,             # >=8 Barkskips, >=6 Worldtree
    "Purple Canopy Tree": 8,    # >=6 Worldtree, >=4 Emberroot / Sporewood
    "Sporewood Tree": 6,        # >=4 Worldtree
    "Emberroot Tree": 4,        # coverage >0 for Ashroot; also burns the soil
}

# The species whose coverage opens the first tier of the tree. Nothing deeper
# can unlock until these clear their gates, and on the big maps those gates are
# large (Blue Moss needs Rose Bush over 1% of the WHOLE grid = 600 cells on L4).
# Measured on L4: Rose Bush stalled at 290 cells, so Blue Moss never unlocked and
# the entire chain behind it stayed locked at 10 species.
# ORDER MATTERS. Placement onto an occupied cell is DENIED (the engine log says
# "plant already occupies cell"; the problem statement wrongly says the existing
# plant is replaced). So whoever plants first keeps the ground.
#
# Measured on Level 2: Grass went in first, spread across the map, and Lavender's
# later placements were refused - it finished on 108 cells = 1.5%, just under the
# 2% the Nectaris animal needs. No Nectaris meant no Crimson Vine, no Orange
# Blossom, and only 6 of 28 species ever unlocked. Slow species go FIRST.
GATE_DRIVERS = ["Lavender", "Rose Bush", "Dwarf Sunflower", "Grass"]

SEED_PRIORITY = [
    "Grass", "Rose Bush", "Lavender", "Dwarf Sunflower",
    "Blue Moss", "Crimson Vine", "Glowcap Fungus",
    "Oak Tree", "Purple Canopy Tree",
    "Whiteveil Mycelium", "Silver Fern", "Moonpetal Lily",
    "Orange Blossom", "Ironthorn Shrub", "Razorgrass", "Stone Reed",
    "Mire Bloom", "Thornheart Bramble",
    "Sporewood Tree", "Emberroot Tree", "Ghost Orchid", "Skyvine",
    "Amber Fern", "Living Topiary", "Sunshard Bloom", "Crystal Cactus",
    "Ashroot Bramble", "Starcap Colony", "Worldtree Sapling", "Phoenix Bloom",
    "Bloodbloom",
]

# Species that spread fast enough to fill the reserve on their own, so they need
# fewer hand-placed seeds than the slow ones.
FAST_SPREADERS = frozenset({
    "Grass", "Razorgrass", "Crimson Vine", "Skyvine", "Whiteveil Mycelium",
    "Blue Moss", "Starcap Colony",
})


def _scatter(cells: list[int]) -> list[int]:
    """Deterministic low-discrepancy ordering - spreads seeds across the map."""
    return sorted(cells, key=lambda i: (i * 2654435761) % 4294967296)


class CascadePlanner:
    def __init__(self, spec: LevelSpec, *,
                 cascade_start: int = 0,
                 paint_start: int | None = None,
                 work_fraction: float = 0.30,
                 seeds_per_species: int = 6,
                 stagger: float = 0.0,
                 fast_weight: float = 0.45,
                 gate_push: bool = False,
                 gate_seeds: int = 8,
                 fast_cap: int | None = None,
                 pregate_start: int | None = None,
                 pregate: tuple = ("Rose Bush", "Lavender"),
                 weights: dict | None = None,
                 gate_quota_override: dict | None = None,
                 runaway: frozenset = frozenset({"Oak Tree", "Razorgrass"}),
                 runaway_release: int = 12,
                 skip: frozenset[str] = frozenset()):
        self.spec = spec
        # Nothing is planted before `cascade_start`, which keeps the map at full
        # nutrients for longer. Later starts mean healthier soil at scoring time
        # but less time for the unlock chain to run - the trade is swept in
        # solvers/level2.py rather than guessed.
        self.cascade_start = cascade_start
        # the paint must finish inside the ~100-tick nutrient lifetime, with a
        # couple of ticks of margin (see solvers/level1.py for the same logic)
        self.paint_start = paint_start if paint_start is not None else spec.ticks - 97
        self.seeds_per_species = seeds_per_species
        # Fraction of the paint window that fast spreaders are held back for.
        # We only hand-place ~10% of the final cells on the big levels, so the
        # mix is decided by SPREAD, not by our placements: a fast species seeded
        # at the start of the window takes a third of the map. Holding it back
        # lets the slow species claim ground first, which is what the entropy
        # term rewards.
        self.stagger = stagger
        # A seed of a fast spreader becomes ~200 cells by the final tick; a slow
        # one becomes a handful. Treating every species' deficit equally
        # therefore hands the map to whatever grows quickest. This weight makes
        # the planner act as if fast species already hold more than they do, so
        # they get fewer seeds and the mix ends up closer to even.
        self.fast_weight = fast_weight
        self.gate_push = gate_push
        self.gate_seeds = gate_seeds
        # Hard ceiling on how many seeds a FAST spreader may ever receive. One
        # Grass seed becomes ~200 cells by the final tick, so a handful of them
        # is already more than a fair share; every seed beyond that is bought
        # straight out of the entropy term.
        self.fast_cap = fast_cap
        self._seeded: dict[str, int] = {}
        self._handplanted: dict[str, int] = {}
        # A narrow head-start for the SLOW species that gate the first tier.
        # Blue Moss needs Rose Bush over 1% of the whole grid - 600 cells on L4 -
        # but Rose Bush matures in 10 ticks and spreads in a Row at range 1, so
        # in a 97-tick window it only manages ~290 and the entire tree behind it
        # stays shut. Giving just these two a head start costs a little coverage
        # (they are slow, so the damage stays local) and may open several tiers.
        self.pregate_start = pregate_start
        self.pregate = pregate
        # Per-species seed allocation weights. A seed of a fast spreader becomes
        # far more cells than a seed of a slow one, so equal seeding produces a
        # wildly unbalanced final garden - and entropy is what we are paid for.
        # Optionally fitted by solvers/experimental/balance.py (which measured
        # WORSE entropy each round - the final mix is driven by spread, not by
        # how many seeds we hand out).
        self.weights = dict(weights or {})
        # Hand-quotas assume ZERO spread, which is right on a small map but very
        # wrong on a big one: on L3 Grass turns 720 seeds into 9,853 cells (13.7x)
        # and ends up at 73% of the garden, crushing entropy and spawning
        # Monocryx. On the big levels the quota for a fast spreader must be set
        # from its multiplier, not from the raw threshold.
        self.gate_quota_override = dict(gate_quota_override or {})
        # RUNAWAY CONTROL - the single biggest lever, measured from the engine's
        # own evaluation logs:
        #     Level 3: Oak 68.8% + Razorgrass 18.8% of the whole garden
        #     Level 4: Razorgrass 56.9% + Oak 24.9%
        # Density was already maxed (0.81-0.88 against 0.81-0.90 plantable), and
        # score = 0.8 * entropy * density, so these two were destroying the only
        # term still worth anything.
        #
        # We cannot stop them spreading, but we can deny them the TIME to do it.
        # Oak matures at 20 ticks and Razorgrass at 2; released in the final few
        # ticks they still occupy the cells we place them on - so they still
        # count as species - but never mature, never spread, and never shade.
        # Oak's shade is what killed every Grass plant on Level 3, which in turn
        # killed the Loamcrawlers animal and the whole unlock tree behind it.
        self.runaway = frozenset(runaway)
        self.runaway_release = runaway_release

        self.by_index = plants_by_index()
        self.by_name = plants_by_name()

        reachable = reachable_for_level(spec)
        self.usable = [n for n in SEED_PRIORITY
                       if n in reachable and n not in skip]

        self.plan = ActionPlan(spec)
        self._cursor: dict[str, int] = {}
        self._taken: set[int] = set()

        # -- split the map -------------------------------------------------
        from engine.world import World
        probe = World(spec)
        all_plantable = _scatter(probe.plantable_indices(frozenset({0, 1})))
        split = int(len(all_plantable) * work_fraction)
        self.work_cells = set(all_plantable[:split])
        self.reserve_cells = all_plantable[split:]
        self._reserve_set = set(self.reserve_cells)
        self._soil_cache: dict[int, list[int]] = {}
        self._probe = probe

    # -- cell selection ----------------------------------------------------

    def _zone_cells(self, name: str, reserve) -> list[int]:
        """Cells of the right soil in the requested zone, scattered.

        reserve=True  -> the pristine reserve only
        reserve=False -> the work zone only
        reserve=None  -> anywhere plantable (used by the paint phase)
        """
        p = self.by_name[name]
        key = hash((p.preferred_soil, reserve))
        if key not in self._soil_cache:
            viable = self._probe.plantable_indices(p.preferred_soil)
            if reserve is None:
                pool = viable
            else:
                allowed = self._reserve_set if reserve else self.work_cells
                pool = [c for c in viable if c in allowed]
            self._soil_cache[key] = _scatter(pool)
        return self._soil_cache[key]

    def _pick(self, world, name: str, wanted: int, *, reserve,
              overwrite: bool = True, min_nutrients: float = 20.0) -> list[int]:
        """Next `wanted` viable cells, preferring empty ones.

        `self._taken` stops two species claiming the same cell in one tick, which
        the engine would silently drop as a duplicate.
        """
        p = self.by_name[name]
        cells = self._zone_cells(name, reserve)
        n = len(cells)
        if not n or wanted <= 0:
            return []

        empty: list[int] = []
        occupied: list[int] = []
        start = self._cursor.get((name, reserve), 0)
        i = 0
        while i < n and len(empty) < wanted:
            cell = cells[(start + i) % n]
            i += 1
            if cell in self._taken or world.nutrients[cell] < min_nutrients:
                continue
            occupant = world.plant[cell]
            if occupant == EMPTY:
                empty.append(cell)
            elif overwrite and occupant != p.index and len(occupied) < wanted:
                occupied.append(cell)

        self._cursor[(name, reserve)] = (start + i) % n
        out = (empty + occupied)[:wanted]
        self._taken.update(out)
        return out

    # -- phase 1: unlock cascade, confined to the work zone ----------------

    def _cascade(self, tick, sim, budget):
        """Drive the first-tier gates hard before switching to balanced painting.

        Spreading seeds evenly across every species leaves each one short of its
        threshold, so the deeper tree never opens. Here the whole budget goes to
        the handful of species that gate everything else, until they are past
        their targets.

        OFF BY DEFAULT: measured on L3 this LOWERS the score (350M -> 293M).
        Concentrating the budget on four starters lets those four dominate the
        final mix, and the entropy lost outweighs the species gained. Kept
        because it is the right lever on a map where a first-tier gate is the
        actual blocker - see solvers/level4.py.
        """
        world = sim.world
        counts = world.counts()
        cmax = self.spec.cmax
        picks = []

        if self.gate_push:
            # Count only what we have HAND-PLANTED, not the world population.
            # Counting the world lets weak spread flatter us into stopping early,
            # which is precisely how the first Level 2 submission ended up with
            # 28 Grass against a 281-cell requirement and unlocked nothing.
            for name in GATE_DRIVERS:
                if len(picks) >= budget:
                    break
                if name not in sim.unlocked:
                    continue
                p = self.by_name[name]
                frac = self.gate_quota_override.get(
                    name, GATE_TARGETS.get(name, 0.05))
                quota = int(frac * cmax) + 1
                if self._handplanted.get(name, 0) >= quota:
                    continue
                want = min(budget - len(picks), self.gate_seeds)
                for cell in self._pick(world, name, want, reserve=None,
                                       min_nutrients=8.0):
                    picks.append((p.index, cell))
                    self._handplanted[name] = self._handplanted.get(name, 0) + 1
            if picks:
                return picks

        for name in self.usable:
            if len(picks) >= budget:
                break
            if name not in sim.unlocked:
                continue
            if name in self.runaway and tick < self.spec.ticks - self.runaway_release:
                continue
            p = self.by_name[name]
            have = counts.get(p.index, 0)

            target_count = COUNT_TARGETS.get(name)
            target_cov = GATE_TARGETS.get(name)

            if target_count is not None and have < target_count:
                want = min(target_count - have, 3, budget - len(picks))
            elif target_cov is not None and have / cmax < target_cov:
                want = min(self.seeds_per_species, budget - len(picks))
            elif have == 0:
                want = min(2, budget - len(picks))       # get it on the board
            else:
                continue

            for cell in self._pick(world, name, want, reserve=False,
                                   min_nutrients=5.0):
                picks.append((p.index, cell))
        return picks

    # -- phase 2: paint the pristine reserve -------------------------------

    def _paint(self, tick, sim, budget):
        """The scored phase: scatter seeds across the WHOLE map and let spread
        close the gaps.

        Measured on L2: scattering ~1,900 seeds from tick 403 onto pristine soil
        reaches 87.5% coverage by tick 500 - essentially every plantable cell -
        because each seed becomes a spreading colony. Restricting the paint to a
        reserve zone, or demanding high nutrients, throws that away.

        Species are chosen by deficit so the final mix is balanced, which is what
        the entropy term rewards. Fast spreaders are deliberately under-seeded:
        they will claim more than their share of the gaps on their own.
        """
        world = sim.world
        counts = world.counts()
        live = [n for n in self.usable if n in sim.unlocked]
        if not live:
            return []

        total = max(1, sum(counts.values()))
        share_target = 1.0 / len(live)

        wsum = sum(self.weights.get(n, 1.0) for n in live)

        def deficit(n):
            have = counts.get(self.by_name[n].index, 0) / total
            want = self.weights.get(n, 1.0) / wsum if wsum else share_target
            return have - want

        # hold fast spreaders back until late in the window
        if self.stagger > 0:
            window = self.spec.ticks - self.paint_start
            release = self.paint_start + int(window * self.stagger)
            if tick < release:
                slow = [n for n in live if n not in FAST_SPREADERS]
                if slow:
                    live = slow

        picks = []
        for name in sorted(live, key=deficit):
            if len(picks) >= budget:
                break
            if name in self.runaway and tick < self.spec.ticks - self.runaway_release:
                continue
            if (self.fast_cap is not None and name in FAST_SPREADERS
                    and self._seeded.get(name, 0) >= self.fast_cap):
                continue
            want = min(budget - len(picks), 3)
            for cell in self._pick(world, name, want, reserve=None,
                                   overwrite=True, min_nutrients=8.0):
                picks.append((self.by_name[name].index, cell))
                self._seeded[name] = self._seeded.get(name, 0) + 1
        return picks

    # -- driver ------------------------------------------------------------

    def policy(self, tick, sim):
        budget = 20
        self._taken = set()
        if self.pregate_start is not None and self.pregate_start <= tick < self.cascade_start:
            picks = []
            world = sim.world
            per = max(1, budget // max(1, len(self.pregate)))
            for name in self.pregate:
                if name not in sim.unlocked or len(picks) >= budget:
                    continue
                for cell in self._pick(world, name, min(per, budget - len(picks)),
                                       reserve=None, min_nutrients=8.0):
                    picks.append((self.by_name[name].index, cell))
            cols = self.spec.cols
            return [(i, c // cols, c % cols) for i, c in picks[:budget]]
        if tick < self.cascade_start:
            return []
        if tick >= self.paint_start:
            picks = self._paint(tick, sim, budget)
        else:
            picks = self._cascade(tick, sim, budget)

        cols = self.spec.cols
        return [(idx, cell // cols, cell % cols) for idx, cell in picks[:budget]]

    def build(self) -> ActionPlan:
        sim = Simulation(self.spec, self.plan)
        sim.run(policy=self.policy)
        self.simulation = sim
        return self.plan
