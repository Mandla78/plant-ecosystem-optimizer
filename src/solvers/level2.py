from __future__ import annotations
from engine.data import LevelSpec
from .cascade import CascadePlanner

SKIP = frozenset({"Crystal Cactus", "Phoenix Bloom"})

CASCADE_START = 401
PAINT_START = 470
SEEDS_PER_SPECIES = 40
GATE_SEEDS = 250 

def build(spec: LevelSpec):
    planner = CascadePlanner(
        spec,
        cascade_start=CASCADE_START,
        paint_start=PAINT_START,
        seeds_per_species=SEEDS_PER_SPECIES,
        gate_push=True,
        gate_seeds=GATE_SEEDS,
        skip=SKIP,
    )
    return planner.build()