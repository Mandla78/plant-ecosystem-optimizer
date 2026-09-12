from __future__ import annotations
from engine.data import LevelSpec
from .cascade import CascadePlanner

CASCADE_START = 695
PAINT_START = 705
SEEDS_PER_SPECIES = 300
GATE_SEEDS = 800
GRASS_QUOTA = 0.012

def build(spec: LevelSpec):
    planner = CascadePlanner(
        spec,
        cascade_start=CASCADE_START,
        paint_start=PAINT_START,
        seeds_per_species=SEEDS_PER_SPECIES,
        gate_push=True,
        gate_seeds=GATE_SEEDS,
        gate_quota_override={"Grass": GRASS_QUOTA},
    )
    return planner.build()