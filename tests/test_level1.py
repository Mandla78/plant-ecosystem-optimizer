"""Level 1 - Greenhouse. 50x50, 500 ticks, no animals, no events.

Level 1 is fully determined: only the 5 starting species can ever exist, so
H is capped at log_31(5) = 0.4687 and C is capped at the 1,800 plantable cells.
We therefore assert against the KNOWN OPTIMUM, not against a loose threshold -
if a change drops us below it, that is a regression.
"""

from __future__ import annotations

import math
import unittest

from common import LevelContract, plants_by_index
from engine.data import load_level, plant
from engine.scoring import max_entropy_for
from engine.world import World

PLANTABLE_CELLS = 1800
CMAX = 2500
PERFECT_H = math.log(5) / math.log(31)      # 0.46866...


class TestLevel1(LevelContract):
    LEVEL = 1

    MIN_SPECIES = 5
    MIN_COVERAGE = PLANTABLE_CELLS / CMAX - 0.001     # 72%
    MIN_FINAL_SCORE = 0.28

    # -- the optimum -------------------------------------------------------

    def test_reaches_the_entropy_ceiling(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        """5 species in exactly equal proportions is the best L1 can ever do."""
        _, _, score = self.simulated()
        self.assertEqual(score.species, 5)
        self.assertAlmostEqual(score.entropy, PERFECT_H, places=4)
        self.assertAlmostEqual(score.entropy, max_entropy_for(5), places=9)

    def test_fills_every_plantable_cell(self):
        _, _, score = self.simulated()
        self.assertEqual(score.population, PLANTABLE_CELLS)

    def test_species_are_evenly_split(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        _, world, _ = self.simulated()
        counts = world.counts()
        self.assertEqual(len(counts), 5)
        for idx, n in counts.items():
            name = plants_by_index()[idx].name
            self.assertEqual(n, PLANTABLE_CELLS // 5,
                             f"{name} has {n}, expected an even 360")

    def test_uses_the_whole_placement_budget_efficiently(self):
        """1,800 placements for 1,800 cells - not one wasted."""
        total = sum(len(e["plants"]) for e in self.doc["actions"])
        self.assertEqual(total, PLANTABLE_CELLS)

    # -- the reasoning behind the design ------------------------------------

    def test_plants_nothing_before_the_paint_window(self):
        """Planting early drains a cell's nutrients and leaves dead matter, so
        the cell may be exhausted by the final tick. L1 deliberately waits."""
        first_tick = min(e["tick"] for e in self.doc["actions"])
        self.assertGreaterEqual(first_tick, 400,
                                "early planting wastes nutrients on L1")

    def test_paint_window_leaves_a_survival_margin(self):
        """The first plants must still be alive at the final tick. They live
        ~100 ticks; starting at 403 leaves 2-3 ticks of margin against an
        off-by-one in the official tick accounting."""
        first_tick = min(e["tick"] for e in self.doc["actions"])
        age_at_scoring = self.spec.ticks - first_tick
        self.assertLessEqual(age_at_scoring, 98,
                             "first plants would starve before scoring")
        self.assertGreaterEqual(age_at_scoring, 90,
                                "starting later than necessary costs longevity")

    def test_oak_is_planted_last_so_it_never_matures(self):
        """Oak matures at 20 and then shades radius 4; Grass has
        no_shade_survival. Keeping every Oak under age 20 at scoring time means
        it counts for C and species diversity without killing our Grass."""
        oak_index = plant("Oak Tree").index
        oak_ticks = [e["tick"] for e in self.doc["actions"]
                     if any(p["plant_index"] == oak_index for p in e["plants"])]
        self.assertTrue(oak_ticks, "no Oak planted")
        youngest_age = self.spec.ticks - min(oak_ticks)
        self.assertLess(youngest_age, plant("Oak Tree").time_to_maturity,
                        "an Oak reaches maturity and will shade out the Grass")

    def test_no_grass_dies_of_shade(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        """Direct check on the outcome rather than the mechanism."""
        _, world, _ = self.simulated()
        counts = world.counts()
        self.assertEqual(counts.get(plant("Grass").index), 360)

    # -- level facts this solution relies on --------------------------------

    def test_clay_beds_are_unusable_and_correctly_skipped(self):
        """360 clay cells exist; none of the 5 starting species accept clay."""
        spec = load_level(1)
        clay = sum(1 for c in spec.cells if c["terrain"] == 0 and c["soil"] == 2)
        self.assertEqual(clay, 360)
        for name in ("Grass", "Rose Bush", "Lavender", "Dwarf Sunflower", "Oak Tree"):
            self.assertNotIn(2, plant(name).preferred_soil)

    def test_plantable_budget_is_1800(self):
        world = World(load_level(1))
        self.assertEqual(len(world.plantable_indices(plant("Grass").preferred_soil)),
                         PLANTABLE_CELLS)

    def test_level_has_no_unlocks_available(self):
        """Animals off + no events = the 5 starters are all we will ever have."""
        spec = load_level(1)
        self.assertFalse(spec.animals_enabled)
        self.assertEqual([c for c in spec.commands if c["type"] == "event"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
