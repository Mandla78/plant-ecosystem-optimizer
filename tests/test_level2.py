"""Level 2 - Garden Growth. 70x100, 500 ticks, animals on, Rain at tick 250.

Reachable: 28 of 31 species.
Blocked: Crystal Cactus (needs Drought), Phoenix Bloom (needs Ash Eclipse),
Bloodbloom (needs Rhizorends, blocked by the group-name bug in the data).

The LevelContract tests SKIP until src/solvers/level2.py and its solution.json
exist, so the suite stays green while we build. TestLevel2Facts runs
immediately - it guards the assumptions the solver is being written against.
"""

from __future__ import annotations

import unittest

from common import LevelContract
from engine.analysis import reachable_for_level, unlock_waves
from engine.data import load_level, plants_by_name

REACHABLE_SPECIES = 28
UNREACHABLE = {"Crystal Cactus", "Phoenix Bloom", "Bloodbloom"}
# Bloodbloom is a free lottery ticket: if their engine normalises the group name
# it unlocks, and a rejected planting costs nothing. The other two are certain.
CERTAINLY_UNREACHABLE = {"Crystal Cactus", "Phoenix Bloom"}


class TestLevel2(LevelContract):
    LEVEL = 2

    # Deliberately modest to start; tighten once the solver is tuned.
    # Recalibrated after the first submission scored 79M against a 534M
    # prediction. The old thresholds were set from a spread model we now know
    # was ~2.4x too generous, so they were asserting fiction.
    MIN_SPECIES = 12
    MIN_COVERAGE = 0.70
    MIN_FINAL_SCORE = 0.34
    MIN_ENTROPY_RATIO = 0.80

    def test_does_not_waste_budget_on_impossible_species(self):
        """Every action is one slot out of a hard 20/tick budget; planting a
        species this level can never unlock throws that slot away."""
        banned = {plants_by_name()[n].index for n in CERTAINLY_UNREACHABLE}
        used = {p["plant_index"] for e in self.doc["actions"] for p in e["plants"]}
        self.assertEqual(used & banned, set())

    def test_plants_in_the_scoring_window(self):
        """Only the final tick is scored and plants live ~100 ticks, so the
        last 100 ticks are where the score is actually made."""
        late = [e for e in self.doc["actions"] if e["tick"] >= self.spec.ticks - 100]
        self.assertTrue(late, "nothing planted in the scoring window")

    def test_every_unlock_gate_is_secured_by_hand(self):
        """THE test for this level - it is the exact failure that cost us 79M.

        Our first submission planted 28 Grass and trusted spread to carry it
        past the 280 cells the Loamcrawlers animal requires. Real spread is far
        weaker than we modelled, Grass never got there, no animal appeared, and
        64% of our actions were silently discarded.

        So: every threshold must be cleared by HAND-PLACED cells alone, with
        spread counted as zero. Then the unlocks cannot fail however wrong our
        spread model turns out to be."""
        from collections import Counter
        counts = Counter(p["plant_index"] for e in self.doc["actions"]
                         for p in e["plants"])
        cmax = self.spec.cmax
        required = [
            ("Grass", 0.05, "Verdelopes"),
            ("Grass", 0.04, "Loamcrawlers and Grazeleths"),
            ("Rose Bush", 0.04, "Ironthorn Shrub"),
            ("Lavender", 0.02, "Nectaris"),
            ("Dwarf Sunflower", 0.03, "Solwings"),
        ]
        for name, threshold, why in required:
            have = counts.get(plants_by_name()[name].index, 0)
            need = int(threshold * cmax) + 1
            self.assertGreaterEqual(
                have, need,
                f"{name}: {have} hand-planted, {need} needed for {why} - "
                "this gate depends on spread and may not fire"
            )

    def test_starts_late_enough_that_the_garden_survives(self):
        """Plants live ~100 ticks, so most of what we place must still be alive
        at the final tick. We start a little before T-100 deliberately: with
        spread this weak the early cohort's descendants are worth more than the
        cohort itself."""
        first = min(e["tick"] for e in self.doc["actions"])
        self.assertGreaterEqual(first, 330, "too early - the map will thrash")
        self.assertLessEqual(first, 420, "too late - nothing has time to spread")

    def test_relies_on_spread_not_brute_force(self):
        """We place ~1,900 seeds but finish with ~6,800 populated cells - the
        rest is natural spread from those colonies. If this ratio collapses the
        seeding has become too dense to be efficient."""
        placed = sum(len(e["plants"]) for e in self.doc["actions"])
        _, _, score = self.simulated()
        self.assertGreater(score.population, placed * 2,
                           "spread is not doing its share of the filling")


class TestLevel2Facts(unittest.TestCase):
    """Assumptions behind the Level 2 strategy - these run right now."""

    @classmethod
    def setUpClass(cls):
        cls.spec = load_level(2)

    def test_grid_and_timing(self):
        self.assertEqual((self.spec.rows, self.spec.cols), (70, 100))
        self.assertEqual(self.spec.ticks, 500)
        self.assertTrue(self.spec.animals_enabled)

    def test_only_rain_fires(self):
        events = {c["event"]: c["tick"] for c in self.spec.commands
                  if c["type"] == "event"}
        self.assertEqual(events, {"Rain": 250})

    def test_does_not_end_in_winter(self):
        """Unlike L3/L4, Rose Bush and Lavender CAN still spread at the end."""
        self.assertEqual(self.spec.season_at(self.spec.ticks - 1), "Spring")

    def test_reachable_species_recomputed_from_the_graph(self):
        reachable = reachable_for_level(self.spec)
        self.assertEqual(len(reachable), REACHABLE_SPECIES)
        from engine.data import load_plants
        self.assertEqual({p.name for p in load_plants()} - reachable, UNREACHABLE)

    def test_unlock_cascade_has_four_waves(self):
        waves = unlock_waves(frozenset({"Rain"}), True)
        self.assertEqual(len(waves), 4)
        self.assertIn("Blue Moss", waves[0][0])
        self.assertIn("Whiteveil Mycelium", waves[1][0])
        self.assertIn("Worldtree Sapling", waves[3][0])

    def test_six_animals_arrive_from_the_starting_plants_alone(self):
        """The whole early game: carpet with the 5 starters and the animal
        block shows up on its own."""
        first_wave_animals = unlock_waves(frozenset({"Rain"}), True)[0][1]
        for name in ("Nectaris", "Verdelopes", "Loamcrawlers", "Virexids",
                     "Grazeleths", "Solwings"):
            self.assertIn(name, first_wave_animals)


if __name__ == "__main__":
    unittest.main(verbosity=2)
