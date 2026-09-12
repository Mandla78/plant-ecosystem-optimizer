"""Level 4 - Forest. 200x300, 800 ticks, animals on, every event.

Events: Rain@50, Ash Eclipse@250, Drought@280, Earthquake@700.
Reachable: 30 of 31 (only Bloodbloom is blocked).

L4 is defined by one number: 16,000 total placements against 54,012 plantable
cells. We can hand-touch at most 30% of the map, and a single 0.05 coverage gate
is 3,000 cells. So L4 is won by seed GEOMETRY and natural spread, and the manual
budget is spent on the unlock chain plus the final diversity top-up.
"""

from __future__ import annotations

import unittest

from common import LevelContract
from engine.analysis import reachable_for_level
from engine.data import load_level, plants_by_name

REACHABLE_SPECIES = 30
UNREACHABLE = {"Bloodbloom"}
PLANTABLE_CELLS = 54012


class TestLevel4(LevelContract):
    LEVEL = 4

    # Measured result. 10 of 30 species is the weakest point of the whole
    # submission: on a 60,000-cell map a first-tier gate needs 600+ cells and
    # Rose Bush stalls short of it, so most of the tree never opens.
    MIN_SPECIES = 10
    MIN_COVERAGE = 0.85
    MIN_FINAL_SCORE = 0.30
    MIN_ENTROPY_RATIO = 0.74

    def test_plants_in_the_scoring_window(self):
        late = [e for e in self.doc["actions"] if e["tick"] >= self.spec.ticks - 100]
        self.assertTrue(late, "nothing planted in the scoring window")

    def test_spends_its_placements_where_they_survive(self):
        """We deliberately use only ~1,900 of the 16,000 available placements.

        The other 14,000 sit in ticks whose plants would be dead long before
        scoring, and using them would drain the very cells the final garden
        needs. What matters is not how many slots we spend but how much
        population they produce: ~1,900 seeds become ~52,000 cells through
        natural spread."""
        placed = sum(len(e["plants"]) for e in self.doc["actions"])
        _, _, score = self.simulated()
        self.assertGreater(score.population, placed * 10,
                           "seeds are not multiplying - spread has stopped working")

    def test_skyvine_established_before_the_earthquake(self):
        """The quake at tick 700 opens cracks, and Skyvine (crack_spread) is
        the only species that can cross them."""
        skyvine = plants_by_name()["Skyvine"].index
        ticks = [e["tick"] for e in self.doc["actions"]
                 if any(p["plant_index"] == skyvine for p in e["plants"])]
        if not ticks:
            self.skipTest("solver does not use Skyvine")
        self.assertLess(min(ticks), 700)

    def test_winter_sensitive_species_are_hand_placed_at_the_end(self):
        """Rose Bush and Lavender cannot SPREAD after tick 700, so the only way
        they appear in the scored state is direct placement."""
        sensitive = {plants_by_name()[n].index for n in ("Rose Bush", "Lavender")}
        late = {p["plant_index"] for e in self.doc["actions"]
                if e["tick"] >= 700 for p in e["plants"]}
        self.assertTrue(late & sensitive)

    def test_keeps_the_soil_pristine_until_late(self):
        first = min(e["tick"] for e in self.doc["actions"])
        self.assertGreaterEqual(first, 650)


class TestLevel4Facts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_level(4)

    def test_grid_and_timing(self):
        self.assertEqual((self.spec.rows, self.spec.cols), (200, 300))
        self.assertEqual(self.spec.ticks, 800)
        self.assertEqual(self.spec.cmax, 60000)

    def test_every_event_fires(self):
        events = {c["event"]: c["tick"] for c in self.spec.commands
                  if c["type"] == "event"}
        self.assertEqual(events, {"Rain": 50, "Ash Eclipse": 250,
                                  "Drought": 280, "Earthquake": 700})

    def test_ends_in_winter(self):
        self.assertEqual(self.spec.season_at(self.spec.ticks - 1), "Winter")

    def test_reachable_species(self):
        reachable = reachable_for_level(self.spec)
        self.assertEqual(len(reachable), REACHABLE_SPECIES)
        from engine.data import load_plants
        self.assertEqual({p.name for p in load_plants()} - reachable, UNREACHABLE)

    def test_manual_budget_cannot_cover_the_map(self):
        """The single fact that dictates the whole L4 strategy."""
        budget = self.spec.ticks * 20
        self.assertEqual(budget, 16000)
        self.assertLess(budget, PLANTABLE_CELLS * 0.35,
                        "if we could paint the map, L4 would be an L1-style problem")

    def test_phoenix_bloom_burn_cost_is_significant(self):
        """Phoenix Bloom needs Ashroot coverage > 0.04 on BURNT soil. Each
        Emberroot burns roughly a 5x5 disc, so this is ~100 Emberroots and ~4%
        of the map sterilised - possibly not worth one extra species."""
        ashroot_cells = int(0.04 * self.spec.cmax)
        emberroots_needed = ashroot_cells / 25
        self.assertGreater(ashroot_cells, 2000)
        self.assertGreater(emberroots_needed, 90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
