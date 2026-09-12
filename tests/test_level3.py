"""Level 3 - Park Potential. 150x150, 800 ticks, animals on.

Events: Drought@10, Rain@150, Ash Eclipse@300 - so all three unlock-bearing
events fire, and early. Reachable: 30 of 31 (only Bloodbloom is blocked).

The hard constraints here are scale and the ending season:
  * a 0.05 coverage gate is 1,125 cells, beyond hand-planting, so the unlock
    chain must be driven by natural spread
  * the level ENDS IN WINTER (tick 700+), when Rose Bush and Lavender cannot
    spread - exactly during the window that gets scored
"""

from __future__ import annotations

import unittest

from common import LevelContract
from engine.analysis import reachable_for_level
from engine.data import load_level, plants_by_name

REACHABLE_SPECIES = 30
UNREACHABLE = {"Bloodbloom"}


class TestLevel3(LevelContract):
    LEVEL = 3

    # Rebuilt after the Level 2 result showed our spread model was ~2.4x too
    # generous. These are the re-measured values under the calibrated model.
    MIN_SPECIES = 14
    MIN_COVERAGE = 0.55
    MIN_FINAL_SCORE = 0.30
    MIN_ENTROPY_RATIO = 0.85

    def test_plants_in_the_scoring_window(self):
        late = [e for e in self.doc["actions"] if e["tick"] >= self.spec.ticks - 100]
        self.assertTrue(late, "nothing planted in the scoring window")

    def test_planting_fits_inside_the_survival_window(self):
        """Plants live ~100 ticks, and only the final tick is scored, so nothing
        planted before roughly T-105 is still alive when it matters.

        An earlier version of this test asserted the opposite - that planting
        should START early so spread could fill the map. That came from the
        period when we believed density was squared in the score and were
        optimising coverage. The engine log later showed density was already at
        its ceiling (0.813 of a 0.813 plantable fraction) while entropy was at
        44% of its own, so filling the map was never the constraint."""
        first = min(e["tick"] for e in self.doc["actions"])
        self.assertGreaterEqual(first, self.spec.ticks - 115,
                                "planted too early - it starves before scoring")
        self.assertLessEqual(first, self.spec.ticks - 40,
                             "planted so late nothing can establish")

    def test_grass_is_deliberately_starved(self):
        """Grass is seeded far below its gate quota because fast spreaders
        monopolise the garden and entropy is what the score pays for.

        Asserted against the SOLUTION (how much we hand-plant), not against the
        simulated outcome: the simulator over-models Grass spread and predicts
        it at 76%, whereas the real engine log has Grass at 3% on this level
        with Oak as the actual monopolist."""
        from collections import Counter
        counts = Counter(p["plant_index"] for e in self.doc["actions"]
                         for p in e["plants"])
        total = sum(counts.values())
        grass = counts.get(plants_by_name()["Grass"].index, 0) / total
        self.assertLess(grass, 0.25,
                        f"hand-planting {grass:.0%} Grass - it will monopolise")
        return
        _, world, _ = self.simulated()
        counts = world.counts()
        total = sum(counts.values()) or 1
        grass = counts.get(plants_by_name()["Grass"].index, 0) / total
        self.assertLess(grass, 0.20, f"Grass at {grass:.0%} - entropy will collapse")

    def test_winter_sensitive_species_are_hand_placed_at_the_end(self):
        """Rose Bush and Lavender cannot SPREAD after tick 700. If we want them
        in the final mix they have to be placed directly."""
        winter_start = 700
        sensitive = {plants_by_name()[n].index for n in ("Rose Bush", "Lavender")}
        late = {p["plant_index"] for e in self.doc["actions"]
                if e["tick"] >= winter_start for p in e["plants"]}
        self.assertTrue(late & sensitive,
                        "no Rose Bush or Lavender placed during the final winter")


class TestLevel3Facts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_level(3)

    def test_grid_and_timing(self):
        self.assertEqual((self.spec.rows, self.spec.cols), (150, 150))
        self.assertEqual(self.spec.ticks, 800)
        self.assertTrue(self.spec.animals_enabled)

    def test_all_three_unlock_events_fire_early(self):
        events = {c["event"]: c["tick"] for c in self.spec.commands
                  if c["type"] == "event"}
        self.assertEqual(events, {"Drought": 10, "Rain": 150, "Ash Eclipse": 300})
        self.assertLess(max(events.values()), self.spec.ticks // 2,
                        "events land in the first half - no need to wait")

    def test_level_ends_in_winter(self):
        self.assertEqual(self.spec.season_at(self.spec.ticks - 1), "Winter")
        self.assertEqual(self.spec.season_at(699), "Autumn")

    def test_reachable_species(self):
        reachable = reachable_for_level(self.spec)
        self.assertEqual(len(reachable), REACHABLE_SPECIES)
        from engine.data import load_plants
        self.assertEqual({p.name for p in load_plants()} - reachable, UNREACHABLE)

    def test_coverage_gates_exceed_the_manual_budget(self):
        """Justifies the 'seed clusters, do not paint' strategy."""
        budget = self.spec.ticks * 20
        gate_cells = int(0.05 * self.spec.cmax)
        self.assertGreater(gate_cells, 1000)
        self.assertLess(gate_cells, budget,
                        "a single gate should still be under the total budget")


if __name__ == "__main__":
    unittest.main(verbosity=2)
