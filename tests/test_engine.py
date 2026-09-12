"""Tests for the shared engine: data loading, geometry, scoring maths.

These are the tests that catch a bug which would silently corrupt EVERY level,
so they matter more than any single level's numbers.
"""

from __future__ import annotations

import math
import unittest

from common import ROOT, load_level                     # noqa: F401
from engine.actions import ActionPlan
from engine.data import (
    MAX_PLANTS_PER_TICK,
    STARTING_PLANTS,
    TOTAL_SPECIES,
    load_animals,
    load_classifications,
    load_plants,
    load_unlock_conditions,
    plant,
    plants_by_index,
    resolve_group,
)
from engine.scoring import entropy, max_entropy_for
from engine.sim import offsets_for
from engine.world import World


class TestDataset(unittest.TestCase):
    def test_31_species_and_contiguous_indices(self):
        plants = load_plants()
        self.assertEqual(len(plants), TOTAL_SPECIES)
        self.assertEqual([p.index for p in plants], list(range(1, TOTAL_SPECIES + 1)))

    def test_starting_plants_have_no_unlock_condition(self):
        conditions = load_unlock_conditions()
        for name in STARTING_PLANTS:
            self.assertNotIn(name, conditions,
                             f"{name} is supposed to be available from tick 0")

    def test_every_unlock_names_a_real_plant(self):
        names = {p.name for p in load_plants()}
        for name in load_unlock_conditions():
            self.assertIn(name, names)

    def test_known_plant_properties(self):
        """Spot-check values the whole strategy depends on."""
        oak = plant("Oak Tree")
        self.assertEqual(oak.index, 12)
        self.assertEqual(oak.time_to_maturity, 20)
        self.assertEqual(oak.special_value("shade_radius"), 4)

        grass = plant("Grass")
        self.assertEqual(grass.time_to_maturity, 1)
        self.assertTrue(grass.has_weakness("no_shade_survival"))

        # Mire Bloom is the ONLY plant that accepts clay - it is the only way
        # to use the clay rings around water.
        clay_capable = [p.name for p in load_plants() if 2 in p.preferred_soil]
        self.assertEqual(clay_capable, ["Mire Bloom"])

        # Ashroot is the only burnt-soil plant
        burnt_capable = [p.name for p in load_plants() if 3 in p.preferred_soil]
        self.assertEqual(burnt_capable, ["Ashroot Bramble"])

    def test_orange_blossom_is_slower_in_summer(self):
        """Its conditional_modifier RAISES spread_rate 3 -> 4, and a higher
        spread_rate means it spreads LESS often. Easy thing to get backwards."""
        ob = plant("Orange Blossom")
        self.assertEqual(ob.spread_rate, 3)
        self.assertEqual(ob.spread_rate_for("Summer"), 4)
        self.assertEqual(ob.spread_rate_for("Winter"), 3)

    def test_the_group_name_bug_exists_in_the_source_data(self):
        """animals.json asks for 'Shallow-root Species'; classifications.json
        defines 'Shallowroot Species'. This mismatch is what blocks Bloodbloom."""
        groups = load_classifications()
        self.assertIn("Shallowroot Species", groups)
        self.assertNotIn("Shallow-root Species", groups)

    def test_strict_mode_is_the_default_and_does_not_bridge_the_bug(self):
        """We plan for 30 species, not 31 - so the default must model the
        pessimistic reading where the misspelled group resolves to nothing."""
        import engine.data as data_module
        self.assertTrue(data_module.STRICT_GROUP_NAMES)
        self.assertEqual(resolve_group("Shallow-root Species"),
                         ("Shallow-root Species",))
        # correctly spelled groups still resolve normally
        self.assertEqual(resolve_group("Shallowroot Species"),
                         load_classifications()["Shallowroot Species"])

    def test_tolerant_mode_bridges_the_bug_when_enabled(self):
        """The optimistic reading is available for what-if analysis."""
        import engine.data as data_module
        data_module.STRICT_GROUP_NAMES = False
        try:
            self.assertEqual(resolve_group("Shallow-root Species"),
                             load_classifications()["Shallowroot Species"])
        finally:
            data_module.STRICT_GROUP_NAMES = True

    def test_species_ceiling_differs_between_the_two_readings(self):
        """Quantifies exactly what the data bug costs us: one species."""
        import engine.data as data_module
        from engine.analysis import reachable_species

        events = frozenset({"Rain", "Drought", "Ash Eclipse"})
        strict = reachable_species(events, True)
        data_module.STRICT_GROUP_NAMES = False
        try:
            tolerant = reachable_species(events, True)
        finally:
            data_module.STRICT_GROUP_NAMES = True

        self.assertEqual(len(strict), 30)
        self.assertEqual(len(tolerant), 31)
        self.assertEqual(tolerant - strict, {"Bloodbloom"})

    def test_all_ten_animals_load(self):
        animals = load_animals()
        self.assertEqual(len(animals), 10)
        self.assertIn("Monocryx", {a.name for a in animals})


class TestSpreadGeometry(unittest.TestCase):
    def test_von_neumann(self):
        self.assertEqual(set(offsets_for("VonNeumann", 1)),
                         {(-1, 0), (1, 0), (0, -1), (0, 1)})

    def test_moore_is_the_full_disc(self):
        offsets = offsets_for("Moore", 1)
        self.assertEqual(len(offsets), 8)
        self.assertNotIn((0, 0), offsets)
        self.assertEqual(len(offsets_for("Moore", 2)), 24)

    def test_row_is_horizontal_and_column_is_vertical(self):
        self.assertTrue(all(dr == 0 for dr, _ in offsets_for("Row", 3)))
        self.assertTrue(all(dc == 0 for _, dc in offsets_for("Column", 3)))

    def test_crosshatch_is_diagonal(self):
        self.assertTrue(all(abs(dr) == abs(dc) for dr, dc in offsets_for("CrossHatch", 2)))

    def test_range_scales(self):
        self.assertEqual(len(offsets_for("VonNeumann", 4)), 16)


class TestScoringMaths(unittest.TestCase):
    def test_entropy_of_single_species_is_zero(self):
        self.assertEqual(entropy({1: 100}), 0.0)

    def test_entropy_of_empty_grid_is_zero(self):
        self.assertEqual(entropy({}), 0.0)

    def test_balanced_species_hit_the_log_ceiling(self):
        for s in (2, 5, 10, 31):
            counts = {i: 100 for i in range(1, s + 1)}
            self.assertAlmostEqual(entropy(counts), max_entropy_for(s), places=9)

    def test_all_31_balanced_scores_exactly_one(self):
        counts = {i: 7 for i in range(1, 32)}
        self.assertAlmostEqual(entropy(counts), 1.0, places=9)

    def test_imbalance_lowers_entropy(self):
        balanced = entropy({1: 50, 2: 50})
        skewed = entropy({1: 95, 2: 5})
        self.assertGreater(balanced, skewed)

    def test_entropy_uses_base_31_not_base_species_count(self):
        """A 5-species garden must NOT score 1.0 - the log base is always 31."""
        counts = {i: 10 for i in range(1, 6)}
        self.assertAlmostEqual(entropy(counts), math.log(5) / math.log(31), places=9)
        self.assertLess(entropy(counts), 0.47)


class TestWorldDefaults(unittest.TestCase):
    def test_unlisted_cells_default_to_plantable_dirt(self):
        """The level files only list NON-default cells. This assumption is
        load-bearing for every level - see docs/levels.txt."""
        spec = load_level(1)
        world = World(spec)
        listed = {(c["row"], c["col"]) for c in spec.cells}
        unlisted = [i for i in range(world.size)
                    if world.rc(i) not in listed]
        self.assertGreater(len(unlisted), 0)
        for i in unlisted[:50]:
            self.assertTrue(world.is_passable(i))
            self.assertEqual(world.soil[i], 0)

    def test_level1_plantable_budget(self):
        """Level 1 has exactly 1,800 cells the starting species can use."""
        spec = load_level(1)
        world = World(spec)
        cells = world.plantable_indices(plant("Grass").preferred_soil)
        self.assertEqual(len(cells), 1800)

    def test_listed_terrain_is_applied(self):
        spec = load_level(1)
        world = World(spec)
        for cell in spec.cells[:100]:
            i = world.idx(cell["row"], cell["col"])
            self.assertEqual(world.terrain[i], cell["terrain"])
            self.assertEqual(world.soil[i], cell["soil"])


class TestActionPlan(unittest.TestCase):
    def setUp(self):
        self.spec = load_level(1)
        self.plan = ActionPlan(self.spec)

    def test_enforces_20_per_tick(self):
        for i in range(30):
            self.plan.add(5, 1, i, 0)
        self.assertEqual(len(self.plan.actions_for(5)), MAX_PLANTS_PER_TICK)
        self.assertEqual(self.plan.dropped_full, 10)

    def test_rejects_duplicate_cell_in_same_tick(self):
        self.assertTrue(self.plan.add(3, 1, 10, 10))
        self.assertFalse(self.plan.add(3, 2, 10, 10))
        self.assertEqual(self.plan.dropped_dupe, 1)

    def test_rejects_out_of_bounds(self):
        self.assertFalse(self.plan.add(0, 1, -1, 0))
        self.assertFalse(self.plan.add(0, 1, 0, self.spec.cols))
        self.assertFalse(self.plan.add(self.spec.ticks, 1, 0, 0))
        self.assertEqual(self.plan.dropped_bounds, 3)

    def test_serialises_to_the_documented_schema(self):
        self.plan.add(7, 12, 4, 9)
        doc = self.plan.to_dict()
        self.assertEqual(doc, {"actions": [
            {"tick": 7, "plants": [{"plant_index": 12, "row": 4, "col": 9}]}
        ]})

    def test_ticks_are_emitted_in_order(self):
        for tick in (9, 2, 5):
            self.plan.add(tick, 1, 0, tick)
        ticks = [e["tick"] for e in self.plan.to_dict()["actions"]]
        self.assertEqual(ticks, sorted(ticks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
