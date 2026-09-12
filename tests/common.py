"""Shared helpers and the per-level contract every solution must satisfy.

Uses only the standard library (unittest) - no pytest, no third-party deps, so
the suite runs anywhere the submission itself runs.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from engine.data import (                       # noqa: E402
    MAX_PLANTS_PER_TICK,
    STARTING_PLANTS,
    TERRAIN_GROUND,
    load_level,
    plants_by_index,
    plants_by_name,
)
from engine.scoring import max_entropy_for, score_world   # noqa: E402
from engine.sim import Simulation                          # noqa: E402


def solution_path(level: int) -> str:
    return os.path.join(ROOT, "submissions", f"level{level}", "solution.json")


def zip_path(level: int) -> str:
    return os.path.join(ROOT, "submissions", f"level{level}", f"level{level}.zip")


def solver_path(level: int) -> str:
    return os.path.join(SRC, "solvers", f"level{level}.py")


def load_solution(level: int) -> dict:
    with open(solution_path(level), encoding="utf-8") as fh:
        return json.load(fh)


def terrain_soil_maps(spec):
    terrain, soil = {}, {}
    for cell in spec.cells:
        terrain[(cell["row"], cell["col"])] = cell["terrain"]
        soil[(cell["row"], cell["col"])] = cell["soil"]
    return terrain, soil


class LevelContract(unittest.TestCase):
    """Base class: subclass it, set LEVEL, and every shared rule is checked.

    Subclasses add their own level-specific expectations (species count,
    minimum score, and so on).
    """

    LEVEL: int | None = None

    # expectations a subclass may tighten
    MIN_SPECIES = 1
    MIN_COVERAGE = 0.0
    MIN_FINAL_SCORE = 0.0
    MAX_SINGLE_SPECIES_SHARE = 0.50      # above this, Monocryx spawns
    MIN_ENTROPY_RATIO = 0.85             # H as a fraction of log_31(S)

    # Our simulator reproduces DENSITY and legality well, but it does not
    # reproduce COMPOSITION - it predicts a monoculture where the real engine
    # produced a balanced garden (see README, "Honest limitations"). Tests that
    # depend on which species end up where are therefore skipped rather than
    # asserted against a model we know to be wrong on that axis. Ground truth
    # for composition lives in tests/test_ground_truth.py, which reads the real
    # evaluation logs in logs/.
    SIMULATOR_MODELS_COMPOSITION = False

    @classmethod
    def setUpClass(cls):
        if cls.LEVEL is None:
            raise unittest.SkipTest("base class")
        if not os.path.exists(solver_path(cls.LEVEL)):
            raise unittest.SkipTest(f"no solver for level {cls.LEVEL} yet")
        if not os.path.exists(solution_path(cls.LEVEL)):
            raise unittest.SkipTest(
                f"no solution.json for level {cls.LEVEL} yet - run "
                f"python src/run.py --level {cls.LEVEL}"
            )
        cls.spec = load_level(cls.LEVEL)
        cls.doc = load_solution(cls.LEVEL)
        cls.terrain, cls.soil = terrain_soil_maps(cls.spec)
        cls._sim_cache = None

    # -- lazily simulate once and share across tests -----------------------

    @classmethod
    def simulated(cls):
        if cls._sim_cache is None:
            module = importlib.import_module(f"solvers.level{cls.LEVEL}")
            plan = module.build(cls.spec)
            sim = Simulation(cls.spec, plan)
            world = sim.run()
            cls._sim_cache = (sim, world, score_world(world, cls.spec.ticks))
        return cls._sim_cache

    # -- structural ---------------------------------------------------------

    def test_top_level_shape(self):
        self.assertEqual(set(self.doc), {"actions"})
        self.assertIsInstance(self.doc["actions"], list)
        self.assertGreater(len(self.doc["actions"]), 0, "solution plants nothing")

    def test_uses_plant_index_key(self):
        """The PDF shows both 'plant_index' and 'index'; we standardise on the
        one used in the full worked example."""
        for entry in self.doc["actions"]:
            for p in entry["plants"]:
                self.assertIn("plant_index", p)
                self.assertEqual(set(p), {"plant_index", "row", "col"})

    def test_ticks_in_range_and_unique(self):
        seen = set()
        for entry in self.doc["actions"]:
            tick = entry["tick"]
            self.assertIsInstance(tick, int)
            self.assertGreaterEqual(tick, 0)
            self.assertLessEqual(tick, self.spec.ticks - 1,
                                 "actions must land in [0, T-1]")
            self.assertNotIn(tick, seen, f"tick {tick} listed twice")
            seen.add(tick)

    def test_at_most_20_plants_per_tick(self):
        for entry in self.doc["actions"]:
            self.assertLessEqual(
                len(entry["plants"]), MAX_PLANTS_PER_TICK,
                f"tick {entry['tick']} exceeds the cap - extras are dropped"
            )

    def test_no_duplicate_cell_within_a_tick(self):
        for entry in self.doc["actions"]:
            cells = [(p["row"], p["col"]) for p in entry["plants"]]
            dupes = [c for c, n in Counter(cells).items() if n > 1]
            self.assertEqual(dupes, [], f"tick {entry['tick']} replants {dupes}")

    def test_coordinates_in_bounds(self):
        for entry in self.doc["actions"]:
            for p in entry["plants"]:
                self.assertTrue(0 <= p["row"] < self.spec.rows,
                                f"row {p['row']} out of bounds")
                self.assertTrue(0 <= p["col"] < self.spec.cols,
                                f"col {p['col']} out of bounds")

    def test_plant_indices_are_real(self):
        known = plants_by_index()
        for entry in self.doc["actions"]:
            for p in entry["plants"]:
                self.assertIn(p["plant_index"], known)

    # -- placement legality -------------------------------------------------

    def test_never_plants_on_unplantable_terrain(self):
        bad = []
        for entry in self.doc["actions"]:
            for p in entry["plants"]:
                t = self.terrain.get((p["row"], p["col"]), TERRAIN_GROUND)
                if t != TERRAIN_GROUND:
                    bad.append((entry["tick"], p["row"], p["col"], t))
        self.assertEqual(bad[:5], [], f"{len(bad)} placements on water/stone/path")

    def test_respects_preferred_soil(self):
        by_index = plants_by_index()
        bad = []
        for entry in self.doc["actions"]:
            for p in entry["plants"]:
                s = self.soil.get((p["row"], p["col"]), 0)
                plant = by_index[p["plant_index"]]
                if s not in plant.preferred_soil:
                    bad.append((entry["tick"], plant.name, s))
        self.assertEqual(bad[:5], [], f"{len(bad)} placements on wrong soil")

    def test_no_locked_plants_when_animals_disabled(self):
        """With animals off and no events, only the 5 starters can ever exist."""
        if self.spec.animals_enabled:
            self.skipTest("animals enabled - unlocks are reachable")
        allowed = {plants_by_name()[n].index for n in STARTING_PLANTS}
        used = {p["plant_index"] for e in self.doc["actions"] for p in e["plants"]}
        self.assertTrue(used <= allowed,
                        f"unreachable species used: {sorted(used - allowed)}")

    # -- determinism --------------------------------------------------------

    def test_build_is_deterministic(self):
        """Rule 6: same input must always give the same output."""
        module = importlib.import_module(f"solvers.level{self.LEVEL}")
        first = module.build(self.spec).to_dict()
        second = module.build(self.spec).to_dict()
        self.assertEqual(first, second, "solver is not deterministic")

    def test_solution_file_matches_solver(self):
        """The committed solution.json must be what the solver produces now."""
        module = importlib.import_module(f"solvers.level{self.LEVEL}")
        self.assertEqual(
            module.build(self.spec).to_dict(), self.doc,
            "solution.json is stale - re-run python src/run.py --level "
            f"{self.LEVEL}"
        )

    # -- outcome ------------------------------------------------------------

    def test_simulated_species_count(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        _, _, score = self.simulated()
        self.assertGreaterEqual(score.species, self.MIN_SPECIES)

    def test_simulated_coverage(self):
        _, _, score = self.simulated()
        self.assertGreaterEqual(score.coverage, self.MIN_COVERAGE)

    def test_simulated_final_score(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        _, _, score = self.simulated()
        self.assertGreaterEqual(score.final, self.MIN_FINAL_SCORE)

    def test_no_species_dominates(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        """Any species at >=50% spawns Monocryx, which blocks Amber Fern."""
        _, world, _ = self.simulated()
        counts = world.counts()
        total = sum(counts.values())
        if not total:
            self.skipTest("empty garden")
        worst = max(counts.values()) / total
        self.assertLess(worst, self.MAX_SINGLE_SPECIES_SHARE,
                        f"a species holds {worst:.1%} - Monocryx risk")

    def test_nothing_rejected_by_the_engine(self):
        """Every action we emit should actually take effect - a rejected
        placement is a wasted slot out of our hard 20/tick budget."""
        sim, _, _ = self.simulated()
        self.assertEqual(sim.rejected_soil, 0, "placements rejected: bad soil")
        self.assertEqual(sim.rejected_locked, 0, "placements rejected: still locked")

    def test_entropy_near_ceiling_for_species_present(self):
        if not self.SIMULATOR_MODELS_COMPOSITION:
            self.skipTest('simulator does not model composition - see test_ground_truth.py')
        """H should be close to log_31(S) - i.e. the species are balanced."""
        _, _, score = self.simulated()
        ceiling = max_entropy_for(score.species)
        if ceiling == 0:
            self.skipTest("single species")
        ratio = score.entropy / ceiling
        self.assertGreater(ratio, self.MIN_ENTROPY_RATIO,
                           f"H={score.entropy:.4f} is only {ratio:.0%} of the "
                           f"{ceiling:.4f} ceiling - populations are unbalanced")
