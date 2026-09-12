"""Regression guards against the REAL engine, not against our simulator.

`logs/` holds the evaluation logs the official engine returned for submitted
solutions. They print the engine's own statistics - entropy, density, the score
and the exact final plant counts - which makes them the only trustworthy record
of what actually happened.

These tests do two things:

  1. Verify our understanding of the scoring function still reproduces the
     engine's numbers exactly (this is how alpha = 1 was established).
  2. Pin the best score achieved on each level, so a future change that would
     have scored worse is visible.
"""

from __future__ import annotations

import math
import os
import re
import unittest

from common import ROOT

LOG_DIR = os.path.join(ROOT, "logs")

GRID_TO_LEVEL = {(50, 50): 1, (70, 100): 2, (150, 150): 3, (200, 300): 4}

# Best score each level actually achieved, from the logs in this repo.
BEST = {1: 209636426, 2: 291811331, 3: 199661742, 4: 249282871}


def parse_logs():
    out = {}
    if not os.path.isdir(LOG_DIR):
        return out
    for name in os.listdir(LOG_DIR):
        if not name.endswith(".log"):
            continue
        txt = open(os.path.join(LOG_DIR, name), encoding="utf-8", errors="ignore").read()
        grid = re.search(r"loaded level \((\d+)x(\d+), (\d+) ticks\)", txt)
        stats = re.search(r"PlantSim statistics: \{(.*)", txt, re.S)
        if not (grid and stats):
            continue
        blob = stats.group(1)

        def num(key):
            m = re.search(rf"'{key}': ([0-9.eE+-]+)", blob)
            return float(m.group(1)) if m else None

        level = GRID_TO_LEVEL[(int(grid.group(1)), int(grid.group(2)))]
        arr = re.search(r"plant_counts': array\(\[(.*?)\]\)", blob, re.S)
        counts = [float(x) for x in re.findall(r"[0-9.]+(?:e\+?\d+)?", arr.group(1).replace("\n", " "))] if arr else []
        out[level] = {
            "score": num("score"), "main": num("main_score"),
            "longevity": num("longevity_score"), "entropy": num("entropy"),
            "density": num("density_factor"), "C": num("total_plants_planted_C"),
            "cmax": num("c_max_or_grid_size"), "counts": counts,
        }
    return out


class TestScoringModel(unittest.TestCase):
    """Our understanding of the scoring function, checked against the engine."""

    @classmethod
    def setUpClass(cls):
        cls.logs = parse_logs()
        if not cls.logs:
            raise unittest.SkipTest("no evaluation logs in logs/")

    def test_main_score_is_entropy_times_density(self):
        """This is the evidence that ALPHA = 1.

        main = entropy * density^alpha with alpha undisclosed. Calibrating from
        Level 1 alone suggested alpha ~ 2; that was wrong, because it assumed
        our simulated garden matched the real one. Here the engine prints
        entropy, density and main_score itself, and alpha=1 reproduces it.
        """
        for level, s in sorted(self.logs.items()):
            self.assertAlmostEqual(s["entropy"] * s["density"], s["main"], places=9,
                                   msg=f"level {level}: alpha is not 1")

    def test_final_score_is_the_documented_weighted_sum(self):
        for level, s in sorted(self.logs.items()):
            self.assertAlmostEqual(0.8 * s["main"] + 0.2 * s["longevity"],
                                   s["score"], places=9, msg=f"level {level}")

    def test_entropy_matches_the_reported_plant_counts(self):
        """Confirms entropy is Shannon entropy in base 31 over the final counts."""
        for level, s in sorted(self.logs.items()):
            counts = [c for c in s["counts"] if c > 0]
            if not counts:
                continue
            total = sum(counts)
            h = -sum((c / total) * math.log(c / total) / math.log(31) for c in counts)
            self.assertAlmostEqual(h, s["entropy"], places=6, msg=f"level {level}")

    def test_density_is_population_over_grid_size(self):
        for level, s in sorted(self.logs.items()):
            self.assertAlmostEqual(s["C"] / s["cmax"], s["density"], places=9,
                                   msg=f"level {level}")


class TestAchievedScores(unittest.TestCase):
    """Pins what each level actually scored, so a regression is visible."""

    @classmethod
    def setUpClass(cls):
        cls.logs = parse_logs()
        if not cls.logs:
            raise unittest.SkipTest("no evaluation logs in logs/")

    def test_recorded_scores_match_the_logs(self):
        for level, expected in BEST.items():
            if level not in self.logs:
                continue
            self.assertEqual(round(self.logs[level]["score"] * 1e9), expected,
                             f"level {level} score changed")

    def test_density_was_effectively_maxed_out(self):
        """Density was never the bottleneck - entropy was.

        Plantable fractions are 0.72 / 0.88 / 0.81 / 0.90 for levels 1-4, and we
        reached 0.72 / 0.87 / 0.81 / 0.88. Diversity is where the remaining
        score lives.
        """
        plantable = {1: 0.72, 2: 0.876, 3: 0.813, 4: 0.900}
        for level, s in sorted(self.logs.items()):
            self.assertGreater(s["density"], plantable[level] * 0.95,
                               f"level {level} density well under the plantable ceiling")

    def test_entropy_was_the_bottleneck(self):
        """Every level finished far below its entropy ceiling."""
        for level, s in sorted(self.logs.items()):
            species = sum(1 for c in s["counts"] if c > 0)
            ceiling = math.log(species) / math.log(31) if species > 1 else 0.0
            self.assertLess(s["entropy"], ceiling,
                            f"level {level} somehow exceeded its own ceiling")


if __name__ == "__main__":
    unittest.main(verbosity=2)
