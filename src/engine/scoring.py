"""The scoring function from the problem statement.

    H      = -SUM p_i * log_N(p_i)          N = 31, ALWAYS (all species in game)
    main   = H * (C / Cmax) ** alpha
    longev = (1 / Cmax) * SUM (l_ij / T) ** k
    final  = 0.8 * main + 0.2 * longev

alpha and k are not published. We default both to 1.0 and expose them so we can
compare strategies under different assumptions - see docs/scoring.txt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import TOTAL_SPECIES

DEFAULT_ALPHA = 1.0   # PROVEN from the L3 evaluation log: main = entropy * density
DEFAULT_K = 1.0


@dataclass(frozen=True)
class ScoreBreakdown:
    entropy: float            # H
    species: int              # distinct species present at the final tick
    population: int           # C
    cmax: int
    coverage: float           # C / Cmax
    size_factor: float        # (C/Cmax) ** alpha
    main: float
    longevity: float
    final: float
    alpha: float
    k: float

    def __str__(self) -> str:
        return (
            f"species={self.species:>3}  H={self.entropy:.4f}  "
            f"C={self.population:>6}/{self.cmax}  cov={self.coverage:6.2%}  "
            f"main={self.main:.4f}  longev={self.longevity:.4f}  FINAL={self.final:.4f}"
        )


def entropy(counts: dict[int, int]) -> float:
    """Shannon entropy in base N=31 over the population proportions."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    log_n = math.log(TOTAL_SPECIES)
    h = 0.0
    for n_i in counts.values():
        if n_i <= 0:
            continue
        p = n_i / total
        h -= p * (math.log(p) / log_n)
    return h


def max_entropy_for(species_count: int) -> float:
    """H when `species_count` species are present in equal proportions."""
    if species_count <= 1:
        return 0.0
    return math.log(species_count) / math.log(TOTAL_SPECIES)


def longevity(lifespans: list[int], cmax: int, ticks: int, k: float = DEFAULT_K) -> float:
    if cmax <= 0 or ticks <= 0:
        return 0.0
    total = 0.0
    for span in lifespans:
        if span > 0:
            total += (span / ticks) ** k
    return total / cmax


def score_world(world, ticks: int, alpha: float = DEFAULT_ALPHA,
                k: float = DEFAULT_K) -> ScoreBreakdown:
    """Score the final state of a World."""
    counts = world.counts()
    population = sum(counts.values())
    cmax = world.spec.cmax

    h = entropy(counts)
    coverage = population / cmax if cmax else 0.0
    size_factor = coverage ** alpha
    main = h * size_factor

    lifespans = [age for p, age in zip(world.plant, world.age) if p]
    lifespans += [age for p, age in zip(world.sub_plant, world.sub_age) if p]
    lon = longevity(lifespans, cmax, ticks, k)

    return ScoreBreakdown(
        entropy=h,
        species=len(counts),
        population=population,
        cmax=cmax,
        coverage=coverage,
        size_factor=size_factor,
        main=main,
        longevity=lon,
        final=0.8 * main + 0.2 * lon,
        alpha=alpha,
        k=k,
    )
