"""Iteratively fit per-species seed weights so the FINAL garden is balanced.

A seed of Grass becomes far more cells than a seed of Rose Bush, so seeding
every species equally hands the map to whatever grows fastest - and entropy,
which is what the score actually pays for, collapses.

This is plain iterative proportional fitting: build, measure what each species
actually ended up with, push the weights of the over-represented species down
and the under-represented ones up, repeat. A handful of rounds is enough.

Deterministic - no randomness - so Rule 6 still holds.
"""

from __future__ import annotations

from engine.data import plants_by_index
from engine.scoring import score_world


def balance_weights(make_planner, spec, rounds: int = 6, alpha: float = 2.0,
                    verbose: bool = False):
    """Return (best_weights, best_plan, best_score) after `rounds` refinements."""
    by_index = plants_by_index()
    weights: dict[str, float] = {}
    best = None

    for r in range(rounds):
        planner = make_planner(weights)
        plan = planner.build()
        world = planner.simulation.world
        score = score_world(world, spec.ticks, alpha=alpha)

        if best is None or score.final > best[2].final:
            best = (dict(weights), plan, score)

        counts = {by_index[i].name: n for i, n in world.counts().items()}
        total = sum(counts.values()) or 1
        live = [n for n in counts if counts[n] > 0]
        if not live:
            break
        target = 1.0 / len(live)

        # push each species toward an equal share; damped to avoid oscillation
        for name in live:
            share = counts[name] / total
            ratio = target / share if share > 0 else 4.0
            ratio = max(0.25, min(4.0, ratio))
            weights[name] = max(0.02, weights.get(name, 1.0) * (ratio ** 0.5))

        if verbose:
            print(f"  round {r}: species={score.species} H={score.entropy:.4f} "
                  f"cov={score.coverage:.1%} -> {score.final * 1e9:,.0f}", flush=True)

    return best
