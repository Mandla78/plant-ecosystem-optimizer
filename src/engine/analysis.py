"""Static analysis of the unlock graph.

Answers "what could this level ever reach, and in what order" WITHOUT running a
simulation, by assuming every coverage/count threshold can eventually be met for
a species we already have. That optimistic fixpoint gives us:

  * the species ceiling per level (used by docs/unlock_tree.txt and the tests)
  * the wave order the solvers plant in

Used by both the solvers and the test suite, so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from .data import (
    STARTING_PLANTS,
    expand_species,
    load_animals,
    load_plants,
    load_unlock_conditions,
)


def _satisfied_animal(node: dict[str, Any], have: frozenset[str]) -> bool:
    kind = node.get("type")
    if kind in ("AND", "OR"):
        results = [_satisfied_animal(c, have) for c in node.get("conditions", ())]
        return all(results) if kind == "AND" else any(results)
    if kind == "coverage":
        species = node["species"]
        species = species if isinstance(species, list) else [species]
        return all(s in have for s in species)
    if kind == "group_coverage":
        return bool(expand_species(node["species_group"]) & have)
    if kind == "count":
        if "species_group" in node:
            return bool(expand_species(node["species_group"]) & have)
        return node["species"] in have
    if kind == "dominance":
        # Monocryx: we can always choose to let a species dominate, but we never
        # want to. Treated as reachable so species_absent checks stay honest.
        return True
    return False


def _satisfied_plant(node: dict[str, Any], have: frozenset[str],
                     animals: frozenset[str], events: frozenset[str]) -> bool:
    if "op" in node:
        op = node["op"]
        if op == "NOT":
            return not _satisfied_plant(node["child"], have, animals, events)
        results = [_satisfied_plant(c, have, animals, events)
                   for c in node.get("children", ())]
        return all(results) if op == "AND" else any(results)

    kind = node["type"]
    if kind in ("coverage", "count"):
        return node["plant"] in have
    if kind == "species_present":
        return node["species"] in animals
    if kind == "species_absent":
        # we deliberately keep populations balanced, so Monocryx stays away
        return True
    if kind == "event":
        return node["event"] in events
    if kind == "feature_count":
        # dead matter and burnt soil are both farmable on demand
        return True
    return False


def available_animals(have: frozenset[str], animals_enabled: bool) -> frozenset[str]:
    if not animals_enabled:
        return frozenset()
    return frozenset(
        a.name for a in load_animals() if _satisfied_animal(a.requirements, have)
    )


def unlock_waves(events: frozenset[str], animals_enabled: bool
                 ) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Successive unlock waves as (newly unlocked plants, animals present)."""
    conditions = load_unlock_conditions()
    all_names = frozenset(p.name for p in load_plants())
    have = frozenset(STARTING_PLANTS)

    waves: list[tuple[frozenset[str], frozenset[str]]] = []
    while True:
        animals = available_animals(have, animals_enabled)
        new = frozenset(
            name for name in all_names - have
            if name in conditions
            and _satisfied_plant(conditions[name], have, animals, events)
        )
        if not new:
            break
        waves.append((new, animals))
        have |= new
    return waves


def reachable_species(events: frozenset[str], animals_enabled: bool) -> frozenset[str]:
    have = set(STARTING_PLANTS)
    for new, _ in unlock_waves(events, animals_enabled):
        have |= new
    return frozenset(have)


def reachable_for_level(spec) -> frozenset[str]:
    events = frozenset(
        c["event"] for c in spec.commands if c["type"] == "event"
    )
    return reachable_species(events, spec.animals_enabled)


def planting_order(spec) -> list[str]:
    """Species in the order they become available - the solvers' build order."""
    events = frozenset(c["event"] for c in spec.commands if c["type"] == "event")
    order = list(STARTING_PLANTS)
    for new, _ in unlock_waves(events, spec.animals_enabled):
        order.extend(sorted(new))
    return order
