"""Loaders for the static challenge data (plants, animals, groups, unlocks).

Everything here is read-only and cached. No randomness: Rule 6 requires the
solution to be deterministic and reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root() -> str:
    """Locate the directory holding problem-statement/.

    In the repo this module lives at src/engine/data.py, but in the submission
    ZIP it is flattened to engine/data.py - so we walk upwards instead of
    assuming a fixed depth.
    """
    candidate = _HERE
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "problem-statement")):
            return candidate
        candidate = os.path.abspath(os.path.join(candidate, ".."))
    return os.path.abspath(os.path.join(_HERE, "..", ".."))


_ROOT = _find_root()

RESOURCE_DIR = os.path.join(_ROOT, "problem-statement", "additional-resources")
LEVEL_DIR = os.path.join(_ROOT, "problem-statement", "levels")

# ---------------------------------------------------------------------------
# constants pulled out of the spec
# ---------------------------------------------------------------------------

TOTAL_SPECIES = 31          # N in the entropy formula - fixed, NOT per level
MAX_PLANTS_PER_TICK = 20
START_NUTRIENTS = 100

SOIL_DIRT, SOIL_MUD, SOIL_CLAY, SOIL_BURNT = 0, 1, 2, 3

# Terrain ids are undocumented; these were decoded from the level files.
# See docs/levels.txt for the evidence.
TERRAIN_GROUND = 0
TERRAIN_WATER = 1
TERRAIN_STONE = 2
TERRAIN_PATH = 4
TERRAIN_CRACK = 3           # never present at tick 0; created by Earthquake

PASSABLE_TERRAIN = frozenset({TERRAIN_GROUND})

STARTING_PLANTS = ("Grass", "Rose Bush", "Dwarf Sunflower", "Lavender", "Oak Tree")


# ---------------------------------------------------------------------------
# plants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Plant:
    name: str
    index: int
    time_to_maturity: int
    spread_rate: int
    spread_mechanism: str
    spread_type: str
    spread_range: int
    root_type: str
    invasiveness_rank: int
    preferred_soil: frozenset[int]
    weaknesses: dict[str, dict[str, Any]]
    special: dict[str, dict[str, Any]]
    conditional_modifiers: tuple[dict[str, Any], ...] = ()
    role: str = ""

    def has_weakness(self, name: str) -> bool:
        return name in self.weaknesses

    def has_special(self, name: str) -> bool:
        return name in self.special

    def special_value(self, name: str, default: Any = None) -> Any:
        rule = self.special.get(name)
        return default if rule is None else rule.get("value", default)

    def weakness_value(self, name: str, default: Any = None) -> Any:
        rule = self.weaknesses.get(name)
        return default if rule is None else rule.get("value", default)

    def spread_rate_for(self, season: str | None) -> int:
        """Apply conditional_modifiers. Only season_* conditions exist today."""
        rate = self.spread_rate
        if season:
            want = f"season_{season.lower()}"
            for mod in self.conditional_modifiers:
                if mod.get("condition") == want and "spread_rate" in mod:
                    rate = mod["spread_rate"]
        return rate

    def can_root_in(self, soil: int) -> bool:
        return soil in self.preferred_soil


def _rules_to_map(rules: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rule in rules or ():
        out[rule["type"]] = {k: v for k, v in rule.items() if k != "type"}
    return out


@lru_cache(maxsize=1)
def load_plants() -> tuple[Plant, ...]:
    with open(os.path.join(RESOURCE_DIR, "plant_dataset.json"), encoding="utf-8") as fh:
        raw = json.load(fh)

    plants = []
    for entry in raw:
        growth = entry["growth"]
        rules = entry.get("rules", {})
        plants.append(
            Plant(
                name=entry["plant"],
                index=entry["index"],
                time_to_maturity=growth["time_to_maturity"],
                spread_rate=growth["spread_rate"],
                spread_mechanism=growth["spread_mechanism"],
                spread_type=growth["spread_type"],
                spread_range=growth["spread_range"],
                root_type=growth["root_type"],
                invasiveness_rank=growth["invasiveness_rank"],
                preferred_soil=frozenset(entry["preferred_soil"]),
                weaknesses=_rules_to_map(rules.get("weaknesses")),
                special=_rules_to_map(rules.get("special")),
                conditional_modifiers=tuple(growth.get("conditional_modifiers", ())),
                role=entry.get("role", ""),
            )
        )
    plants.sort(key=lambda p: p.index)
    return tuple(plants)


@lru_cache(maxsize=1)
def plants_by_name() -> dict[str, Plant]:
    return {p.name: p for p in load_plants()}


@lru_cache(maxsize=1)
def plants_by_index() -> dict[int, Plant]:
    return {p.index: p for p in load_plants()}


def plant(name_or_index: str | int) -> Plant:
    if isinstance(name_or_index, int):
        return plants_by_index()[name_or_index]
    return plants_by_name()[name_or_index]


# ---------------------------------------------------------------------------
# classifications (species groups)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_classifications() -> dict[str, tuple[str, ...]]:
    path = os.path.join(RESOURCE_DIR, "classifications.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: tuple(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def _normalised_groups() -> dict[str, tuple[str, ...]]:
    """Group lookup tolerant of the hyphen inconsistency in the source data.

    animals.json asks for "Shallow-root Species"; classifications.json defines
    "Shallowroot Species". We index both spellings so our own reasoning is not
    derailed by it. NOTE: the real engine may NOT do this, which is why
    Bloodbloom is treated as unreachable in docs/unlock_tree.txt.
    """
    out: dict[str, tuple[str, ...]] = {}
    for key, members in load_classifications().items():
        out[key] = members
        out[key.lower()] = members
        out[key.replace("-", "").lower()] = members
        out[key.replace(" ", "").replace("-", "").lower()] = members
    return out


# The source data is inconsistent: animals.json asks for "Shallow-root Species"
# but classifications.json defines "Shallowroot Species". We do NOT know whether
# the official engine bridges that gap.
#   STRICT (default): match group names literally, so the misspelled reference
#     resolves to nothing, Rhizorends never appears and Bloodbloom stays locked.
#     This is the conservative assumption our plan is built on (30 species).
#   TOLERANT: normalise the spelling, which makes Bloodbloom reachable (31).
# Flip this to explore the optimistic case; see docs/unlock_tree.txt.
STRICT_GROUP_NAMES = True


def resolve_group(name: str) -> tuple[str, ...]:
    """Expand a group name to member species; a plain species name maps to itself."""
    if STRICT_GROUP_NAMES:
        groups = load_classifications()
        return groups.get(name, (name,))

    groups = _normalised_groups()
    for key in (name, name.lower(), name.replace("-", "").lower(),
                name.replace(" ", "").replace("-", "").lower()):
        if key in groups:
            return groups[key]
    return (name,)


def expand_species(names: list[str] | tuple[str, ...] | str) -> frozenset[str]:
    """Expand a list that may mix species names and group names."""
    if isinstance(names, str):
        names = [names]
    out: set[str] = set()
    for name in names:
        out.update(resolve_group(name))
    return frozenset(out)


# ---------------------------------------------------------------------------
# unlock conditions
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_unlock_conditions() -> dict[str, dict[str, Any]]:
    path = os.path.join(RESOURCE_DIR, "plant_unlock_conditions.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {entry["plant"]: entry["unlock"] for entry in raw}


# ---------------------------------------------------------------------------
# animals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Animal:
    id: str
    name: str
    requirements: dict[str, Any]
    effects: tuple[dict[str, Any], ...] = field(default=())


@lru_cache(maxsize=1)
def load_animals() -> tuple[Animal, ...]:
    with open(os.path.join(RESOURCE_DIR, "animals.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    return tuple(
        Animal(
            id=entry["id"],
            name=entry["name"],
            requirements=entry["requirements"],
            effects=tuple(entry.get("effects", ())),
        )
        for entry in raw
    )


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelSpec:
    number: int
    rows: int
    cols: int
    ticks: int
    animals_enabled: bool
    cells: tuple[dict[str, int], ...]
    commands: tuple[dict[str, Any], ...]

    @property
    def cmax(self) -> int:
        return self.rows * self.cols

    def season_at(self, tick: int) -> str:
        season = "Spring"          # levels start in spring; first change is at 100
        for cmd in self.commands:
            if cmd["type"] == "season" and cmd["tick"] <= tick:
                season = cmd["season"]
        return season

    def events_by_tick(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for cmd in self.commands:
            if cmd["type"] == "event":
                out.setdefault(cmd["tick"], []).append(cmd["event"])
        return out

    def events_before(self, tick: int) -> frozenset[str]:
        """Events that have fired at or before `tick` (they latch permanently)."""
        return frozenset(
            cmd["event"]
            for cmd in self.commands
            if cmd["type"] == "event" and cmd["tick"] <= tick
        )


@lru_cache(maxsize=8)
def load_level(number: int) -> LevelSpec:
    with open(os.path.join(LEVEL_DIR, f"{number}.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    return LevelSpec(
        number=number,
        rows=raw["rows"],
        cols=raw["cols"],
        ticks=raw["ticks"],
        animals_enabled=raw["animals_enabled"],
        cells=tuple(raw["cells"]),
        commands=tuple(raw["commands"]),
    )
