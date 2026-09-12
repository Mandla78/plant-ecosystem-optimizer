"""Approaches that were built and measured but did not ship.

Kept deliberately: each one rules out a branch of the search space, and
docs/results.txt records what every one of them cost when measured.

  balance.py    iterative proportional fitting of per-species seed weights.
                Made entropy WORSE each round - the final mix is driven by
                spread, not by how many seeds we hand out.
  gated.py      plants counted quotas of the gate-driving species before
                spending anything on the species they unlock. Right idea, and
                it is where the hand-secured-gate logic in cascade.py came
                from, but the standalone planner was beaten by the cascade.
  territory.py  splits the map into blocks and gives each species its own,
                so spread fronts meet at boundaries instead of one species
                sweeping the map. Written last, never fully tuned.
"""
