================================================================================
 TESTS
================================================================================

RUN
---
    python tests/run_tests.py            all tests
    python tests/run_tests.py 1          level 1 only
    python tests/run_tests.py 2          level 2 only
    python tests/run_tests.py engine     engine only
    python tests/run_tests.py -q         quiet

Standard library unittest only - no pytest, no pip install needed.

LAYOUT
------
    common.py          shared helpers + LevelContract, the rule set EVERY
                       level's solution must satisfy
    test_engine.py     the shared engine: dataset sanity, spread geometry,
                       entropy maths, world defaults, action budget
    test_level1.py     Level 1 (greenhouse)
    test_level2.py     Level 2 (garden)
    test_level3.py     Level 3 (park)
    test_level4.py     Level 4 (forest)

HOW THE LEVEL TESTS ARE STRUCTURED
----------------------------------
Each test_levelN.py has two classes:

  TestLevelN        inherits LevelContract. SKIPS until src/solvers/levelN.py
                    and submissions/levelN/solution.json both exist, then
                    checks the solution against every documented constraint
                    plus that level's own expectations.

  TestLevelNFacts   runs immediately, with or without a solver. These pin the
                    assumptions the strategy is built on - grid size, event
                    schedule, ending season, reachable species count. If the
                    organisers were to change a level file, these fail first.

So a level with no solver yet shows as SKIPPED, not failed, and the suite stays
green while we build.

WHAT LevelContract CHECKS (shared by all four levels)
-----------------------------------------------------
  schema          top-level shape is {"actions": [...]}
                  every plant entry is exactly {plant_index, row, col}
  budget          at most 20 plants per tick (extras would be silently dropped)
                  no tick listed twice, no cell planted twice in one tick
  bounds          ticks in [0, T-1], coordinates inside the grid
  legality        never plants on water / stone / path
                  never plants a species on soil it does not accept
                  never uses a locked species when animals are disabled
  determinism     building twice produces identical output (Rule 6)
                  the committed solution.json matches what the solver produces
                  now - catches a stale file
  outcome         minimum species count, coverage and final score
                  no species at or above 50% (that spawns Monocryx, which
                  hard-blocks the Amber Fern unlock)
                  H is within 15% of log_31(S) - i.e. populations are balanced
                  zero placements rejected by the engine - a rejected action is
                  a wasted slot out of a hard budget

NOTES
-----
* Level 1 asserts against the KNOWN OPTIMUM (H = log_31(5) = 0.4687, all 1,800
  plantable cells, an even 360 per species), not a loose threshold. Anything
  less is a regression.

* The MIN_* thresholds on levels 2-4 are placeholders, set low on purpose.
  Tighten them as each solver improves so they lock in progress.

* test_engine.py pins the STRICT_GROUP_NAMES default. The source data is
  inconsistent - animals.json asks for "Shallow-root Species" while
  classifications.json defines "Shallowroot Species". Strict mode (the default)
  models the pessimistic reading in which Rhizorends never appears and
  Bloodbloom stays locked, giving a 30-species ceiling. The tests assert that
  the two readings differ by exactly Bloodbloom, so the cost of the bug is
  measured rather than assumed.
================================================================================
