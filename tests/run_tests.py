"""Run the whole suite, or one level.

    python tests/run_tests.py            # everything
    python tests/run_tests.py 1          # level 1 only
    python tests/run_tests.py engine     # engine only
    python tests/run_tests.py -q         # quiet

Standard library only - no pytest required.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main() -> int:
    args = [a for a in sys.argv[1:]]
    verbosity = 1 if "-q" in args else 2
    args = [a for a in args if a != "-q"]

    loader = unittest.TestLoader()
    if not args:
        suite = loader.discover(HERE, pattern="test_*.py")
    else:
        names = []
        for a in args:
            names.append(f"test_level{a}" if a.isdigit() else f"test_{a}")
        suite = unittest.TestSuite(loader.loadTestsFromName(n) for n in names)

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print()
    print(f"ran={result.testsRun}  failures={len(result.failures)}  "
          f"errors={len(result.errors)}  skipped={len(result.skipped)}")
    if result.skipped:
        print("skipped (expected until the matching solver exists):")
        seen = set()
        for test, reason in result.skipped:
            key = reason.split(" - ")[0]
            if key not in seen:
                seen.add(key)
                print(f"  {reason}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
