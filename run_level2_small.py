# run_level2_small.py

from __future__ import annotations

import json
from pathlib import Path

from level2_loader import load_level2_graph
from pcgs import PCGS


DATASET_DIR = Path(
    "dataset_level2_small"
)


def run():

    solver = PCGS()

    files = sorted(
        DATASET_DIR.rglob("*.json")
    )

    print("=" * 80)
    print("PCGS LEVEL-2 SMALL BENCHMARK")
    print("=" * 80)

    total = 0
    passed = 0

    aggregate = {
        "expanded_states": 0,
        "generated_states": 0,
        "policy_pruned": 0,
        "dominance_pruned": 0,
        "bound_pruned": 0,
        "cycle_pruned": 0,
    }

    for path in files:

        graph = load_level2_graph(path)

        result = solver.search(graph)

        expected = graph.metadata[
            "expected"
        ]

        ok = True

        if result.found != expected["found"]:
            ok = False

        if result.found:
            expected_cost = float(
                expected["cost"]
            )

            if abs(
                result.cost - expected_cost
            ) > 1e-6:
                ok = False

        total += 1

        if ok:
            passed += 1
        else:
            print(
                f"[FAIL] {path}"
            )
            print(
                f"  expected={expected}"
            )
            print(
                f"  actual cost={result.cost}"
            )

        for key in aggregate:
            aggregate[key] += getattr(
                result,
                key,
            )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Instances:              {total}"
    )

    print(
        f"Passed:                 "
        f"{passed}/{total}"
    )

    print(
        f"Accuracy:               "
        f"{passed / total * 100:.2f}%"
    )

    print()
    print(
        f"Generated states:       "
        f"{aggregate['generated_states']}"
    )

    print(
        f"Expanded states:        "
        f"{aggregate['expanded_states']}"
    )

    print(
        f"Policy pruned:          "
        f"{aggregate['policy_pruned']}"
    )

    print(
        f"Dominance pruned:       "
        f"{aggregate['dominance_pruned']}"
    )

    print(
        f"Bound pruned:           "
        f"{aggregate['bound_pruned']}"
    )

    print(
        f"Cycle pruned:           "
        f"{aggregate['cycle_pruned']}"
    )

    print()

    if passed == total:
        print(
            "RESULT: PASS"
        )
    else:
        print(
            "RESULT: FAIL"
        )


if __name__ == "__main__":
    run()