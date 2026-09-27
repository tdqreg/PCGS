from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from level2_loader import load_level2_graph
from pcgs import PCGS


DATASET_DIR = Path("dataset_level2_small")
OUTPUT_DIR = Path("ablation_results")

CONFIGS = {
    "PCGS-full": {
        "use_dominance": True,
        "use_bound": True,
    },

    "PCGS-no-dominance": {
        "use_dominance": False,
        "use_bound": True,
    },

    "PCGS-no-bound": {
        "use_dominance": True,
        "use_bound": False,
    },

    "State-space": {
        "use_dominance": False,
        "use_bound": False,
    },
}


def run_algorithm(graph, config):

    solver = PCGS(
        use_policy_pruning=True,
        use_dominance=config["use_dominance"],
        use_bound=config["use_bound"],
    )

    return solver.search(graph)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DATASET_DIR.rglob("*.json")
    )

    rows = []

    print("=" * 90)
    print("PCGS LEVEL-2 ABLATION")
    print("=" * 90)

    for config_name, config in CONFIGS.items():

        print()
        print(
            f"Running: {config_name}"
        )

        total = 0
        correct = 0

        runtimes = []
        generated = []
        expanded = []
        policy_pruned = []
        dominance_pruned = []
        bound_pruned = []

        for path in files:

            graph = load_level2_graph(path)

            result = run_algorithm(
                graph,
                config,
            )

            expected = graph.metadata[
                "expected"
            ]

            ok = (
                result.found
                == expected["found"]
            )

            if ok and result.found:

                ok = (
                    abs(
                        result.cost
                        - float(
                            expected["cost"]
                        )
                    )
                    <= 1e-6
                )

            total += 1

            if ok:
                correct += 1

            runtimes.append(
                result.runtime_ms
            )

            generated.append(
                result.generated_states
            )

            expanded.append(
                result.expanded_states
            )

            policy_pruned.append(
                result.policy_pruned
            )

            dominance_pruned.append(
                result.dominance_pruned
            )

            bound_pruned.append(
                result.bound_pruned
            )

            rows.append({
                "instance": str(path),
                "algorithm": config_name,
                "correct": int(ok),
                "found": int(result.found),
                "cost": (
                    result.cost
                    if result.cost is not None
                    else ""
                ),
                "runtime_ms": result.runtime_ms,
                "generated_states":
                    result.generated_states,
                "expanded_states":
                    result.expanded_states,
                "policy_pruned":
                    result.policy_pruned,
                "dominance_pruned":
                    result.dominance_pruned,
                "bound_pruned":
                    result.bound_pruned,
            })

        print(
            f"  Accuracy: "
            f"{correct}/{total} "
            f"({correct / total * 100:.2f}%)"
        )

        print(
            f"  Mean runtime: "
            f"{statistics.mean(runtimes):.4f} ms"
        )

        print(
            f"  Mean generated: "
            f"{statistics.mean(generated):.2f}"
        )

        print(
            f"  Mean expanded: "
            f"{statistics.mean(expanded):.2f}"
        )

        print(
            f"  Mean policy pruned: "
            f"{statistics.mean(policy_pruned):.2f}"
        )

        print(
            f"  Mean dominance pruned: "
            f"{statistics.mean(dominance_pruned):.2f}"
        )

        print(
            f"  Mean bound pruned: "
            f"{statistics.mean(bound_pruned):.2f}"
        )

    output_file = (
        OUTPUT_DIR
        / "ablation_results.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 90)
    print(
        f"Saved: {output_file}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()