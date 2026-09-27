from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DATASET_DIR = Path("dataset_full")


def load_json_files(dataset_dir: Path):
    files = sorted(dataset_dir.rglob("*.json"))

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            yield path, json.load(f)


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    return statistics.median(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def print_section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    records = list(load_json_files(DATASET_DIR))

    if not records:
        print("No JSON files found.")
        return

    # ------------------------------------------------------------------
    # Global containers
    # ------------------------------------------------------------------

    total_nodes = []
    total_edges = []

    path_depths = []
    path_costs = []

    valid_path_counts = []
    invalid_path_counts = []

    structural_edges_by_relation = Counter()
    violations = Counter()
    scenarios = Counter()
    sizes = Counter()

    valid_costs_by_scenario = defaultdict(list)
    invalid_costs_by_scenario = defaultdict(list)

    depth_by_scenario = defaultdict(list)
    cost_by_scenario = defaultdict(list)

    edges_by_size = defaultdict(list)
    nodes_by_size = defaultdict(list)

    files_by_size_scenario = defaultdict(int)

    source_nodes = Counter()
    target_nodes = Counter()

    metadata_keys = Counter()

    # ------------------------------------------------------------------
    # Read dataset
    # ------------------------------------------------------------------

    for file_path, graph in records:
        size = len(graph["nodes"])

        # Prefer directory scenario, fallback to metadata.
        scenario = file_path.parent.name

        if not scenario:
            scenario = graph.get("metadata", {}).get(
                "scenario",
                "unknown",
            )

        sizes[size] += 1
        scenarios[scenario] += 1

        files_by_size_scenario[(size, scenario)] += 1

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        total_nodes.append(len(nodes))
        total_edges.append(len(edges))

        nodes_by_size[size].append(len(nodes))
        edges_by_size[size].append(len(edges))

        source_nodes[graph.get("source_id")] += 1
        target_nodes[graph.get("target_id")] += 1

        # Metadata
        metadata = graph.get("metadata", {})

        for key in metadata:
            metadata_keys[key] += 1

        # Edge relations
        for edge in edges:
            relation = edge.get("relation", "unknown")
            structural_edges_by_relation[relation] += 1

        # Valid paths
        valid_paths = graph.get("valid_paths", [])

        valid_path_counts.append(len(valid_paths))

        for path in valid_paths:
            cost = float(path["expected_cost"])

            path_costs.append(cost)
            path_depths.append(
                max(0, len(path.get("node_ids", [])) - 1)
            )

            valid_costs_by_scenario[scenario].append(cost)
            depth_by_scenario[scenario].append(
                max(0, len(path.get("node_ids", [])) - 1)
            )
            cost_by_scenario[scenario].append(cost)

        # Invalid paths
        invalid_paths = graph.get("invalid_paths", [])

        invalid_path_counts.append(len(invalid_paths))

        for path in invalid_paths:
            cost = float(path["expected_cost"])

            invalid_costs_by_scenario[scenario].append(cost)

            violation = path.get("violation")

            if violation is None:
                violation = "none"

            violations[violation] += 1

    # ------------------------------------------------------------------
    # Dataset overview
    # ------------------------------------------------------------------

    print_section("PCGS DATASET STATISTICS")

    print(f"Dataset directory:       {DATASET_DIR}")
    print(f"Total JSON instances:    {len(records):,}")

    print(
        f"Total nodes:             {sum(total_nodes):,}"
    )

    print(
        f"Total edges:             {sum(total_edges):,}"
    )

    print(
        f"Average nodes / graph:   {safe_mean(total_nodes):.2f}"
    )

    print(
        f"Average edges / graph:   {safe_mean(total_edges):.2f}"
    )

    print(
        f"Average valid paths:     {safe_mean(valid_path_counts):.2f}"
    )

    print(
        f"Average invalid paths:   {safe_mean(invalid_path_counts):.2f}"
    )

    # ------------------------------------------------------------------
    # Distribution by size
    # ------------------------------------------------------------------

    print_section("DISTRIBUTION BY GRAPH SIZE")

    print(
        f"{'Size':>8} "
        f"{'Graphs':>10} "
        f"{'Avg Nodes':>12} "
        f"{'Avg Edges':>12}"
    )

    print("-" * 48)

    for size in sorted(sizes):
        print(
            f"{size:>8} "
            f"{sizes[size]:>10,} "
            f"{safe_mean(nodes_by_size[size]):>12.2f} "
            f"{safe_mean(edges_by_size[size]):>12.2f}"
        )

    # ------------------------------------------------------------------
    # Distribution by scenario
    # ------------------------------------------------------------------

    print_section("DISTRIBUTION BY SCENARIO")

    print(
        f"{'Scenario':<30} "
        f"{'Graphs':>10} "
        f"{'Valid':>10} "
        f"{'Invalid':>10}"
    )

    print("-" * 64)

    for scenario in sorted(scenarios):
        scenario_valid = sum(
            len(
                graph.get("valid_paths", [])
            )
            for path, graph in records
            if path.parent.name == scenario
        )

        scenario_invalid = sum(
            len(
                graph.get("invalid_paths", [])
            )
            for path, graph in records
            if path.parent.name == scenario
        )

        print(
            f"{scenario:<30} "
            f"{scenarios[scenario]:>10,} "
            f"{scenario_valid:>10,} "
            f"{scenario_invalid:>10,}"
        )

    # ------------------------------------------------------------------
    # Path statistics
    # ------------------------------------------------------------------

    print_section("PATH STATISTICS")

    print(
        f"All declared valid paths: {len(path_costs):,}"
    )

    print(
        f"Valid path cost:"
    )

    print(
        f"  Min:      {safe_min(path_costs):.4f}"
    )

    print(
        f"  Mean:     {safe_mean(path_costs):.4f}"
    )

    print(
        f"  Median:   {safe_median(path_costs):.4f}"
    )

    print(
        f"  Max:      {safe_max(path_costs):.4f}"
    )

    print()

    print(
        f"Valid path depth:"
    )

    print(
        f"  Min:      {safe_min(path_depths):.0f}"
    )

    print(
        f"  Mean:     {safe_mean(path_depths):.2f}"
    )

    print(
        f"  Median:   {safe_median(path_depths):.0f}"
    )

    print(
        f"  Max:      {safe_max(path_depths):.0f}"
    )

    # ------------------------------------------------------------------
    # Valid path statistics by scenario
    # ------------------------------------------------------------------

    print_section("VALID PATH COST BY SCENARIO")

    print(
        f"{'Scenario':<30} "
        f"{'Paths':>10} "
        f"{'Min':>10} "
        f"{'Mean':>10} "
        f"{'Max':>10}"
    )

    print("-" * 76)

    for scenario in sorted(valid_costs_by_scenario):
        values = valid_costs_by_scenario[scenario]

        print(
            f"{scenario:<30} "
            f"{len(values):>10,} "
            f"{safe_min(values):>10.3f} "
            f"{safe_mean(values):>10.3f} "
            f"{safe_max(values):>10.3f}"
        )

    # ------------------------------------------------------------------
    # Invalid path statistics
    # ------------------------------------------------------------------

    print_section("INVALID PATH STATISTICS")

    total_invalid = sum(
        len(values)
        for values in invalid_costs_by_scenario.values()
    )

    print(
        f"All declared invalid paths: {total_invalid:,}"
    )

    if total_invalid:
        all_invalid_costs = [
            cost
            for values in invalid_costs_by_scenario.values()
            for cost in values
        ]

        print(
            f"Invalid path cost:"
        )

        print(
            f"  Min:      {safe_min(all_invalid_costs):.4f}"
        )

        print(
            f"  Mean:     {safe_mean(all_invalid_costs):.4f}"
        )

        print(
            f"  Median:   {safe_median(all_invalid_costs):.4f}"
        )

        print(
            f"  Max:      {safe_max(all_invalid_costs):.4f}"
        )

    # ------------------------------------------------------------------
    # Violation distribution
    # ------------------------------------------------------------------

    print_section("POLICY VIOLATION DISTRIBUTION")

    total_violations = sum(violations.values())

    print(
        f"{'Violation':<30} "
        f"{'Count':>10} "
        f"{'Percentage':>14}"
    )

    print("-" * 58)

    for violation, count in sorted(
        violations.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        percentage = (
            100.0 * count / total_violations
            if total_violations
            else 0.0
        )

        print(
            f"{violation:<30} "
            f"{count:>10,} "
            f"{percentage:>13.2f}%"
        )

    # ------------------------------------------------------------------
    # Edge relation distribution
    # ------------------------------------------------------------------

    print_section("EDGE RELATION DISTRIBUTION")

    total_edges_count = sum(
        structural_edges_by_relation.values()
    )

    print(
        f"{'Relation':<30} "
        f"{'Count':>10} "
        f"{'Percentage':>14}"
    )

    print("-" * 58)

    for relation, count in sorted(
        structural_edges_by_relation.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        percentage = (
            100.0 * count / total_edges_count
            if total_edges_count
            else 0.0
        )

        print(
            f"{relation:<30} "
            f"{count:>10,} "
            f"{percentage:>13.2f}%"
        )

    # ------------------------------------------------------------------
    # Scenario x size matrix
    # ------------------------------------------------------------------

    print_section("SCENARIO × SIZE MATRIX")

    scenario_names = sorted(scenarios)
    size_values = sorted(sizes)

    header = f"{'Scenario':<30}"

    for size in size_values:
        header += f"{size:>10}"

    print(header)
    print("-" * (30 + 10 * len(size_values)))

    for scenario in scenario_names:
        row = f"{scenario:<30}"

        for size in size_values:
            count = files_by_size_scenario.get(
                (size, scenario),
                0,
            )

            row += f"{count:>10}"

        print(row)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    print_section("METADATA KEYS")

    for key, count in sorted(
        metadata_keys.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"{key:<40} {count:>10,}"
        )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    print_section("SANITY CHECKS")

    expected_instances = (
        len(sizes)
        * len(scenarios)
        * 100
    )

    print(
        f"Expected instances:     {expected_instances:,}"
    )

    print(
        f"Actual instances:       {len(records):,}"
    )

    print(
        f"Instance count matches: "
        f"{len(records) == expected_instances}"
    )

    print(
        f"All node counts valid:   "
        f"{all(n > 0 for n in total_nodes)}"
    )

    print(
        f"All edge counts valid:   "
        f"{all(e >= 0 for e in total_edges)}"
    )

    print(
        f"All valid costs > 0:     "
        f"{all(c > 0 for c in path_costs)}"
    )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STATISTICS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()