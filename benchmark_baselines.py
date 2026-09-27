from __future__ import annotations

import csv
import heapq
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_dataset import load_graph
from oracle_search import oracle_search


DATASET_DIR = Path("dataset_full")
RESULTS_DIR = Path("benchmark_results")

ORACLE_MAX_PATHS = 100_000


# ============================================================
# Utility
# ============================================================

def mean(values):
    return statistics.mean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def safe_min(values):
    return min(values) if values else None


def safe_max(values):
    return max(values) if values else None


def format_cost(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_runtime(value):
    return f"{value:.4f}"


# ============================================================
# Graph helpers
# ============================================================

def build_adjacency(graph):
    adjacency = defaultdict(list)

    for edge in graph.edges:
        adjacency[edge.source].append(edge)

    return adjacency


def reconstruct_path(previous, target):
    path = []

    current = target

    while current is not None:
        path.append(current)

        if current not in previous:
            break

        current = previous[current]

    path.reverse()

    if not path or path[0] != "SOURCE":
        return None

    return path


# ============================================================
# Structural Dijkstra
# ============================================================

def structural_dijkstra(graph):
    """
    Shortest SOURCE -> TARGET path without authorization constraints.
    """

    adjacency = build_adjacency(graph)

    source = graph.source_id
    target = graph.target_id

    distances = {
        source: 0.0,
    }

    previous = {
        source: None,
    }

    queue = [
        (0.0, source)
    ]

    visited = set()

    while queue:
        distance, node_id = heapq.heappop(queue)

        if node_id in visited:
            continue

        visited.add(node_id)

        if node_id == target:
            path = reconstruct_path(
                previous,
                target,
            )

            return {
                "found": True,
                "cost": distance,
                "path": path,
                "visited_nodes": len(visited),
            }

        for edge in adjacency.get(node_id, []):
            next_node = edge.target

            new_distance = (
                distance
                + float(edge.cost)
            )

            if new_distance < distances.get(
                next_node,
                float("inf"),
            ):
                distances[next_node] = new_distance

                previous[next_node] = node_id

                heapq.heappush(
                    queue,
                    (
                        new_distance,
                        next_node,
                    ),
                )

    return {
        "found": False,
        "cost": None,
        "path": None,
        "visited_nodes": len(visited),
    }


# ============================================================
# Authorization semantics
# ============================================================

def edge_authorized(graph, edge):
    """
    Independent implementation of the PCGS authorization semantics.

    Every edge must satisfy the policy induced by:
      - department
      - classification
      - source_id
      - relation

    This function intentionally does not use oracle_search().
    """

    source = graph.nodes[edge.source]
    target = graph.nodes[edge.target]

    relation = edge.relation

    # --------------------------------------------------------
    # Explicit deny
    # --------------------------------------------------------

    deny_roles = getattr(
        target,
        "deny_roles",
        None,
    )

    if deny_roles:
        if "employee" in deny_roles:
            return False

    # --------------------------------------------------------
    # Semantic relation
    # --------------------------------------------------------

    if relation == "semantic":
        return (
            source.department
            == target.department
        )

    # --------------------------------------------------------
    # Hierarchical relation
    # --------------------------------------------------------

    if relation == "hierarchical":
        return (
            target.classification
            <= source.classification
        )

    # --------------------------------------------------------
    # Temporal relation
    # --------------------------------------------------------

    if relation == "temporal":
        return (
            target.classification
            <= source.classification
        )

    # --------------------------------------------------------
    # Source relation
    # --------------------------------------------------------

    if relation == "source":
        return (
            source.source_id
            == target.source_id
        )

    # --------------------------------------------------------
    # Reference relation
    # --------------------------------------------------------

    if relation == "reference":
        # Reference edges do not independently grant
        # authorization in the current policy model.
        return False

    return False


def build_authorized_adjacency(graph):
    adjacency = defaultdict(list)

    for edge in graph.edges:
        if edge_authorized(graph, edge):
            adjacency[edge.source].append(edge)

    return adjacency


# ============================================================
# Policy-filtered Dijkstra
# ============================================================

def policy_filtered_dijkstra(graph):
    """
    Filter unauthorized edges first, then run Dijkstra.
    """

    adjacency = build_authorized_adjacency(graph)

    source = graph.source_id
    target = graph.target_id

    distances = {
        source: 0.0,
    }

    previous = {
        source: None,
    }

    queue = [
        (0.0, source)
    ]

    visited = set()

    while queue:
        distance, node_id = heapq.heappop(queue)

        if node_id in visited:
            continue

        visited.add(node_id)

        if node_id == target:
            path = reconstruct_path(
                previous,
                target,
            )

            return {
                "found": True,
                "cost": distance,
                "path": path,
                "visited_nodes": len(visited),
            }

        for edge in adjacency.get(node_id, []):
            next_node = edge.target

            new_distance = (
                distance
                + float(edge.cost)
            )

            if new_distance < distances.get(
                next_node,
                float("inf"),
            ):
                distances[next_node] = new_distance

                previous[next_node] = node_id

                heapq.heappush(
                    queue,
                    (
                        new_distance,
                        next_node,
                    ),
                )

    return {
        "found": False,
        "cost": None,
        "path": None,
        "visited_nodes": len(visited),
    }


# ============================================================
# Oracle
# ============================================================

def oracle_authorized_shortest_path(graph):
    """
    Run the independent exhaustive oracle and select
    the minimum-cost authorized path.
    """

    result = oracle_search(
        graph,
        max_paths=ORACLE_MAX_PATHS,
    )

    if not result.valid_paths:
        return {
            "found": False,
            "cost": None,
            "path": None,
            "structural_path_count": (
                result.structural_path_count
            ),
            "truncated": result.truncated,
        }

    best = min(
        result.valid_paths,
        key=lambda path: path.cost,
    )

    return {
        "found": True,
        "cost": best.cost,
        "path": best.node_ids,
        "visited_nodes": None,
        "structural_path_count": (
            result.structural_path_count
        ),
        "truncated": result.truncated,
    }


# ============================================================
# Single graph benchmark
# ============================================================

def benchmark_graph(
    file_path: Path,
    graph,
):
    scenario = file_path.parent.name
    size = len(graph.nodes)

    # --------------------------------------------------------
    # Structural shortest path
    # --------------------------------------------------------

    start = time.perf_counter()

    structural = structural_dijkstra(graph)

    structural_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Policy-filtered shortest path
    # --------------------------------------------------------

    start = time.perf_counter()

    filtered = policy_filtered_dijkstra(graph)

    filtered_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    start = time.perf_counter()

    oracle = oracle_authorized_shortest_path(
        graph
    )

    oracle_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Cost comparison
    # --------------------------------------------------------

    authorized_cost = oracle["cost"]
    structural_cost = structural["cost"]
    filtered_cost = filtered["cost"]

    structural_gap = None
    structural_ratio = None

    if (
        structural_cost is not None
        and authorized_cost is not None
    ):
        structural_gap = (
            authorized_cost
            - structural_cost
        )

        if authorized_cost > 0:
            structural_ratio = (
                structural_cost
                / authorized_cost
            )

    filtered_gap = None

    if (
        filtered_cost is not None
        and authorized_cost is not None
    ):
        filtered_gap = (
            filtered_cost
            - authorized_cost
        )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    filtered_matches_oracle = (
        filtered["found"]
        == oracle["found"]
    )

    if (
        filtered["found"]
        and oracle["found"]
    ):
        filtered_matches_oracle = (
            abs(
                filtered["cost"]
                - oracle["cost"]
            )
            <= 1e-3
        )

    # Structural path is unauthorized
    # when it exists but differs from authorized
    # shortest-path cost/path.

    structural_is_unauthorized = False

    if structural["found"]:
        if not oracle["found"]:
            structural_is_unauthorized = True
        else:
            structural_is_unauthorized = (
                structural["cost"]
                < oracle["cost"] - 1e-3
            )

    return {
        "file": str(file_path),
        "size": size,
        "scenario": scenario,

        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),

        "structural_found": structural["found"],
        "structural_cost": structural_cost,
        "structural_runtime_ms": (
            structural_runtime * 1000.0
        ),
        "structural_visited_nodes": (
            structural["visited_nodes"]
        ),
        "structural_path": (
            "->".join(structural["path"])
            if structural["path"]
            else ""
        ),

        "filtered_found": filtered["found"],
        "filtered_cost": filtered_cost,
        "filtered_runtime_ms": (
            filtered_runtime * 1000.0
        ),
        "filtered_visited_nodes": (
            filtered["visited_nodes"]
        ),
        "filtered_path": (
            "->".join(filtered["path"])
            if filtered["path"]
            else ""
        ),

        "oracle_found": oracle["found"],
        "oracle_cost": authorized_cost,
        "oracle_runtime_ms": (
            oracle_runtime * 1000.0
        ),
        "oracle_structural_path_count": (
            oracle.get(
                "structural_path_count"
            )
        ),
        "oracle_truncated": (
            oracle.get("truncated", False)
        ),
        "oracle_path": (
            "->".join(oracle["path"])
            if oracle["path"]
            else ""
        ),

        "structural_gap": structural_gap,
        "structural_ratio": structural_ratio,

        "filtered_gap": filtered_gap,

        "structural_is_unauthorized": (
            structural_is_unauthorized
        ),

        "filtered_matches_oracle": (
            filtered_matches_oracle
        ),
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(results):
    by_scenario = defaultdict(list)
    by_size = defaultdict(list)
    by_size_scenario = defaultdict(list)

    for result in results:
        by_scenario[
            result["scenario"]
        ].append(result)

        by_size[
            result["size"]
        ].append(result)

        by_size_scenario[
            (
                result["size"],
                result["scenario"],
            )
        ].append(result)

    return (
        by_scenario,
        by_size,
        by_size_scenario,
    )


def print_group_statistics(
    title,
    groups,
):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'Group':<30}"
        f"{'N':>8}"
        f"{'Struct':>12}"
        f"{'Auth':>12}"
        f"{'Gap':>12}"
        f"{'Filter':>12}"
        f"{'Oracle ms':>12}"
    )

    print("-" * 100)

    for group, rows in sorted(
        groups.items(),
        key=lambda item: str(item[0]),
    ):
        structural_costs = [
            r["structural_cost"]
            for r in rows
            if r["structural_cost"] is not None
        ]

        authorized_costs = [
            r["oracle_cost"]
            for r in rows
            if r["oracle_cost"] is not None
        ]

        gaps = [
            r["structural_gap"]
            for r in rows
            if r["structural_gap"] is not None
        ]

        filtered_costs = [
            r["filtered_cost"]
            for r in rows
            if r["filtered_cost"] is not None
        ]

        oracle_runtimes = [
            r["oracle_runtime_ms"]
            for r in rows
        ]

        label = str(group)

        print(
            f"{label:<30}"
            f"{len(rows):>8}"
            f"{format_cost(mean(structural_costs)):>12}"
            f"{format_cost(mean(authorized_costs)):>12}"
            f"{format_cost(mean(gaps)):>12}"
            f"{format_cost(mean(filtered_costs)):>12}"
            f"{mean(oracle_runtimes):>12.3f}"
        )


# ============================================================
# Main
# ============================================================

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DATASET_DIR.rglob("*.json")
    )

    print("=" * 78)
    print("PCGS BASELINE BENCHMARK")
    print("=" * 78)
    print(f"Dataset:       {DATASET_DIR}")
    print(f"Instances:     {len(files):,}")
    print()

    results = []

    for index, file_path in enumerate(files, start=1):
        graph = load_graph(file_path)

        result = benchmark_graph(
            file_path,
            graph,
        )

        results.append(result)

        if index % 100 == 0:
            print(
                f"Processed {index:,}/{len(files):,}"
            )

    # --------------------------------------------------------
    # Save per-instance CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "benchmark_results.csv"
    )

    fieldnames = list(results[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    (
        by_scenario,
        by_size,
        by_size_scenario,
    ) = aggregate_results(results)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    structural_found = sum(
        r["structural_found"]
        for r in results
    )

    authorized_found = sum(
        r["oracle_found"]
        for r in results
    )

    filtered_matches = sum(
        r["filtered_matches_oracle"]
        for r in results
    )

    unauthorized_structural = sum(
        r["structural_is_unauthorized"]
        for r in results
    )

    structural_gaps = [
        r["structural_gap"]
        for r in results
        if r["structural_gap"] is not None
    ]

    filtered_gaps = [
        r["filtered_gap"]
        for r in results
        if r["filtered_gap"] is not None
    ]

    print()
    print("=" * 78)
    print("OVERALL RESULTS")
    print("=" * 78)

    print(
        f"Instances:                         "
        f"{len(results):,}"
    )

    print(
        f"Structural paths found:            "
        f"{structural_found:,}"
    )

    print(
        f"Authorized paths found:            "
        f"{authorized_found:,}"
    )

    print(
        f"Structural shortest path unauthorized: "
        f"{unauthorized_structural:,}"
    )

    print(
        f"Structural unauthorized rate:       "
        f"{100.0 * unauthorized_structural / len(results):.2f}%"
    )

    print(
        f"Mean structural cost:               "
        f"{format_cost(mean([r['structural_cost'] for r in results if r['structural_cost'] is not None]))}"
    )

    print(
        f"Mean authorized cost:               "
        f"{format_cost(mean([r['oracle_cost'] for r in results if r['oracle_cost'] is not None]))}"
    )

    print(
        f"Mean structural cost gap:           "
        f"{format_cost(mean(structural_gaps))}"
    )

    print(
        f"Mean filtered cost gap:             "
        f"{format_cost(mean(filtered_gaps))}"
    )

    print(
        f"Filter matches oracle:              "
        f"{filtered_matches:,}/{len(results):,} "
        f"({100.0 * filtered_matches / len(results):.2f}%)"
    )

    # --------------------------------------------------------
    # Scenario statistics
    # --------------------------------------------------------

    print_group_statistics(
        "RESULTS BY SCENARIO",
        by_scenario,
    )

    # --------------------------------------------------------
    # Size statistics
    # --------------------------------------------------------

    print_group_statistics(
        "RESULTS BY GRAPH SIZE",
        by_size,
    )

    # --------------------------------------------------------
    # Important scenario analysis
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("STRUCTURAL SHORTEST PATH VS AUTHORIZED SHORTEST PATH")
    print("=" * 100)

    interesting_scenarios = [
        "safe_single_hop",
        "safe_multi_hop",
        "multiple_valid_paths",
        "invalid_cheap_shortcut",
        "explicit_deny",
        "mixed_relation_conflict",
        "no_valid_path",
    ]

    print(
        f"{'Scenario':<30}"
        f"{'Structural':>14}"
        f"{'Authorized':>14}"
        f"{'Gap':>14}"
        f"{'Unauthorized':>14}"
    )

    print("-" * 90)

    for scenario in interesting_scenarios:
        rows = by_scenario.get(
            scenario,
            [],
        )

        structural = [
            r["structural_cost"]
            for r in rows
            if r["structural_cost"] is not None
        ]

        authorized = [
            r["oracle_cost"]
            for r in rows
            if r["oracle_cost"] is not None
        ]

        gaps = [
            r["structural_gap"]
            for r in rows
            if r["structural_gap"] is not None
        ]

        unauthorized = sum(
            r["structural_is_unauthorized"]
            for r in rows
        )

        print(
            f"{scenario:<30}"
            f"{format_cost(mean(structural)):>14}"
            f"{format_cost(mean(authorized)):>14}"
            f"{format_cost(mean(gaps)):>14}"
            f"{unauthorized:>14}"
        )

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {
        "dataset": str(DATASET_DIR),
        "instances": len(results),

        "structural_paths_found": structural_found,
        "authorized_paths_found": authorized_found,

        "structural_unauthorized_count": (
            unauthorized_structural
        ),

        "structural_unauthorized_rate": (
            unauthorized_structural
            / len(results)
        ),

        "mean_structural_cost": mean([
            r["structural_cost"]
            for r in results
            if r["structural_cost"] is not None
        ]),

        "mean_authorized_cost": mean([
            r["oracle_cost"]
            for r in results
            if r["oracle_cost"] is not None
        ]),

        "mean_structural_gap": mean(
            structural_gaps
        ),

        "mean_filtered_gap": mean(
            filtered_gaps
        ),

        "filter_matches_oracle": filtered_matches,
        "filter_match_rate": (
            filtered_matches
            / len(results)
        ),

        "by_scenario": {},
        "by_size": {},
    }

    for scenario, rows in by_scenario.items():
        summary["by_scenario"][scenario] = {
            "instances": len(rows),

            "structural_found": sum(
                r["structural_found"]
                for r in rows
            ),

            "authorized_found": sum(
                r["oracle_found"]
                for r in rows
            ),

            "structural_unauthorized": sum(
                r["structural_is_unauthorized"]
                for r in rows
            ),

            "mean_structural_cost": mean([
                r["structural_cost"]
                for r in rows
                if r["structural_cost"] is not None
            ]),

            "mean_authorized_cost": mean([
                r["oracle_cost"]
                for r in rows
                if r["oracle_cost"] is not None
            ]),

            "mean_structural_gap": mean([
                r["structural_gap"]
                for r in rows
                if r["structural_gap"] is not None
            ]),
        }

    for size, rows in by_size.items():
        summary["by_size"][str(size)] = {
            "instances": len(rows),

            "mean_structural_runtime_ms": mean([
                r["structural_runtime_ms"]
                for r in rows
            ]),

            "mean_filtered_runtime_ms": mean([
                r["filtered_runtime_ms"]
                for r in rows
            ]),

            "mean_oracle_runtime_ms": mean([
                r["oracle_runtime_ms"]
                for r in rows
            ]),

            "mean_structural_visited_nodes": mean([
                r["structural_visited_nodes"]
                for r in rows
            ]),

            "mean_filtered_visited_nodes": mean([
                r["filtered_visited_nodes"]
                for r in rows
            ]),
        }

    summary_path = (
        RESULTS_DIR
        / "benchmark_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)
    print(f"Per-instance results: {csv_path}")
    print(f"Summary:              {summary_path}")


if __name__ == "__main__":
    main()