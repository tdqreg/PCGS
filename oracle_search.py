# oracle_search.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from models import Graph, Edge, PathRecord


# ============================================================
# Configuration
# ============================================================

TOLERANCE = 1e-6


# ============================================================
# Result types
# ============================================================

@dataclass(frozen=True)
class OraclePath:
    """
    A path discovered independently by the oracle.

    This is deliberately independent of PathRecord.
    """

    node_ids: Tuple[str, ...]
    edge_ids: Tuple[str, ...]
    cost: float

    @property
    def depth(self) -> int:
        return len(self.edge_ids)


@dataclass
class OracleResult:

    valid_paths: List[OraclePath]
    violating_paths: List[OraclePath]

    # Number of SOURCE -> TARGET structural paths considered.
    structural_path_count: int

    # True when enumeration was stopped because max_paths
    # was reached.
    truncated: bool

    @property
    def valid_path_count(self) -> int:
        return len(self.valid_paths)

    @property
    def invalid_path_count(self) -> int:
        return len(self.violating_paths)


# ============================================================
# Policy Oracle
# ============================================================

class PCGSOracle:
    """
    Independent ground-truth oracle for the PCGS benchmark.

    The oracle does NOT use:
        graph.valid_paths
        graph.invalid_paths
        PathRecord.expected_valid
        PathRecord.violation

    Instead, it examines the actual graph and evaluates every
    reachable SOURCE -> TARGET path according to the benchmark
    policy semantics.
    """

    VALID_RELATIONS = {
        "semantic",
        "reference",
        "temporal",
        "hierarchical",
        "source",
    }

    def __init__(
        self,
        graph: Graph,
        max_paths: int = 100_000,
    ):

        self.graph = graph
        self.max_paths = max_paths

        self.edge_map: Dict[str, Edge] = (
            graph.edge_map
        )

        self.adjacency: Dict[str, List[Edge]] = {}

        for edge in graph.edges:

            self.adjacency.setdefault(
                edge.source,
                [],
            ).append(edge)

        # ----------------------------------------------------
        # Only nodes that can structurally reach TARGET are
        # relevant to SOURCE -> TARGET search.
        # ----------------------------------------------------

        self.reverse_adjacency: Dict[
            str, List[str]
        ] = {}

        for edge in graph.edges:

            self.reverse_adjacency.setdefault(
                edge.target,
                [],
            ).append(edge.source)

        self.target_reachable = (
            self._compute_target_reachable()
        )

    # ========================================================
    # Reverse reachability
    # ========================================================

    def _compute_target_reachable(self) -> Set[str]:

        reachable: Set[str] = {
            self.graph.target_id
        }

        stack = [
            self.graph.target_id
        ]

        while stack:

            current = stack.pop()

            for previous in self.reverse_adjacency.get(
                current,
                [],
            ):

                if previous in reachable:
                    continue

                reachable.add(previous)
                stack.append(previous)

        return reachable

    # ========================================================
    # Policy checking
    # ========================================================

    def check_edge_policy(
        self,
        edge: Edge,
    ) -> List[str]:
        """
        Check the policy transition represented by one edge.

        The returned list contains violation identifiers.

        IMPORTANT:
        This function uses only graph node/edge attributes.
        It does not inspect any generated PathRecord.
        """

        violations: List[str] = []

        source_node = self.graph.nodes.get(
            edge.source
        )

        target_node = self.graph.nodes.get(
            edge.target
        )

        if source_node is None:

            violations.append(
                f"missing_node:{edge.source}"
            )

            return violations

        if target_node is None:

            violations.append(
                f"missing_node:{edge.target}"
            )

            return violations

        # ----------------------------------------------------
        # Unknown relation
        # ----------------------------------------------------

        if edge.relation not in self.VALID_RELATIONS:

            violations.append(
                f"unknown_relation:{edge.relation}"
            )

            return violations

        # ----------------------------------------------------
        # Hierarchical propagation
        # ----------------------------------------------------

        if edge.relation == "hierarchical":

            if (
                target_node.classification
                > source_node.classification
            ):

                violations.append(
                    "hierarchical_classification_violation"
                )

        # ----------------------------------------------------
        # Semantic propagation
        # ----------------------------------------------------

        elif edge.relation == "semantic":

            if (
                source_node.department
                != target_node.department
            ):

                violations.append(
                    "cross_department_violation"
                )

        # ----------------------------------------------------
        # Source propagation
        # ----------------------------------------------------

        elif edge.relation == "source":

            if (
                source_node.source_id
                != target_node.source_id
            ):

                violations.append(
                    "source_mismatch"
                )

        # ----------------------------------------------------
        # Temporal propagation
        # ----------------------------------------------------

        elif edge.relation == "temporal":

            if (
                target_node.classification
                > source_node.classification
            ):

                violations.append(
                    "temporal_classification_violation"
                )

        return violations

    # ========================================================
    # Target authorization
    # ========================================================

    def check_target_policy(self) -> List[str]:

        target = self.graph.nodes.get(
            self.graph.target_id
        )

        if target is None:

            return ["missing_target"]

        violations = []

        # This matches the benchmark's explicit-deny
        # semantics.
        if "employee" in target.deny_roles:

            violations.append(
                "explicit_deny"
            )

        return violations

    # ========================================================
    # Path policy
    # ========================================================

    def check_path_policy(
        self,
        edge_ids: List[str],
    ) -> List[str]:

        violations: List[str] = []

        # ----------------------------------------------------
        # Edge violations
        # ----------------------------------------------------

        for edge_id in edge_ids:

            edge = self.edge_map.get(edge_id)

            if edge is None:

                violations.append(
                    f"unknown_edge:{edge_id}"
                )

                continue

            violations.extend(
                self.check_edge_policy(edge)
            )

        # ----------------------------------------------------
        # Target-level policy
        # ----------------------------------------------------

        violations.extend(
            self.check_target_policy()
        )

        return sorted(
            set(violations)
        )

    # ========================================================
    # Structural DFS
    # ========================================================

    def _enumerate_paths(
        self,
        max_paths: int,
    ):

        source = self.graph.source_id
        target = self.graph.target_id

        if source not in self.graph.nodes:
            return

        if target not in self.graph.nodes:
            return

        # If SOURCE cannot reach TARGET, no path exists.
        if source not in self.target_reachable:
            return

        path_nodes: List[str] = [
            source
        ]

        path_edges: List[str] = []

        visited: Set[str] = {
            source
        }

        yielded = 0

        def dfs(
            current: str,
            cost: float,
        ):

            nonlocal yielded

            if yielded >= max_paths:
                return

            # ------------------------------------------------
            # Target reached
            # ------------------------------------------------

            if current == target:

                yielded += 1

                yield (
                    tuple(path_nodes),
                    tuple(path_edges),
                    cost,
                )

                return

            # ------------------------------------------------
            # Traverse outgoing edges
            # ------------------------------------------------

            for edge in self.adjacency.get(
                current,
                [],
            ):

                if yielded >= max_paths:
                    return

                next_node = edge.target

                # Ignore edges that cannot eventually reach
                # TARGET.
                if next_node not in self.target_reachable:
                    continue

                # Simple-path restriction.
                if next_node in visited:
                    continue

                visited.add(next_node)
                path_nodes.append(next_node)
                path_edges.append(edge.edge_id)

                yield from dfs(
                    next_node,
                    cost + edge.cost,
                )

                path_edges.pop()
                path_nodes.pop()
                visited.remove(next_node)

        yield from dfs(
            source,
            0.0,
        )

    # ========================================================
    # Public search
    # ========================================================

    def search(self) -> OracleResult:

        valid_paths: List[OraclePath] = []
        violating_paths: List[OraclePath] = []

        structural_path_count = 0
        truncated = False

        for (
            node_ids,
            edge_ids,
            cost,
        ) in self._enumerate_paths(
            self.max_paths
        ):

            structural_path_count += 1

            path = OraclePath(
                node_ids=node_ids,
                edge_ids=edge_ids,
                cost=round(cost, 4),
            )

            violations = self.check_path_policy(
                list(edge_ids)
            )

            if violations:

                violating_paths.append(path)

            else:

                valid_paths.append(path)

            if (
                structural_path_count
                >= self.max_paths
            ):

                truncated = True
                break

        return OracleResult(
            valid_paths=valid_paths,
            violating_paths=violating_paths,
            structural_path_count=structural_path_count,
            truncated=truncated,
        )


# ============================================================
# Convenience function
# ============================================================

def oracle_search(
    graph: Graph,
    max_paths: int = 100_000,
) -> OracleResult:

    oracle = PCGSOracle(
        graph=graph,
        max_paths=max_paths,
    )

    return oracle.search()


# ============================================================
# Comparison helpers
# ============================================================

def path_signature(
    path: OraclePath,
):
    return (
        tuple(path.node_ids),
        tuple(path.edge_ids),
    )


def declared_path_signature(
    path: PathRecord,
):
    return (
        tuple(path.node_ids),
        tuple(path.edge_ids),
    )


def compare_declared_with_oracle(
    graph: Graph,
    oracle_result: OracleResult,
) -> dict:
    """
    Compare generator-declared paths against the actual
    oracle solution set.

    This function is useful for dataset auditing.
    """

    declared_valid = {
        declared_path_signature(path)
        for path in graph.valid_paths
    }

    declared_invalid = {
        declared_path_signature(path)
        for path in graph.invalid_paths
    }

    oracle_valid = {
        path_signature(path)
        for path in oracle_result.valid_paths
    }

    oracle_invalid = {
        path_signature(path)
        for path in oracle_result.violating_paths
    }

    return {
        "declared_valid_count": len(
            declared_valid
        ),

        "oracle_valid_count": len(
            oracle_valid
        ),

        "declared_invalid_count": len(
            declared_invalid
        ),

        "oracle_invalid_count": len(
            oracle_invalid
        ),

        "missing_declared_valid": len(
            oracle_valid - declared_valid
        ),

        "undeclared_valid_paths": len(
            declared_valid - oracle_valid
        ),

        "missing_declared_invalid": len(
            oracle_invalid - declared_invalid
        ),

        "undeclared_invalid_paths": len(
            declared_invalid - oracle_invalid
        ),

        "exact_valid_match": (
            declared_valid == oracle_valid
        ),

        "exact_invalid_match": (
            declared_invalid == oracle_invalid
        ),
    }


# ============================================================
# Human-readable summary
# ============================================================

def print_oracle_result(
    result: OracleResult,
) -> None:

    print(
        "Oracle search result:"
    )

    print(
        f"  Structural SOURCE→TARGET paths: "
        f"{result.structural_path_count}"
    )

    print(
        f"  Policy-valid paths: "
        f"{result.valid_path_count}"
    )

    print(
        f"  Policy-violating paths: "
        f"{result.invalid_path_count}"
    )

    print(
        f"  Truncated: "
        f"{result.truncated}"
    )

    if result.valid_paths:

        print()
        print(
            "Valid paths:"
        )

        for index, path in enumerate(
            result.valid_paths,
            start=1,
        ):

            print(
                f"  [{index}] "
                f"cost={path.cost:.4f}, "
                f"depth={path.depth}, "
                f"path={' -> '.join(path.node_ids)}"
            )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    from audit_dataset import load_graph

    parser = argparse.ArgumentParser(
        description=(
            "Independent PCGS ground-truth "
            "oracle search."
        )
    )

    parser.add_argument(
        "graph",
        type=Path,
        help="Path to a graph JSON file.",
    )

    parser.add_argument(
        "--max-paths",
        type=int,
        default=100_000,
        help=(
            "Maximum number of structural paths "
            "to enumerate."
        ),
    )

    args = parser.parse_args()

    graph = load_graph(args.graph)

    result = oracle_search(
        graph,
        max_paths=args.max_paths,
    )

    print_oracle_result(
        result
    )

    print()

    comparison = compare_declared_with_oracle(
        graph,
        result,
    )

    print(
        "Declared vs oracle:"
    )

    for key, value in comparison.items():

        print(
            f"  {key}: {value}"
        )