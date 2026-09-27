from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from models import Graph, PathRecord
from scenario_specs import SizeProfile, ScenarioSpec


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def merge(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)

    @property
    def error_count(self) -> int:
        return len(self.errors)


class GraphValidator:
    """
    Structural and semantic validator for generated PCGS graphs.

    The validator is used during generation to reject invalid
    synthetic instances. It is not the final policy oracle.
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
        profile: SizeProfile,
        scenario: ScenarioSpec,
    ):
        self.profile = profile
        self.scenario = scenario

    # =========================================================
    # Public entry point
    # =========================================================

    def validate(self, graph: Graph) -> ValidationResult:
        """
        Run every validation layer.
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        validators = [
            self.validate_exact_node_count,
            self.validate_structure,
            self.validate_path_policy,
            self.validate_cost_order,
            self.validate_edge_density,
            self.validate_scenario,
        ]

        for validator in validators:
            current_result = validator(graph)
            result.merge(current_result)

        result.valid = len(result.errors) == 0

        return result

    # =========================================================
    # 1. Exact node count
    # =========================================================

    def validate_exact_node_count(
        self,
        graph: Graph,
    ) -> ValidationResult:
        """
        Verify that the graph has exactly the requested number
        of nodes.
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        expected_count = self.profile.num_nodes
        actual_count = len(graph.nodes)

        if actual_count != expected_count:
            result.add_error(
                "Invalid node count: "
                f"expected={expected_count}, "
                f"actual={actual_count}"
            )

        required_nodes = {
            graph.source_id,
            graph.target_id,
        }

        missing_nodes = required_nodes - set(graph.nodes.keys())

        if missing_nodes:
            result.add_error(
                "Missing required nodes: "
                f"{sorted(missing_nodes)}"
            )

        if graph.source_id == graph.target_id:
            result.add_error(
                "Source and target must be different."
            )

        return result

    # =========================================================
    # 2. Structural validation
    # =========================================================

    def validate_structure(
        self,
        graph: Graph,
    ) -> ValidationResult:
        """
        Validate graph structure:

        - Unique node IDs
        - Unique edge IDs
        - Existing edge endpoints
        - No self-loops
        - Positive edge costs
        - Valid relation types
        - Path continuity
        - Path edge/node consistency
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        # -----------------------------------------------------
        # Node IDs
        # -----------------------------------------------------

        node_ids = list(graph.nodes.keys())

        if len(node_ids) != len(set(node_ids)):
            result.add_error(
                "Duplicate node IDs detected."
            )

        # -----------------------------------------------------
        # Edge IDs
        # -----------------------------------------------------

        edge_ids = [edge.edge_id for edge in graph.edges]

        if len(edge_ids) != len(set(edge_ids)):
            result.add_error(
                "Duplicate edge IDs detected."
            )

        # -----------------------------------------------------
        # Edges
        # -----------------------------------------------------

        known_nodes = set(graph.nodes.keys())

        for edge in graph.edges:
            if edge.source not in known_nodes:
                result.add_error(
                    f"Edge {edge.edge_id} has unknown source "
                    f"{edge.source}."
                )

            if edge.target not in known_nodes:
                result.add_error(
                    f"Edge {edge.edge_id} has unknown target "
                    f"{edge.target}."
                )

            if edge.source == edge.target:
                result.add_error(
                    f"Self-loop detected at {edge.source} "
                    f"for edge {edge.edge_id}."
                )

            if edge.cost <= 0:
                result.add_error(
                    f"Edge {edge.edge_id} has non-positive cost "
                    f"{edge.cost}."
                )

            if edge.relation not in self.VALID_RELATIONS:
                result.add_error(
                    f"Edge {edge.edge_id} has unknown relation "
                    f"{edge.relation}."
                )

        # -----------------------------------------------------
        # Path records
        # -----------------------------------------------------

        all_paths = (
            list(graph.valid_paths)
            + list(graph.invalid_paths)
        )

        edge_map = graph.edge_map

        for path in all_paths:
            self._validate_single_path_structure(
                graph=graph,
                path=path,
                edge_map=edge_map,
                result=result,
            )

        return result

    def _validate_single_path_structure(
        self,
        graph: Graph,
        path: PathRecord,
        edge_map: dict,
        result: ValidationResult,
    ) -> None:
        """
        Validate the internal consistency of one path record.
        """

        if not path.node_ids:
            result.add_error(
                f"Path {path.path_id} has no nodes."
            )
            return

        if path.node_ids[0] != graph.source_id:
            result.add_error(
                f"Path {path.path_id} does not start at "
                f"source {graph.source_id}."
            )

        if path.node_ids[-1] != graph.target_id:
            result.add_error(
                f"Path {path.path_id} does not end at "
                f"target {graph.target_id}."
            )

        expected_edge_count = len(path.node_ids) - 1

        if len(path.edge_ids) != expected_edge_count:
            result.add_error(
                f"Path {path.path_id} has inconsistent length: "
                f"nodes={len(path.node_ids)}, "
                f"edges={len(path.edge_ids)}."
            )
            return

        total_cost = 0.0

        for index, edge_id in enumerate(path.edge_ids):
            if edge_id not in edge_map:
                result.add_error(
                    f"Path {path.path_id} references unknown edge "
                    f"{edge_id}."
                )
                continue

            edge = edge_map[edge_id]

            expected_source = path.node_ids[index]
            expected_target = path.node_ids[index + 1]

            if edge.source != expected_source:
                result.add_error(
                    f"Path {path.path_id}, edge {edge_id}: "
                    f"expected source={expected_source}, "
                    f"actual source={edge.source}."
                )

            if edge.target != expected_target:
                result.add_error(
                    f"Path {path.path_id}, edge {edge_id}: "
                    f"expected target={expected_target}, "
                    f"actual target={edge.target}."
                )

            total_cost += edge.cost

        tolerance = 1e-6

        if abs(total_cost - path.expected_cost) > tolerance:
            result.add_error(
                f"Path {path.path_id} has incorrect expected cost: "
                f"recorded={path.expected_cost:.6f}, "
                f"actual={total_cost:.6f}."
            )

    # =========================================================
    # 3. Policy validation
    # =========================================================

    def validate_path_policy(
        self,
        graph: Graph,
    ) -> ValidationResult:
        result = ValidationResult(
            valid=True,
            errors=[],
        )

        edge_map = graph.edge_map

        all_paths = (
            list(graph.valid_paths)
            + list(graph.invalid_paths)
        )

        for path in all_paths:
            violations = self._get_path_policy_violations(
                graph=graph,
                path=path,
                edge_map=edge_map,
            )

            actual_valid = len(violations) == 0

            # Explicit deny scenario is a special case:
            # the path can be structurally valid but denied
            # by the target policy.
            if self.scenario.explicit_deny:
                if path.expected_valid:
                    if "explicit_deny" in violations:
                        # Structural validity and authorization
                        # validity are different.
                        continue

            if actual_valid != path.expected_valid:
                result.add_error(
                    f"Path {path.path_id} policy mismatch: "
                    f"expected_valid={path.expected_valid}, "
                    f"actual_valid={actual_valid}, "
                    f"violations={violations}."
                )

        return result

    def _get_path_policy_violations(
        self,
        graph: Graph,
        path: PathRecord,
        edge_map: dict,
    ) -> List[str]:
        """
        Return all policy violations found on a path.

        This function does not rely on PathRecord.violation
        or PathRecord.expected_valid.
        """

        violations: List[str] = []

        if not path.node_ids:
            return ["empty_path"]

        # -----------------------------------------------------
        # Explicit deny
        # -----------------------------------------------------

        target_node = graph.nodes.get(graph.target_id)

        if target_node is None:
            violations.append("missing_target")
            return violations

        if target_node.deny_roles:
            # The generated benchmark uses "employee" as the
            # target role for explicit deny scenarios.
            if "employee" in target_node.deny_roles:
                violations.append("explicit_deny")

        # -----------------------------------------------------
        # Edge-level and target-level propagation rules
        # -----------------------------------------------------

        for index, edge_id in enumerate(path.edge_ids):
            edge = edge_map.get(edge_id)

            if edge is None:
                violations.append(
                    f"unknown_edge:{edge_id}"
                )
                continue

            source_id = path.node_ids[index]
            target_id = path.node_ids[index + 1]

            source_node = graph.nodes.get(source_id)
            target_node = graph.nodes.get(target_id)

            if source_node is None or target_node is None:
                violations.append(
                    f"missing_node:{source_id}->{target_id}"
                )
                continue

            # Hierarchical propagation:
            # target classification must not exceed source
            # classification.
            if edge.relation == "hierarchical":
                if (
                    target_node.classification
                    > source_node.classification
                ):
                    violations.append(
                        "hierarchical_classification_violation"
                    )

            # Semantic propagation:
            # this benchmark treats cross-department semantic
            # propagation as invalid.
            if edge.relation == "semantic":
                if (
                    source_node.department
                    != target_node.department
                ):
                    violations.append(
                        "cross_department_violation"
                    )

            # Source propagation:
            # source IDs must match.
            if edge.relation == "source":
                if (
                    source_node.source_id
                    != target_node.source_id
                ):
                    violations.append(
                        "source_mismatch"
                    )

            # Temporal propagation:
            # this benchmark rejects temporal propagation to
            # a higher classification.
            if edge.relation == "temporal":
                if (
                    target_node.classification
                    > source_node.classification
                ):
                    violations.append(
                        "temporal_classification_violation"
                    )

        return sorted(set(violations))

    # =========================================================
    # 4. Cost ordering
    # =========================================================

    def validate_cost_order(
        self,
        graph: Graph,
    ) -> ValidationResult:
        """
        Validate scenario-specific cost ordering.

        For invalid_cheap_shortcut:
            min(invalid path cost) < min(valid path cost)

        For multiple_valid_paths:
            valid paths should have distinct costs and respect
            the configured minimum cost gap.

        For other scenarios:
            ensure all path costs are positive and finite.
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        valid_costs = [
            path.expected_cost
            for path in graph.valid_paths
        ]

        invalid_costs = [
            path.expected_cost
            for path in graph.invalid_paths
        ]

        all_costs = valid_costs + invalid_costs

        for cost in all_costs:
            if cost <= 0:
                result.add_error(
                    f"Path cost must be positive: {cost}."
                )

        # -----------------------------------------------------
        # Invalid cheap shortcut
        # -----------------------------------------------------

        if self.scenario.require_invalid_cheaper_than_valid:
            if not valid_costs:
                result.add_error(
                    "Cost-order validation requires at least "
                    "one valid path."
                )

            if not invalid_costs:
                result.add_error(
                    "Cost-order validation requires at least "
                    "one invalid path."
                )

            if valid_costs and invalid_costs:
                cheapest_valid = min(valid_costs)
                cheapest_invalid = min(invalid_costs)

                if not cheapest_invalid < cheapest_valid:
                    result.add_error(
                        "Invalid shortcut is not cheaper than "
                        "the valid path: "
                        f"invalid={cheapest_invalid:.6f}, "
                        f"valid={cheapest_valid:.6f}."
                    )

        # -----------------------------------------------------
        # Multiple valid paths
        # -----------------------------------------------------

        if self.scenario.name == "multiple_valid_paths":
            if len(valid_costs) < 2:
                result.add_error(
                    "Multiple-valid-path scenario requires "
                    "at least two valid paths."
                )

            unique_costs = set(valid_costs)

            if len(unique_costs) != len(valid_costs):
                result.add_error(
                    "Multiple valid paths must have distinct "
                    "costs."
                )

            sorted_costs = sorted(valid_costs)

            for previous, current in zip(
                sorted_costs,
                sorted_costs[1:],
            ):
                gap = current - previous
                EPS = 1e-3

                if gap + EPS < self.scenario.valid_cost_gap:
                    result.add_error(
                        "Valid path cost gap is too small: "
                        f"previous={previous:.6f}, "
                        f"current={current:.6f}, "
                        f"gap={gap:.6f}, "
                        f"required={self.scenario.valid_cost_gap:.6f}."
                    )

        return result

    # =========================================================
    # 5. Edge density
    # =========================================================

    def validate_edge_density(
        self,
        graph: Graph,
    ) -> ValidationResult:
        """
        Validate actual edge density.

        Directed graph:
            density = |E| / (|V| * (|V| - 1))

        The generated graph may exceed the nominal density
        because mandatory path edges must be retained.

        We therefore use a configurable tolerance:
            max(nominal_density * 2, 0.25)
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        if node_count <= 1:
            result.add_error(
                "Cannot calculate edge density for a graph "
                "with fewer than two nodes."
            )
            return result

        possible_edges = node_count * (node_count - 1)

        actual_density = edge_count / possible_edges
        nominal_density = self.profile.edge_density

        maximum_allowed_density = max(
            nominal_density * 2.0,
            0.25,
        )

        if actual_density > maximum_allowed_density:
            result.add_error(
                "Edge density is too high: "
                f"actual={actual_density:.6f}, "
                f"nominal={nominal_density:.6f}, "
                f"maximum_allowed={maximum_allowed_density:.6f}."
            )

        if actual_density <= 0:
            result.add_error(
                "Graph must contain at least one edge."
            )

        # Store measured values for later analysis.
        graph.metadata["actual_edge_density"] = actual_density
        graph.metadata["nominal_edge_density"] = nominal_density
        graph.metadata[
            "maximum_allowed_edge_density"
        ] = maximum_allowed_density

        return result

    # =========================================================
    # 6. Scenario validation
    # =========================================================

    def validate_scenario(
        self,
        graph: Graph,
    ) -> ValidationResult:
        """
        Validate scenario-specific constraints.
        """

        result = ValidationResult(
            valid=True,
            errors=[],
        )

        valid_count = len(graph.valid_paths)
        invalid_count = len(graph.invalid_paths)

        # -----------------------------------------------------
        # Number of valid paths
        # -----------------------------------------------------

        if not (
            self.scenario.min_valid_paths
            <= valid_count
            <= self.scenario.max_valid_paths
        ):
            result.add_error(
                "Invalid number of valid paths: "
                f"actual={valid_count}, "
                f"expected_range=("
                f"{self.scenario.min_valid_paths}, "
                f"{self.scenario.max_valid_paths})."
            )

        # -----------------------------------------------------
        # Number of invalid paths
        # -----------------------------------------------------

        if not (
            self.scenario.min_invalid_paths
            <= invalid_count
            <= self.scenario.max_invalid_paths
        ):
            result.add_error(
                "Invalid number of invalid paths: "
                f"actual={invalid_count}, "
                f"expected_range=("
                f"{self.scenario.min_invalid_paths}, "
                f"{self.scenario.max_invalid_paths})."
            )

        # -----------------------------------------------------
        # Explicit deny
        # -----------------------------------------------------

        if self.scenario.explicit_deny:
            target = graph.nodes.get(graph.target_id)

            if target is None:
                result.add_error(
                    "Explicit-deny scenario has no target."
                )
            elif "employee" not in target.deny_roles:
                result.add_error(
                    "Explicit-deny scenario requires target "
                    "deny role 'employee'."
                )

        # -----------------------------------------------------
        # Required violation
        # -----------------------------------------------------

        if self.scenario.required_violation:
            expected_violation = (
                self.scenario.required_violation
            )

            for path in graph.invalid_paths:
                if path.violation != expected_violation:
                    result.add_error(
                        f"Path {path.path_id} has violation "
                        f"{path.violation}, expected "
                        f"{expected_violation}."
                    )

        # -----------------------------------------------------
        # No valid path
        # -----------------------------------------------------

        if self.scenario.name == "no_valid_path":
            if valid_count != 0:
                result.add_error(
                    "no_valid_path scenario must not contain "
                    "valid paths."
                )

        return result