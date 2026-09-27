from __future__ import annotations

import random
from typing import Dict, List, Set

from models import (
    Edge,
    Graph,
    Node,
    PathRecord,
)

from scenario_specs import (
    ScenarioSpec,
    SizeProfile,
)


class GenericGraphGenerator:

    DEPARTMENTS = (
        "engineering",
        "finance",
        "hr",
        "legal",
        "sales",
    )

    RELATIONS = (
        "semantic",
        "reference",
        "temporal",
        "hierarchical",
        "source",
    )

    VALID_RELATION = "semantic"

    def __init__(
        self,
        profile: SizeProfile,
        scenario: ScenarioSpec,
        seed: int,
    ):
        self.profile = profile
        self.scenario = scenario
        self.rng = random.Random(seed)

        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

        self.valid_paths: List[PathRecord] = []
        self.invalid_paths: List[PathRecord] = []

        self.edge_counter = 0
        self.path_counter = 0

        self.source_id = "SOURCE"
        self.target_id = "TARGET"

        self.reserved_nodes: Set[str] = set()

        self.path_nodes: Set[str] = {
            self.source_id,
            self.target_id,
        }

    # ========================================================
    # Basic factories
    # ========================================================

    def _create_node(
        self,
        node_id: str,
        department: str = "engineering",
        classification: int = 1,
        source_id: str = "src-main",
        direct_roles: tuple[str, ...] = (),
        propagation_roles: tuple[str, ...] = (),
        deny_roles: tuple[str, ...] = (),
        is_path_node: bool = False,
    ) -> Node:

        if node_id in self.nodes:
            raise ValueError(
                f"Duplicate node ID: {node_id}"
            )

        node = Node(
            node_id=node_id,
            department=department,
            classification=classification,
            source_id=source_id,
            direct_roles=direct_roles,
            propagation_roles=propagation_roles,
            deny_roles=deny_roles,
        )

        self.nodes[node_id] = node

        if is_path_node:
            self.path_nodes.add(node_id)

        self.reserved_nodes.add(node_id)

        return node

    def _create_edge(
        self,
        source: str,
        target: str,
        relation: str,
        cost: float,
    ) -> Edge:

        self.edge_counter += 1

        edge = Edge(
            edge_id=f"E{self.edge_counter}",
            source=source,
            target=target,
            relation=relation,
            cost=round(cost, 4),
        )

        self.edges.append(edge)

        return edge

    def _new_intermediate_id(
        self,
        path_kind: str,
        path_index: int,
        step_index: int,
    ) -> str:

        return (
            f"{path_kind}_{path_index}_{step_index}"
        )

    # ========================================================
    # Core nodes
    # ========================================================

    def create_core_nodes(self) -> None:

        self._create_node(
            self.source_id,
            department="engineering",
            classification=1,
            source_id="src-main",
            direct_roles=("employee",),
            propagation_roles=("employee",),
        )

        deny_roles = (
            ("employee",)
            if self.scenario.explicit_deny
            else ()
        )

        self._create_node(
            self.target_id,
            department="engineering",
            classification=1,
            source_id="src-main",
            direct_roles=(),
            propagation_roles=("employee",),
            deny_roles=deny_roles,
        )

    # ========================================================
    # Path depth and costs
    # ========================================================

    def choose_depth(
        self,
        path_kind: str,
    ) -> int:

        if self.scenario.name == "safe_single_hop":
            return 1

        # Multiple valid paths must not collapse into
        # multiple copies of SOURCE -> TARGET.
        if self.scenario.name == "multiple_valid_paths":
            min_depth = max(
                2,
                self.profile.min_path_depth,
            )

            return self.rng.randint(
                min_depth,
                self.profile.max_path_depth,
            )

        if path_kind == "invalid":
            return self.rng.randint(
                self.profile.min_path_depth,
                self.profile.max_path_depth,
            )

        return self.rng.randint(
            self.profile.min_path_depth,
            self.profile.max_path_depth,
        )

    def calculate_edge_cost(
        self,
        path_kind: str,
        path_index: int,
        depth: int,
    ) -> float:

        if (
            path_kind == "invalid"
            and self.scenario.require_invalid_cheaper_than_valid
        ):
            return self.scenario.invalid_shortcut_cost

        if path_kind == "valid":
            return (
                self.profile.min_edge_cost
                + path_index
                * self.scenario.valid_cost_gap
            )

        return self.profile.min_edge_cost

    # ========================================================
    # Valid path construction
    # ========================================================

    def create_valid_path(
        self,
        path_index: int,
    ) -> PathRecord:

        depth = self.choose_depth("valid")

        path_id = f"VALID_{path_index}"

        current = self.source_id

        node_ids = [current]
        edge_ids: List[str] = []

        total_path_cost = (
            10.0
            + path_index
            * self.scenario.valid_cost_gap
        )

        edge_cost = total_path_cost / depth

        for step in range(1, depth):

            node_id = self._new_intermediate_id(
                "V",
                path_index,
                step,
            )

            self._create_node(
                node_id,
                department="engineering",
                classification=1,
                source_id="src-main",
                propagation_roles=("employee",),
                is_path_node=True,
            )

            edge = self._create_edge(
                source=current,
                target=node_id,
                relation=self.VALID_RELATION,
                cost=edge_cost,
            )

            node_ids.append(node_id)
            edge_ids.append(edge.edge_id)

            current = node_id

        final_edge = self._create_edge(
            source=current,
            target=self.target_id,
            relation=self.VALID_RELATION,
            cost=edge_cost,
        )

        node_ids.append(self.target_id)
        edge_ids.append(final_edge.edge_id)

        total_cost = sum(
            edge.cost
            for edge in self.edges
            if edge.edge_id in edge_ids
        )

        record = PathRecord(
            path_id=path_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            expected_valid=True,
            expected_cost=round(total_cost, 4),
        )

        self.valid_paths.append(record)

        return record

    # ========================================================
    # Invalid path construction
    # ========================================================

    def create_invalid_path(
        self,
        path_index: int,
    ) -> PathRecord:

        depth = max(
            2,
            self.choose_depth("invalid"),
        )

        path_id = f"INVALID_{path_index}"

        violation = (
            self.scenario.required_violation
            or "hierarchical"
        )

        current = self.source_id

        node_ids = [current]
        edge_ids: List[str] = []

        total_path_cost = (
            self.scenario.invalid_shortcut_cost
        )

        edge_cost = total_path_cost / depth

        violation_step = max(
            1,
            depth // 2,
        )

        for step in range(1, depth):

            node_id = self._new_intermediate_id(
                "I",
                path_index,
                step,
            )

            department = "engineering"
            classification = 1
            source_id = "src-main"
            relation = self.VALID_RELATION

            if step == violation_step:

                if violation == "hierarchical":

                    classification = 3
                    relation = "hierarchical"

                elif violation == "cross_department":

                    department = "finance"
                    relation = "semantic"

                elif violation == "source":

                    source_id = "src-other"
                    relation = "source"

                elif violation == "temporal":

                    classification = 3
                    relation = "temporal"

                else:
                    raise ValueError(
                        f"Unsupported violation: {violation}"
                    )

            self._create_node(
                node_id,
                department=department,
                classification=classification,
                source_id=source_id,
                propagation_roles=("employee",),
                is_path_node=True,
            )

            edge = self._create_edge(
                source=current,
                target=node_id,
                relation=relation,
                cost=edge_cost,
            )

            node_ids.append(node_id)
            edge_ids.append(edge.edge_id)

            current = node_id

        final_edge = self._create_edge(
            source=current,
            target=self.target_id,
            relation=self.VALID_RELATION,
            cost=edge_cost,
        )

        node_ids.append(self.target_id)
        edge_ids.append(final_edge.edge_id)

        total_cost = sum(
            edge.cost
            for edge in self.edges
            if edge.edge_id in edge_ids
        )

        record = PathRecord(
            path_id=path_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            expected_valid=False,
            expected_cost=round(total_cost, 4),
            violation=violation,
        )

        self.invalid_paths.append(record)

        return record

    # ========================================================
    # Exact node count
    # ========================================================

    def fill_decoy_nodes(self) -> None:

        while len(self.nodes) < self.profile.num_nodes:

            node_id = f"D{len(self.nodes)}"

            self._create_node(
                node_id,
                department=self.rng.choice(
                    self.DEPARTMENTS
                ),
                classification=self.rng.randint(
                    1,
                    3,
                ),
                source_id=(
                    f"src-{self.rng.randint(1, 5)}"
                ),
                direct_roles=(),
                propagation_roles=(),
            )

    # ========================================================
    # Edge helpers
    # ========================================================

    def _existing_edge_pairs(
        self,
    ) -> Set[tuple[str, str]]:

        return {
            (
                edge.source,
                edge.target,
            )
            for edge in self.edges
        }

    def calculate_target_edge_count(self) -> int:

        node_count = len(self.nodes)

        if node_count <= 1:
            return 0

        possible_edges = (
            node_count
            * (node_count - 1)
        )

        target_count = int(
            possible_edges
            * self.profile.edge_density
        )

        return max(
            target_count,
            len(self.edges),
        )

    # ========================================================
    # Policy evaluation
    # ========================================================

    def _edge_is_policy_valid(
        self,
        source_id: str,
        target_id: str,
        relation: str,
    ) -> bool:

        source = self.nodes[source_id]
        target = self.nodes[target_id]

        if relation == "semantic":

            return (
                source.department
                == target.department
            )

        if relation == "hierarchical":

            return (
                target.classification
                <= source.classification
            )

        if relation == "source":

            return (
                source.source_id
                == target.source_id
            )

        if relation == "temporal":

            return (
                target.classification
                <= source.classification
            )

        # "reference" is intentionally not a
        # policy-valid relation.
        return False

    def _path_is_policy_valid(
        self,
        node_path: List[str],
    ) -> bool:

        if (
            len(node_path) < 2
            or node_path[0] != self.source_id
            or node_path[-1] != self.target_id
        ):
            return False

        target = self.nodes[
            self.target_id
        ]

        if "employee" in target.deny_roles:
            return False

        edge_map = {
            (
                edge.source,
                edge.target,
            ): edge
            for edge in self.edges
        }

        for source_id, target_id in zip(
            node_path,
            node_path[1:],
        ):

            edge = edge_map.get(
                (
                    source_id,
                    target_id,
                )
            )

            if edge is None:
                return False

            if not self._edge_is_policy_valid(
                source_id,
                target_id,
                edge.relation,
            ):
                return False

        return True

    # ========================================================
    # Policy-aware reachability
    # ========================================================

    def _build_policy_adjacency(
        self,
    ) -> Dict[str, List[Edge]]:

        adjacency: Dict[str, List[Edge]] = {
            node_id: []
            for node_id in self.nodes
        }

        for edge in self.edges:

            if not self._edge_is_policy_valid(
                edge.source,
                edge.target,
                edge.relation,
            ):
                continue

            adjacency[
                edge.source
            ].append(edge)

        return adjacency

    def _build_policy_reverse_adjacency(
        self,
    ) -> Dict[str, List[Edge]]:

        reverse: Dict[str, List[Edge]] = {
            node_id: []
            for node_id in self.nodes
        }

        for edge in self.edges:

            if not self._edge_is_policy_valid(
                edge.source,
                edge.target,
                edge.relation,
            ):
                continue

            reverse[
                edge.target
            ].append(edge)

        return reverse

    def _valid_forward_reachability(
        self,
    ) -> Set[str]:

        """
        Return nodes reachable from SOURCE using
        policy-valid edges.

        This performs graph reachability only.
        It does not enumerate individual paths.
        """

        target = self.nodes[
            self.target_id
        ]

        if "employee" in target.deny_roles:
            return set()

        adjacency = (
            self._build_policy_adjacency()
        )

        reachable: Set[str] = {
            self.source_id
        }

        stack = [
            self.source_id
        ]

        while stack:

            current = stack.pop()

            for edge in adjacency.get(
                current,
                [],
            ):

                if edge.target in reachable:
                    continue

                reachable.add(
                    edge.target
                )

                stack.append(
                    edge.target
                )

        return reachable

    def _valid_backward_reachability(
        self,
    ) -> Set[str]:

        """
        Return nodes from which TARGET can be
        reached using policy-valid edges.

        This performs graph reachability only.
        It does not enumerate individual paths.
        """

        target = self.nodes[
            self.target_id
        ]

        if "employee" in target.deny_roles:
            return set()

        reverse = (
            self._build_policy_reverse_adjacency()
        )

        reachable: Set[str] = {
            self.target_id
        }

        stack = [
            self.target_id
        ]

        while stack:

            current = stack.pop()

            for edge in reverse.get(
                current,
                [],
            ):

                if edge.source in reachable:
                    continue

                reachable.add(
                    edge.source
                )

                stack.append(
                    edge.source
                )

        return reachable

    # ========================================================
    # Candidate edge validation
    # ========================================================

    def _candidate_edge_allowed(
        self,
        source: str,
        target: str,
        relation: str,
        forward_reachable: Set[str],
        backward_reachable: Set[str],
    ) -> bool:

        # SOURCE remains the only graph entry point.
        if source == self.source_id:
            return False

        # TARGET remains a sink.
        if target == self.target_id:
            return False

        # If TARGET is explicitly denied, no authorized
        # SOURCE -> TARGET path can exist.
        #
        # In this scenario, decoy structural paths are
        # allowed because they are all unauthorized.
        if (
            "employee"
            in self.nodes[
                self.target_id
            ].deny_roles
        ):
            return True

        # If SOURCE cannot policy-reach the candidate
        # source, this edge cannot participate in a
        # policy-valid SOURCE -> TARGET path.
        if source not in forward_reachable:
            return True

        # If candidate target cannot policy-reach TARGET,
        # this edge cannot complete a valid path.
        if target not in backward_reachable:
            return True

        # If the candidate edge itself violates policy,
        # it cannot create an authorized path.
        if not self._edge_is_policy_valid(
            source,
            target,
            relation,
        ):
            return True

        # All three conditions are satisfied:
        #
        # SOURCE -> ... -> source
        # source -> target
        # target -> ... -> TARGET
        #
        # Therefore this candidate would introduce
        # another policy-valid SOURCE -> TARGET path.
        return False

    # ========================================================
    # Controlled decoy edges
    # ========================================================

    def create_decoy_edges(self) -> None:

        target_edge_count = (
            self.calculate_target_edge_count()
        )

        if len(self.edges) >= target_edge_count:
            return

        # ----------------------------------------------------
        # IMPORTANT INVARIANT
        #
        # Decoy edges are allowed ONLY between decoy nodes.
        #
        # They must never touch:
        #   SOURCE
        #   TARGET
        #   valid-path nodes
        #   invalid-path nodes
        #
        # This guarantees that decoy edges cannot create
        # additional SOURCE -> TARGET policy-valid paths.
        # ----------------------------------------------------

        decoy_ids = [
            node_id
            for node_id in self.nodes
            if node_id not in self.path_nodes
        ]

        existing_pairs = (
            self._existing_edge_pairs()
        )

        if len(decoy_ids) < 2:
            return

        possible_decoy_pairs = [
            (source, target)
            for source in decoy_ids
            for target in decoy_ids
            if source != target
        ]

        self.rng.shuffle(
            possible_decoy_pairs
        )

        pair_index = 0

        while (
            len(self.edges) < target_edge_count
            and pair_index < len(possible_decoy_pairs)
        ):

            source, target = (
                possible_decoy_pairs[pair_index]
            )

            pair_index += 1

            if (
                source,
                target,
            ) in existing_pairs:
                continue

            relation = self.rng.choice(
                self.RELATIONS
            )

            cost = self.rng.uniform(
                self.profile.min_edge_cost,
                self.profile.max_edge_cost,
            )

            self._create_edge(
                source=source,
                target=target,
                relation=relation,
                cost=cost,
            )

            existing_pairs.add(
                (
                    source,
                    target,
                )
            )

    # ========================================================
    # Main generation
    # ========================================================

    def generate(self) -> Graph:

        self.create_core_nodes()

        valid_count = self.rng.randint(
            self.scenario.min_valid_paths,
            self.scenario.max_valid_paths,
        )

        invalid_count = self.rng.randint(
            self.scenario.min_invalid_paths,
            self.scenario.max_invalid_paths,
        )

        for path_index in range(
            valid_count
        ):

            self.create_valid_path(
                path_index
            )

        for path_index in range(
            invalid_count
        ):

            self.create_invalid_path(
                path_index
            )

        self.fill_decoy_nodes()

        self.create_decoy_edges()

        graph = Graph(
            nodes=self.nodes,
            edges=self.edges,
            source_id=self.source_id,
            target_id=self.target_id,
            valid_paths=self.valid_paths,
            invalid_paths=self.invalid_paths,
            metadata={
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "edge_density": (
                    self.profile.edge_density
                ),
                "scenario": self.scenario.name,
            },
        )

        return graph