from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from policy_oracle import Chunk, Edge, User


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class GraphConfig:
    min_nodes: int
    max_nodes: int

    min_depth: int
    max_depth: int

    min_branching: int
    max_branching: int

    min_decoys: int
    max_decoys: int

    min_cost: float
    max_cost: float

    min_cost_gap: float
    max_cost_gap: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str

    # Number of valid backbone paths.
    min_valid_paths: int = 1
    max_valid_paths: int = 1

    # Number of deliberately invalid paths.
    min_invalid_paths: int = 0
    max_invalid_paths: int = 0

    # Number of unrelated decoy nodes.
    min_decoys: int = 0
    max_decoys: int = 0

    # Whether target is propagation-only.
    propagation_target: bool = True

    # Whether target is explicitly denied.
    explicit_deny: bool = False

    # Whether there must be at least one cheaper invalid path.
    invalid_cheap_shortcut: bool = False

    # Whether competing valid paths must exist.
    competing_valid_paths: bool = False


# ============================================================
# Node factory
# ============================================================

class NodeFactory:

    def __init__(self, user: User):
        self.user = user
        self.counter = 0

    def next_id(self, prefix="N"):
        value = f"{prefix}{self.counter}"
        self.counter += 1
        return value

    def make_source(self):
        return Chunk(
            chunk_id="S",
            department=self.user.department,
            classification=min(
                2,
                self.user.clearance,
            ),
            source_id="doc-S",
            direct_roles=(self.user.role,),
            propagation_roles=(self.user.role,),
        )

    def make_propagation_node(
        self,
        node_id: str,
        department: str | None = None,
        classification: int | None = None,
    ):
        return Chunk(
            chunk_id=node_id,
            department=(
                department
                or self.user.department
            ),
            classification=(
                classification
                if classification is not None
                else min(
                    2,
                    self.user.clearance,
                )
            ),
            source_id=f"doc-{node_id}",
            direct_roles=(),
            propagation_roles=(
                self.user.role,
            ),
        )

    def make_decoy(self, node_id: str):
        return Chunk(
            chunk_id=node_id,
            department=self.user.department,
            classification=2,
            source_id=f"decoy-{node_id}",
            direct_roles=(),
            propagation_roles=(),
        )


# ============================================================
# Edge factory
# ============================================================

class EdgeFactory:

    def __init__(
        self,
        rng: random.Random,
        config: GraphConfig,
    ):
        self.rng = rng
        self.config = config
        self.counter = 0

    def next_id(self):
        value = f"e{self.counter}"
        self.counter += 1
        return value

    def cost(self):
        return round(
            self.rng.uniform(
                self.config.min_cost,
                self.config.max_cost,
            ),
            3,
        )

    def gap(self):
        return self.rng.uniform(
            self.config.min_cost_gap,
            self.config.max_cost_gap,
        )

    def make(
        self,
        source,
        target,
        relation,
        cost=None,
    ):
        return Edge(
            edge_id=self.next_id(),
            source=source,
            target=target,
            relation=relation,
            cost=round(
                cost if cost is not None else self.cost(),
                3,
            ),
        )


# ============================================================
# Generic graph
# ============================================================

class GenericGraphGenerator:

    VALID_RELATIONS = (
        "semantic",
        "reference",
        "temporal",
    )

    INVALID_RELATIONS = (
        "hierarchical",
    )

    def __init__(
        self,
        rng: random.Random,
        config: GraphConfig,
        user: User,
    ):
        self.rng = rng
        self.config = config
        self.user = user

        self.nodes: Dict[str, Chunk] = {}
        self.edges: List[Edge] = []

        self.node_factory = NodeFactory(user)
        self.edge_factory = EdgeFactory(
            rng,
            config,
        )

    # --------------------------------------------------------
    # Node operations
    # --------------------------------------------------------

    def add_node(self, node: Chunk):
        self.nodes[node.chunk_id] = node

    # --------------------------------------------------------
    # Edge operations
    # --------------------------------------------------------

    def add_edge(
        self,
        source,
        target,
        relation,
        cost=None,
    ):
        self.edges.append(
            self.edge_factory.make(
                source,
                target,
                relation,
                cost,
            )
        )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    def create_backbone(
        self,
        depth: int,
    ) -> List[str]:

        path = ["S"]

        previous = "S"

        for i in range(depth - 1):

            node_id = (
                f"B{i}"
            )

            node = (
                self.node_factory
                .make_propagation_node(
                    node_id
                )
            )

            self.add_node(node)

            path.append(node_id)

            relation = (
                self.VALID_RELATIONS[
                    i % len(
                        self.VALID_RELATIONS
                    )
                ]
            )

            self.add_edge(
                previous,
                node_id,
                relation,
            )

            previous = node_id

        return path

    # --------------------------------------------------------
    # Alternative valid path
    # --------------------------------------------------------

    def create_valid_path(
        self,
        source: str,
        target: str,
        depth: int,
        prefix: str,
    ) -> List[str]:

        path = [source]

        previous = source

        for i in range(depth - 1):

            node_id = (
                f"{prefix}{i}"
            )

            node = (
                self.node_factory
                .make_propagation_node(
                    node_id
                )
            )

            self.add_node(node)

            path.append(node_id)

            relation = (
                self.VALID_RELATIONS[
                    i % len(
                        self.VALID_RELATIONS
                    )
                ]
            )

            self.add_edge(
                previous,
                node_id,
                relation,
            )

            previous = node_id

        self.add_edge(
            previous,
            target,
            self.VALID_RELATIONS[
                depth % len(
                    self.VALID_RELATIONS
                )
            ],
        )

        path.append(target)

        return path

    # --------------------------------------------------------
    # Invalid path by policy-conflicting edge
    # --------------------------------------------------------

    def create_invalid_path(
        self,
        source: str,
        target: str,
        depth: int,
        prefix: str,
        cheap: bool = False,
    ) -> List[str]:

        path = [source]

        previous = source

        for i in range(depth - 1):

            node_id = (
                f"{prefix}{i}"
            )

            node = (
                self.node_factory
                .make_propagation_node(
                    node_id,
                    classification=2,
                )
            )

            self.add_node(node)

            path.append(node_id)

            # First edges are valid.
            if i == depth - 2:

                # Final edge intentionally violates
                # hierarchical policy.
                #
                # source classification = 2
                # target classification = 3
                #
                # target <= source is false.
                relation = "hierarchical"

                cost = (
                    self.edge_factory.cost()
                )

                if cheap:
                    cost *= 0.2

                self.add_edge(
                    previous,
                    node_id,
                    relation,
                    cost,
                )

            else:

                self.add_edge(
                    previous,
                    node_id,
                    "semantic",
                )

            previous = node_id

        # Target is class 3 so hierarchical edge
        # becomes invalid.
        target_chunk = self.nodes[target]

        if target_chunk.classification != 3:

            self.nodes[target] = Chunk(
                chunk_id=target_chunk.chunk_id,
                department=target_chunk.department,
                classification=3,
                source_id=target_chunk.source_id,
                direct_roles=target_chunk.direct_roles,
                propagation_roles=target_chunk.propagation_roles,
                deny_roles=target_chunk.deny_roles,
            )

        self.add_edge(
            previous,
            target,
            "hierarchical",
            (
                self.edge_factory.cost() * 0.2
                if cheap
                else self.edge_factory.cost()
            ),
        )

        path.append(target)

        return path

    # --------------------------------------------------------
    # Decoy nodes
    # --------------------------------------------------------

    def create_decoys(
        self,
        count: int,
    ):

        node_ids = list(
            self.nodes.keys()
        )

        for _ in range(count):

            node_id = (
                self.node_factory
                .next_id("D")
            )

            self.add_node(
                self.node_factory
                .make_decoy(
                    node_id
                )
            )

            # Connect decoy to an existing node.
            if node_ids:

                source = self.rng.choice(
                    node_ids
                )

                self.add_edge(
                    source,
                    node_id,
                    self.rng.choice(
                        self.VALID_RELATIONS
                    ),
                )

            node_ids.append(node_id)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    def create_target(
        self,
        explicit_deny=False,
    ):

        classification = 2

        deny_roles = ()

        if explicit_deny:
            deny_roles = (
                self.user.role,
            )

        target = Chunk(
            chunk_id="T",
            department=self.user.department,
            classification=classification,
            source_id="doc-T",
            direct_roles=(),
            propagation_roles=(
                self.user.role,
            ),
            deny_roles=deny_roles,
        )

        self.add_node(target)

    # --------------------------------------------------------
    # Build complete graph
    # --------------------------------------------------------

    def generate(
        self,
        spec: ScenarioSpec,
    ):

        depth = self.rng.randint(
            self.config.min_depth,
            self.config.max_depth,
        )

        depth = max(
            depth,
            2,
        )

        self.add_node(
            self.node_factory
            .make_source()
        )

        self.create_target(
            explicit_deny=spec.explicit_deny
        )

        # ----------------------------------------------------
        # Main valid path
        # ----------------------------------------------------

        backbone = self.create_backbone(
            depth
        )

        # Connect backbone to target.
        last = backbone[-1]

        self.add_edge(
            last,
            "T",
            self.VALID_RELATIONS[
                depth % len(
                    self.VALID_RELATIONS
                )
            ],
        )

        # ----------------------------------------------------
        # Additional valid paths
        # ----------------------------------------------------

        num_valid = self.rng.randint(
            spec.min_valid_paths,
            spec.max_valid_paths,
        )

        for i in range(
            max(0, num_valid - 1)
        ):

            alternative_depth = self.rng.randint(
                max(2, depth - 1),
                depth + 2,
            )

            self.create_valid_path(
                source="S",
                target="T",
                depth=alternative_depth,
                prefix=f"V{i}_",
            )

        # ----------------------------------------------------
        # Invalid paths
        # ----------------------------------------------------

        num_invalid = self.rng.randint(
            spec.min_invalid_paths,
            spec.max_invalid_paths,
        )

        for i in range(
            num_invalid
        ):

            self.create_invalid_path(
                source="S",
                target="T",
                depth=max(2, depth),
                prefix=f"I{i}_",
                cheap=(
                    spec.invalid_cheap_shortcut
                    and i == 0
                ),
            )

        # ----------------------------------------------------
        # Decoys
        # ----------------------------------------------------

        num_decoys = self.rng.randint(
            max(
                self.config.min_decoys,
                spec.min_decoys,
            ),
            min(
                self.config.max_decoys,
                max(
                    spec.max_decoys,
                    self.config.max_decoys,
                ),
            ),
        )

        self.create_decoys(
            num_decoys
        )

        return (
            list(self.nodes.values()),
            self.edges,
            ["S"],
            "T",
            {
                "depth": depth,
                "num_nodes": len(
                    self.nodes
                ),
                "num_edges": len(
                    self.edges
                ),
                "num_valid_paths":
                    num_valid,
                "num_invalid_paths":
                    num_invalid,
                "num_decoys":
                    num_decoys,
            },
        )