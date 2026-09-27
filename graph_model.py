from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Node
# ============================================================

@dataclass
class Node:
    """
    Graph node with authorization-related attributes.
    """

    node_id: str

    department: str

    classification: int

    source_id: str

    direct_roles: list[str] = field(
        default_factory=list
    )

    propagation_roles: list[str] = field(
        default_factory=list
    )

    deny_roles: list[str] = field(
        default_factory=list
    )


# ============================================================
# Edge
# ============================================================

@dataclass
class Edge:
    """
    Directed graph edge.
    """

    edge_id: str

    source: str

    target: str

    relation: str

    cost: float


# ============================================================
# Path Record
# ============================================================

@dataclass
class PathRecord:
    """
    Dataset-declared path.

    path_id:
        Unique path identifier.

    node_ids:
        Ordered node sequence.

    edge_ids:
        Ordered edge sequence.

    expected_cost:
        Expected total cost.

    expected_valid:
        Generator-declared validity.

    violation:
        Optional policy violation label.
    """

    path_id: str

    node_ids: list[str]

    edge_ids: list[str]

    expected_cost: float

    expected_valid: bool

    violation: str | None = None


# ============================================================
# Graph
# ============================================================

@dataclass
class Graph:
    nodes: dict[str, Node]
    edges: list[Edge]
    source_id: str
    target_id: str
    valid_paths: list[PathRecord] = field( default_factory=list )
    invalid_paths: list[PathRecord] = field( default_factory=list )
    metadata: dict[str, Any] = field( default_factory=dict )
    # Derived index.
    edge_map: dict[str, Edge] = field( init=False, repr=False, )

    def __post_init__(self):
        """
        Build an edge-id -> Edge lookup table. This is derived from self.edges so it does not need to exist in the serialized JSON dataset.
        """
        self.edge_map = { edge.edge_id: edge for edge in self.edges }