# models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Node:
    node_id: str
    department: str
    classification: int
    source_id: str

    direct_roles: Tuple[str, ...] = ()
    propagation_roles: Tuple[str, ...] = ()
    deny_roles: Tuple[str, ...] = ()


@dataclass
class Edge:
    edge_id: str
    source: str
    target: str
    relation: str
    cost: float


@dataclass
class PathRecord:
    path_id: str
    node_ids: List[str]
    edge_ids: List[str]

    expected_valid: bool
    expected_cost: float

    violation: str | None = None


@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]

    source_id: str
    target_id: str

    valid_paths: List[PathRecord] = field(default_factory=list)
    invalid_paths: List[PathRecord] = field(default_factory=list)

    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def edge_map(self) -> Dict[str, Edge]:
        return {
            edge.edge_id: edge
            for edge in self.edges
        }

RELATIONS = (
    "semantic",
    "hierarchical",
    "temporal",
    "source",
)