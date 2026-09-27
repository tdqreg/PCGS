from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Node:
    node_id: str
    node_type: str = "INTERMEDIATE"


@dataclass
class Edge:
    edge_id: str
    source: str
    target: str
    cost: float

    # Level-2 authorization fields
    required_grants: frozenset[str] = field(
        default_factory=frozenset
    )

    min_clearance: int = 0

    grant: str | None = None

    grant_clearance: int | None = None

    deny: bool = False


@dataclass
class Graph:
    nodes: dict[str, Node]
    edges: list[Edge]
    source_id: str
    target_id: str
    metadata: dict[str, Any]

    def __post_init__(self):
        self.edge_map = {
            edge.edge_id: edge
            for edge in self.edges
        }


def load_level2_graph(
    path: str | Path,
) -> Graph:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------
    #
    # Current Level-2 generator uses:
    #
    # [
    #     {
    #         "node_id": "SOURCE",
    #         "type": "SOURCE"
    #     },
    #     ...
    # ]
    #
    # Support dict form as well for backward compatibility.
    # ---------------------------------------------------------

    raw_nodes = data["nodes"]

    if isinstance(raw_nodes, dict):
        node_items = raw_nodes.values()
    else:
        node_items = raw_nodes

    nodes: dict[str, Node] = {}

    for item in node_items:

        node_id = item["node_id"]

        nodes[node_id] = Node(
            node_id=node_id,
            node_type=item.get(
                "type",
                "INTERMEDIATE",
            ),
        )

    # ---------------------------------------------------------
    # Edges
    # ---------------------------------------------------------

    edges: list[Edge] = []

    for item in data["edges"]:

        required_grants = frozenset(
            item.get(
                "required_grants",
                [],
            )
        )

        grant_clearance = item.get(
            "grant_clearance"
        )

        if grant_clearance is not None:
            grant_clearance = int(
                grant_clearance
            )

        edges.append(
            Edge(
                edge_id=item["edge_id"],
                source=item["source"],
                target=item["target"],
                cost=float(
                    item["cost"]
                ),
                required_grants=required_grants,
                min_clearance=int(
                    item.get(
                        "min_clearance",
                        0,
                    )
                ),
                grant=item.get(
                    "grant"
                ),
                grant_clearance=grant_clearance,
                deny=bool(
                    item.get(
                        "deny",
                        False,
                    )
                ),
            )
        )

    # ---------------------------------------------------------
    # Graph-level validation
    # ---------------------------------------------------------

    source_id = data["source_id"]
    target_id = data["target_id"]

    if source_id not in nodes:
        raise ValueError(
            f"Source node '{source_id}' "
            f"not found in graph: {path}"
        )

    if target_id not in nodes:
        raise ValueError(
            f"Target node '{target_id}' "
            f"not found in graph: {path}"
        )

    node_ids = set(nodes)

    for edge in edges:

        if edge.source not in node_ids:
            raise ValueError(
                f"Edge {edge.edge_id} references "
                f"unknown source node "
                f"'{edge.source}'"
            )

        if edge.target not in node_ids:
            raise ValueError(
                f"Edge {edge.edge_id} references "
                f"unknown target node "
                f"'{edge.target}'"
            )

        if edge.cost <= 0:
            raise ValueError(
                f"Edge {edge.edge_id} has "
                f"non-positive cost: {edge.cost}"
            )

    return Graph(
        nodes=nodes,
        edges=edges,
        source_id=source_id,
        target_id=target_id,
        metadata=data.get(
            "metadata",
            {},
        ),
    )