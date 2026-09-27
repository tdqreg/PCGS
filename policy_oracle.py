from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class User:
    user_id: str
    role: str
    department: str
    clearance: int
    cross_department: bool


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    department: str
    classification: int
    source_id: str

    # Direct access and propagation are deliberately separated.
    direct_roles: Tuple[str, ...]
    propagation_roles: Tuple[str, ...]

    deny_roles: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: str
    target: str
    relation: str
    cost: float


# ============================================================
# Direct authorization
# ============================================================

def explicit_deny(chunk: Chunk, user: User) -> bool:
    if user.role in chunk.deny_roles:
        return True

    if chunk.classification > user.clearance:
        return True

    return False


def direct_decision(
    chunk: Chunk,
    user: User,
) -> str:

    if explicit_deny(chunk, user):
        return "DENY"

    if user.role not in chunk.direct_roles:
        return "DEFER"

    if (
        chunk.department != user.department
        and not user.cross_department
    ):
        return "DEFER"

    return "ALLOW"


# ============================================================
# Edge policy
# ============================================================

def edge_allowed(
    edge: Edge,
    source: Chunk,
    target: Chunk,
    user: User,
) -> bool:

    if edge.relation in {"semantic", "reference"}:
        return (
            source.department == target.department
            or user.cross_department
        )

    if edge.relation == "temporal":
        return (
            abs(
                source.classification
                - target.classification
            ) <= 1
        )

    if edge.relation == "hierarchical":
        return (
            target.classification
            <= source.classification
        )

    if edge.relation == "source":
        return (
            source.source_id == target.source_id
            or user.cross_department
        )

    return False


# ============================================================
# Propagation policy
# ============================================================

def propagation_allowed(
    chunk: Chunk,
    user: User,
) -> bool:

    if explicit_deny(chunk, user):
        return False

    if user.role not in chunk.propagation_roles:
        return False

    if chunk.classification > user.clearance:
        return False

    if (
        chunk.department != user.department
        and not user.cross_department
    ):
        return False

    return True


def transition_allowed(
    edge: Edge,
    source: Chunk,
    target: Chunk,
    user: User,
) -> bool:

    if not edge_allowed(
        edge,
        source,
        target,
        user,
    ):
        return False

    if not propagation_allowed(
        target,
        user,
    ):
        return False

    return True


# ============================================================
# Minimum-cost policy-valid path
# ============================================================

def minimum_cost_path(
    chunks: Dict[str, Chunk],
    edges: List[Edge],
    source_ids: List[str],
    target_id: str,
    user: User,
) -> dict:

    adjacency: Dict[str, List[Edge]] = {
        chunk_id: []
        for chunk_id in chunks
    }

    for edge in edges:
        adjacency.setdefault(
            edge.source,
            []
        ).append(edge)

    best_cost: Dict[str, float] = {}
    parent: Dict[str, Optional[str]] = {}
    parent_edge: Dict[str, Optional[str]] = {}

    queue = []

    for source_id in source_ids:
        best_cost[source_id] = 0.0
        parent[source_id] = None
        parent_edge[source_id] = None

        heapq.heappush(
            queue,
            (0.0, source_id),
        )

    while queue:

        current_cost, current = heapq.heappop(queue)

        if (
            current_cost
            != best_cost.get(current)
        ):
            continue

        if current == target_id:
            break

        for edge in adjacency.get(
            current,
            [],
        ):

            source = chunks[current]
            target = chunks[edge.target]

            if not transition_allowed(
                edge,
                source,
                target,
                user,
            ):
                continue

            new_cost = (
                current_cost
                + edge.cost
            )

            if new_cost < best_cost.get(
                edge.target,
                float("inf"),
            ):

                best_cost[edge.target] = new_cost
                parent[edge.target] = current
                parent_edge[edge.target] = edge.edge_id

                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        edge.target,
                    ),
                )

    if target_id not in best_cost:

        return {
            "authorized": False,
            "authorization_type": "none",
            "path": [],
            "edge_ids": [],
            "cost": None,
        }

    path = []
    edge_ids = []

    current = target_id

    while current is not None:

        path.append(current)

        if parent_edge[current] is not None:
            edge_ids.append(
                parent_edge[current]
            )

        current = parent[current]

    path.reverse()
    edge_ids.reverse()

    return {
        "authorized": True,
        "authorization_type": "propagation",
        "path": path,
        "edge_ids": edge_ids,
        "cost": best_cost[target_id],
    }


# ============================================================
# Complete oracle
# ============================================================

def evaluate_instance(
    user: User,
    chunks: List[Chunk],
    edges: List[Edge],
    source_ids: List[str],
    target_id: str,
) -> dict:

    chunk_map = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    target = chunk_map[target_id]

    direct = direct_decision(
        target,
        user,
    )

    # Explicit denial always dominates.
    if direct == "DENY":

        return {
            "direct_decision": direct,
            "oracle_authorized": False,
            "authorization_type": "deny",
            "oracle_path": [],
            "oracle_edge_ids": [],
            "oracle_cost": None,
        }

    # Direct permission does not require graph search.
    if direct == "ALLOW":

        return {
            "direct_decision": direct,
            "oracle_authorized": True,
            "authorization_type": "direct",
            "oracle_path": [target_id],
            "oracle_edge_ids": [],
            "oracle_cost": 0.0,
        }

    # Otherwise try explicit propagation.
    result = minimum_cost_path(
        chunks=chunk_map,
        edges=edges,
        source_ids=source_ids,
        target_id=target_id,
        user=user,
    )

    return {
        "direct_decision": direct,
        "oracle_authorized": result["authorized"],
        "authorization_type": result[
            "authorization_type"
        ],
        "oracle_path": result["path"],
        "oracle_edge_ids": result["edge_ids"],
        "oracle_cost": result["cost"],
    }