from __future__ import annotations

import heapq
import math
import time

from dataclasses import dataclass
from typing import Any

from policy_engine import AuthState, PolicyEngine


EPS = 1e-9


@dataclass
class SearchState:
    node_id: str
    auth_state: AuthState
    cost: float
    path_nodes: tuple[str, ...]
    path_edges: tuple[str, ...]


@dataclass
class PCGSResult:
    found: bool
    cost: float | None
    path_nodes: list[str]
    path_edges: list[str]

    runtime_ms: float

    expanded_states: int
    generated_states: int

    policy_pruned: int
    dominance_pruned: int
    bound_pruned: int
    cycle_pruned: int

    max_frontier_size: int

    final_auth_state: AuthState | None = None


class LowerBound:
    """
    Structural minimum-cost lower bound.

    Authorization constraints are ignored, therefore:

        LB(v) <= actual authorized remaining cost from v to TARGET

    assuming all edge costs are non-negative.

    This makes the bound admissible for branch-and-bound.
    """

    def __init__(self, graph: Any):
        self.graph = graph
        self.lb = self._compute()

    def _compute(self) -> dict[str, float]:

        reverse = {
            node_id: []
            for node_id in self.graph.nodes
        }

        for edge in self.graph.edges:
            reverse[edge.target].append(edge)

        target = self.graph.target_id

        dist = {
            node_id: math.inf
            for node_id in self.graph.nodes
        }

        dist[target] = 0.0

        heap = [(0.0, target)]

        while heap:

            current_cost, node = heapq.heappop(heap)

            if current_cost > dist[node] + EPS:
                continue

            for edge in reverse[node]:

                predecessor = edge.source

                new_cost = (
                    current_cost
                    + float(edge.cost)
                )

                if new_cost < dist[predecessor] - EPS:

                    dist[predecessor] = new_cost

                    heapq.heappush(
                        heap,
                        (
                            new_cost,
                            predecessor,
                        ),
                    )

        return dist

    def get(self, node_id: str) -> float:
        return self.lb.get(
            node_id,
            math.inf,
        )


class DominanceManager:
    """
    Maintains authorization states that have been generated
    for each graph node.

    Candidate (state_1, cost_1) is dominated by
    (state_2, cost_2) when:

        cost_2 <= cost_1

    and

        state_2 >= state_1

    according to PolicyEngine.state_dominates().

    We intentionally retain previously inserted states in the
    frontier instead of physically deleting them.

    This avoids stale heap entries interacting with the
    dominance structure. A dominated heap entry may still be
    expanded, but no future candidate will be incorrectly
    discarded because of it.
    """

    def __init__(self, policy: PolicyEngine):

        self.policy = policy

        self.frontier: dict[
            str,
            list[tuple[AuthState, float]],
        ] = {}

    def is_dominated(
        self,
        node_id: str,
        state: AuthState,
        cost: float,
    ) -> bool:

        candidates = self.frontier.get(
            node_id,
            [],
        )

        for existing_state, existing_cost in candidates:

            # Do not treat the exact same state/cost as
            # a reason to reject itself.
            if (
                abs(existing_cost - cost) <= EPS
                and existing_state == state
            ):
                continue

            if existing_cost <= cost + EPS:

                if self.policy.state_dominates(
                    existing_state,
                    state,
                ):
                    return True

        return False

    def insert(
        self,
        node_id: str,
        state: AuthState,
        cost: float,
    ) -> None:

        candidates = self.frontier.setdefault(
            node_id,
            [],
        )

        candidates.append(
            (state, cost)
        )

    def frontier_size(self) -> int:

        return sum(
            len(states)
            for states in self.frontier.values()
        )


class PCGS:
    """
    Policy-Constrained Graph Search.

    Search state:

        (node, authorization_state)

    Main components:

        1. Policy-aware transition
        2. Policy pruning
        3. Dominance pruning
        4. Admissible cost-bound pruning

    The three main components can be disabled independently
    for ablation experiments.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        use_policy_pruning: bool = True,
        use_dominance: bool = True,
        use_bound: bool = True,
    ):

        self.policy = (
            policy_engine
            if policy_engine is not None
            else PolicyEngine()
        )

        self.use_policy_pruning = (
            use_policy_pruning
        )

        self.use_dominance = (
            use_dominance
        )

        self.use_bound = (
            use_bound
        )

    def _build_adjacency(
        self,
        graph: Any,
    ):

        adjacency = {
            node_id: []
            for node_id in graph.nodes
        }

        for edge in graph.edges:

            adjacency[
                edge.source
            ].append(edge)

        return adjacency

    def search(
        self,
        graph: Any,
    ) -> PCGSResult:

        start_time = time.perf_counter()

        source = graph.source_id
        target = graph.target_id

        adjacency = self._build_adjacency(
            graph
        )

        # ------------------------------------------------------
        # Lower bound
        # ------------------------------------------------------

        lower_bound = LowerBound(
            graph
        )

        # ------------------------------------------------------
        # Initial authorization state
        # ------------------------------------------------------

        initial_state = (
            self.policy.initial_state(
                graph
            )
        )

        # ------------------------------------------------------
        # Dominance manager
        # ------------------------------------------------------

        dominance = DominanceManager(
            self.policy
        )

        # ------------------------------------------------------
        # Initial search state
        # ------------------------------------------------------

        start = SearchState(
            node_id=source,
            auth_state=initial_state,
            cost=0.0,
            path_nodes=(source,),
            path_edges=(),
        )

        queue = []

        counter = 0

        heapq.heappush(
            queue,
            (
                0.0,
                counter,
                start,
            ),
        )

        # Register initial state only when
        # dominance pruning is enabled.
        if self.use_dominance:

            dominance.insert(
                source,
                initial_state,
                0.0,
            )

        # ------------------------------------------------------
        # Search statistics
        # ------------------------------------------------------

        best_cost = math.inf
        best_state = None

        expanded_states = 0
        generated_states = 1

        policy_pruned = 0
        dominance_pruned = 0
        bound_pruned = 0
        cycle_pruned = 0

        max_frontier_size = 1

        # ------------------------------------------------------
        # Main search
        # ------------------------------------------------------

        while queue:

            current_cost, _, current = (
                heapq.heappop(queue)
            )

            # --------------------------------------------------
            # Global incumbent bound
            # --------------------------------------------------

            if (
                current_cost
                > best_cost + EPS
            ):
                continue

            expanded_states += 1

            # --------------------------------------------------
            # Goal test
            # --------------------------------------------------

            if current.node_id == target:

                if (
                    current.cost
                    < best_cost - EPS
                ):

                    best_cost = current.cost
                    best_state = current

                continue

            # --------------------------------------------------
            # Bound pruning for current state
            # --------------------------------------------------

            if self.use_bound:

                remaining_lb = (
                    lower_bound.get(
                        current.node_id
                    )
                )

                if (
                    current.cost
                    + remaining_lb
                    >= best_cost - EPS
                ):

                    bound_pruned += 1
                    continue

            # --------------------------------------------------
            # Expand outgoing edges
            # --------------------------------------------------

            for edge in adjacency.get(
                current.node_id,
                [],
            ):

                next_node = edge.target

                # --------------------------------------------------
                # Policy transition
                # --------------------------------------------------

                transition = self.policy.transition(
                    graph,
                    current.auth_state,
                    edge,
                )

                if not transition.allowed:

                    if self.use_policy_pruning:
                        policy_pruned += 1
                        continue

                    next_state = current.auth_state

                else:

                    next_state = transition.next_state

                # --------------------------------------------------
                # Candidate cost
                # --------------------------------------------------

                next_cost = (
                    current.cost
                    + float(edge.cost)
                )

                generated_states += 1

                # --------------------------------------------------
                # Lower-bound pruning
                # --------------------------------------------------

                if self.use_bound:

                    next_lb = lower_bound.get(
                        next_node
                    )

                    if (
                        next_cost
                        + next_lb
                        >= best_cost - EPS
                    ):

                        bound_pruned += 1
                        continue

                # --------------------------------------------------
                # Dominance pruning
                # --------------------------------------------------

                if self.use_dominance:

                    if dominance.is_dominated(
                        next_node,
                        next_state,
                        next_cost,
                    ):

                        dominance_pruned += 1
                        continue

                # --------------------------------------------------
                # Register candidate
                # --------------------------------------------------

                if self.use_dominance:

                    dominance.insert(
                        next_node,
                        next_state,
                        next_cost,
                    )

                # --------------------------------------------------
                # Create search state
                # --------------------------------------------------

                counter += 1

                next_search_state = SearchState(
                    node_id=next_node,
                    auth_state=next_state,
                    cost=next_cost,
                    path_nodes=(
                        current.path_nodes
                        + (next_node,)
                    ),
                    path_edges=(
                        current.path_edges
                        + (edge.edge_id,)
                    ),
                )

                heapq.heappush(
                    queue,
                    (
                        next_cost,
                        counter,
                        next_search_state,
                    ),
                )

                max_frontier_size = max(
                    max_frontier_size,
                    len(queue),
                )

        # ------------------------------------------------------
        # Finish
        # ------------------------------------------------------

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # ------------------------------------------------------
        # No authorized path
        # ------------------------------------------------------

        if best_state is None:

            return PCGSResult(
                found=False,
                cost=None,
                path_nodes=[],
                path_edges=[],
                runtime_ms=runtime_ms,
                expanded_states=expanded_states,
                generated_states=generated_states,
                policy_pruned=policy_pruned,
                dominance_pruned=dominance_pruned,
                bound_pruned=bound_pruned,
                cycle_pruned=cycle_pruned,
                max_frontier_size=max_frontier_size,
                final_auth_state=None,
            )

        # ------------------------------------------------------
        # Authorized path found
        # ------------------------------------------------------

        return PCGSResult(
            found=True,
            cost=best_state.cost,
            path_nodes=list(
                best_state.path_nodes
            ),
            path_edges=list(
                best_state.path_edges
            ),
            runtime_ms=runtime_ms,
            expanded_states=expanded_states,
            generated_states=generated_states,
            policy_pruned=policy_pruned,
            dominance_pruned=dominance_pruned,
            bound_pruned=bound_pruned,
            cycle_pruned=cycle_pruned,
            max_frontier_size=max_frontier_size,
            final_auth_state=(
                best_state.auth_state
            ),
        )


def pcgs_search(
    graph: Any,
) -> PCGSResult:

    return PCGS().search(graph)