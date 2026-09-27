from __future__ import annotations

import heapq
import itertools
import sys
from pathlib import Path
from typing import Any

from level2_loader import load_level2_graph
from policy_engine import PolicyEngine, AuthState
from pcgs import PCGS


# ============================================================
# Configuration
# ============================================================

INSTANCE = Path(
    "dataset_level2_scaling/50/random_mixed/00074.json"
)

EPS = 1e-9


# ============================================================
# Helpers
# ============================================================

def state_repr(state: AuthState) -> str:
    grants = ",".join(sorted(state.grants))
    return (
        f"grants={{{grants}}}, "
        f"clearance={state.clearance}, "
        f"denied={state.denied}"
    )


def build_adjacency(graph):
    adjacency: dict[str, list[Any]] = {
        node_id: [] for node_id in graph.nodes
    }

    for edge in graph.edges:
        adjacency[edge.source].append(edge)

    return adjacency


def edge_repr(edge) -> str:
    return (
        f"{edge.edge_id}: "
        f"{edge.source} -> {edge.target}, "
        f"cost={edge.cost}, "
        f"req={sorted(edge.required_grants)}, "
        f"min_clearance={edge.min_clearance}, "
        f"grant={edge.grant}, "
        f"grant_clearance={edge.grant_clearance}, "
        f"deny={edge.deny}"
    )


# ============================================================
# Path replay / verification
# ============================================================

def verify_path(graph, edge_ids, label):
    """
    Replay a path using the SAME PolicyEngine used by PCGS.

    This is the most important diagnostic:
    if the oracle says a path is valid but this function
    rejects it, the oracle semantics are inconsistent.
    """

    print()
    print("=" * 80)
    print(f"VERIFY PATH: {label}")
    print("=" * 80)

    engine = PolicyEngine()
    state = engine.initial_state(graph)
    total_cost = 0.0

    print(f"Initial state: {state_repr(state)}")
    print()

    if not edge_ids:
        print("Path contains no edges.")
        return False, total_cost

    current_node = graph.source_id

    for index, edge_id in enumerate(edge_ids, start=1):

        if edge_id not in graph.edge_map:
            print(f"[ERROR] Unknown edge: {edge_id}")
            return False, total_cost

        edge = graph.edge_map[edge_id]

        print(f"[{index}] {edge_repr(edge)}")

        if edge.source != current_node:
            print(
                f"  [ERROR] Path discontinuity: "
                f"expected source={current_node}, "
                f"got source={edge.source}"
            )
            return False, total_cost

        before = state

        result = engine.transition(
            graph,
            state,
            edge,
        )

        print(f"  state before : {state_repr(before)}")
        print(f"  allowed      : {result.allowed}")
        print(f"  reason       : {result.reason}")
        print(f"  state after  : {state_repr(result.next_state)}")

        if not result.allowed:
            print()
            print(
                f"[INVALID] Path rejected at edge {edge.edge_id}"
            )
            print(f"Cost accumulated before rejection: {total_cost}")
            return False, total_cost

        total_cost += edge.cost
        state = result.next_state
        current_node = edge.target

        print(f"  total cost   : {total_cost}")
        print()

    if current_node != graph.target_id:
        print(
            f"[INVALID] Path terminates at {current_node}, "
            f"not TARGET={graph.target_id}"
        )
        return False, total_cost

    print(
        f"[VALID] Path reaches TARGET with cost={total_cost}"
    )
    print(f"Final state: {state_repr(state)}")

    return True, total_cost


# ============================================================
# Independent exhaustive state-space oracle
# ============================================================

def oracle_search(graph):
    """
    Exhaustive Dijkstra over (node, AuthState).

    No dominance pruning.
    No lower-bound pruning.

    This is intentionally simple and independent of PCGS.
    """

    engine = PolicyEngine()
    adjacency = build_adjacency(graph)

    initial_state = engine.initial_state(graph)

    counter = itertools.count()

    # heap item:
    # (cost, tie_breaker, node_id, AuthState, path_edge_ids)
    heap = [
        (
            0.0,
            next(counter),
            graph.source_id,
            initial_state,
            (),
        )
    ]

    # IMPORTANT:
    # State identity includes BOTH node and authorization state.
    best_cost: dict[
        tuple[str, AuthState],
        float,
    ] = {
        (graph.source_id, initial_state): 0.0
    }

    expanded = 0

    while heap:

        cost, _, node, state, path = heapq.heappop(heap)

        key = (node, state)

        known = best_cost.get(key)

        if known is None or cost > known + EPS:
            continue

        expanded += 1

        # Goal test
        if node == graph.target_id:
            return {
                "found": True,
                "cost": cost,
                "path": path,
                "expanded": expanded,
                "state": state,
            }

        for edge in adjacency.get(node, []):

            result = engine.transition(
                graph,
                state,
                edge,
            )

            if not result.allowed:
                continue

            next_state = result.next_state
            next_node = edge.target
            next_cost = cost + edge.cost

            next_key = (next_node, next_state)

            old_cost = best_cost.get(next_key)

            if old_cost is None or next_cost < old_cost - EPS:

                best_cost[next_key] = next_cost

                heapq.heappush(
                    heap,
                    (
                        next_cost,
                        next(counter),
                        next_node,
                        next_state,
                        path + (edge.edge_id,),
                    ),
                )

    return {
        "found": False,
        "cost": None,
        "path": (),
        "expanded": expanded,
        "state": None,
    }


# ============================================================
# Run PCGS variants
# ============================================================


def run_pcgs(graph, label, **kwargs):
    print()
    print("=" * 80)
    print(f"RUNNING {label}")
    print("=" * 80)

    solver = PCGS(**kwargs)

    result = solver.search(graph)

    # PCGSResult field names may differ from the benchmark CSV.
    # Print only fields that actually exist.
    print(f"found     : {getattr(result, 'found', None)}")
    print(f"cost      : {getattr(result, 'cost', None)}")

    for field in [
        "generated",
        "expanded",
        "policy_pruned",
        "dominance_pruned",
        "bound_pruned",
        "states_generated",
        "states_expanded",
        "path_edges",
        "path",
    ]:
        if hasattr(result, field):
            print(f"{field:12}: {getattr(result, field)}")

    path = extract_pcgs_path(result)

    if path:
        print(f"normalized path: {path}")
    else:
        print("normalized path: <not available>")

    return result


# ============================================================
# Extract path from PCGS result
# ============================================================

def extract_pcgs_path(result):
    """
    PCGS implementations sometimes expose path_edges or path.
    Normalize both possibilities.
    """

    if hasattr(result, "path_edges"):
        path = result.path_edges

        if path:
            return tuple(
                edge.edge_id if hasattr(edge, "edge_id") else edge
                for edge in path
            )

    if hasattr(result, "path"):
        path = result.path

        if path:
            return tuple(
                edge.edge_id if hasattr(edge, "edge_id") else edge
                for edge in path
            )

    return ()


# ============================================================
# Print graph summary
# ============================================================

def print_graph_summary(graph):
    print()
    print("=" * 80)
    print("GRAPH SUMMARY")
    print("=" * 80)

    print(f"Source : {graph.source_id}")
    print(f"Target : {graph.target_id}")
    print(f"Nodes  : {len(graph.nodes)}")
    print(f"Edges  : {len(graph.edges)}")

    print()
    print("Policy:")
    for key, value in graph.metadata.get("policy", {}).items():
        print(f"  {key}: {value}")

    print()
    print("Benchmark metadata:")
    for key, value in graph.metadata.items():
        if key != "policy":
            print(f"  {key}: {value}")


# ============================================================
# Inspect parallel edges
# ============================================================

def inspect_parallel_edges(graph):
    """
    Check whether multiple edges connect the same source/target.
    This matters because collapsing parallel edges would be incorrect:
    two edges can have different grants/requirements even if their
    endpoints are identical.
    """

    pairs: dict[tuple[str, str], list[Any]] = {}

    for edge in graph.edges:
        key = (edge.source, edge.target)
        pairs.setdefault(key, []).append(edge)

    parallel = {
        key: edges
        for key, edges in pairs.items()
        if len(edges) > 1
    }

    print()
    print("=" * 80)
    print("PARALLEL EDGE CHECK")
    print("=" * 80)

    if not parallel:
        print("No parallel edges.")
        return

    print(f"Parallel endpoint pairs: {len(parallel)}")

    for (source, target), edges in parallel.items():
        print()
        print(f"{source} -> {target}: {len(edges)} edges")

        for edge in edges:
            print(f"  {edge_repr(edge)}")


# ============================================================
# Compare paths
# ============================================================

def print_path_details(graph, path, label):
    print()
    print("=" * 80)
    print(f"PATH DETAILS: {label}")
    print("=" * 80)

    if not path:
        print("<empty path>")
        return

    for i, edge_id in enumerate(path, start=1):

        edge = graph.edge_map.get(edge_id)

        if edge is None:
            print(f"{i}. UNKNOWN EDGE {edge_id}")
        else:
            print(f"{i}. {edge_repr(edge)}")


# ============================================================
# Main
# ============================================================

def main():

    instance = INSTANCE

    if len(sys.argv) > 1:
        instance = Path(sys.argv[1])

    if not instance.exists():
        print(f"[ERROR] Instance not found:")
        print(f"  {instance}")
        sys.exit(1)

    print("=" * 80)
    print("PCGS COUNTEREXAMPLE DEBUGGER")
    print("=" * 80)
    print(f"Instance: {instance}")

    graph = load_level2_graph(instance)

    print_graph_summary(graph)
    inspect_parallel_edges(graph)

    # --------------------------------------------------------
    # 1. Oracle
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING EXHAUSTIVE ORACLE")
    print("=" * 80)

    oracle = oracle_search(graph)

    print(f"found    : {oracle['found']}")
    print(f"cost     : {oracle['cost']}")
    print(f"expanded : {oracle['expanded']}")
    print(f"path     : {oracle['path']}")

    if oracle["state"] is not None:
        print(
            f"final state: "
            f"{state_repr(oracle['state'])}"
        )

    oracle_path = tuple(oracle["path"])

    print_path_details(
        graph,
        oracle_path,
        "ORACLE",
    )

    # --------------------------------------------------------
    # 2. Verify oracle path using PolicyEngine
    # --------------------------------------------------------

    oracle_valid, oracle_verified_cost = verify_path(
        graph,
        oracle_path,
        "ORACLE PATH",
    )

    print()
    print(
        f"Oracle reported cost : {oracle['cost']}"
    )
    print(
        f"Verified path cost   : {oracle_verified_cost}"
    )
    print(
        f"Verified authorization: {oracle_valid}"
    )

    if (
        oracle["found"]
        and oracle_valid
        and abs(
            oracle_verified_cost - oracle["cost"]
        ) > EPS
    ):
        print(
            "[WARNING] Oracle path cost does not match "
            "its reported cost!"
        )

    # --------------------------------------------------------
    # 3. State-space
    # --------------------------------------------------------

    state_space = run_pcgs(
        graph,
        "STATE-SPACE / NO DOMINANCE / NO BOUND",
        use_policy_pruning=True,
        use_dominance=False,
        use_bound=False,
    )

    state_path = extract_pcgs_path(state_space)

    print_path_details(
        graph,
        state_path,
        "STATE-SPACE",
    )

    if state_path:
        verify_path(
            graph,
            state_path,
            "STATE-SPACE PATH",
        )

    # --------------------------------------------------------
    # 4. PCGS no dominance
    # --------------------------------------------------------

    no_dominance = run_pcgs(
        graph,
        "PCGS-NO-DOMINANCE",
        use_policy_pruning=True,
        use_dominance=False,
        use_bound=True,
    )

    no_dom_path = extract_pcgs_path(no_dominance)

    print_path_details(
        graph,
        no_dom_path,
        "PCGS-NO-DOMINANCE",
    )

    if no_dom_path:
        verify_path(
            graph,
            no_dom_path,
            "PCGS-NO-DOMINANCE PATH",
        )

    # --------------------------------------------------------
    # 5. PCGS no bound
    # --------------------------------------------------------

    no_bound = run_pcgs(
        graph,
        "PCGS-NO-BOUND",
        use_policy_pruning=True,
        use_dominance=True,
        use_bound=False,
    )

    no_bound_path = extract_pcgs_path(no_bound)

    print_path_details(
        graph,
        no_bound_path,
        "PCGS-NO-BOUND",
    )

    if no_bound_path:
        verify_path(
            graph,
            no_bound_path,
            "PCGS-NO-BOUND PATH",
        )

    # --------------------------------------------------------
    # 6. Full PCGS
    # --------------------------------------------------------

    full = run_pcgs(
        graph,
        "PCGS-FULL",
        use_policy_pruning=True,
        use_dominance=True,
        use_bound=True,
    )

    full_path = extract_pcgs_path(full)

    print_path_details(
        graph,
        full_path,
        "PCGS-FULL",
    )

    if full_path:
        verify_path(
            graph,
            full_path,
            "PCGS-FULL PATH",
        )

    # --------------------------------------------------------
    # 7. Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"Oracle       : found={oracle['found']}, "
        f"cost={oracle['cost']}"
    )

    print(
        f"State-space  : found={state_space.found}, "
        f"cost={state_space.cost}"
    )

    print(
        f"No dominance : found={no_dominance.found}, "
        f"cost={no_dominance.cost}"
    )

    print(
        f"No bound     : found={no_bound.found}, "
        f"cost={no_bound.cost}"
    )

    print(
        f"PCGS full    : found={full.found}, "
        f"cost={full.cost}"
    )

    print()

    if oracle["found"]:

        oracle_cost = oracle["cost"]

        methods = {
            "State-space": state_space,
            "PCGS-no-dominance": no_dominance,
            "PCGS-no-bound": no_bound,
            "PCGS-full": full,
        }

        for label, result in methods.items():

            if result.cost is None:
                print(
                    f"[MISMATCH] {label}: "
                    f"did not find a path"
                )

            elif abs(result.cost - oracle_cost) > EPS:
                print(
                    f"[MISMATCH] {label}: "
                    f"{result.cost} != oracle {oracle_cost}"
                )

            else:
                print(
                    f"[OK] {label}: "
                    f"matches oracle"
                )

    else:
        print(
            "Oracle found no authorized path."
        )

    print()
    print("=" * 80)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 80)

    if (
        oracle["found"]
        and oracle_valid
        and state_space.cost is not None
        and state_space.cost > oracle["cost"] + EPS
    ):
        print(
            "[IMPORTANT] Oracle has a valid cheaper path, "
            "but State-space does not find it."
        )
        print(
            "This indicates a problem in the state-space "
            "search, graph representation, or policy semantics."
        )

    elif (
        oracle["found"]
        and not oracle_valid
    ):
        print(
            "[IMPORTANT] Oracle's reported path is rejected "
            "by PolicyEngine."
        )
        print(
            "This strongly indicates an Oracle/policy-semantics mismatch."
        )

    elif (
        oracle["found"]
        and state_space.cost is not None
        and abs(state_space.cost - oracle["cost"]) <= EPS
    ):
        print(
            "[OK] State-space agrees with Oracle."
        )
        print(
            "If PCGS variants disagree, investigate "
            "dominance/bound pruning."
        )

    else:
        print(
            "[INFO] Further inspection required."
        )


if __name__ == "__main__":
    main()