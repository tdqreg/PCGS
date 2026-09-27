from __future__ import annotations

import csv
import heapq
import multiprocessing as mp
import time
from itertools import count
from pathlib import Path
from queue import Empty

from level2_loader import load_level2_graph
from policy_engine import AuthState, PolicyEngine
from pcgs import PCGS


# ============================================================
# Configuration
# ============================================================

ROOT = Path("dataset_level2_scaling")

SIZES = [50, 100, 500, 1000, 5000]

SCENARIOS = [
    "random_state",
    "random_dominance",
    "random_mixed",
]

EPS = 1e-9

# Maximum time allowed for the exhaustive oracle
# per benchmark instance.
ORACLE_TIMEOUT_SECONDS = 60.0

# Maximum time allowed for EACH PCGS configuration
# per benchmark instance.
METHOD_TIMEOUT_SECONDS = 60.0

OUTPUT_DIR = Path("scaling_results")

OUTPUT_FILE = (
    OUTPUT_DIR
    / "scaling_results.csv"
)


# ============================================================
# Authorization-state key
# ============================================================

def state_key(
    node_id: str,
    state: AuthState,
):
    """
    Hashable representation of a graph node + authorization state.
    """

    return (
        node_id,
        tuple(sorted(state.grants)),
        state.clearance,
        state.denied,
    )


# ============================================================
# Exhaustive oracle
# ============================================================

def exhaustive_oracle(graph):
    """
    Independent correctness oracle.

    Dijkstra over the complete authorization-state space.

    No:
        - dominance pruning
        - lower-bound pruning

    is performed.

    Heap entries:

        (cost, tie_breaker, node_id, auth_state)

    The tie-breaker guarantees that AuthState objects are
    never compared by Python.
    """

    policy = PolicyEngine()

    adjacency = {}

    for edge in graph.edges:
        adjacency.setdefault(
            edge.source,
            [],
        ).append(edge)

    initial = policy.initial_state(graph)

    queue = []

    tie_breaker = count()

    heapq.heappush(
        queue,
        (
            0.0,
            next(tie_breaker),
            graph.source_id,
            initial,
        ),
    )

    best = {
        state_key(
            graph.source_id,
            initial,
        ): 0.0
    }

    expanded = 0

    while queue:

        cost, _, node_id, state = (
            heapq.heappop(queue)
        )

        key = state_key(
            node_id,
            state,
        )

        if (
            cost
            > best.get(
                key,
                float("inf"),
            )
            + EPS
        ):
            continue

        if node_id == graph.target_id:
            return (
                True,
                cost,
                expanded,
            )

        expanded += 1

        for edge in adjacency.get(
            node_id,
            [],
        ):

            transition = policy.transition(
                graph,
                state,
                edge,
            )

            if not transition.allowed:
                continue

            next_state = (
                transition.next_state
            )

            next_cost = (
                cost
                + edge.cost
            )

            next_key = state_key(
                edge.target,
                next_state,
            )

            if (
                next_cost
                <
                best.get(
                    next_key,
                    float("inf"),
                )
                - EPS
            ):

                best[next_key] = (
                    next_cost
                )

                heapq.heappush(
                    queue,
                    (
                        next_cost,
                        next(tie_breaker),
                        edge.target,
                        next_state,
                    ),
                )

    return (
        False,
        None,
        expanded,
    )


# ============================================================
# Oracle worker
# ============================================================

def oracle_worker(
    graph,
    result_queue,
):
    """
    Worker process for the exhaustive oracle.
    """

    try:

        start = time.perf_counter()

        found, cost, expanded = (
            exhaustive_oracle(graph)
        )

        runtime = (
            time.perf_counter()
            - start
        ) * 1000.0

        result_queue.put(
            {
                "status": "ok",
                "found": found,
                "cost": cost,
                "expanded": expanded,
                "runtime_ms": runtime,
            }
        )

    except Exception as exc:

        result_queue.put(
            {
                "status": "error",
                "found": None,
                "cost": None,
                "expanded": None,
                "runtime_ms": 0.0,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


# ============================================================
# Oracle timeout wrapper
# ============================================================

def run_oracle_with_timeout(
    graph,
    timeout_seconds=ORACLE_TIMEOUT_SECONDS,
):
    """
    Execute exhaustive_oracle() in a separate process.

    Returns:

        status = "ok"
        status = "timeout"
        status = "error"
    """

    ctx = mp.get_context("spawn")

    result_queue = ctx.Queue()

    process = ctx.Process(
        target=oracle_worker,
        args=(
            graph,
            result_queue,
        ),
    )

    start = time.perf_counter()

    process.start()

    process.join(
        timeout_seconds
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    if process.is_alive():

        process.terminate()
        process.join()

        return {
            "status": "timeout",
            "found": None,
            "cost": None,
            "expanded": None,
            "runtime_ms": elapsed,
            "error": (
                "oracle timeout after "
                f"{timeout_seconds:.1f}s"
            ),
        }

    try:

        result = result_queue.get_nowait()

    except Empty:

        return {
            "status": "error",
            "found": None,
            "cost": None,
            "expanded": None,
            "runtime_ms": elapsed,
            "error": (
                "oracle process exited "
                "without returning a result"
            ),
        }

    if "runtime_ms" not in result:
        result["runtime_ms"] = elapsed

    return result


# ============================================================
# PCGS worker
# ============================================================

def pcgs_worker(
    graph,
    use_dominance,
    use_bound,
    result_queue,
):
    """
    Worker process for one PCGS configuration.
    """

    try:

        solver = PCGS(
            use_policy_pruning=True,
            use_dominance=use_dominance,
            use_bound=use_bound,
        )

        start = time.perf_counter()

        result = solver.search(graph)

        runtime = (
            time.perf_counter()
            - start
        ) * 1000.0

        result_queue.put(
            {
                "status": "ok",
                "found": result.found,
                "cost": result.cost,
                "runtime_ms": runtime,
                "generated": (
                    result.generated_states
                ),
                "expanded": (
                    result.expanded_states
                ),
                "policy_pruned": (
                    result.policy_pruned
                ),
                "dominance_pruned": (
                    result.dominance_pruned
                ),
                "bound_pruned": (
                    result.bound_pruned
                ),
            }
        )

    except Exception as exc:

        result_queue.put(
            {
                "status": "error",
                "found": None,
                "cost": None,
                "runtime_ms": 0.0,
                "generated": None,
                "expanded": None,
                "policy_pruned": None,
                "dominance_pruned": None,
                "bound_pruned": None,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


# ============================================================
# PCGS timeout wrapper
# ============================================================

def run_pcgs_with_timeout(
    graph,
    use_dominance,
    use_bound,
    timeout_seconds=METHOD_TIMEOUT_SECONDS,
):
    """
    Run one PCGS configuration in an isolated process.

    Returns:

        status = "ok"
        status = "timeout"
        status = "error"
    """

    ctx = mp.get_context("spawn")

    result_queue = ctx.Queue()

    process = ctx.Process(
        target=pcgs_worker,
        args=(
            graph,
            use_dominance,
            use_bound,
            result_queue,
        ),
    )

    start = time.perf_counter()

    process.start()

    process.join(
        timeout_seconds
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    if process.is_alive():

        process.terminate()
        process.join()

        return {
            "status": "timeout",
            "found": None,
            "cost": None,
            "runtime_ms": elapsed,
            "generated": None,
            "expanded": None,
            "policy_pruned": None,
            "dominance_pruned": None,
            "bound_pruned": None,
            "error": (
                "method timeout after "
                f"{timeout_seconds:.1f}s"
            ),
        }

    # --------------------------------------------------------
    # Worker returned normally
    # --------------------------------------------------------

    try:

        result = result_queue.get_nowait()

    except Empty:

        return {
            "status": "error",
            "found": None,
            "cost": None,
            "runtime_ms": elapsed,
            "generated": None,
            "expanded": None,
            "policy_pruned": None,
            "dominance_pruned": None,
            "bound_pruned": None,
            "error": (
                "method process exited "
                "without returning a result"
            ),
        }

    if "runtime_ms" not in result:
        result["runtime_ms"] = elapsed

    return result


# ============================================================
# Correctness comparison
# ============================================================

def check_correctness(
    result,
    oracle,
):
    """
    Compare a completed method against a completed oracle.

    Returns:
        1    correct
        0    incorrect
        ""   correctness not evaluable
    """

    if oracle["status"] != "ok":
        return ""

    if result["status"] != "ok":
        return ""

    correct = (
        result["found"]
        == oracle["found"]
    )

    if (
        correct
        and oracle["found"]
        and result["cost"] is not None
    ):

        correct = (
            abs(
                result["cost"]
                - oracle["cost"]
            )
            <= 1e-6
        )

    return int(correct)


# ============================================================
# Process one benchmark instance
# ============================================================

def process_file(
    path: Path,
    index: int,
    total: int,
):

    graph = load_level2_graph(
        path
    )

    size = len(graph.nodes)

    scenario = graph.metadata.get(
        "scenario",
        "unknown",
    )

    print()
    print(
        f"[{index}/{total}] "
        f"size={size} "
        f"scenario={scenario} "
        f"file={path.name}"
    )

    # ========================================================
    # Oracle
    # ========================================================

    print(
        "    Oracle: running...",
        end=" ",
        flush=True,
    )

    oracle = run_oracle_with_timeout(
        graph
    )

    if oracle["status"] == "ok":

        print(
            "OK "
            f"({oracle['runtime_ms']:.2f} ms, "
            f"expanded={oracle['expanded']})"
        )

    elif oracle["status"] == "timeout":

        print(
            "TIMEOUT "
            f"({oracle['runtime_ms'] / 1000.0:.1f}s)"
        )

    else:

        print(
            "ERROR "
            f"({oracle.get('error', 'unknown error')})"
        )

    # ========================================================
    # Configurations
    # ========================================================

    configurations = {
        "PCGS-full": (
            True,
            True,
        ),
        "PCGS-no-dominance": (
            False,
            True,
        ),
        "PCGS-no-bound": (
            True,
            False,
        ),
        "State-space": (
            False,
            False,
        ),
    }

    rows = []

    for name, (
        use_dominance,
        use_bound,
    ) in configurations.items():

        print(
            f"    {name}: running...",
            end=" ",
            flush=True,
        )

        result = run_pcgs_with_timeout(
            graph,
            use_dominance=use_dominance,
            use_bound=use_bound,
        )

        status = result[
            "status"
        ]

        # ----------------------------------------------------
        # Completed
        # ----------------------------------------------------

        if status == "ok":

            print(
                "OK "
                f"({result['runtime_ms']:.2f} ms, "
                f"generated={result['generated']}, "
                f"expanded={result['expanded']})"
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        elif status == "timeout":

            print(
                "TIMEOUT "
                f"({result['runtime_ms'] / 1000.0:.1f}s)"
            )

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        else:

            print(
                "ERROR "
                f"({result.get('error', 'unknown error')})"
            )

        correct = check_correctness(
            result,
            oracle,
        )

        rows.append(
            {
                "file": str(path),
                "size": size,
                "scenario": scenario,
                "method": name,
                "status": status,
                "correct": correct,
                "oracle_status": (
                    oracle["status"]
                ),
                "oracle_found": (
                    int(oracle["found"])
                    if oracle["status"] == "ok"
                    else ""
                ),
                "oracle_cost": (
                    oracle["cost"]
                    if (
                        oracle["status"] == "ok"
                        and oracle["cost"]
                        is not None
                    )
                    else ""
                ),
                "cost": (
                    result["cost"]
                    if (
                        result["status"]
                        == "ok"
                        and result["cost"]
                        is not None
                    )
                    else ""
                ),
                "runtime_ms": (
                    result["runtime_ms"]
                ),
                "generated": (
                    result["generated"]
                    if result["status"]
                    == "ok"
                    else ""
                ),
                "expanded": (
                    result["expanded"]
                    if result["status"]
                    == "ok"
                    else ""
                ),
                "policy_pruned": (
                    result["policy_pruned"]
                    if result["status"]
                    == "ok"
                    else ""
                ),
                "dominance_pruned": (
                    result["dominance_pruned"]
                    if result["status"]
                    == "ok"
                    else ""
                ),
                "bound_pruned": (
                    result["bound_pruned"]
                    if result["status"]
                    == "ok"
                    else ""
                ),
                "oracle_expanded": (
                    oracle["expanded"]
                    if oracle["status"]
                    == "ok"
                    else ""
                ),
                "oracle_runtime_ms": (
                    oracle["runtime_ms"]
                ),
                "error": (
                    result.get(
                        "error",
                        "",
                    )
                ),
                "oracle_error": (
                    oracle.get(
                        "error",
                        "",
                    )
                ),
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    files = []

    for size in SIZES:

        for scenario in SCENARIOS:

            directory = (
                ROOT
                / str(size)
                / scenario
            )

            if not directory.exists():
                continue

            files.extend(
                sorted(
                    directory.glob(
                        "*.json"
                    )
                )
            )

    total = len(files)

    print(
        f"Found {total} benchmark instances."
    )

    print(
        f"Oracle timeout: "
        f"{ORACLE_TIMEOUT_SECONDS:.1f}s"
    )

    print(
        f"Per-method timeout: "
        f"{METHOD_TIMEOUT_SECONDS:.1f}s"
    )

    if total == 0:

        print(
            "No benchmark instances found."
        )

        return

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    all_rows = []

    benchmark_start = (
        time.perf_counter()
    )

    # ========================================================
    # Process all instances
    # ========================================================

    for index, path in enumerate(
        files,
        1,
    ):

        try:

            rows = process_file(
                path,
                index,
                total,
            )

            all_rows.extend(
                rows
            )

        except Exception as exc:

            print()

            print(
                "    INSTANCE ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "    Skipping instance "
                "and continuing..."
            )

            continue

    # ========================================================
    # Write CSV
    # ========================================================

    fieldnames = [
        "file",
        "size",
        "scenario",
        "method",
        "status",
        "correct",
        "oracle_status",
        "oracle_found",
        "oracle_cost",
        "cost",
        "runtime_ms",
        "generated",
        "expanded",
        "policy_pruned",
        "dominance_pruned",
        "bound_pruned",
        "oracle_expanded",
        "oracle_runtime_ms",
        "error",
        "oracle_error",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # ========================================================
    # Summary
    # ========================================================

    elapsed_total = (
        time.perf_counter()
        - benchmark_start
    )

    oracle_ok = sum(
        1
        for row in all_rows
        if row["oracle_status"]
        == "ok"
    )

    oracle_timeout = sum(
        1
        for row in all_rows
        if row["oracle_status"]
        == "timeout"
    )

    oracle_error = sum(
        1
        for row in all_rows
        if row["oracle_status"]
        == "error"
    )

    method_ok = sum(
        1
        for row in all_rows
        if row["status"]
        == "ok"
    )

    method_timeout = sum(
        1
        for row in all_rows
        if row["status"]
        == "timeout"
    )

    method_error = sum(
        1
        for row in all_rows
        if row["status"]
        == "error"
    )

    state_space_timeouts = sum(
        1
        for row in all_rows
        if (
            row["method"]
            == "State-space"
            and row["status"]
            == "timeout"
        )
    )

    correctness_rows = [
        row
        for row in all_rows
        if row["correct"] != ""
    ]

    correct_count = sum(
        int(row["correct"])
        for row in correctness_rows
    )

    print()
    print("=" * 72)
    print(
        "LEVEL-2 SCALING BENCHMARK COMPLETE"
    )
    print("=" * 72)

    print(
        f"Instances discovered:      {total}"
    )

    print(
        f"Rows written:              {len(all_rows)}"
    )

    print()
    print(
        f"Oracle OK:                  {oracle_ok}"
    )

    print(
        f"Oracle timeout:             {oracle_timeout}"
    )

    print(
        f"Oracle error:               {oracle_error}"
    )

    print()
    print(
        f"Method OK:                  {method_ok}"
    )

    print(
        f"Method timeout:             {method_timeout}"
    )

    print(
        f"Method error:               {method_error}"
    )

    print()
    print(
        f"State-space timeouts:       "
        f"{state_space_timeouts}"
    )

    if correctness_rows:

        print()
        print(
            f"Correctness checked:        "
            f"{len(correctness_rows)}"
        )

        print(
            f"Correct:                    "
            f"{correct_count}/"
            f"{len(correctness_rows)} "
            f"("
            f"{100.0 * correct_count / len(correctness_rows):.2f}%"
            f")"
        )

    print()
    print(
        f"Total runtime:              "
        f"{elapsed_total:.1f}s"
    )

    print()
    print(
        f"Results written to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()