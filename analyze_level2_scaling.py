from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT = Path(
    "scaling_results/scaling_results.csv"
)

METHODS = [
    "PCGS-full",
    "PCGS-no-dominance",
    "PCGS-no-bound",
    "State-space",
]

SIZES = [50, 100, 500, 1000, 5000]

SCENARIOS = [
    "random_dominance",
    "random_mixed",
    "random_state",
]

# Completed executions include both:
#   NORMAL      : oracle found a valid path
#   VALID_NO_PATH: oracle correctly determined that no path exists
#
# TIMEOUT and ORACLE_FAILURE are excluded from metric means.
COMPLETED_STATES = {
    "NORMAL",
    "VALID_NO_PATH",
}

PATH_STATES = {
    "NORMAL",
}


def classify_row(row):
    """
    Classify one result row.

    Priority:
        1. TIMEOUT
        2. ORACLE_FAILURE
        3. VALID_NO_PATH
        4. NORMAL
        5. INVALID
    """

    status = (row.get("status") or "").strip().lower()
    oracle_status = (row.get("oracle_status") or "").strip().lower()

    if status == "timeout":
        return "TIMEOUT"

    if oracle_status and oracle_status != "ok":
        return "ORACLE_FAILURE"

    oracle_found = (row.get("oracle_found") or "").strip()

    if oracle_found == "0":
        return "VALID_NO_PATH"

    if oracle_found == "1":
        return "NORMAL"

    return "INVALID"


def to_float(value):
    """
    Convert CSV value to float.

    Blank or malformed values return None.
    Never silently converts missing data to zero.
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    """
    Convert CSV value to int.

    Blank or malformed values return None.
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values):
    """
    Mean without silently converting an empty set to zero.
    """

    if not values:
        return None

    return sum(values) / len(values)


def pct(numerator, denominator):
    if denominator == 0:
        return None

    return 100.0 * numerator / denominator


def fmt(value, digits=2):
    """
    Format numeric values.

    None is printed as '-'.
    """

    if value is None:
        return "-"

    return f"{value:.{digits}f}"


def load_rows():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    with INPUT.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        row["_state"] = classify_row(row)

    return rows


def group_by(rows, fields):
    groups = defaultdict(list)

    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        groups[key].append(row)

    return groups


def state_statistics(rows):
    """
    Execution status statistics.

    COMPLETED = NORMAL + VALID_NO_PATH
    """

    total = len(rows)

    normal = sum(
        1
        for row in rows
        if row["_state"] == "NORMAL"
    )

    no_path = sum(
        1
        for row in rows
        if row["_state"] == "VALID_NO_PATH"
    )

    timeout = sum(
        1
        for row in rows
        if row["_state"] == "TIMEOUT"
    )

    oracle_failure = sum(
        1
        for row in rows
        if row["_state"] == "ORACLE_FAILURE"
    )

    invalid = sum(
        1
        for row in rows
        if row["_state"] == "INVALID"
    )

    completed = normal + no_path

    return {
        "total": total,
        "normal": normal,
        "no_path": no_path,
        "timeout": timeout,
        "oracle_failure": oracle_failure,
        "invalid": invalid,
        "completed": completed,
        "completion_rate": pct(completed, total),
        "timeout_rate": pct(timeout, total),
        "no_path_rate": pct(no_path, total),
    }


def correctness_statistics(rows):
    """
    Correctness is evaluated over completed executions.

    NORMAL and VALID_NO_PATH are both valid completed cases.
    TIMEOUT is excluded because no final algorithm decision exists.
    """

    completed_rows = [
        row
        for row in rows
        if row["_state"] in COMPLETED_STATES
    ]

    evaluated = 0
    correct = 0
    missing = 0

    for row in completed_rows:

        value = (row.get("correct") or "").strip()

        if value == "":
            missing += 1
            continue

        try:
            value = int(float(value))
        except (TypeError, ValueError):
            missing += 1
            continue

        evaluated += 1

        if value == 1:
            correct += 1

    return {
        "completed": len(completed_rows),
        "evaluated": evaluated,
        "correct": correct,
        "missing": missing,
        "accuracy": pct(correct, evaluated),
    }


def metric_values(
    rows,
    field,
    states=COMPLETED_STATES,
):
    """
    Return numeric metric values for the requested execution states.

    Missing or malformed values are ignored from the mean,
    but they are NOT converted to zero.
    """

    values = []

    for row in rows:

        if row["_state"] not in states:
            continue

        value = to_float(row.get(field))

        if value is not None:
            values.append(value)

    return values


def metric_summary(
    rows,
    field,
    states=COMPLETED_STATES,
):
    values = metric_values(
        rows,
        field,
        states=states,
    )

    return {
        "n": len(values),
        "mean": mean(values),
    }


def print_execution_status(rows):
    print("\n" + "=" * 100)
    print("EXECUTION STATUS")
    print("=" * 100)

    groups = group_by(
        rows,
        ["size", "scenario", "method"],
    )

    for size in SIZES:

        print(f"\nSIZE = {size}")

        for scenario in SCENARIOS:

            print(f"\n  SCENARIO = {scenario}")

            for method in METHODS:

                group = groups.get(
                    (str(size), scenario, method),
                    [],
                )

                stats = state_statistics(group)

                print(
                    f"    {method:<20}"
                    f" Total={stats['total']:>3}"
                    f"  Normal={stats['normal']:>3}"
                    f"  NoPath={stats['no_path']:>3}"
                    f"  Timeout={stats['timeout']:>3}"
                    f"  Completed={stats['completed']:>3}"
                    f"  Completion={fmt(stats['completion_rate'])}%"
                    f"  TimeoutRate={fmt(stats['timeout_rate'])}%"
                )


def print_correctness(rows):
    print("\n" + "=" * 100)
    print("CORRECTNESS")
    print("=" * 100)

    groups = group_by(
        rows,
        ["size", "scenario", "method"],
    )

    for size in SIZES:

        print(f"\nSIZE = {size}")

        for scenario in SCENARIOS:

            print(f"\n  SCENARIO = {scenario}")

            for method in METHODS:

                group = groups.get(
                    (str(size), scenario, method),
                    [],
                )

                stats = correctness_statistics(group)

                print(
                    f"    {method:<20}"
                    f" Completed={stats['completed']:>3}"
                    f" Eval={stats['evaluated']:>3}"
                    f" Correct={stats['correct']:>3}"
                    f" Missing={stats['missing']:>3}"
                    f" Accuracy={fmt(stats['accuracy'])}%"
                )


def print_scaling_analysis(rows):
    print("\n" + "=" * 100)
    print("SCALING ANALYSIS")
    print("=" * 100)

    groups = group_by(
        rows,
        ["size", "method"],
    )

    for size in SIZES:

        print(f"\nSIZE = {size}")

        print(
            f"{'Method':<20}"
            f"{'N':>6}"
            f"{'Done':>7}"
            f"{'Timeout':>9}"
            f"{'Completion':>12}"
            f"{'Runtime(ms)':>16}"
            f"{'Generated':>16}"
            f"{'Expanded':>16}"
            f"{'Policy':>14}"
            f"{'Dominance':>14}"
            f"{'Bound':>14}"
        )

        print("-" * 150)

        for method in METHODS:

            group = groups.get(
                (str(size), method),
                [],
            )

            stats = state_statistics(group)

            runtime = metric_summary(
                group,
                "runtime_ms",
                states=COMPLETED_STATES,
            )

            generated = metric_summary(
                group,
                "generated",
                states=COMPLETED_STATES,
            )

            expanded = metric_summary(
                group,
                "expanded",
                states=COMPLETED_STATES,
            )

            policy = metric_summary(
                group,
                "policy_pruned",
                states=COMPLETED_STATES,
            )

            dominance = metric_summary(
                group,
                "dominance_pruned",
                states=COMPLETED_STATES,
            )

            bound = metric_summary(
                group,
                "bound_pruned",
                states=COMPLETED_STATES,
            )

            print(
                f"{method:<20}"
                f"{stats['total']:>6}"
                f"{stats['completed']:>7}"
                f"{stats['timeout']:>9}"
                f"{fmt(stats['completion_rate']):>11}%"
                f"{fmt(runtime['mean']):>10} (n={runtime['n']:<3})"
                f"{fmt(generated['mean']):>10} (n={generated['n']:<3})"
                f"{fmt(expanded['mean']):>10} (n={expanded['n']:<3})"
                f"{fmt(policy['mean']):>9} (n={policy['n']:<3})"
                f"{fmt(dominance['mean']):>9} (n={dominance['n']:<3})"
                f"{fmt(bound['mean']):>9} (n={bound['n']:<3})"
            )


def print_path_cost_analysis(rows):
    print("\n" + "=" * 100)
    print("PATH COST ANALYSIS")
    print("=" * 100)

    groups = group_by(
        rows,
        ["size", "scenario", "method"],
    )

    for size in SIZES:

        print(f"\nSIZE = {size}")

        for scenario in SCENARIOS:

            print(f"\n  SCENARIO = {scenario}")

            print(
                f"    {'Method':<20}"
                f"{'PathN':>8}"
                f"{'OracleCost':>15}"
                f"{'AlgCost':>15}"
                f"{'Gap':>15}"
            )

            print("    " + "-" * 75)

            for method in METHODS:

                group = groups.get(
                    (str(size), scenario, method),
                    [],
                )

                # Cost is meaningful only when a valid path exists.
                oracle_values = metric_values(
                    group,
                    "oracle_cost",
                    states=PATH_STATES,
                )

                alg_values = metric_values(
                    group,
                    "cost",
                    states=PATH_STATES,
                )

                gap_values = []

                for row in group:

                    if row["_state"] not in PATH_STATES:
                        continue

                    oracle_cost = to_float(
                        row.get("oracle_cost")
                    )

                    alg_cost = to_float(
                        row.get("cost")
                    )

                    if (
                        oracle_cost is not None
                        and alg_cost is not None
                    ):
                        gap_values.append(
                            alg_cost - oracle_cost
                        )

                path_n = len(
                    [
                        row
                        for row in group
                        if row["_state"] == "NORMAL"
                    ]
                )

                print(
                    f"    {method:<20}"
                    f"{path_n:>8}"
                    f"{fmt(mean(oracle_values)):>15}"
                    f"{fmt(mean(alg_values)):>15}"
                    f"{fmt(mean(gap_values)):>15}"
                )


def print_dominance_reduction(rows):
    """
    Paired-instance analysis for dominance pruning.

    For each dataset instance, compare:
        State-space generated
        vs.
        PCGS-full generated

    Only instances where BOTH methods have:
        - a completed execution
        - a valid numeric 'generated' value

    are included.

    Per-instance reduction:

        reduction =
            (state_space_generated - pcgs_generated)
            / state_space_generated * 100

    Report:
        N paired
        Mean reduction
        Median reduction
        Min reduction
        Max reduction
    """

    print("\n" + "=" * 100)
    print("DOMINANCE PRUNING REDUCTION - PAIRED INSTANCE ANALYSIS")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Build lookup:
    #
    #   (file, method) -> row
    #
    # The dataset filename uniquely identifies the same graph instance
    # across all four methods.
    # ------------------------------------------------------------------

    lookup = {}

    for row in rows:

        file_id = row.get("file", "")
        method = row.get("method", "")

        lookup[(file_id, method)] = row

    # ------------------------------------------------------------------
    # Group files by size and scenario.
    # ------------------------------------------------------------------

    groups = defaultdict(list)

    for row in rows:

        file_id = row.get("file", "")
        size = row.get("size", "")
        scenario = row.get("scenario", "")

        groups[(size, scenario)].append(file_id)

    for size in SIZES:

        print(f"\nSIZE = {size}")

        for scenario in SCENARIOS:

            file_ids = groups.get(
                (str(size), scenario),
                [],
            )

            # Remove duplicates while preserving deterministic order.
            file_ids = sorted(set(file_ids))

            reductions = []

            for file_id in file_ids:

                pcgs_row = lookup.get(
                    (file_id, "PCGS-full")
                )

                state_row = lookup.get(
                    (file_id, "State-space")
                )

                if pcgs_row is None or state_row is None:
                    continue

                # Both executions must be completed.
                if pcgs_row["_state"] not in COMPLETED_STATES:
                    continue

                if state_row["_state"] not in COMPLETED_STATES:
                    continue

                pcgs_generated = to_float(
                    pcgs_row.get("generated")
                )

                state_generated = to_float(
                    state_row.get("generated")
                )

                # Missing metrics are excluded, never treated as zero.
                if (
                    pcgs_generated is None
                    or state_generated is None
                ):
                    continue

                # Avoid division by zero.
                if state_generated <= 0:
                    continue

                reduction = (
                    100.0
                    * (
                        state_generated
                        - pcgs_generated
                    )
                    / state_generated
                )

                reductions.append(reduction)

            if not reductions:

                print(
                    f"  {scenario:<20}"
                    f" N paired=0"
                    f"  Mean=-"
                    f"  Median=-"
                    f"  Min=-"
                    f"  Max=-"
                )

                continue

            reductions.sort()

            print(
                f"  {scenario:<20}"
                f" N paired={len(reductions):>3}"
                f"  Mean={mean(reductions):>8.2f}%"
                f"  Median={__import__('statistics').median(reductions):>8.2f}%"
                f"  Min={min(reductions):>8.2f}%"
                f"  Max={max(reductions):>8.2f}%"
            )

def print_method_summary(rows):
    print("\n" + "=" * 100)
    print("METHOD SUMMARY")
    print("=" * 100)

    groups = group_by(
        rows,
        ["method"],
    )

    print(
        f"{'Method':<20}"
        f"{'N':>7}"
        f"{'Done':>7}"
        f"{'Timeout':>9}"
        f"{'Completion':>12}"
        f"{'NoPath':>10}"
        f"{'Correctness':>13}"
        f"{'Runtime(ms)':>16}"
    )

    print("-" * 100)

    for method in METHODS:

        group = groups.get(
            (method,),
            [],
        )

        state = state_statistics(group)
        correctness = correctness_statistics(group)

        runtime = metric_summary(
            group,
            "runtime_ms",
            states=COMPLETED_STATES,
        )

        print(
            f"{method:<20}"
            f"{state['total']:>7}"
            f"{state['completed']:>7}"
            f"{state['timeout']:>9}"
            f"{fmt(state['completion_rate']):>11}%"
            f"{fmt(state['no_path_rate']):>9}%"
            f"{fmt(correctness['accuracy']):>12}%"
            f"{fmt(runtime['mean']):>10} (n={runtime['n']})"
        )


def print_metric_completeness(rows):
    """
    Check whether completed rows contain the execution metrics.

    This is intentionally separate from the main analysis so that
    missing execution metrics are visible instead of becoming zero.
    """

    fields = [
        "runtime_ms",
        "generated",
        "expanded",
        "policy_pruned",
        "dominance_pruned",
        "bound_pruned",
    ]

    print("\n" + "=" * 100)
    print("EXECUTION METRIC COMPLETENESS")
    print("=" * 100)

    groups = group_by(
        rows,
        ["size", "scenario", "method"],
    )

    any_missing = False

    for size in SIZES:

        for scenario in SCENARIOS:

            for method in METHODS:

                group = groups.get(
                    (str(size), scenario, method),
                    [],
                )

                completed = [
                    row
                    for row in group
                    if row["_state"] in COMPLETED_STATES
                ]

                for field in fields:

                    missing = sum(
                        1
                        for row in completed
                        if to_float(row.get(field)) is None
                    )

                    if missing > 0:

                        any_missing = True

                        print(
                            f"  size={size:<5}"
                            f" scenario={scenario:<20}"
                            f" method={method:<20}"
                            f" field={field:<20}"
                            f" missing={missing}"
                            f"/{len(completed)}"
                        )

    if not any_missing:
        print(
            "No missing execution metrics among completed executions."
        )


def main():

    rows = load_rows()

    print(
        f"Loaded {len(rows)} result rows."
    )

    print_execution_status(rows)
    print_correctness(rows)
    print_scaling_analysis(rows)
    print_path_cost_analysis(rows)
    print_dominance_reduction(rows)
    print_method_summary(rows)
    print_metric_completeness(rows)


if __name__ == "__main__":
    main()