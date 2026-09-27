from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path(
"scaling_results/scaling_results.csv"
)

EXPECTED_METHODS = {
"PCGS-full",
"PCGS-no-dominance",
"PCGS-no-bound",
"State-space",
}

# Metrics that should exist for a completed successful run

EXECUTION_METRICS = [
"generated",
"expanded",
"policy_pruned",
"dominance_pruned",
"bound_pruned",
"runtime_ms",
]

# Metrics whose existence depends on whether a path exists

PATH_METRICS = [
"cost",
]

ORACLE_METRICS = [
"oracle_cost",
"oracle_expanded",
"oracle_runtime_ms",
]

def is_blank(value):
    return value is None or value.strip() == ""

def get(row, field):
# """
# Safely retrieve a CSV field.

# ```
# This also makes the script robust if the CSV contains
# accidental whitespace around column names.
# """
    value = row.get(field, "")
    if value is None:
        return ""
    return value.strip()

def mean(values):
# """
# Return None for an empty collection.

# ```
# IMPORTANT:
# Missing data must not silently become 0.0.
# """
    if not values:
        return None

    return sum(values) / len(values)

def fmt(value, digits=2):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"

def load_rows():
    with INPUT.open(
    "r",
    encoding="utf-8",
    newline="",
    ) as f:


        reader = csv.DictReader(f)

        rows = list(reader)

        if reader.fieldnames is None:
            raise RuntimeError(
                "CSV has no header."
            )

        return rows

# ----------------------------------------------------------------------

# Classification

# ----------------------------------------------------------------------

def classify_row(row):
# """
# Classify a result row according to the semantics of the experiment.

# ```
# Categories:

# NORMAL
#     Completed execution with the expected metrics.

# VALID_NO_PATH
#     Oracle confirms that no valid path exists.
#     Missing cost is therefore expected.

# ALGORITHM_NO_PATH
#     Oracle has a valid path, but the algorithm completed without
#     finding one. This is NOT a missing metric. It is an algorithmic
#     result and should be evaluated through correctness.

# TIMEOUT
#     Execution was censored by timeout. Missing execution metrics
#     are expected and should not be called missing metrics.

# TRUE_MISSING
#     A metric that should exist is absent.
# """

    status = get(row, "status")
    oracle_status = get(row, "oracle_status")
    oracle_found = get(row, "oracle_found")
    correct = get(row, "correct")
    cost = get(row, "cost")
    oracle_cost = get(row, "oracle_cost")

    # --------------------------------------------------------------
    # Timeout
    # --------------------------------------------------------------

    if status.lower() == "timeout":
        return "TIMEOUT"

    # --------------------------------------------------------------
    # Oracle did not complete
    # --------------------------------------------------------------

    if oracle_status.lower() != "ok":
        return "ORACLE_FAILURE"

    # --------------------------------------------------------------
    # Valid no-path case
    # --------------------------------------------------------------

    if oracle_found == "0":

        # For a no-path instance, oracle_cost being empty is correct.
        #
        # The algorithm should also normally report no path.
        # Whether correct is 1 or 0 is evaluated separately.
        return "VALID_NO_PATH"

    # --------------------------------------------------------------
    # Oracle says a path exists
    # --------------------------------------------------------------

    if oracle_found == "1":

        # Oracle found a valid path, therefore oracle_cost is mandatory.
        if is_blank(oracle_cost):
            return "TRUE_MISSING_ORACLE_COST"

        # Algorithm did not produce a cost.
        #
        # This is not "missing" in the same sense as a broken CSV if
        # the algorithm explicitly completed with no path.
        #
        # We use correct=0 as the signal that the algorithm failed
        # to produce the expected path.
        if is_blank(cost):

            if correct == "0":
                return "ALGORITHM_NO_PATH"

            # correct=1 while cost is absent is internally inconsistent.
            return "TRUE_MISSING_COST"

        return "NORMAL"

    # --------------------------------------------------------------
    # Unknown oracle_found value
    # --------------------------------------------------------------

    return "TRUE_MISSING_ORACLE_FOUND"

# ----------------------------------------------------------------------

# Summary

# ----------------------------------------------------------------------

def print_summary(rows):
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_rows = len(rows)

    files = {
        get(row, "file")
        for row in rows
        if not is_blank(get(row, "file"))
    }

    print(
        f"Total CSV rows: {total_rows}"
    )

    print(
        f"Unique dataset files: {len(files)}"
    )

    print(
        f"Expected methods per file: "
        f"{len(EXPECTED_METHODS)}"
    )

    expected_rows = (
        len(files)
        * len(EXPECTED_METHODS)
    )

    print(
        f"Expected rows from observed files: "
        f"{expected_rows}"
    )

    if total_rows == expected_rows:
        print(
            "Row count is consistent with "
            f"{len(EXPECTED_METHODS)} methods per file."
        )
    else:
        print(
            "WARNING: Row count does NOT match "
            "expected file × method coverage."
        )

# ----------------------------------------------------------------------

# Method coverage

# ----------------------------------------------------------------------

def print_missing_methods(rows):
    print()
    print("=" * 100)
    print("MISSING METHODS")
    print("=" * 100)

    methods_by_file = defaultdict(set)

    for row in rows:

        file = get(row, "file")
        method = get(row, "method")

        methods_by_file[file].add(method)

    found = False

    for file in sorted(methods_by_file):

        missing = (
            EXPECTED_METHODS
            - methods_by_file[file]
        )

        if missing:

            found = True

            print(
                f"{file}"
            )

            print(
                f"  missing: "
                f"{', '.join(sorted(missing))}"
            )

    if not found:
        print(
            "No files with missing methods."
        )


# ----------------------------------------------------------------------

# Duplicate rows

# ----------------------------------------------------------------------

def print_duplicates(rows):
    print()
    print("=" * 100)
    print("DUPLICATE FILE × METHOD")
    print("=" * 100)

    counter = Counter(
        (
            get(row, "file"),
            get(row, "method"),
        )
        for row in rows
    )

    duplicates = {
        key: count
        for key, count in counter.items()
        if count > 1
    }

    if not duplicates:

        print(
            "No duplicate file × method rows."
        )

        return

    for (file, method), count in sorted(
        duplicates.items()
    ):

        print(
            f"{file} | {method} | count={count}"
        )

# ----------------------------------------------------------------------

# Semantic classification

# ----------------------------------------------------------------------

def print_classification(rows):
    print()
    print("=" * 100)
    print("RESULT CLASSIFICATION")
    print("=" * 100)

    counts = Counter(
        classify_row(row)
        for row in rows
    )

    labels = [
        "NORMAL",
        "VALID_NO_PATH",
        "ALGORITHM_NO_PATH",
        "TIMEOUT",
        "ORACLE_FAILURE",
        "TRUE_MISSING_ORACLE_COST",
        "TRUE_MISSING_COST",
        "TRUE_MISSING_ORACLE_FOUND",
    ]

    for label in labels:

        print(
            f"{label:<30} "
            f"{counts[label]:>6}"
        )
# ----------------------------------------------------------------------

# Valid no-path cases

# ----------------------------------------------------------------------

def print_no_path_cases(rows):


    print()
    print("=" * 100)
    print("VALID NO-PATH CASES")
    print("=" * 100)

    groups = defaultdict(list)

    for row in rows:

        if classify_row(row) != "VALID_NO_PATH":
            continue

        key = (
            get(row, "size"),
            get(row, "scenario"),
        )

        groups[key].append(row)

    if not groups:

        print(
            "No valid no-path cases found."
        )

        return

    print(
        f"{'Size':>7} "
        f"{'Scenario':<24} "
        f"{'Rows':>8} "
        f"{'Files':>8}"
    )

    print("-" * 55)

    for key in sorted(
        groups,
        key=lambda x: (
            int(x[0]),
            x[1],
        ),
    ):

        group = groups[key]

        files = {
            get(row, "file")
            for row in group
        }

        print(
            f"{int(key[0]):7d} "
            f"{key[1]:<24} "
            f"{len(group):8d} "
            f"{len(files):8d}"
        )


# ----------------------------------------------------------------------

# True missing metrics

# ----------------------------------------------------------------------

def print_true_missing(rows):
    print()
    print("=" * 100)
    print("TRUE MISSING METRICS")
    print("=" * 100)

    missing = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        classification = classify_row(row)

        # ----------------------------------------------------------
        # Missing oracle cost
        # ----------------------------------------------------------

        if (
            classification
            == "TRUE_MISSING_ORACLE_COST"
        ):

            missing.append(
                (
                    row_number,
                    get(row, "file"),
                    get(row, "size"),
                    get(row, "scenario"),
                    get(row, "method"),
                    "oracle_cost",
                    "oracle_found=1, oracle_status=ok",
                )
            )

        # ----------------------------------------------------------
        # Missing algorithm cost
        # ----------------------------------------------------------

        elif (
            classification
            == "TRUE_MISSING_COST"
        ):

            missing.append(
                (
                    row_number,
                    get(row, "file"),
                    get(row, "size"),
                    get(row, "scenario"),
                    get(row, "method"),
                    "cost",
                    (
                        "correct=1 but cost is empty"
                    ),
                )
            )

        # ----------------------------------------------------------
        # Missing oracle_found
        # ----------------------------------------------------------

        elif (
            classification
            == "TRUE_MISSING_ORACLE_FOUND"
        ):

            missing.append(
                (
                    row_number,
                    get(row, "file"),
                    get(row, "size"),
                    get(row, "scenario"),
                    get(row, "method"),
                    "oracle_found",
                    "oracle_status=ok",
                )
            )

    if not missing:

        print(
            "No true missing metrics found."
        )

        return

    for item in missing:

        (
            row_number,
            file,
            size,
            scenario,
            method,
            metric,
            reason,
        ) = item

        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : {file}"
        )

        print(
            f"size    : {size}"
        )

        print(
            f"scenario: {scenario}"
        )

        print(
            f"method  : {method}"
        )

        print(
            f"metric  : {metric}"
        )

        print(
            f"reason  : {reason}"
        )

        print()

# ----------------------------------------------------------------------

# Timeout / censored executions

# ----------------------------------------------------------------------

def print_timeouts(rows):
    print()
    print("=" * 100)
    print("TIMEOUT / CENSORED EXECUTIONS")
    print("=" * 100)

    timeouts = [
        (row_number, row)
        for row_number, row in enumerate(
            rows,
            start=2,
        )
        if classify_row(row) == "TIMEOUT"
    ]

    if not timeouts:

        print(
            "No timeout rows."
        )

        return

    print(
        f"Total timeout rows: "
        f"{len(timeouts)}"
    )

    for row_number, row in timeouts:

        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : {get(row, 'file')}"
        )

        print(
            f"size    : {get(row, 'size')}"
        )

        print(
            f"scenario: {get(row, 'scenario')}"
        )

        print(
            f"method  : {get(row, 'method')}"
        )

        print(
            f"oracle  : {get(row, 'oracle_status')}"
        )

        print()

# ----------------------------------------------------------------------

# Internal consistency

# ----------------------------------------------------------------------

def print_inconsistencies(rows):

    print()
    print("=" * 100)
    print("SEMANTIC INCONSISTENCIES")
    print("=" * 100)

    inconsistencies = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        status = get(row, "status")
        oracle_status = get(row, "oracle_status")
        oracle_found = get(row, "oracle_found")
        correct = get(row, "correct")
        cost = get(row, "cost")
        oracle_cost = get(row, "oracle_cost")

        # Timeout is intentionally excluded from ordinary
        # completeness checks.
        if status.lower() == "timeout":
            continue

        # Oracle failure is reported separately.
        if oracle_status.lower() != "ok":
            continue

        # ----------------------------------------------------------
        # oracle_found = 1 requires oracle_cost
        # ----------------------------------------------------------

        if (
            oracle_found == "1"
            and is_blank(oracle_cost)
        ):

            inconsistencies.append(
                (
                    row_number,
                    row,
                    "oracle_found=1 but oracle_cost is empty",
                )
            )

        # ----------------------------------------------------------
        # oracle_found = 0 means oracle_cost should be empty
        # ----------------------------------------------------------

        if (
            oracle_found == "0"
            and not is_blank(oracle_cost)
        ):

            inconsistencies.append(
                (
                    row_number,
                    row,
                    "oracle_found=0 but oracle_cost is present",
                )
            )

        # ----------------------------------------------------------
        # If correct=1 and oracle_found=1, algorithm should have
        # a path and therefore a cost.
        # ----------------------------------------------------------

        if (
            correct == "1"
            and oracle_found == "1"
            and is_blank(cost)
        ):

            inconsistencies.append(
                (
                    row_number,
                    row,
                    "correct=1 and oracle_found=1 but cost is empty",
                )
            )

        # ----------------------------------------------------------
        # If correct=1 and oracle_found=0, algorithm should also
        # have correctly established no path.
        #
        # We do not require cost here.
        # ----------------------------------------------------------

        # No cost check intentionally.

    if not inconsistencies:

        print(
            "No semantic inconsistencies found."
        )

        return

    for row_number, row, reason in inconsistencies:

        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : {get(row, 'file')}"
        )

        print(
            f"size    : {get(row, 'size')}"
        )

        print(
            f"scenario: {get(row, 'scenario')}"
        )

        print(
            f"method  : {get(row, 'method')}"
        )

        print(
            f"reason  : {reason}"
        )

        print()


# ----------------------------------------------------------------------

# Files with != 4 rows

# ----------------------------------------------------------------------

def print_file_row_counts(rows):
    print()
    print("=" * 100)
    print("FILES WITH METHOD COUNT != 4")
    print("=" * 100)

    counts = Counter(
        get(row, "file")
        for row in rows
    )

    bad = {
        file: count
        for file, count in counts.items()
        if count != len(EXPECTED_METHODS)
    }

    if not bad:

        print(
            "Every file has exactly "
            f"{len(EXPECTED_METHODS)} rows."
        )

        return

    for file, count in sorted(bad.items()):

        print(
            f"{file}: {count} rows"
        )


# ----------------------------------------------------------------------

# Main

# ----------------------------------------------------------------------

def main():
    rows = load_rows()

    print(
        f"Loaded {len(rows)} result rows."
    )

    print_summary(rows)

    print_missing_methods(rows)

    print_file_row_counts(rows)

    print_duplicates(rows)

    print_classification(rows)

    print_no_path_cases(rows)

    print_true_missing(rows)

    print_timeouts(rows)

    print_inconsistencies(rows)

if __name__ == "__main__":
    main()