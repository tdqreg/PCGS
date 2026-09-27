from __future__ import annotations

import csv
from collections import defaultdict
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


IMPORTANT_FIELDS = [
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
]


def load_rows():

    with INPUT.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def print_summary(rows):

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Total CSV rows: {len(rows)}"
    )

    files = {
        row.get("file", "").strip()
        for row in rows
        if row.get("file", "").strip()
    }

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

    if len(rows) != expected_rows:

        print(
            f"WARNING: row count mismatch "
            f"({len(rows)} vs {expected_rows})"
        )

    else:

        print(
            "Row count is consistent "
            "with 4 methods per file."
        )


def print_missing_correct(rows):

    print()
    print("=" * 100)
    print("MISSING CORRECT")
    print("=" * 100)

    found = False

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        correct = row.get(
            "correct",
            "",
        ).strip()

        if correct != "":
            continue

        found = True

        print()
        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : {row.get('file', '')}"
        )

        print(
            f"size    : {row.get('size', '')}"
        )

        print(
            f"scenario: {row.get('scenario', '')}"
        )

        print(
            f"method  : {row.get('method', '')}"
        )

        print(
            f"status  : {row.get('status', '')}"
        )

        print(
            f"error   : {row.get('**error', '')}"
        )

        print(
            f"oracle  : {row.get('oracle_status', '')}"
        )

    if not found:

        print(
            "No rows with missing correct."
        )


def print_missing_methods(rows):

    print()
    print("=" * 100)
    print("MISSING METHODS")
    print("=" * 100)

    file_methods = defaultdict(set)

    file_info = {}

    for row in rows:

        file = row.get(
            "file",
            "",
        ).strip()

        method = row.get(
            "method",
            "",
        ).strip()

        if not file:
            continue

        file_methods[file].add(
            method
        )

        file_info[file] = (
            row.get("size", ""),
            row.get("scenario", ""),
        )

    found = False

    for file in sorted(file_methods):

        present = file_methods[file]

        missing = (
            EXPECTED_METHODS - present
        )

        if not missing:
            continue

        found = True

        size, scenario = (
            file_info[file]
        )

        print()
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
            f"present : "
            f"{', '.join(sorted(present))}"
        )

        print(
            f"MISSING : "
            f"{', '.join(sorted(missing))}"
        )

    if not found:

        print(
            "No files with missing methods."
        )


def print_incomplete_rows(rows):

    print()
    print("=" * 100)
    print("INCOMPLETE RESULT ROWS")
    print("=" * 100)

    found = False

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        missing = []

        for field in IMPORTANT_FIELDS:

            value = row.get(
                field,
                "",
            ).strip()

            if value == "":
                missing.append(field)

        if not missing:
            continue

        found = True

        print()
        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : "
            f"{row.get('file', '')}"
        )

        print(
            f"size    : "
            f"{row.get('size', '')}"
        )

        print(
            f"scenario: "
            f"{row.get('scenario', '')}"
        )

        print(
            f"method  : "
            f"{row.get('method', '')}"
        )

        print(
            f"status  : "
            f"{row.get('status', '')}"
        )

        print(
            f"MISSING : "
            f"{', '.join(missing)}"
        )

        error = row.get(
            "**error",
            "",
        ).strip()

        oracle_error = row.get(
            "**oracle_error",
            "",
        ).strip()

        if error:
            print(
                f"error   : {error}"
            )

        if oracle_error:
            print(
                f"oracle_error: "
                f"{oracle_error}"
            )

    if not found:

        print(
            "No incomplete result rows."
        )


def print_failed_status(rows):

    print()
    print("=" * 100)
    print("NON-OK STATUS")
    print("=" * 100)

    found = False

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        status = row.get(
            "status",
            "",
        ).strip()

        if status in {
            "",
            "ok",
        }:
            continue

        found = True

        print()
        print(
            f"CSV row: {row_number}"
        )

        print(
            f"file    : "
            f"{row.get('file', '')}"
        )

        print(
            f"method  : "
            f"{row.get('method', '')}"
        )

        print(
            f"status  : {status}"
        )

        print(
            f"error   : "
            f"{row.get('**error', '')}"
        )

    if not found:

        print(
            "No non-ok status rows."
        )


def print_duplicates(rows):

    print()
    print("=" * 100)
    print("DUPLICATE FILE × METHOD")
    print("=" * 100)

    groups = defaultdict(list)

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        key = (
            row.get("file", "").strip(),
            row.get("method", "").strip(),
        )

        groups[key].append(
            row_number
        )

    found = False

    for key, row_numbers in groups.items():

        if len(row_numbers) <= 1:
            continue

        found = True

        print()
        print(
            f"file  : {key[0]}"
        )

        print(
            f"method: {key[1]}"
        )

        print(
            f"rows  : {row_numbers}"
        )

    if not found:

        print(
            "No duplicate file × method pairs."
        )


def print_files_with_bad_method_count(rows):

    print()
    print("=" * 100)
    print("FILES WITH METHOD COUNT != 4")
    print("=" * 100)

    file_methods = defaultdict(list)

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        file = row.get(
            "file",
            "",
        ).strip()

        method = row.get(
            "method",
            "",
        ).strip()

        file_methods[file].append(
            (method, row_number)
        )

    found = False

    for file in sorted(file_methods):

        entries = file_methods[file]

        if len(entries) == 4:
            continue

        found = True

        print()
        print(
            f"file: {file}"
        )

        print(
            f"count: {len(entries)}"
        )

        for method, row_number in entries:

            print(
                f"  row={row_number} "
                f"method={method}"
            )

    if not found:

        print(
            "Every file has exactly 4 rows."
        )


def main():

    rows = load_rows()

    print(
        f"Loaded {len(rows)} result rows."
    )

    print_summary(rows)

    print_missing_correct(rows)

    print_missing_methods(rows)

    print_incomplete_rows(rows)

    print_failed_status(rows)

    print_duplicates(rows)

    print_files_with_bad_method_count(rows)


if __name__ == "__main__":
    main()