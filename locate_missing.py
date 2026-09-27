from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


INPUT = Path(
    "scaling_results/scaling_results.csv"
)


EXPECTED_METHODS = {
    "State-space",
    "PCGS-full",
    "PCGS-no-dominance",
    "PCGS-no-bound",
}


def load_rows():

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def print_schema(rows):

    print()
    print("=" * 100)
    print("CSV SCHEMA")
    print("=" * 100)

    if not rows:
        print("CSV is empty.")
        return

    for key in rows[0].keys():
        print(f"- {key}")


def print_missing_correct(rows):

    print()
    print("=" * 100)
    print("ROWS WITH MISSING CORRECT")
    print("=" * 100)

    found = False

    for i, row in enumerate(rows, start=2):

        correct = row.get(
            "correct",
            "",
        ).strip()

        if correct != "":
            continue

        found = True

        print()
        print(f"CSV row: {i}")

        for key, value in row.items():
            print(
                f"  {key:<24}: {value}"
            )

    if not found:
        print("No rows with missing correct.")


def print_missing_metrics(rows):

    metrics = [
        "generated",
        "expanded",
        "dominance_pruned",
        "bound_pruned",
        "policy_pruned",
        "runtime_ms",
    ]

    print()
    print("=" * 100)
    print("ROWS WITH MISSING METRICS")
    print("=" * 100)

    found = False

    for i, row in enumerate(rows, start=2):

        missing = []

        for metric in metrics:

            if row.get(
                metric,
                "",
            ).strip() == "":
                missing.append(metric)

        if not missing:
            continue

        found = True

        print()
        print(f"CSV row: {i}")

        print(
            f"  size={row.get('size', '')}"
        )

        print(
            f"  scenario={row.get('scenario', '')}"
        )

        print(
            f"  method={row.get('method', '')}"
        )

        print(
            f"  missing={', '.join(missing)}"
        )

        for key, value in row.items():

            if key in metrics:
                continue

            print(
                f"  {key:<24}: {value}"
            )

    if not found:
        print("No rows with missing metrics.")


def print_group_counts(rows):

    print()
    print("=" * 100)
    print("GROUP COUNTS")
    print("=" * 100)

    groups = defaultdict(int)

    for row in rows:

        key = (
            row.get("size", ""),
            row.get("scenario", ""),
            row.get("method", ""),
        )

        groups[key] += 1

    for key in sorted(
        groups,
        key=lambda x: (
            int(x[0]) if x[0].isdigit() else 0,
            x[1],
            x[2],
        ),
    ):

        print(
            f"size={key[0]:>5} "
            f"scenario={key[1]:<24} "
            f"method={key[2]:<24} "
            f"rows={groups[key]}"
        )


def print_missing_methods(rows):

    print()
    print("=" * 100)
    print("GROUPS WITH MISSING METHODS")
    print("=" * 100)

    groups = defaultdict(set)

    for row in rows:

        key = (
            row.get("size", ""),
            row.get("scenario", ""),
        )

        method = row.get(
            "method",
            "",
        ).strip()

        groups[key].add(method)

    found = False

    for key in sorted(
        groups,
        key=lambda x: (
            int(x[0]) if x[0].isdigit() else 0,
            x[1],
        ),
    ):

        methods = groups[key]

        missing = (
            EXPECTED_METHODS - methods
        )

        if not missing:
            continue

        found = True

        print(
            f"size={key[0]:>5} "
            f"scenario={key[1]:<24}"
        )

        print(
            f"  present: {sorted(methods)}"
        )

        print(
            f"  missing: {sorted(missing)}"
        )

    if not found:
        print(
            "No groups with missing methods."
        )


def print_duplicates(rows):

    print()
    print("=" * 100)
    print("POSSIBLE DUPLICATES")
    print("=" * 100)

    counts = Counter()

    for row in rows:

        key = (
            row.get("size", ""),
            row.get("scenario", ""),
            row.get("method", ""),
            row.get("instance_id", ""),
            row.get("seed", ""),
        )

        counts[key] += 1

    found = False

    for key, count in counts.items():

        if count <= 1:
            continue

        found = True

        print(
            f"count={count} "
            f"key={key}"
        )

    if not found:
        print("No duplicates detected.")


def main():

    rows = load_rows()

    print(
        f"Loaded {len(rows)} result rows."
    )

    print_schema(rows)

    print_group_counts(rows)

    print_missing_correct(rows)

    print_missing_metrics(rows)

    print_missing_methods(rows)

    print_duplicates(rows)


if __name__ == "__main__":
    main()