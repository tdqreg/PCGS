import argparse
import json
from pathlib import Path

from policy_oracle import (
    Chunk,
    Edge,
    User,
    evaluate_instance,
)


def load_user(data):
    return User(
        user_id=data["user_id"],
        role=data["role"],
        department=data["department"],
        clearance=data["clearance"],
        cross_department=data[
            "cross_department"
        ],
    )


def load_chunk(data):
    return Chunk(
        chunk_id=data["chunk_id"],
        department=data["department"],
        classification=data[
            "classification"
        ],
        source_id=data["source_id"],
        direct_roles=tuple(
            data["direct_roles"]
        ),
        propagation_roles=tuple(
            data["propagation_roles"]
        ),
        deny_roles=tuple(
            data.get(
                "deny_roles",
                [],
            )
        ),
    )


def load_edge(data):
    return Edge(
        edge_id=data["edge_id"],
        source=data["source"],
        target=data["target"],
        relation=data["relation"],
        cost=data["cost"],
    )


def process_instance(raw):

    user = load_user(
        raw["user"]
    )

    chunks = [
        load_chunk(x)
        for x in raw["chunks"]
    ]

    edges = [
        load_edge(x)
        for x in raw["edges"]
    ]

    truth = evaluate_instance(
        user=user,
        chunks=chunks,
        edges=edges,
        source_ids=raw[
            "source_ids"
        ],
        target_id=raw[
            "target_id"
        ],
    )

    return {
        **raw,
        "ground_truth": truth,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="dataset/raw_all.jsonl",
    )

    parser.add_argument(
        "--output",
        default="dataset/ground_truth.jsonl",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    count = 0

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as src,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as dst,
    ):

        for line in src:

            if not line.strip():
                continue

            raw = json.loads(
                line
            )

            result = process_instance(
                raw
            )

            dst.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print(
        f"Generated ground truth "
        f"for {count} instances."
    )


if __name__ == "__main__":
    main()