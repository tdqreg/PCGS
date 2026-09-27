# generate_dataset.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from graph_generator import GenericGraphGenerator
from graph_validator import GraphValidator
from scenario_specs import (
    SCENARIOS,
    SIZE_PROFILES,
    get_size_profile
)


MAX_ATTEMPTS = 100


def serialize_graph(graph) -> dict:

    return {
        "source_id": graph.source_id,
        "target_id": graph.target_id,

        "nodes": [
            asdict(node)
            for node in graph.nodes.values()
        ],

        "edges": [
            asdict(edge)
            for edge in graph.edges
        ],

        "valid_paths": [
            asdict(path)
            for path in graph.valid_paths
        ],

        "invalid_paths": [
            asdict(path)
            for path in graph.invalid_paths
        ],

        "metadata": graph.metadata,
    }


def generate_one(
        num_nodes: int,
        scenario_name: str,
        seed: int,
    ):
        profile = get_size_profile(num_nodes)
        scenario = SCENARIOS[scenario_name]

        validator = GraphValidator(
            profile=profile,
            scenario=scenario,
        )

        last_errors = []

        for attempt in range(MAX_ATTEMPTS):
            attempt_seed = seed + attempt

            generator = GenericGraphGenerator(
                profile=profile,
                scenario=scenario,
                seed=attempt_seed,
            )

            graph = generator.generate()

            validation_result = validator.validate(graph)

            if validation_result.valid:
                return graph

            last_errors = validation_result.errors

            # In lỗi ở một số lần thử đầu tiên
            if attempt < 3:
                print(
                    f"\n[INVALID] "
                    f"attempt={attempt + 1}, "
                    f"seed={attempt_seed}, "
                    f"nodes={num_nodes}, "
                    f"scenario={scenario_name}"
                )

                for error in validation_result.errors:
                    print(f"  - {error}")

        error_message = (
            "Unable to generate valid graph after "
            f"{MAX_ATTEMPTS} attempts. "
            f"nodes={num_nodes}, "
            f"scenario={scenario_name}, "
            f"initial_seed={seed}\n"
            "Last validation errors:\n"
            + "\n".join(
                f"  - {error}"
                for error in last_errors
            )
        )

        raise RuntimeError(error_message)

def save_graph(
    graph,
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serialize_graph(graph),
            file,
            ensure_ascii=False,
            indent=2,
        )


def generate_dataset(
    output_dir: str,
    sizes: list[int],
    scenarios: list[str],
    instances_per_configuration: int,
    base_seed: int = 42,
):

    output_root = Path(output_dir)

    instance_id = 0

    for num_nodes in sizes:

        for scenario_name in scenarios:

            for index in range(
                instances_per_configuration
            ):

                seed = (
                    base_seed
                    + instance_id * 1000
                    + index
                )

                graph = generate_one(
                    num_nodes=num_nodes,
                    scenario_name=scenario_name,
                    seed=seed,
                )

                output_path = (
                    output_root
                    / str(num_nodes)
                    / scenario_name
                    / f"{index:05d}.json"
                )

                save_graph(
                    graph,
                    output_path,
                )

                instance_id += 1


if __name__ == "__main__":

    generate_dataset(
        output_dir="dataset",
        sizes=[10, 50, 100, 500, 1000],
        scenarios=list(SCENARIOS.keys()),
        instances_per_configuration=100,
    )