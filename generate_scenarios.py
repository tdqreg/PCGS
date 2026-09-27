from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from policy_oracle import (
    Chunk,
    Edge,
    User,
)


# ============================================================
# Parameterized difficulty
# ============================================================

@dataclass(frozen=True)
class ScenarioConfig:

    name: str

    min_nodes: int
    max_nodes: int

    min_depth: int
    max_depth: int

    min_decoys: int
    max_decoys: int

    min_branching: int
    max_branching: int

    min_cost: float
    max_cost: float

    min_cost_gap: float
    max_cost_gap: float


CONFIGS = {

    "easy": ScenarioConfig(
        name="easy",

        min_nodes=5,
        max_nodes=10,

        min_depth=1,
        max_depth=3,

        min_decoys=0,
        max_decoys=2,

        min_branching=1,
        max_branching=2,

        min_cost=0.2,
        max_cost=1.5,

        min_cost_gap=0.5,
        max_cost_gap=1.5,
    ),

    "medium": ScenarioConfig(
        name="medium",

        min_nodes=10,
        max_nodes=20,

        min_depth=3,
        max_depth=5,

        min_decoys=2,
        max_decoys=5,

        min_branching=2,
        max_branching=4,

        min_cost=0.1,
        max_cost=1.5,

        min_cost_gap=0.2,
        max_cost_gap=1.0,
    ),

    "hard": ScenarioConfig(
        name="hard",

        min_nodes=20,
        max_nodes=50,

        min_depth=5,
        max_depth=10,

        min_decoys=5,
        max_decoys=15,

        min_branching=3,
        max_branching=6,

        min_cost=0.05,
        max_cost=1.5,

        min_cost_gap=0.05,
        max_cost_gap=0.5,
    ),
}


# ============================================================
# Helpers
# ============================================================

def random_cost(
    rng: random.Random,
    config: ScenarioConfig,
) -> float:

    return round(
        rng.uniform(
            config.min_cost,
            config.max_cost,
        ),
        3,
    )


def random_gap(
    rng: random.Random,
    config: ScenarioConfig,
) -> float:

    return round(
        rng.uniform(
            config.min_cost_gap,
            config.max_cost_gap,
        ),
        3,
    )


def make_user(
    cross_department=False,
) -> User:

    return User(
        user_id="u0",
        role="analyst",
        department="engineering",
        clearance=3,
        cross_department=cross_department,
    )


def chunk(
    chunk_id,
    department="engineering",
    classification=2,
    source_id="doc-A",
    direct_roles=(),
    propagation_roles=(),
    deny_roles=(),
):

    return Chunk(
        chunk_id=chunk_id,
        department=department,
        classification=classification,
        source_id=source_id,
        direct_roles=tuple(direct_roles),
        propagation_roles=tuple(
            propagation_roles
        ),
        deny_roles=tuple(
            deny_roles
        ),
    )


def edge(
    edge_id,
    source,
    target,
    relation,
    cost,
):

    return Edge(
        edge_id=edge_id,
        source=source,
        target=target,
        relation=relation,
        cost=round(cost, 3),
    )


# ============================================================
# Scenario 1
# ============================================================

def safe_single_hop(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
            source_id="doc-S",
        ),

        # IMPORTANT:
        # no direct role, but propagation is allowed.
        chunk(
            "T",
            direct_roles=(),
            propagation_roles=("analyst",),
            source_id="doc-T",
        ),
    ]

    edges = [
        edge(
            "e0",
            "S",
            "T",
            "semantic",
            random_cost(rng, config),
        )
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 1,
            "decoys": 0,
            "branching": 1,
        },
    )


# ============================================================
# Scenario 2
# ============================================================

def safe_multi_hop(
    rng,
    config,
):

    user = make_user()

    depth = rng.randint(
        max(2, config.min_depth),
        max(2, config.max_depth),
    )

    chunks = [
        chunk(
            "S",
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
            source_id="doc-S",
        )
    ]

    node_ids = ["S"]

    for i in range(1, depth):

        node_id = f"X{i}"

        node_ids.append(node_id)

        chunks.append(
            chunk(
                node_id,
                propagation_roles=("analyst",),
                source_id=f"doc-{i}",
            )
        )

    node_ids.append("T")

    chunks.append(
        chunk(
            "T",
            propagation_roles=("analyst",),
            source_id="doc-T",
        )
    )

    relations = [
        "semantic",
        "reference",
        "temporal",
    ]

    edges = []

    for i in range(
        len(node_ids) - 1
    ):

        edges.append(
            edge(
                f"e{i}",
                node_ids[i],
                node_ids[i + 1],
                relations[
                    i % len(relations)
                ],
                random_cost(
                    rng,
                    config,
                ),
            )
        )

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": depth,
            "decoys": 0,
            "branching": 1,
        },
    )


# ============================================================
# Scenario 3
# ============================================================

def multiple_valid_paths(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
        ),

        chunk(
            "A",
            propagation_roles=("analyst",),
        ),

        chunk(
            "B",
            propagation_roles=("analyst",),
        ),

        chunk(
            "T",
            propagation_roles=("analyst",),
        ),
    ]

    cheap = random_cost(
        rng,
        config,
    )

    gap = random_gap(
        rng,
        config,
    )

    expensive = cheap + gap

    edges = [

        edge(
            "eSA",
            "S",
            "A",
            "semantic",
            cheap,
        ),

        edge(
            "eAT",
            "A",
            "T",
            "reference",
            cheap,
        ),

        edge(
            "eSB",
            "S",
            "B",
            "semantic",
            expensive,
        ),

        edge(
            "eBT",
            "B",
            "T",
            "reference",
            expensive,
        ),
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 2,
            "decoys": 1,
            "branching": 2,
            "cost_gap": gap,
        },
    )


# ============================================================
# Scenario 4
# ============================================================

def invalid_cheap_shortcut(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            classification=2,
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
        ),

        chunk(
            "X",
            classification=2,
            propagation_roles=("analyst",),
        ),

        chunk(
            "Y",
            classification=2,
            propagation_roles=("analyst",),
        ),

        # T is propagation eligible.
        # Direct access is still DEFER.
        chunk(
            "T",
            classification=3,
            propagation_roles=("analyst",),
        ),
    ]

    cheap = random_cost(
        rng,
        config,
    )

    gap = random_gap(
        rng,
        config,
    )

    expensive = cheap + gap

    edges = [

        # Cheap but INVALID.
        #
        # hierarchical:
        # X classification = 2
        # T classification = 3
        #
        # Therefore:
        #
        # 3 <= 2 -> false
        edge(
            "eSX",
            "S",
            "X",
            "semantic",
            cheap,
        ),

        edge(
            "eXT",
            "X",
            "T",
            "hierarchical",
            cheap,
        ),

        # Expensive but VALID.
        edge(
            "eSY",
            "S",
            "Y",
            "reference",
            expensive,
        ),

        edge(
            "eYT",
            "Y",
            "T",
            "reference",
            expensive,
        ),
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 2,
            "decoys": 1,
            "branching": 2,
            "cost_gap": gap,
        },
    )


# ============================================================
# Scenario 5
# ============================================================

def explicit_deny(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
        ),

        chunk(
            "X",
            propagation_roles=("analyst",),
        ),

        chunk(
            "T",
            propagation_roles=("analyst",),

            # Explicit policy denial.
            deny_roles=("analyst",),
        ),
    ]

    edges = [

        edge(
            "eSX",
            "S",
            "X",
            "semantic",
            random_cost(
                rng,
                config,
            ),
        ),

        edge(
            "eXT",
            "X",
            "T",
            "reference",
            random_cost(
                rng,
                config,
            ),
        ),
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 2,
            "decoys": 0,
            "branching": 1,
        },
    )


# ============================================================
# Scenario 6
# ============================================================

def mixed_relation_conflict(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            classification=2,
            direct_roles=("analyst",),
            propagation_roles=("analyst",),
        ),

        chunk(
            "X",
            classification=2,
            propagation_roles=("analyst",),
        ),

        chunk(
            "Y",
            classification=2,
            propagation_roles=("analyst",),
        ),

        chunk(
            "T",
            classification=3,
            propagation_roles=("analyst",),
        ),
    ]

    cheap = random_cost(
        rng,
        config,
    )

    gap = random_gap(
        rng,
        config,
    )

    valid = cheap + gap

    edges = [

        # First path is invalid.
        edge(
            "eSX",
            "S",
            "X",
            "semantic",
            cheap,
        ),

        edge(
            "eXT",
            "X",
            "T",
            "hierarchical",
            cheap,
        ),

        # Second path is valid.
        edge(
            "eSY",
            "S",
            "Y",
            "reference",
            valid,
        ),

        edge(
            "eYT",
            "Y",
            "T",
            "reference",
            valid,
        ),
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 2,
            "decoys": 1,
            "branching": 2,
            "cost_gap": gap,
        },
    )


# ============================================================
# Scenario 7
# ============================================================

def no_valid_path(
    rng,
    config,
):

    user = make_user()

    chunks = [

        chunk(
            "S",
            department="engineering",
            propagation_roles=("analyst",),
            direct_roles=("analyst",),
        ),

        # Cross-department propagation is impossible.
        chunk(
            "X",
            department="sales",
            propagation_roles=("analyst",),
        ),

        chunk(
            "T",
            department="sales",
            propagation_roles=("analyst",),
        ),
    ]

    edges = [

        edge(
            "eSX",
            "S",
            "X",
            "semantic",
            random_cost(
                rng,
                config,
            ),
        ),

        edge(
            "eXT",
            "X",
            "T",
            "reference",
            random_cost(
                rng,
                config,
            ),
        ),
    ]

    return (
        user,
        chunks,
        edges,
        ["S"],
        "T",
        {
            "depth": 2,
            "decoys": 0,
            "branching": 1,
        },
    )


SCENARIOS = {
    "safe_single_hop": safe_single_hop,
    "safe_multi_hop": safe_multi_hop,
    "multiple_valid_paths": multiple_valid_paths,
    "invalid_cheap_shortcut": invalid_cheap_shortcut,
    "explicit_deny": explicit_deny,
    "mixed_relation_conflict": mixed_relation_conflict,
    "no_valid_path": no_valid_path,
}


# ============================================================
# Instance generation
# ============================================================

def generate_instance(
    difficulty,
    scenario_name,
    seed,
    index,
):

    rng = random.Random(seed)

    config = CONFIGS[
        difficulty
    ]

    builder = SCENARIOS[
        scenario_name
    ]

    (
        user,
        chunks,
        edges,
        source_ids,
        target_id,
        parameters,
    ) = builder(
        rng,
        config,
    )

    return {

        "instance_id":
            f"{difficulty}-"
            f"{scenario_name}-"
            f"{index:06d}",

        "difficulty":
            difficulty,

        "scenario":
            scenario_name,

        "seed":
            seed,

        "user":
            asdict(user),

        "chunks":
            [asdict(c) for c in chunks],

        "edges":
            [asdict(e) for e in edges],

        "source_ids":
            source_ids,

        "target_id":
            target_id,

        "parameters":
            parameters,

    }


def generate_dataset(
    output_dir,
    count_per_scenario=100,
    seed=42,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "raw_all.jsonl"
    )

    total = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for difficulty in CONFIGS:

            for scenario_index, scenario_name in enumerate(
                SCENARIOS
            ):

                for i in range(
                    count_per_scenario
                ):

                    instance_seed = (
                        seed
                        + scenario_index * 1_000_000
                        + i
                        + list(CONFIGS).index(
                            difficulty
                        ) * 10_000_000
                    )

                    instance = generate_instance(
                        difficulty,
                        scenario_name,
                        instance_seed,
                        i,
                    )

                    f.write(
                        json.dumps(
                            instance,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total += 1

    metadata = {

        "schema_version":
            "pcgs-parameterized-v1",

        "total_instances":
            total,

        "count_per_scenario":
            count_per_scenario,

        "scenarios":
            list(SCENARIOS.keys()),

        "difficulty_configs":
            {
                name: asdict(config)
                for name, config
                in CONFIGS.items()
            },
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Generated {total} instances."
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="dataset",
    )

    parser.add_argument(
        "--count-per-scenario",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_dataset(
        Path(args.output),
        args.count_per_scenario,
        args.seed,
    )


if __name__ == "__main__":
    main()