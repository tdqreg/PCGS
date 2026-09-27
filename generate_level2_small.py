from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


OUTPUT_DIR = Path("dataset_level2_small")
INSTANCES_PER_SCENARIO = 100
BASE_SEED = 42


SCENARIO_SEEDS = {
    "state_dependent_grant": 101,
    "competing_authorization_states": 202,
    "dominated_authorization_states": 303,
    "policy_dead_end": 404,
    "deep_dominance": 505,
    "bound_pruning": 606,
}


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def node(
    node_id: str,
    node_type: str = "INTERMEDIATE",
) -> dict:
    return {
        "node_id": node_id,
        "type": node_type,
    }


def edge(
    edge_id: str,
    source: str,
    target: str,
    cost: float,
    required_grants: list[str] | None = None,
    min_clearance: int = 0,
    grant: str | None = None,
    grant_clearance: int | None = None,
) -> dict:

    result = {
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "cost": float(cost),
        "required_grants": (
            list(required_grants)
            if required_grants is not None
            else []
        ),
        "min_clearance": int(min_clearance),
    }

    if grant is not None:
        result["grant"] = grant

    if grant_clearance is not None:
        result["grant_clearance"] = int(
            grant_clearance
        )

    return result


def make_graph(
    nodes,
    edges,
    scenario,
    expected_found,
    expected_cost,
    description,
):
    source_id = "SOURCE"
    target_id = "TARGET"

    # --------------------------------------------------------------
    # Ensure SOURCE exists.
    # --------------------------------------------------------------

    node_ids = {
        item["node_id"]
        for item in nodes
    }

    if source_id not in node_ids:
        nodes.insert(
            0,
            node(
                source_id,
                "SOURCE",
            ),
        )

    # --------------------------------------------------------------
    # Ensure TARGET exists.
    # --------------------------------------------------------------

    node_ids = {
        item["node_id"]
        for item in nodes
    }

    if target_id not in node_ids:
        nodes.append(
            node(
                target_id,
                "TARGET",
            ),
        )

    # --------------------------------------------------------------
    # Normalize SOURCE / TARGET types.
    # --------------------------------------------------------------

    for item in nodes:

        if item["node_id"] == source_id:
            item["type"] = "SOURCE"

        elif item["node_id"] == target_id:
            item["type"] = "TARGET"

    return {
        "nodes": nodes,
        "edges": edges,

        "source_id": source_id,
        "target_id": target_id,

        "metadata": {
            "level2": True,
            "scenario": scenario,

            "expected": {
                "found": expected_found,
                "cost": expected_cost,
            },

            "description": description,

            "policy": {
                "authorization_semantics":
                    "explicit_propagation_grant",

                "path_semantics":
                    "every_edge",

                "search_objective":
                    "minimum_cost",

                "monotonic_authorization":
                    True,
            },
        },
    }


# ----------------------------------------------------------------------
# Scenario 1
# ----------------------------------------------------------------------

def generate_state_dependent_grant(
    rng: random.Random,
) -> dict:

    nodes = [
        node("A"),
        node("B"),
        node("DIRECT"),
    ]

    edges = [

        # Authorization-producing path.
        edge(
            "e1",
            "SOURCE",
            "A",
            4.0,
            grant="manager",
        ),

        edge(
            "e2",
            "A",
            "TARGET",
            6.0,
            required_grants=["manager"],
        ),

        # Structurally cheaper path.
        edge(
            "e3",
            "SOURCE",
            "DIRECT",
            1.0,
        ),

        edge(
            "e4",
            "DIRECT",
            "TARGET",
            1.0,
            required_grants=["manager"],
        ),
    ]

    return make_graph(
        nodes,
        edges,
        "state_dependent_grant",
        True,
        10.0,
        (
            "A cheap direct route reaches the target structurally, "
            "but the final edge requires a manager grant that is "
            "obtained only through the alternative route."
        ),
    )


# ----------------------------------------------------------------------
# Scenario 2
# ----------------------------------------------------------------------

def generate_competing_authorization_states(
    rng: random.Random,
) -> dict:

    nodes = [
        node("A"),
        node("B"),
        node("X"),
    ]

    edges = [

        edge(
            "e1",
            "SOURCE",
            "A",
            2.0,
            grant="employee",
        ),

        edge(
            "e2",
            "A",
            "X",
            2.0,
        ),

        edge(
            "e3",
            "SOURCE",
            "B",
            5.0,
            grant="employee",
        ),

        edge(
            "e4",
            "B",
            "X",
            5.0,
            grant="confidential",
        ),

        edge(
            "e5",
            "X",
            "TARGET",
            5.0,
            required_grants=[
                "employee",
                "confidential",
            ],
        ),
    ]

    return make_graph(
        nodes,
        edges,
        "competing_authorization_states",
        True,
        15.0,
        (
            "Two non-dominated authorization states reach X. "
            "The cheaper state lacks confidential authorization, "
            "while the more expensive state obtains it."
        ),
    )


# ----------------------------------------------------------------------
# Scenario 3
# ----------------------------------------------------------------------

def generate_dominated_authorization_states(
    rng: random.Random,
) -> dict:

    nodes = [
        node("STRONG"),
        node("WEAK"),
        node("X"),
    ]

    edges = [

        # Strong state.
        edge(
            "e1",
            "SOURCE",
            "STRONG",
            2.0,
            grant="employee",
        ),

        edge(
            "e2",
            "STRONG",
            "X",
            3.0,
            grant="viewer",
        ),

        # Alternative transition gives another authorization
        # grant at the same node.
        #
        # Positive cost is required because the Level-2 loader
        # rejects non-positive edge costs.
        edge(
            "e3",
            "STRONG",
            "X",
            0.1,
            grant="confidential",
        ),

        # Weak state.
        edge(
            "e4",
            "SOURCE",
            "WEAK",
            4.0,
            grant="employee",
        ),

        edge(
            "e5",
            "WEAK",
            "X",
            6.0,
            grant="viewer",
        ),

        edge(
            "e6",
            "X",
            "TARGET",
            5.0,
            required_grants=[
                "employee",
                "viewer",
            ],
        ),
    ]

    return make_graph(
        nodes,
        edges,
        "dominated_authorization_states",
        True,
        10.0,
        (
            "A stronger authorization state reaches X at lower "
            "cost than a weaker state and therefore dominates it."
        ),
    )


# ----------------------------------------------------------------------
# Scenario 4
# ----------------------------------------------------------------------

def generate_policy_dead_end(
    rng: random.Random,
) -> dict:

    nodes = [
        node("CHEAP"),
        node("SAFE"),
    ]

    edges = [

        # Cheap structural route.
        edge(
            "e1",
            "SOURCE",
            "CHEAP",
            1.0,
        ),

        edge(
            "e2",
            "CHEAP",
            "TARGET",
            1.0,
            required_grants=["manager"],
        ),

        # Authorized route.
        edge(
            "e3",
            "SOURCE",
            "SAFE",
            4.0,
            grant="manager",
        ),

        edge(
            "e4",
            "SAFE",
            "TARGET",
            6.0,
            required_grants=["manager"],
        ),
    ]

    return make_graph(
        nodes,
        edges,
        "policy_dead_end",
        True,
        10.0,
        (
            "The structurally cheapest route reaches a policy "
            "dead end. A more expensive route first acquires "
            "the required manager grant."
        ),
    )


# ----------------------------------------------------------------------
# Scenario 5: Deep dominance
# ----------------------------------------------------------------------

def generate_deep_dominance(
    rng: random.Random,
) -> dict:

    nodes = []
    edges = []

    layers = 6

    previous = "SOURCE"

    for i in range(layers):

        strong = f"S{i}"
        medium = f"M{i}"
        weak = f"W{i}"
        merge = f"X{i}"

        nodes.extend([
            node(strong),
            node(medium),
            node(weak),
            node(merge),
        ])

        # ----------------------------------------------------------
        # All three branches obtain g1.
        # ----------------------------------------------------------

        edges.append(
            edge(
                f"s{i}_in",
                previous,
                strong,
                1.0,
                grant=(
                    "g1"
                    if i == 0
                    else None
                ),
            )
        )

        edges.append(
            edge(
                f"m{i}_in",
                previous,
                medium,
                2.0,
                grant=(
                    "g1"
                    if i == 0
                    else None
                ),
            )
        )

        edges.append(
            edge(
                f"w{i}_in",
                previous,
                weak,
                3.0,
                grant=(
                    "g1"
                    if i == 0
                    else None
                ),
            )
        )

        # ----------------------------------------------------------
        # Strong branch obtains a new grant every layer.
        # ----------------------------------------------------------

        strong_grant = f"g{i + 2}"

        edges.append(
            edge(
                f"s{i}_merge",
                strong,
                merge,
                1.0,
                grant=strong_grant,
            )
        )

        # ----------------------------------------------------------
        # Medium branch obtains only some grants.
        # ----------------------------------------------------------

        if i % 2 == 0:

            edges.append(
                edge(
                    f"m{i}_merge",
                    medium,
                    merge,
                    1.0,
                    grant=strong_grant,
                )
            )

        else:

            edges.append(
                edge(
                    f"m{i}_merge",
                    medium,
                    merge,
                    1.0,
                )
            )

        # ----------------------------------------------------------
        # Weak branch obtains no new grant.
        # ----------------------------------------------------------

        edges.append(
            edge(
                f"w{i}_merge",
                weak,
                merge,
                1.0,
            )
        )

        previous = merge

    # --------------------------------------------------------------
    # Final target requires all accumulated grants.
    # --------------------------------------------------------------

    required = [
        f"g{i + 2}"
        for i in range(layers)
    ]

    edges.append(
        edge(
            "final",
            previous,
            "TARGET",
            2.0,
            required_grants=required,
        )
    )

    return make_graph(
        nodes,
        edges,
        "deep_dominance",
        True,
        float(
            layers * 2 + 2
        ),
        (
            "Multiple authorization states repeatedly merge at "
            "the same nodes. Stronger states dominate weaker "
            "states while preserving an optimal authorized path."
        ),
    )


# ----------------------------------------------------------------------
# Scenario 6: Bound pruning
# ----------------------------------------------------------------------

def generate_bound_pruning(
    rng: random.Random,
) -> dict:

    nodes = [
        node("GOOD"),
        node("BAD0"),
        node("BAD1"),
        node("BAD2"),
        node("BAD3"),
        node("BAD4"),
        node("BAD5"),
    ]

    edges = []

    # --------------------------------------------------------------
    # Optimal path:
    #
    # SOURCE -> GOOD -> TARGET
    #
    # cost = 10
    # --------------------------------------------------------------

    edges.append(
        edge(
            "good_1",
            "SOURCE",
            "GOOD",
            2.0,
        )
    )

    edges.append(
        edge(
            "good_2",
            "GOOD",
            "TARGET",
            8.0,
        )
    )

    # --------------------------------------------------------------
    # Expensive branches.
    # --------------------------------------------------------------

    for i in range(6):

        bad = f"BAD{i}"

        edges.append(
            edge(
                f"bad_{i}_in",
                "SOURCE",
                bad,
                3.0 + i,
            )
        )

        edges.append(
            edge(
                f"bad_{i}_out",
                bad,
                "TARGET",
                20.0 + i,
            )
        )

    # --------------------------------------------------------------
    # Deeper expensive branches.
    # --------------------------------------------------------------

    for i in range(6):

        bad = f"BAD{i}"
        deep = f"BAD{i}_DEEP"

        nodes.append(
            node(deep)
        )

        edges.append(
            edge(
                f"bad_{i}_deep",
                bad,
                deep,
                2.0,
            )
        )

        edges.append(
            edge(
                f"bad_{i}_deep_out",
                deep,
                "TARGET",
                25.0,
            )
        )

    return make_graph(
        nodes,
        edges,
        "bound_pruning",
        True,
        10.0,
        (
            "A low-cost authorized solution is found before "
            "multiple expensive branches are explored. Their "
            "admissible structural lower bounds make them "
            "eligible for branch-and-bound pruning."
        ),
    )


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

GENERATORS = {
    "state_dependent_grant":
        generate_state_dependent_grant,

    "competing_authorization_states":
        generate_competing_authorization_states,

    "dominated_authorization_states":
        generate_dominated_authorization_states,

    "policy_dead_end":
        generate_policy_dead_end,

    "deep_dominance":
        generate_deep_dominance,

    "bound_pruning":
        generate_bound_pruning,
}


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_instance(
    data: dict,
) -> None:

    nodes = data["nodes"]
    edges = data["edges"]

    source = data["source_id"]
    target = data["target_id"]

    # --------------------------------------------------------------
    # Validate node structure.
    # --------------------------------------------------------------

    assert isinstance(
        nodes,
        list,
    )

    node_ids = set()

    for n in nodes:

        assert "node_id" in n
        assert "type" in n

        node_id = n["node_id"]

        assert node_id not in node_ids

        node_ids.add(
            node_id
        )

    # SOURCE and TARGET must exist.
    assert source in node_ids
    assert target in node_ids

    # --------------------------------------------------------------
    # Validate source / target types.
    # --------------------------------------------------------------

    source_nodes = [
        n
        for n in nodes
        if n["node_id"] == source
    ]

    target_nodes = [
        n
        for n in nodes
        if n["node_id"] == target
    ]

    assert len(source_nodes) == 1
    assert len(target_nodes) == 1

    assert (
        source_nodes[0]["type"]
        == "SOURCE"
    )

    assert (
        target_nodes[0]["type"]
        == "TARGET"
    )

    # --------------------------------------------------------------
    # Validate edges.
    # --------------------------------------------------------------

    assert isinstance(
        edges,
        list,
    )

    edge_ids = set()

    for e in edges:

        assert "edge_id" in e
        assert "source" in e
        assert "target" in e
        assert "cost" in e

        assert (
            e["edge_id"]
            not in edge_ids
        )

        edge_ids.add(
            e["edge_id"]
        )

        assert e["source"] in node_ids
        assert e["target"] in node_ids

        assert e["cost"] >= 0.0

        assert (
            e["source"]
            != e["target"]
        )

        assert isinstance(
            e.get(
                "required_grants",
                [],
            ),
            list,
        )

        assert e.get(
            "min_clearance",
            0,
        ) >= 0

    # --------------------------------------------------------------
    # Validate metadata.
    # --------------------------------------------------------------

    assert "metadata" in data

    metadata = data["metadata"]

    assert metadata.get(
        "level2",
        False,
    ) is True

    assert metadata["scenario"]

    assert "expected" in metadata

    expected = metadata["expected"]

    assert "found" in expected
    assert "cost" in expected

    assert "policy" in metadata

    policy = metadata["policy"]

    assert (
        policy["authorization_semantics"]
        == "explicit_propagation_grant"
    )

    assert (
        policy["path_semantics"]
        == "every_edge"
    )

    assert (
        policy["search_objective"]
        == "minimum_cost"
    )

    assert (
        policy["monotonic_authorization"]
        is True
    )


# ----------------------------------------------------------------------
# Main generation
# ----------------------------------------------------------------------

def main():

    # Always regenerate from scratch.
    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    for scenario, generator in GENERATORS.items():

        scenario_dir = (
            OUTPUT_DIR
            / scenario
        )

        scenario_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        scenario_seed = (
            SCENARIO_SEEDS[
                scenario
            ]
        )

        for index in range(
            INSTANCES_PER_SCENARIO
        ):

            seed = (
                BASE_SEED
                + scenario_seed * 1000
                + index
            )

            rng = random.Random(
                seed
            )

            data = generator(
                rng
            )

            validate_instance(
                data
            )

            output_file = (
                scenario_dir
                / f"{index:05d}.json"
            )

            with output_file.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2,
                )

            total += 1

    print("=" * 80)
    print(
        "LEVEL-2 SMALL DATASET V2"
    )
    print("=" * 80)

    print(
        f"Scenarios: {len(GENERATORS)}"
    )

    print(
        f"Instances per scenario: "
        f"{INSTANCES_PER_SCENARIO}"
    )

    print(
        f"Total instances: {total}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        "Dataset generated successfully."
    )


if __name__ == "__main__":
    main()