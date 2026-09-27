from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SIZES = [50, 100, 500, 1000, 5000]

SCENARIOS = {
    "random_state": {
        "branching": (2, 4),
        "grant_probability": 0.45,
        "require_probability": 0.35,
        "decoy_ratio": 0.35,
    },
    "random_dominance": {
        "branching": (3, 5),
        "grant_probability": 0.60,
        "require_probability": 0.40,
        "decoy_ratio": 0.25,
    },
    "random_mixed": {
        "branching": (2, 5),
        "grant_probability": 0.50,
        "require_probability": 0.50,
        "decoy_ratio": 0.50,
    },
}


GRANTS = [
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "g8",
]


def node(node_id: str, node_type: str = "INTERMEDIATE") -> dict[str, Any]:
    return {
        "node_id": node_id,
        "type": node_type,
    }


def make_edge(
    edge_id: str,
    source: str,
    target: str,
    cost: float,
    required_grants: list[str] | None = None,
    grant: str | None = None,
) -> dict[str, Any]:

    return {
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "cost": float(cost),
        "required_grants": sorted(
            set(required_grants or [])
        ),
        "min_clearance": 0,
        "grant": grant,
    }


class RandomLevel2Generator:

    def __init__(
        self,
        num_nodes: int,
        scenario: str,
        seed: int,
    ):
        self.num_nodes = num_nodes
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)

        self.spec = SCENARIOS[scenario]

        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []

        self.edge_counter = 0

    def edge_id(self) -> str:
        self.edge_counter += 1
        return f"e{self.edge_counter}"

    def add_edge(
        self,
        source: str,
        target: str,
        cost: float | None = None,
        required_grants: list[str] | None = None,
        grant: str | None = None,
    ) -> None:

        if cost is None:
            cost = self.rng.randint(1, 10)

        self.edges.append(
            make_edge(
                edge_id=self.edge_id(),
                source=source,
                target=target,
                cost=cost,
                required_grants=required_grants,
                grant=grant,
            )
        )

    def random_grant(self) -> str | None:
        if self.rng.random() > self.spec["grant_probability"]:
            return None

        return self.rng.choice(GRANTS)

    def random_requirements(self) -> list[str]:
        if self.rng.random() > self.spec["require_probability"]:
            return []

        # Keep requirements intentionally small.
        # This avoids pathological state explosion.
        if self.rng.random() < 0.80:
            return [self.rng.choice(GRANTS)]

        return self.rng.sample(
            GRANTS,
            k=2,
        )

    def build_backbone(self) -> list[str]:

        # We construct a random layered backbone.
        #
        # Each layer contains several alternatives that merge into
        # the next layer. This naturally creates multiple
        # authorization states reaching the same node.

        remaining = self.num_nodes - 2

        layers: list[list[str]] = []

        while remaining > 0:

            low, high = self.spec["branching"]
            width = self.rng.randint(low, high)
            width = min(width, remaining)

            current = []

            for _ in range(width):
                node_id = f"N{len(self.nodes):05d}"
                self.nodes.append(node(node_id))
                current.append(node_id)

            layers.append(current)
            remaining -= width

        return [item for layer in layers for item in layer]

    def build(self) -> dict[str, Any]:

        self.nodes = [
            node("SOURCE", "SOURCE")
        ]

        # Build random layers.
        self.build_backbone()

        self.nodes.append(
            node("TARGET", "TARGET")
        )

        # Recover intermediate nodes.
        intermediate_ids = [
            item["node_id"]
            for item in self.nodes
            if item["node_id"] not in {"SOURCE", "TARGET"}
        ]

        # Partition nodes into approximately equal layers.
        #
        # We use a deterministic random partition so that all
        # backbone edges point forward and therefore do not
        # introduce accidental cycles.
        layers: list[list[str]] = []

        pos = 0

        while pos < len(intermediate_ids):

            low, high = self.spec["branching"]
            width = self.rng.randint(low, high)
            width = min(
                width,
                len(intermediate_ids) - pos,
            )

            layers.append(
                intermediate_ids[pos:pos + width]
            )

            pos += width

        previous = ["SOURCE"]

        # Guaranteed authorization corridor.
        #
        # Each layer has a number of alternatives. One or more
        # branches propagate grants, while requirements are sampled
        # so that at least one continuation remains feasible.
        available_grants: set[str] = set()

        for layer_index, current in enumerate(layers):

            for target in current:

                source = self.rng.choice(previous)

                grant = None

                # Random propagation.
                if self.rng.random() < self.spec["grant_probability"]:
                    grant = self.rng.choice(GRANTS)

                requirements: list[str] = []

                if available_grants and self.rng.random() < self.spec["require_probability"]:
                    requirements = [
                        self.rng.choice(
                            sorted(available_grants)
                        )
                    ]

                # Guarantee the first branch remains usable.
                if target == current[0]:
                    requirements = []

                self.add_edge(
                    source=source,
                    target=target,
                    cost=self.rng.randint(1, 10),
                    required_grants=requirements,
                    grant=grant,
                )

                if grant is not None:
                    available_grants.add(grant)

            # Add additional forward connections.
            for source in previous:
                for target in current:

                    if self.rng.random() < 0.30:

                        grant = self.random_grant()
                        requirements = self.random_requirements()

                        # Avoid making every branch trivially dependent
                        # on an unavailable grant.
                        if target == current[0]:
                            requirements = []

                        self.add_edge(
                            source=source,
                            target=target,
                            cost=self.rng.randint(1, 10),
                            required_grants=requirements,
                            grant=grant,
                        )

            previous = current

        # Connect final layer to TARGET.
        final_layer = layers[-1]

        for source in final_layer:

            requirements = []

            # Some final edges require accumulated permissions.
            if available_grants and self.rng.random() < 0.60:
                requirements = [
                    self.rng.choice(
                        sorted(available_grants)
                    )
                ]

            # Ensure at least one final authorized continuation.
            if source == final_layer[0]:
                requirements = []

            self.add_edge(
                source=source,
                target="TARGET",
                cost=self.rng.randint(1, 10),
                required_grants=requirements,
                grant=None,
            )

        # Add random forward decoys.
        decoy_probability = self.spec["decoy_ratio"]

        candidates = [
            item["node_id"]
            for item in self.nodes
            if item["node_id"] not in {"TARGET"}
        ]

        for source in candidates:

            if self.rng.random() > decoy_probability:
                continue

            # A small number of extra outgoing edges per node.
            num_decoys = self.rng.randint(1, 3)

            for _ in range(num_decoys):

                target = self.rng.choice(candidates)

                if target == source:
                    continue

                self.add_edge(
                    source=source,
                    target=target,
                    cost=self.rng.randint(1, 10),
                    required_grants=self.random_requirements(),
                    grant=self.random_grant(),
                )

        # Remove exact duplicate edges.
        seen = set()
        unique_edges = []

        for edge in self.edges:

            key = (
                edge["source"],
                edge["target"],
                tuple(edge["required_grants"]),
                edge.get("grant"),
                edge["cost"],
            )

            if key in seen:
                continue

            seen.add(key)
            unique_edges.append(edge)

        self.edges = unique_edges

        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "source_id": "SOURCE",
            "target_id": "TARGET",
            "metadata": {
                "level2": True,
                "benchmark": "scaling_randomized",
                "scenario": self.scenario,
                "seed": self.seed,
                "num_nodes": self.num_nodes,
                "policy": {
                    "authorization_semantics": "explicit_propagation_grant",
                    "path_semantics": "every_edge",
                    "search_objective": "minimum_cost",
                    "monotonic_authorization": True,
                    "initial_grants": [],
                    "initial_clearance": 0,
                },
            },
        }


def validate_graph(data: dict[str, Any]) -> None:

    nodes = data["nodes"]
    edges = data["edges"]

    node_ids = {
        item["node_id"]
        for item in nodes
    }

    assert "SOURCE" in node_ids
    assert "TARGET" in node_ids

    assert len(node_ids) == len(nodes)

    for edge in edges:

        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert edge["source"] != edge["target"]

        assert edge["cost"] > 0

        assert isinstance(
            edge["required_grants"],
            list,
        )


def generate_dataset(
    output_dir: str = "dataset_level2_scaling",
    sizes: list[int] | None = None,
    scenarios: list[str] | None = None,
    instances_per_configuration: int = 100,
    base_seed: int = 42000,
) -> None:

    if sizes is None:
        sizes = SIZES

    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    root = Path(output_dir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    for size in sizes:

        for scenario in scenarios:

            directory = root / str(size) / scenario
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            for index in range(instances_per_configuration):

                seed = (
                    base_seed
                    + size * 100000
                    + index * 1000
                    + list(SCENARIOS).index(scenario)
                )

                generator = RandomLevel2Generator(
                    num_nodes=size,
                    scenario=scenario,
                    seed=seed,
                )

                data = generator.build()

                validate_graph(data)

                output_file = (
                    directory
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

    print(
        f"Generated {total} Level-2 scaling instances."
    )


if __name__ == "__main__":
    generate_dataset()
