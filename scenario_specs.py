# scenario_specs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SizeProfile:
    num_nodes: int

    min_path_depth: int
    max_path_depth: int

    edge_density: float

    min_decoy_edges: int
    max_decoy_edges: int

    min_edge_cost: float
    max_edge_cost: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str

    min_valid_paths: int
    max_valid_paths: int

    min_invalid_paths: int
    max_invalid_paths: int

    explicit_deny: bool = False

    require_invalid_cheaper_than_valid: bool = False

    required_violation: str | None = None

    valid_cost_gap: float = 5.0

    invalid_shortcut_cost: float = 1.0


SIZE_PROFILES: Dict[int, SizeProfile] = {
    10: SizeProfile(
        num_nodes=10,
        min_path_depth=1,
        max_path_depth=2,
        edge_density=0.15,
        min_decoy_edges=0,
        max_decoy_edges=2,
        min_edge_cost=1.0,
        max_edge_cost=10.0,
    ),

    50: SizeProfile(
        num_nodes=50,
        min_path_depth=2,
        max_path_depth=5,
        edge_density=0.08,
        min_decoy_edges=2,
        max_decoy_edges=12,
        min_edge_cost=1.0,
        max_edge_cost=10.0,
    ),

    100: SizeProfile(
        num_nodes=100,
        min_path_depth=3,
        max_path_depth=8,
        edge_density=0.05,
        min_decoy_edges=5,
        max_decoy_edges=30,
        min_edge_cost=1.0,
        max_edge_cost=10.0,
    ),

    500: SizeProfile(
        num_nodes=500,
        min_path_depth=4,
        max_path_depth=12,
        edge_density=0.02,
        min_decoy_edges=20,
        max_decoy_edges=120,
        min_edge_cost=1.0,
        max_edge_cost=10.0,
    ),

    1000: SizeProfile(
        num_nodes=1000,
        min_path_depth=5,
        max_path_depth=15,
        edge_density=0.01,
        min_decoy_edges=50,
        max_decoy_edges=300,
        min_edge_cost=1.0,
        max_edge_cost=10.0,
    ),
}


SCENARIOS: Dict[str, ScenarioSpec] = {
    "safe_single_hop": ScenarioSpec(
        name="safe_single_hop",
        min_valid_paths=1,
        max_valid_paths=1,
        min_invalid_paths=0,
        max_invalid_paths=0,
    ),

    "safe_multi_hop": ScenarioSpec(
        name="safe_multi_hop",
        min_valid_paths=1,
        max_valid_paths=1,
        min_invalid_paths=0,
        max_invalid_paths=0,
    ),

    "multiple_valid_paths": ScenarioSpec(
        name="multiple_valid_paths",
        min_valid_paths=2,
        max_valid_paths=3,
        min_invalid_paths=0,
        max_invalid_paths=0,
        valid_cost_gap=5.0,
    ),

    "invalid_cheap_shortcut": ScenarioSpec(
        name="invalid_cheap_shortcut",
        min_valid_paths=1,
        max_valid_paths=1,
        min_invalid_paths=1,
        max_invalid_paths=1,
        require_invalid_cheaper_than_valid=True,
        required_violation="hierarchical",
        invalid_shortcut_cost=1.0,
    ),

    "explicit_deny": ScenarioSpec(
        name="explicit_deny",
        min_valid_paths=1,
        max_valid_paths=1,
        min_invalid_paths=0,
        max_invalid_paths=0,
        explicit_deny=True,
    ),

    "mixed_relation_conflict": ScenarioSpec(
        name="mixed_relation_conflict",
        min_valid_paths=1,
        max_valid_paths=1,
        min_invalid_paths=1,
        max_invalid_paths=1,
        required_violation="hierarchical",
    ),

    "no_valid_path": ScenarioSpec(
        name="no_valid_path",
        min_valid_paths=0,
        max_valid_paths=0,
        min_invalid_paths=1,
        max_invalid_paths=3,
        required_violation="hierarchical",
    ),
}

def get_size_profile(num_nodes: int) -> SizeProfile:
    if num_nodes not in SIZE_PROFILES:
        raise ValueError(
            f"Unsupported graph size: {num_nodes}. "
            f"Supported sizes: {sorted(SIZE_PROFILES.keys())}"
        )

    return SIZE_PROFILES[num_nodes]