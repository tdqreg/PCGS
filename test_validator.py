from generate_dataset import generate_one


def test_exact_node_count():
    graph = generate_one(
        num_nodes=50,
        scenario_name="safe_multi_hop",
        seed=42,
    )

    assert len(graph.nodes) == 50


def test_structure():
    graph = generate_one(
        num_nodes=50,
        scenario_name="safe_multi_hop",
        seed=42,
    )

    assert graph.source_id in graph.nodes
    assert graph.target_id in graph.nodes

    for edge in graph.edges:
        assert edge.source in graph.nodes
        assert edge.target in graph.nodes
        assert edge.source != edge.target
        assert edge.cost > 0


def test_edge_density_metadata():
    graph = generate_one(
        num_nodes=100,
        scenario_name="safe_multi_hop",
        seed=42,
    )

    assert "actual_edge_density" in graph.metadata
    assert graph.metadata["actual_edge_density"] > 0


def test_cost_order():
    graph = generate_one(
        num_nodes=100,
        scenario_name="invalid_cheap_shortcut",
        seed=42,
    )

    valid_costs = [
        path.expected_cost
        for path in graph.valid_paths
    ]

    invalid_costs = [
        path.expected_cost
        for path in graph.invalid_paths
    ]

    assert valid_costs
    assert invalid_costs

    assert min(invalid_costs) < min(valid_costs)


if __name__ == "__main__":
    test_exact_node_count()
    test_structure()
    test_edge_density_metadata()
    test_cost_order()

    print("All validator tests passed.")