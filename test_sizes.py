from generate_dataset import generate_one


def main():
    sizes = [10, 50, 100, 500, 1000]

    for size in sizes:
        graph = generate_one(
            num_nodes=size,
            scenario_name="safe_multi_hop",
            seed=42 + size,
        )

        actual_nodes = len(graph.nodes)

        print(
            f"requested={size}, "
            f"actual={actual_nodes}, "
            f"edges={len(graph.edges)}"
        )

        assert actual_nodes == size, (
            f"Expected {size} nodes, "
            f"got {actual_nodes}"
        )


if __name__ == "__main__":
    main()