from generate_dataset import generate_one


graph = generate_one(
    num_nodes=10,
    scenario_name="safe_single_hop",
    seed=10052,
)

print("Graph generated successfully")
print("Nodes:", len(graph.nodes))
print("Edges:", len(graph.edges))