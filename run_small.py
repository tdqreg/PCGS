from generate_dataset import generate_dataset
from scenario_specs import SCENARIOS


if __name__ == "__main__":
    generate_dataset(
        output_dir="dataset_smoke",
        sizes=[10, 50],
        scenarios=list(SCENARIOS.keys()),
        instances_per_configuration=2,
        base_seed=42,
    )

    print("Small dataset generated successfully.")