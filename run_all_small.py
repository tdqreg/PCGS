from generate_dataset import generate_dataset
from scenario_specs import SCENARIOS


if __name__ == "__main__":
    generate_dataset(
        output_dir="dataset_all_small",
        sizes=[10, 50, 100, 500, 1000],
        scenarios=list(SCENARIOS.keys()),
        instances_per_configuration=2,
        base_seed=42,
    )

    print("All small configurations generated successfully.")