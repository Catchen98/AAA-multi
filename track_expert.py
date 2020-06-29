from datasets.mot import MOT
from expert import get_expert_by_name
from paths import DATASET_PATH, OUTPUT_PATH


def main(expert_name):
    datasets = {
        "MOT15": MOT(DATASET_PATH["MOT15"]),
        # "MOT16": MOT(DATASET_PATH["MOT16"]),
        # "MOT17": MOT(DATASET_PATH["MOT17"]),
    }
    tracker = get_expert_by_name(expert_name)

    for dataset_name, dataset in datasets.items():
        dataset_dir = OUTPUT_PATH / f"{dataset_name}/{expert_name}"
        for seq in dataset:
            if (dataset_dir / f"{seq.name}.txt").exists():
                print(f"Pass {seq.name}")
            else:
                print(f"Start {seq.name}")
                results = tracker.track_seq(dataset_name, seq)
                seq.write_results(results, dataset_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experts")
    parser.add_argument(
        "-n", "--name", type=str, help="The name of the expert",
    )

    args = parser.parse_args()
    main(args.name)
