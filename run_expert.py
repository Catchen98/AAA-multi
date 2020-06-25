from datasets.mot import MOT
from expert import get_expert_by_name


def main(expert_name):
    datasets = {
        "MOT15": MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15"),
        "MOT16": MOT("/home/heonsong/Disk2/Dataset/MOT/MOT16"),
        "MOT17": MOT("/home/heonsong/Disk2/Dataset/MOT/MOT17"),
    }
    tracker = get_expert_by_name(expert_name)

    for dataset_name, dataset in datasets.items():
        for seq in dataset:
            print(f"Start {seq.name}")
            results = tracker.track_seq(dataset_name, seq)
            seq.write_results(results, f"output/{expert_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experts")
    parser.add_argument(
        "-n", "--name", type=str, help="The name of the expert",
    )

    args = parser.parse_args()
    main(args.name)
