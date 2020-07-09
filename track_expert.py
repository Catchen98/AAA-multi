from datasets.mot import MOT
from paths import DATASET_PATH, OUTPUT_PATH
from print_manager import do_not_print


@do_not_print
def get_expert_by_name(name):
    if name == "DAN":
        from experts.dan import DAN as Tracker

        tracker = Tracker("weights/DAN/sst300_0712_83000.pth")
    elif name == "DeepMOT":
        from experts.deepmot import DeepMOT as Tracker

        tracker = Tracker(
            "weights/DeepMOT/trainedSOTtoMOT.pth", "weights/DeepMOT/DAN.pth",
        )
    elif name == "DeepSort":
        from experts.deepsort import DeepSort as Tracker

        tracker = Tracker("weights/DeepSort/mars-small128.pb")
    elif name == "IOU":
        from experts.iou import IOU as Tracker

        tracker = Tracker()
    elif name == "MOTDT":
        from experts.motdt import MOTDT as Tracker

        tracker = Tracker()
    elif name == "Sort":
        from experts.esort import ESort as Tracker

        tracker = Tracker()
    elif name == "Tracktor":
        from experts.tracktor import Tracktor as Tracker

        tracker = Tracker(
            "weights/Tracktor/ResNet_iter_25245.pth",
            "weights/Tracktor/model_epoch_27.model",
            "external/tracking_wo_bnw/experiments/cfgs/tracktor.yaml",
            "weights/Tracktor/sacred_config.yaml",
        )
    elif name == "VIOU":
        from experts.viou import VIOU as Tracker

        tracker = Tracker()
    elif name == "DeepTAMA":
        from experts.deeptama import DeepTAMA as Tracker

        tracker = Tracker()
    else:
        raise ValueError("Invalid name")

    return tracker


def main(expert_name):
    datasets = {
        "MOT15": MOT(DATASET_PATH["MOT15"]),
        "MOT16": MOT(DATASET_PATH["MOT16"]),
        "MOT17": MOT(DATASET_PATH["MOT17"]),
        "MOT20": MOT(DATASET_PATH["MOT20"]),
    }
    tracker = get_expert_by_name(expert_name)

    for dataset_name, dataset in datasets.items():
        dataset_dir = OUTPUT_PATH / f"{dataset_name}/{expert_name}"
        for seq in dataset:
            if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                print(f"Pass {seq.seq_info['seq_name']}")
            else:
                print(f"Start {seq.seq_info['seq_name']}")
                results = tracker.track_seq(seq)
                seq.write_results(results, dataset_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experts")
    parser.add_argument(
        "-n", "--name", type=str, default="DeepMOT", help="The name of the expert"
    )

    args = parser.parse_args()
    main(args.name)
