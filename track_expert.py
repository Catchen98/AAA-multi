import os
import yaml
from pathlib import Path
from datasets.mot import MOT
from datasets.detrac import DETRAC
from evaluate_tracker import eval_tracker
from print_manager import do_not_print


@do_not_print
def get_expert_by_name(config, name):
    if name == "CenterTrack":
        from experts.centertrack import CenterTrack as Tracker

        tracker = Tracker(
            config["CENTERTRACK"]["load_model"],
            config["CENTERTRACK"]["track_thresh"],
            config["CENTERTRACK"]["pre_thresh"],
            config["CENTERTRACK"]["private"],
        )
    elif name == "DAN":
        from experts.dan import DAN as Tracker

        tracker = Tracker(config["DAN"]["model_path"])
    elif name == "DeepMOT":
        from experts.deepmot import DeepMOT as Tracker

        tracker = Tracker(
            config["DEEPMOT"]["obj_detect_weights_path"],
            config["DEEPMOT"]["tracktor_config_path"],
            config["DEEPMOT"]["obj_detect_config_path"],
        )
    elif name == "DeepSORT":
        from experts.deepsort import DeepSORT as Tracker

        tracker = Tracker(
            config["DEEPSORT"]["model"],
            config["DEEPSORT"]["min_confidence"],
            config["DEEPSORT"]["min_detection_height"],
            config["DEEPSORT"]["nms_max_overlap"],
            config["DEEPSORT"]["max_cosine_distance"],
            config["DEEPSORT"]["nn_budget"],
        )
    elif name == "MOTDT":
        from experts.motdt import MOTDT as Tracker

        tracker = Tracker()
    elif name == "Sort":
        from experts.esort import ESort as Tracker

        tracker = Tracker()
    elif name == "Tracktor":
        from experts.tracktor import Tracktor as Tracker

        tracker = Tracker(
            config["TRACKTOR"]["reid_network_weights_path"],
            config["TRACKTOR"]["obj_detect_model_path"],
            config["TRACKTOR"]["tracktor_config_path"],
            config["TRACKTOR"]["reid_config_path"],
        )
    elif name == "DeepTAMA":
        from experts.deeptama import DeepTAMA as Tracker

        tracker = Tracker()
    elif name == "TRMOT":
        from experts.trmot import TRMOT as Tracker

        tracker = Tracker(config["TRMOT"])
    elif name == "UMA":
        from experts.uma import UMA as Tracker

        tracker = Tracker(
            config["UMA"]["life_span"],
            config["UMA"]["occlusion_thres"],
            config["UMA"]["association_thres"],
            config["UMA"]["checkpoint"],
            config["UMA"]["context_amount"],
            config["UMA"]["iou"],
        )
    else:
        raise ValueError("Invalid name")

    return tracker


@do_not_print
def track_seq(tracker, seq):
    return tracker.track_seq(seq)


def main(config_path, expert_name):
    with open(config_path) as c:
        config = yaml.load(c)

    datasets = {
        dataset_name: MOT(config["DATASET_DIR"][dataset_name])
        if dataset_name.startswith("MOT")
        else DETRAC(config["DATASET_DIR"][dataset_name])
        for dataset_name in config["DATASETS"]
    }
    tracker = get_expert_by_name(config, expert_name)

    for dataset_name, dataset in datasets.items():
        dataset_dir = Path(
            os.path.join(config["OUTPUT_DIR"], dataset_name, expert_name)
        )
        for seq in dataset:
            if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                print(f"Pass {seq.seq_info['seq_name']}")
            else:
                print(f"Start {seq.seq_info['seq_name']}")
                results = track_seq(tracker, seq)
                seq.write_results(results, dataset_dir)

        eval_tracker(
            config["DATASET_DIR"],
            config["OUTPUT_DIR"],
            expert_name,
            dataset_name,
            config["EVAL_DIR"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experts")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="experiments/experts.yaml",
        help="The config file of experts",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="Tracktor", help="The name of expert",
    )
    args = parser.parse_args()
    main(args.config, args.name)
