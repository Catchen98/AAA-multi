import os
import yaml
from pathlib import Path
from datasets.mot import MOT
from feedback.neural_solver import NeuralSolver
from evaluate_tracker import eval_tracker
from print_manager import do_not_print
from file_manager import ReadResult


@do_not_print
def track_seq(output_dir, experts_name, tracker, seq):
    tracker.initialize(seq.seq_info)
    experts_reader = [
        ReadResult(
            output_dir,
            seq.seq_info["dataset_name"],
            expert_name,
            seq.seq_info["seq_name"],
        )
        for expert_name in experts_name
    ]
    for frame_idx, (img_path, dets, _) in enumerate(seq):
        expert_results = []
        for reader in experts_reader:
            expert_results.append(reader.get_result_by_frame(frame_idx))
        tracker.step(img_path, dets, None, expert_results)
    return tracker.track(0, len(seq) - 1)


def main(config_path):
    with open(config_path) as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    datasets = {
        dataset_name: MOT(config["DATASET_DIR"][dataset_name])
        for dataset_name in config["DATASETS"]
    }
    tracker = NeuralSolver(
        config["FEEDBACK"]["ckpt_path"],
        config["FEEDBACK"]["frcnn_weights_path"],
        config["FEEDBACK"]["reid_weights_path"],
        config["FEEDBACK"]["tracking_cfg_path"],
        config["FEEDBACK"]["preprocessing_cfg_path"],
        config["OFFLINE"]["use_gt"],
        config["OFFLINE"]["pre_cnn"],
        config["OFFLINE"]["pre_track"],
    )

    if config["OFFLINE"]["pre_track"] == "None":
        tracker.name = tracker.name + f"{config['EXPERTS']}"
    else:
        tracker.name = tracker.name + f"[{config['OFFLINE']['pre_track']}]"

    for dataset_name, dataset in datasets.items():
        dataset_dir = Path(
            os.path.join(config["OUTPUT_DIR"], dataset_name, tracker.name)
        )
        for seq in dataset:
            if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                print(f"Pass {seq.seq_info['seq_name']}")
            else:
                print(f"Start {seq.seq_info['seq_name']}")
                results = track_seq(
                    config["OUTPUT_DIR"], config["EXPERTS"], tracker, seq
                )
                seq.write_results(results, dataset_dir)

        eval_tracker(
            config["DATASET_DIR"],
            config["OUTPUT_DIR"],
            tracker.name,
            dataset_name,
            config["EVAL_DIR"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feedback")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="experiments/feedback.yaml",
        help="The config file of feedback",
    )
    args = parser.parse_args()
    main(args.config)
