import os
import yaml
from pathlib import Path
from datasets.mot import MOT
from feedback.neural_solver import NeuralSolver
from evaluate_tracker import eval_tracker
from print_manager import do_not_print


@do_not_print
def track_seq(tracker, seq):
    tracker.initialize(seq.seq_info)
    for frame_idx, (img_path, dets, _) in enumerate(seq):
        tracker.step(img_path, dets, None)
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
        use_gt=False,
        pre_cnn=False,
        use_pre=True,
    )

    for dataset_name, dataset in datasets.items():
        dataset_dir = Path(
            os.path.join(config["OUTPUT_DIR"], dataset_name, tracker.name)
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
