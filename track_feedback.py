import os
import yaml
from pathlib import Path
from datasets.mot import MOT
from feedback.neural_solver import NeuralSolver
from evaluate_tracker import eval_tracker
from print_manager import do_not_print


@do_not_print
def track_seq(tracker, seq, img_paths, dets):
    return tracker.track(seq.seq_info, img_paths, dets)


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
        config["FEEDBACK"]["prepr_w_tracktor"],
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
                img_paths = []
                dets = []
                for frame_idx, (img_path, det, _) in enumerate(seq):
                    img_paths.append(img_path)
                    if det is None:
                        dets.append([])
                    else:
                        dets.append(det)

                results = track_seq(tracker, seq, img_paths, dets)
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
