from pathlib import Path

import torch
import numpy as np
import random

from datasets.mot import MOT
from algorithms.aaa import AAA
from paths import DATASET_PATH, OUTPUT_PATH
from print_manager import do_not_print
from file_manager import ReadResult, write_results
from evaluate_tracker import eval_tracker


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@do_not_print
def track_seq(experts_name, algorithm, seq):
    algorithm.initialize(seq.seq_info)
    experts_reader = [
        ReadResult(seq.seq_info["dataset_name"], expert_name, seq.seq_info["seq_name"])
        for expert_name in experts_name
    ]

    results = []
    ws = []
    expert_losses = []
    feedbacks = []
    selected_experts = []

    for frame_idx, (img_path, dets, _) in enumerate(seq):
        expert_results = []
        for reader in experts_reader:
            expert_results.append(reader.get_result_by_frame(frame_idx))

        result, w, expert_loss, feedback, selected_expert = algorithm.track(
            img_path, dets, expert_results
        )
        if len(result) > 0:
            frame_result = np.zeros((result.shape[0], result.shape[1] + 1))
            frame_result[:, 1:] = result
            frame_result[:, 0] = frame_idx + 1
            results.append(frame_result)

        frame_w = np.zeros((1, len(w) + 1))
        frame_w[0, 1:] = w
        frame_w[0, 0] = frame_idx + 1
        ws.append(frame_w)

        if expert_loss is not None:
            frame_expert_loss = np.zeros((1, len(expert_loss) + 1))
            frame_expert_loss[0, 1:] = expert_loss
            frame_expert_loss[0, 0] = frame_idx + 1
            expert_losses.append(frame_expert_loss)

        if feedback is not None:
            frame_feedback = np.zeros((feedback.shape[0], feedback.shape[1] + 1))
            frame_feedback[:, 1:] = feedback
            frame_feedback[:, 0] = frame_idx + 1
            feedbacks.append(frame_feedback)

        frame_selected = np.zeros((1, 2))
        frame_selected[0, 1] = selected_expert
        frame_selected[0, 0] = frame_idx + 1
        selected_experts.append(frame_selected)

    results = np.concatenate(results, axis=0)
    ws = np.concatenate(ws, axis=0)
    expert_losses = np.concatenate(expert_losses, axis=0)
    feedbacks = np.concatenate(feedbacks, axis=0)
    selected_experts = np.concatenate(selected_experts, axis=0)

    return results, ws, expert_losses, feedbacks, selected_experts


@do_not_print
def get_algorithm(experts_name, duration, threshold, loss_type):
    config = {
        "detector": {"type": "fixed", "duration": duration},
        "offline": {"reset": True},
        "matching": {"method": "kmeans", "threshold": threshold},
        "loss": {"type": loss_type},
    }
    algorithm = AAA(len(experts_name), config)
    return algorithm


def main(experts_name, duration, threshold, loss_type, result_dir):
    datasets = {
        # "MOT15": MOT(DATASET_PATH["MOT15"]),
        # "MOT16": MOT(DATASET_PATH["MOT16"]),
        "MOT17": MOT(DATASET_PATH["MOT17"]),
    }

    print(f"Duration: {duration}, Threshold: {threshold}, Loss: {loss_type}")
    algorithm = get_algorithm(experts_name, duration, threshold, loss_type)

    for dataset_name, dataset in datasets.items():
        dataset_dir = OUTPUT_PATH / dataset_name / algorithm.name

        for seq in dataset:
            if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                print(f"Pass {seq.seq_info['seq_name']}")
            else:
                print(f"Start {seq.seq_info['seq_name']}")
                results, ws, expert_losses, feedbacks, selected_experts = track_seq(
                    experts_name, algorithm, seq
                )
                seq.write_results(results, dataset_dir)
                write_results(ws, dataset_dir, f"{seq.seq_info['seq_name']}_weight.txt")
                write_results(
                    expert_losses, dataset_dir, f"{seq.seq_info['seq_name']}_loss.txt",
                )
                write_results(
                    feedbacks, dataset_dir, f"{seq.seq_info['seq_name']}_feedback.txt",
                )
                write_results(
                    selected_experts,
                    dataset_dir,
                    f"{seq.seq_info['seq_name']}_selected.txt",
                )
        eval_tracker(algorithm.name, dataset_name, result_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run algorithms")
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default="70",
        help="The duration of the algorithm",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default="0.5",
        help="The threshold of the algorithm",
    )
    parser.add_argument(
        "-l", "--loss_type", type=str, default="fn", help="The loss of the algorithm",
    )

    args = parser.parse_args()

    result_dir = Path("eval")

    experts_name = ["DAN", "DeepSort", "DeepTAMA", "Sort", "MOTDT"]
    main(experts_name, args.duration, args.threshold, args.loss_type, result_dir)
