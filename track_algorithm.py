import os
import yaml
import time
from pathlib import Path

import torch
import numpy as np
import random

from datasets.mot import MOT
from algorithms.aaa import AAA
from print_manager import do_not_print
from file_manager import ReadResult, write_results
from evaluate_tracker import eval_tracker

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@do_not_print
def track_seq(output_dir, experts_name, algorithm, seq):
    algorithm.initialize(seq.seq_info)
    experts_reader = [
        ReadResult(
            output_dir,
            seq.seq_info["dataset_name"],
            expert_name,
            seq.seq_info["seq_name"],
        )
        for expert_name in experts_name
    ]

    results = []
    ws = []
    expert_losses = []
    feedbacks = []
    selected_experts = []
    times = []

    for frame_idx, (img_path, dets, gts) in enumerate(seq):
        expert_results = []
        for reader in experts_reader:
            expert_results.append(reader.get_result_by_frame(frame_idx))

        start_time = time.time()
        result, w, expert_loss, feedback, selected_expert = algorithm.track(
            img_path, dets, gts, expert_results
        )
        end_time = time.time() - start_time
        times.append(end_time)
        # print(f"Frame {frame_idx}: {end_time}s")

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

    return results, ws, expert_losses, feedbacks, selected_experts, times


@do_not_print
def get_algorithm(config):
    return AAA(config)


def main(config_path):
    with open(config_path) as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    durations = [100]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        for duration in durations:
            config["DETECTOR"]["duration"] = duration
            config["DETECTOR"]["threshold"] = threshold

            datasets = {
                dataset_name: MOT(config["DATASET_DIR"][dataset_name])
                for dataset_name in config["DATASETS"]
            }

            algorithm = get_algorithm(config)

            for dataset_name, dataset in datasets.items():
                dataset_dir = Path(
                    os.path.join(config["OUTPUT_DIR"], dataset_name, algorithm.name)
                )

                total_time = []

                for seq in dataset:
                    if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                        print(f"Pass {seq.seq_info['seq_name']}")
                    else:
                        print(f"Start {seq.seq_info['seq_name']}")
                        (results, ws, expert_losses, feedbacks, selected_experts, times) = track_seq(
                            config["OUTPUT_DIR"], config["EXPERTS"], algorithm, seq
                        )
                        seq.write_results(results, dataset_dir)
                        write_results(
                            ws, dataset_dir, f"{seq.seq_info['seq_name']}_weight.txt",
                        )
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
                        np.savetxt(dataset_dir / f"{seq.seq_info['seq_name']}_time.txt", times)
                        total_time += times

                print(f"Total time: {sum(total_time)}s")
                # tracker_dir = os.path.join(config["EVAL_DIR"], "Ours", dataset_name)
                # eval_tracker(
                #     config["DATASET_DIR"],
                #     config["OUTPUT_DIR"],
                #     algorithm.name,
                #     dataset_name,
                #     config["EVAL_DIR"],
                #     tracker_dir,
                # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run algorithms")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="experiments/aaa.yaml",
        help="The config file of the algorithm",
    )
    args = parser.parse_args()
    main(args.config)
