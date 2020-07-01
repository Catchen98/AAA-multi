import os
import numpy as np
import pandas as pd

from datasets.mot import MOT
from algorithms.aaa import AAA
from paths import DATASET_PATH, OUTPUT_PATH
from print_manager import do_not_print


class ReadResult:
    def __init__(self, dataset_name, expert_name, seq_name):
        self.results = pd.read_csv(
            OUTPUT_PATH / dataset_name / expert_name / f"{seq_name}.txt", header=None
        )
        self.results_group = self.results.groupby(0)
        self.frames = list(self.results_group.indices.keys())

    def get_result_by_frame(self, frame_idx):
        if self.frames.count(frame_idx) == 0:
            return []
        else:
            value = self.results_group.get_group(frame_idx).values
            return value[:, 1:6]


def write_results(self, data, output_dir, filename):
    df = pd.DataFrame(data,)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False, header=False)


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
    for frame_idx, (img_path, dets) in enumerate(seq):
        expert_results = []
        for reader in experts_reader:
            expert_results.append(reader.get_result_by_frame(frame_idx))

        result, w, expert_loss, feedback = algorithm.track(
            img_path, dets, expert_results
        )
        if len(result) > 0:
            frame_result = np.zeros((result.shape[0], result.shape[1] + 1))
            frame_result[:, 1:] = result
            frame_result[:, 0] = frame_idx + 1
            results.append(frame_result)

        frame_w = np.zeros((len(w) + 1))
        frame_w[1:] = w
        frame_w[0] = frame_idx + 1
        ws.append(frame_w)

        if expert_loss is not None:
            frame_expert_loss = np.zeros((len(expert_loss) + 1))
            frame_expert_loss[1:] = expert_loss
            frame_expert_loss[0] = frame_idx + 1
            expert_losses.append(frame_expert_loss)

        if feedback is not None:
            frame_feedback = np.zeros((feedback.shape[0], feedback.shape[1] + 1))
            frame_feedback[:, 1:] = feedback
            frame_feedback[:, 0] = frame_idx + 1
            feedbacks.append(frame_feedback)

    results = np.concatenate(results, axis=0)
    ws = np.concatenate(ws, axis=0)
    expert_losses = np.concatenate(expert_losses, axis=0)
    feedbacks = np.concatenate(feedbacks, axis=0)

    return results, ws, expert_losses, feedbacks


@do_not_print
def get_algorithm(experts_name):
    algorithm = AAA(len(experts_name))
    return algorithm


def main(experts_name):
    datasets = {
        "MOT15": MOT(DATASET_PATH["MOT15"]),
        "MOT16": MOT(DATASET_PATH["MOT16"]),
        "MOT17": MOT(DATASET_PATH["MOT17"]),
    }

    algorithm = get_algorithm(experts_name)

    for dataset_name, dataset in datasets.items():
        dataset_dir = OUTPUT_PATH / dataset_name / "Algorithm"

        for seq in dataset:
            if (dataset_dir / f"{seq.seq_info['seq_name']}.txt").exists():
                print(f"Pass {seq.seq_info['seq_name']}")
            else:
                print(f"Start {seq.seq_info['seq_name']}")
                results, ws, expert_losses, feedbacks = track_seq(
                    experts_name, algorithm, seq
                )
                seq.write_results(results, dataset_dir)
                write_results(ws, dataset_dir, f"{seq.seq_info['seq_name']}_weight.txt")
                write_results(
                    expert_losses, dataset_dir, f"{seq.seq_info['seq_name']}_loss.txt"
                )
                write_results(
                    feedbacks, dataset_dir, f"{seq.seq_info['seq_name']}_feedback.txt"
                )


if __name__ == "__main__":
    main(["DAN", "DeepSort", "DeepTAMA", "Sort", "MOTDT"])
