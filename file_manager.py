import os
import pandas as pd

from paths import OUTPUT_PATH


class ReadResult:
    def __init__(self, dataset_name, expert_name, seq_name):
        self.results = pd.read_csv(
            OUTPUT_PATH / dataset_name / expert_name / f"{seq_name}.txt", header=None
        )
        self.results_group = self.results.groupby(0)
        self.frames = list(self.results_group.indices.keys())

    def get_result_by_frame(self, frame_idx):
        if self.frames.count(frame_idx + 1) == 0:
            return []
        else:
            value = self.results_group.get_group(frame_idx + 1).values
            return value[:, 1:6]


def write_results(data, output_dir, filename):
    df = pd.DataFrame(data,)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False, header=False)
