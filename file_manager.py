import os
import pandas as pd


class ReadResult:
    def __init__(self, output_dir, dataset_name, expert_name, seq_name):
        self.results = pd.read_csv(
            os.path.join(output_dir, dataset_name, expert_name, f"{seq_name}.txt"),
            header=None,
            sep=' |,',
        )
        self.results_group = self.results.groupby(0)
        self.frames = list(self.results_group.indices.keys())

    def get_result_by_frame(self, frame_idx, with_frame=False):
        if self.frames.count(frame_idx + 1) == 0:
            return []
        else:
            value = self.results_group.get_group(frame_idx + 1).values
            if with_frame:
                return value[:, :6]
            else:
                return value[:, 1:6]


def write_results(data, output_dir, filename):
    df = pd.DataFrame(data,)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False, header=False)
