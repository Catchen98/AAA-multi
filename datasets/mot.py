"""
https://github.com/shijieS/SST
"""


import os
import configparser
import numpy as np
import pandas as pd


class MOTDataReader:
    def __init__(
        self, seq_info, image_folder, detection_file_name, min_confidence=None
    ):
        self.seq_info = seq_info
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, "{0:06d}.jpg")
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

        self.c = 0

    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if (
            index > len(self.detection_group_keys)
            or self.detection_group_keys.count(index) == 0
        ):
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return self.image_format.format(index)

    def __getitem__(self, item):
        return (
            self.get_image_by_index(item + 1),
            self.get_detection_by_index(item + 1),
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.c < len(self):
            v = self[self.c]
            self.c += 1
            return v
        else:
            raise StopIteration()

    def write_results(self, results, output_dir, filename=None):
        data = np.zeros((len(results), 10))
        data[:, :6] = results
        data[:, 6:] = 1
        data[:, 7:] = -1
        df = pd.DataFrame(data,)

        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.seq_info['seq_name']}.txt"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False, header=False)


class MOT:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.sequence_names = {"train": [], "test": []}
        self.sequences = {"train": [], "test": []}

        for data_type in self.sequence_names.keys():
            main_dir = os.path.join(self.root_dir, data_type)

            self.sequence_names[data_type] = [
                name
                for name in os.listdir(main_dir)
                if os.path.isdir(os.path.join(main_dir, name))
            ]

            for sequence_name in self.sequence_names[data_type]:
                image_dir = os.path.join(main_dir, sequence_name, "img1")
                det_path = os.path.join(main_dir, sequence_name, "det", "det.txt")
                seqinfo_path = os.path.join(main_dir, sequence_name, "seqinfo.ini")
                if os.path.exists(seqinfo_path):
                    config = configparser.ConfigParser()
                    config.read(seqinfo_path)
                    seq_info = {
                        "dataset_name": os.path.basename(self.root_dir),
                        "seq_name": sequence_name,
                        "fps": config.get("Sequence", "frameRate"),
                        "frame_width": config.get("Sequence", "imWidth"),
                        "frame_height": config.get("Sequence", "imHeight"),
                    }
                else:
                    seq_info = {
                        "dataset_name": os.path.basename(self.root_dir),
                        "seq_name": sequence_name,
                    }
                self.sequences[data_type].append(
                    MOTDataReader(seq_info, image_dir, det_path)
                )

        self.all_sequence_names = sum(self.sequence_names.values(), [])
        self.all_sequences = sum(self.sequences.values(), [])

        self.c = 0

    def __len__(self):
        return len(self.sequences["train"])

    def __getitem__(self, item):
        return self.sequences["train"][item]

    def __iter__(self):
        return self

    def __next__(self):
        if self.c < len(self):
            v = self[self.c]
            self.c += 1
            return v
        else:
            raise StopIteration()
