"""
https://github.com/shijieS/SST
"""


import os
import numpy as np
import pandas as pd


class KITTIDataReader:
    def __init__(self, name, image_folder, detection_file_name):
        datatype = {
            0: int,
            1: int,
            2: str,
            3: int,
            4: int,
            5: float,
            6: float,
            7: float,
            8: float,
            9: float,
            10: float,
            11: float,
            12: float,
            13: float,
            14: float,
            15: float,
            16: float,
        }
        self.name = name
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, "{0:06d}.jpg")
        self.detection = pd.read_csv(
            self.detection_file_name, sep=" ", header=None, dtype=datatype
        )

        self.detection = self.detection.iloc[:, 0:17]
        select_type_row = [not (t in ("DontCare")) for t in self.detection[2]]
        # select_score_row = [t > 0 for t in self.detection[17]]
        select_occluded_row = [t in [0, 1] for t in self.detection[4]]
        select_row = select_type_row and select_occluded_row  # and select_score_row

        self.detection = self.detection[select_row]
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

        det = self.detection_group.get_group(index).values
        det[:, [8, 9]] = det[:, [8, 9]] - det[:, [6, 7]]
        return det[:, 4:10]

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
        data[:, 6:] = -1
        data[:, :2] -= 1
        df = pd.DataFrame(data,)

        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.name}.txt"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False, header=False)


class KITTI:
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
                self.sequences[data_type].append(
                    KITTIDataReader(sequence_name, image_dir, det_path)
                )

        self.all_sequence_names = sum(self.sequence_names.values(), [])
        self.all_sequences = sum(self.sequences.values(), [])

        self.c = 0

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, item):
        return self.all_sequences[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self.c < len(self):
            v = self[self.c]
            self.c += 1
            return v
        else:
            raise StopIteration()
