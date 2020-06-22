"""
https://github.com/shijieS/SST
"""


import os
import numpy as np
import pandas as pd
import cv2


class MOTDataReader:
    def __init__(self, image_folder, detection_file_name, min_confidence=None):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, "{0:06d}.jpg")
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

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

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (
            self.get_image_by_index(item + 1),
            self.get_detection_by_index(item + 1),
        )


class MOT:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.sequence_names = {"train": [], "test": []}
        self.sequences = {"train": [], "test": []}

        for data_type in self.sequence_names.keys():
            main_dir = os.path.join(self.root_dir, "test")

            self.sequence_names[data_type] = [
                name
                for name in os.listdir(main_dir)
                if os.path.isdir(os.path.join(main_dir, name))
            ]

            for sequence_name in self.sequence_names[data_type]:
                image_dir = os.path.join(main_dir, sequence_name, "img1")
                det_path = os.path.join(main_dir, sequence_name, "det", "det.txt")
                self.sequences[data_type].append(MOTDataReader(image_dir, det_path))

    def __len__(self):
        return len(self.sequence_names["test"])

    def __getitem__(self, item):
        return (
            self.sequence_names["test"][item],
            self.sequences["test"][item],
        )

    def write_results(self, sequence_name, results, output_dir):
        data = np.zeros((len(results), 10))
        data[:, :6] = results
        data[:, 6:] = -1
        df = pd.DataFrame(data)

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{sequence_name}.txt")
        df.to_csv(file_path, index=False)
