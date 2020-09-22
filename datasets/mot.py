"""
https://github.com/shijieS/SST
"""


import os
import configparser
import numpy as np
import pandas as pd
from PIL import Image


FPS_DICT = {
    "Venice-2": 30,
    "KITTI-17": 10,
    "KITTI-13": 10,
    "ETH-Pedcross2": 14,
    "ETH-Bahnhof": 14,
    "ETH-Sunnyday": 14,
    "TUD-Campus": 25,
    "TUD-Stadtmitte": 25,
    "PETS09-S2L1": 7,
    "ADL-Rundle-6": 30,
    "ADL-Rundle-8": 30,
    "Venice-1": 30,
    "KITTI-19": 10,
    "KITTI-16": 10,
    "ADL-Rundle-3": 30,
    "ADL-Rundle-1": 30,
    "AVG-TownCentre": 2.5,
    "ETH-Crossing": 14,
    "ETH-Linthescher": 14,
    "ETH-Jelmoli": 14,
    "PETS09-S2L2": 7,
    "TUD-Crossing": 25,
}


class MOTDataReader:
    def __init__(
        self, seq_info, image_folder, detection_file_name, label_file_name=None,
    ):
        self.seq_info = seq_info
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, "{0:06d}.jpg")
        self.detection = pd.read_csv(self.detection_file_name, header=None, dtype=float)
        if label_file_name is not None:
            self.gt = pd.read_csv(label_file_name, header=None)
            self.gt_group = self.gt.groupby(0)
            self.gt_group_keys = list(self.gt_group.indices.keys())
        else:
            self.gt = None
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())
        self.seq_info["total_length"] = len(self.detection_group_keys)

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

    def get_label_by_index(self, index):
        if (
            self.gt is None
            or index > len(self.gt_group_keys)
            or self.gt_group_keys.count(index) == 0
        ):
            return None
        return self.gt_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return self.image_format.format(index)

    def __getitem__(self, item):
        return (
            self.get_image_by_index(item + 1),
            self.get_detection_by_index(item + 1),
            self.get_label_by_index(item + 1),
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
                        "fps": float(config.get("Sequence", "frameRate")),
                        "frame_width": float(config.get("Sequence", "imWidth")),
                        "frame_height": float(config.get("Sequence", "imHeight")),
                        "ini_path": seqinfo_path,
                    }
                else:
                    img = Image.open(os.path.join(image_dir, "000001.jpg"))
                    w, h = img.size
                    seq_info = {
                        "dataset_name": os.path.basename(self.root_dir),
                        "seq_name": sequence_name,
                        "fps": FPS_DICT.get(sequence_name, None),
                        "frame_width": w,
                        "frame_height": h,
                    }

                if data_type == "train":
                    gt_path = os.path.join(main_dir, sequence_name, "gt", "gt.txt")
                else:
                    gt_path = None
                self.sequences[data_type].append(
                    MOTDataReader(
                        seq_info, image_dir, det_path, label_file_name=gt_path
                    )
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
