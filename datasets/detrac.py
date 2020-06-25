"""
https://github.com/shijieS/SST
"""


import os
import pandas as pd
import cv2
import numpy as np

"""
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
"""


class DETRACDataReader:
    def __init__(self, name, image_folder, gt_file_name, ignore_file_name):
        datatype = {0: int, 1: int, 2: float, 3: float, 4: float, 5: float}
        datatype_ignore = {0: float, 1: float, 2: float, 3: float}

        self.name = name
        self.image_folder = image_folder
        self.gt_file_name = gt_file_name
        self.ignore_file_name = ignore_file_name
        self.image_format = os.path.join(self.image_folder, "img{0:05d}.jpg")
        self.detection = pd.read_csv(
            self.gt_file_name, sep=",", header=None, dtype=datatype
        )

        # read ignore file
        mask = None
        if ignore_file_name is not None and os.stat(self.ignore_file_name).st_size > 0:
            self.ignore = pd.read_csv(
                self.ignore_file_name, sep=",", header=None, dtype=datatype_ignore
            )
            self.ignore = self.ignore.values
            ls = self.detection.iloc[:, 2].values
            ts = self.detection.iloc[:, 3].values
            rs = self.detection.iloc[:, 4].values
            bs = self.detection.iloc[:, 5].values

            self.ignore = np.array(
                [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in self.ignore]
            )
            for rect in self.ignore:
                left = rect[0]
                top = rect[1]
                right = rect[2]
                bottom = rect[3]
                res = np.logical_and(
                    np.logical_and(
                        np.logical_and(left < ls, ls < right),
                        np.logical_and(left < rs, rs < right),
                    ),
                    np.logical_and(
                        np.logical_and(top < ts, ts < bottom),
                        np.logical_and(top < bs, bs < bottom),
                    ),
                )
                if mask is None:
                    mask = res
                else:
                    mask = np.logical_or(mask, res)
        if mask is not None:
            self.detection = self.detection[np.logical_not(mask)]

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

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (self.get_image_by_index(item), self.get_detection_by_index(item))

    def __iter__(self):
        return self

    def __next__(self):
        if self.c < len(self):
            v = self[self.c]
            self.c += 1
            return v
        else:
            raise StopIteration()

    def write_results(self, results, output_dir):
        data = np.zeros((len(results), 10))
        data[:, :6] = results
        data[:, 6:] = -1
        df = pd.DataFrame(data)

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{self.name}.txt")
        df.to_csv(file_path, index=False, header=False)


class DETRAC:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.sequence_names = {"train": [], "test": []}
        self.sequences = {"train": [], "test": []}

        for data_type in self.sequence_names.keys():
            main_dir = os.path.join(self.root_dir, "test")
            img_dir = os.path.join(main_dir, "imgs")
            det_dir = os.path.join(main_dir, "dets")

            sequence_basenames = [
                name
                for name in os.listdir(img_dir)
                if os.path.isdir(os.path.join(img_dir, name))
            ]
            det_names = [
                name
                for name in os.listdir(det_dir)
                if os.path.isdir(os.path.join(det_dir, name))
            ]

            for sequence_basename in sequence_basenames:
                image_dir = os.path.join(img_dir, sequence_basename)
                for det_name in det_names:
                    det_path = os.path.join(
                        det_dir, det_name, f"{sequence_basename}_Det_{det_name}.txt",
                    )
                    sequence_name = f"{sequence_basename}_Det_{det_name}"
                    self.sequence_names[data_type].append(sequence_name)
                    self.sequences[data_type].append(
                        DETRACDataReader(sequence_name, image_dir, det_path)
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
