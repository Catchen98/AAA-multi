import sys
import pandas as pd
import numpy as np

sys.path.append("external/mot_neural_solver/src")
from mot_neural_solver.data.seq_processing.MOT15loader import FPS_DICT as MOT15_FPS_DICT
from mot_neural_solver.data.seq_processing.MOT17loader import (
    MOV_CAMERA_DICT as MOT17_MOV_CAMERA_DICT,
)
from mot_neural_solver.data.seq_processing.MOT15loader import (
    MOV_CAMERA_DICT as MOT15_MOV_CAMERA_DICT,
)

MOV_CAMERA_DICT = {**MOT15_MOV_CAMERA_DICT, **MOT17_MOV_CAMERA_DICT}
DET_COL_NAMES = ("frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf")


class DataFrameWSeqInfo(pd.DataFrame):
    _metadata = ["seq_info_dict"]

    @property
    def _constructor(self):
        return DataFrameWSeqInfo


class MOTSeqProcessor:
    def __init__(
        self, seq_name, img_paths, det_df, h, w,
    ):
        self.seq_name = seq_name
        self.img_paths = img_paths
        self.det_df = det_df
        self.h = h
        self.w = w

    def _ensure_boxes_in_frame(self):
        initial_bb_top = self.det_df["bb_top"].values.copy()
        initial_bb_left = self.det_df["bb_left"].values.copy()

        self.det_df["bb_top"] = np.maximum(self.det_df["bb_top"].values, 0).astype(int)
        self.det_df["bb_left"] = np.maximum(self.det_df["bb_left"].values, 0).astype(
            int
        )

        bb_top_diff = self.det_df["bb_top"].values - initial_bb_top
        bb_left_diff = self.det_df["bb_left"].values - initial_bb_left

        self.det_df["bb_height"] -= bb_top_diff
        self.det_df["bb_width"] -= bb_left_diff

        img_height, img_width = (
            self.det_df.seq_info_dict["frame_height"],
            self.det_df.seq_info_dict["frame_width"],
        )
        self.det_df["bb_height"] = np.minimum(
            img_height - self.det_df["bb_top"], self.det_df["bb_height"]
        ).astype(int)
        self.det_df["bb_width"] = np.minimum(
            img_width - self.det_df["bb_left"], self.det_df["bb_width"]
        ).astype(int)

    def det_df_loader(self):
        # Number and order of columns is always assumed to be the same
        det_df = self.det_df[self.det_df.columns[: len(DET_COL_NAMES)]]
        det_df.columns = DET_COL_NAMES

        det_df["bb_left"] -= 1  # Coordinates are 1 based
        det_df["bb_top"] -= 1

        # If id already contains an ID assignment (e.g. using tracktor output), keep it
        if len(det_df["id"].unique()) > 1:
            det_df["tracktor_id"] = det_df["id"]

        # Add each image's path (in MOT17Det data dir)
        det_df["frame_path"] = det_df.apply(
            lambda row: self.img_paths[int(row.frame) - 1], axis=1
        )

        if self.seq_name in MOT15_FPS_DICT.keys():
            fps = MOT15_FPS_DICT[self.seq_name]
        seq_info_dict = {
            "fps": fps,
            "mov_camera": MOV_CAMERA_DICT[self.seq_name],
            "frame_height": self.h,
            "frame_width": self.w,
            "is_gt": False,
        }

        return det_df, seq_info_dict, None

    def _get_det_df(self):
        self.det_df, seq_info_dict, self.gt_df = self.det_df_loader()

        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Some further processing
        if self.seq_name in MOT15_MOV_CAMERA_DICT.keys():
            self._ensure_boxes_in_frame()

        # Add some additional box measurements that might be used for graph construction
        self.det_df["bb_bot"] = (
            self.det_df["bb_top"] + self.det_df["bb_height"]
        ).values
        self.det_df["bb_right"] = (
            self.det_df["bb_left"] + self.det_df["bb_width"]
        ).values
        self.det_df["feet_x"] = self.det_df["bb_left"] + 0.5 * self.det_df["bb_width"]
        self.det_df["feet_y"] = self.det_df["bb_top"] + self.det_df["bb_height"]

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        frame_height, frame_width = (
            self.det_df.seq_info_dict["frame_height"],
            self.det_df.seq_info_dict["frame_width"],
        )
        conds = (self.det_df["bb_width"] > 0) & (self.det_df["bb_height"] > 0)
        conds = conds & (self.det_df["bb_right"] > 0) & (self.det_df["bb_bot"] > 0)
        conds = (
            conds
            & (self.det_df["bb_left"] < frame_width)
            & (self.det_df["bb_top"] < frame_height)
        )
        self.det_df = self.det_df[conds].copy()

        self.det_df.sort_values(by="frame", inplace=True)
        self.det_df["detection_id"] = np.arange(
            self.det_df.shape[0]
        )  # This id is used for future tastks

        return self.det_df

    def load_or_process_detections(self):
        # See class header
        self._get_det_df()

        return self.det_df
