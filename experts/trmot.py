import sys
import cv2
import torch
import numpy as np
from types import SimpleNamespace

from experts.expert import Expert

sys.path.append("external/Towards-Realtime-MOT")
from tracker.multitracker import JDETracker


def letterbox(
    img, height=608, width=1088, color=(127.5, 127.5, 127.5)
):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, dw, dh


class TRMOT(Expert):
    def __init__(self, opt):
        super(TRMOT, self).__init__("TRMOT")
        self.width = 1088
        self.height = 608
        self.opt = SimpleNamespace(**opt)

    def initialize(self, seq_info):
        super(TRMOT, self).initialize(seq_info)

        self.opt.img_size = [
            int(seq_info["frame_width"]),
            int(seq_info["frame_height"]),
        ]

        self.tracker = JDETracker(self.opt)

    def track(self, img_path, dets):
        super(TRMOT, self).track(img_path, dets)

        img, img0 = self.preprocess(img_path)

        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = self.tracker.update(blob, img0)
        result = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                result.append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])

        return result

    def preprocess(self, img_path):
        img0 = cv2.imread(img_path)
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img, img0
