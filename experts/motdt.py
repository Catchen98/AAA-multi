import sys
import numpy as np
from scipy.misc import imread

from experts.expert import Expert

sys.path.append("external/MOTDT")
from tracker.mot_tracker import OnlineTracker


class MOTDT(Expert):
    def __init__(self, min_height, min_det_score):
        super(MOTDT, self).__init__("MOTDT")
        self.min_height = min_height

        if min_det_score is None:
            min_det_score = -np.inf
        self.min_det_score = min_det_score

    def initialize(self, seq_info):
        super(MOTDT, self).initialize(seq_info)

        self.tracker = OnlineTracker()

    def track(self, img_path, dets):
        super(MOTDT, self).track(img_path, dets)

        im, tlwhs, scores = self.preprocess(img_path, dets)

        online_targets = self.tracker.update(im, tlwhs, None)

        results = []
        for t in online_targets:
            tracking_id = t.track_id
            bbox = t.tlwh
            results.append([tracking_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return results

    def preprocess(self, img_path, dets):
        im = imread(img_path)  # rgb
        im = im[:, :, ::-1]  # bgr

        if dets is None:
            return im, [], []

        tlwhs = dets[:, 2:6]
        scores = dets[:, 6]

        keep = (tlwhs[:, 3] >= self.min_height) & (scores > self.min_det_score)
        tlwhs = tlwhs[keep]
        scores = scores[keep]

        return im, tlwhs, scores
