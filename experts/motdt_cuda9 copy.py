import sys
import numpy as np
from PIL import Image

from expert import Expert

sys.path.append("external/MOTDT")
from tracker.mot_tracker import OnlineTracker


class MOTDT(Expert):
    def __init__(self):
        super(MOTDT, self).__init__("MOTDT")

    def initialize(self, seq_info):
        super(MOTDT, self).initialize(seq_info)

        self.min_height = 0
        self.min_det_score = -np.inf
        self.tracker = OnlineTracker()

    def track(self, img_path, dets):
        super(MOTDT, self).track(img_path, dets)

        frame, det_tlwhs, det_scores = self.preprocess(img_path, dets)

        online_targets = self.tracker.update(frame, det_tlwhs, None)
        results = []
        for t in online_targets:
            results.append([t.track_id, t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3]])
        return results

    def preprocess(self, img_path, dets):
        im = Image.open(img_path)  # rgb
        im = np.array(im)
        im = im[:, :, ::-1]  # bgr

        if dets is None:
            return im, [], []

        tlwhs = dets[:, 2:6]
        scores = dets[:, 6]

        keep = (tlwhs[:, 3] >= self.min_height) & (scores > self.min_det_score)
        tlwhs = tlwhs[keep]
        scores = scores[keep]

        return im, tlwhs, scores
