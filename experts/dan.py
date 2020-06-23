import sys
import cv2

from experts.expert import Expert

sys.path.append("external/SST")
from tracker import SSTTracker, TrackerConfig
from config.config import config


class DAN(Expert):
    def __init__(self, model_path, choice=None):
        super(DAN, self).__init__("DAN")

        self.choice = choice
        if self.choice is not None:
            TrackerConfig.set_configure(self.choice)
        config["resume"] = model_path

    def initialize(self):
        super(DAN, self).initialize()
        self.tracker = SSTTracker()

    def track(self, img_path, dets):
        super(DAN, self).track(img_path, dets)

        if dets is None:
            return []

        img, dets, h, w = self.preprocess(img_path, dets)
        self.tracker.update(img, dets[:, 2:6], False, self.frame_idx + 1)

        results = []
        for t in self.tracker.tracks:
            n = t.nodes[-1]
            if t.age == 1:
                b = n.get_box(self.tracker.frame_index, self.tracker.recorder)
                results.append([t.id, b[0] * w, b[1] * h, b[2] * w, b[3] * h])
        return results

    def preprocess(self, img_path, dets):
        if len(dets) > config["max_object"]:
            dets = dets[: config["max_object"], :]

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        dets[:, [2, 4]] /= float(w)
        dets[:, [3, 5]] /= float(h)
        return img, dets, h, w
