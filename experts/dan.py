import sys
import cv2

from experts.expert import Expert

sys.path.append("external/SST")
from tracker import SSTTracker, TrackerConfig
from config.config import config, init_test_mot15, init_test_mot17, init_test_ua


class DAN(Expert):
    def __init__(self, model_path):
        super(DAN, self).__init__("DAN")
        self.model_path = model_path

    def initialize(self, seq_info):
        super(DAN, self).initialize(seq_info)
        self.dataset_name = seq_info["dataset_name"]
        self.seq_name = seq_info["seq_name"]

        if self.dataset_name == "MOT15" and self.seq_name == "AVG-TownCentre":
            self.choice = (4, 0, 4, 4, 5, 4)
            init_test_mot15()
        elif self.dataset_name == "MOT15":
            self.choice = (0, 0, 4, 4, 5, 4)
            init_test_mot15()
        elif self.dataset_name == "MOT17":
            self.choice = (0, 0, 4, 0, 3, 3)
            init_test_mot17()
        elif self.dataset_name == "DETRAC":
            self.choice = TrackerConfig.get_ua_choice()
            init_test_ua()
        else:
            self.choice = (0, 0, 4, 0, 3, 3)
            init_test_mot17()

        TrackerConfig.set_configure(self.choice)

        if self.dataset_name == "MOT17":
            self.min_confidence = 0.0
        else:
            self.min_confidence = None

        config["resume"] = self.model_path
        self.tracker = SSTTracker()

    def track(self, img_path, dets):
        super(DAN, self).track(img_path, dets)

        if dets is None:
            return []

        img, dets, h, w = self.preprocess(img_path, dets)

        if len(dets) == 0:
            return []

        self.tracker.update(img, dets, False, self.frame_idx)

        result = []
        for t in self.tracker.tracks:
            n = t.nodes[-1]
            if t.age == 1:
                b = n.get_box(self.tracker.frame_index - 1, self.tracker.recorder)
                result.append([t.id, b[0] * w, b[1] * h, b[2] * w, b[3] * h])

        return result

    def preprocess(self, img_path, dets):
        img = cv2.imread(img_path)

        if self.min_confidence is not None:
            dets = dets[dets[:, 6] > self.min_confidence, :]

        if len(dets) > config["max_object"]:
            dets = dets[: config["max_object"], :]

        h, w, _ = img.shape
        dets[:, [2, 4]] /= float(w)
        dets[:, [3, 5]] /= float(h)

        return img, dets[:, 2:6], h, w
