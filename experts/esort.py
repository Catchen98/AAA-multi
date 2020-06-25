import sys
import numpy as np

from expert import Expert

sys.path.append("external/sort")
from sort import Sort, KalmanBoxTracker


class ESort(Expert):
    def __init__(self):
        super(ESort, self).__init__("Sort")

    def initialize(self, dataset_name, seq_name):
        super(ESort, self).initialize()
        self.tracker = Sort()
        KalmanBoxTracker.count = 0

    def track(self, img_path, dets):
        super(ESort, self).track(img_path, dets)

        dets = self.preprocess(dets)
        results = self.tracker.update(dets)  # x1, y1, x2, y2, ID
        results[:, 2:4] -= results[:, 0:2]  # x, y, w, h, ID
        results = results[:, [4, 0, 1, 2, 3]]
        return results

    def preprocess(self, dets):
        if dets is not None:
            dets = dets[:, 2:7]  # x, y, w, h, score
            dets[:, 2:4] += dets[:, 0:2]  # x, y, w, h to x1, y1, x2, y2
        else:
            dets = np.empty((0, 5))
        return dets
