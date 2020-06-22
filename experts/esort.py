import sys

sys.path.append("external/sort")
from sort import Sort


class ESort:
    def __init__(self):
        super(ESort, self).__init__()

    def initialize(self):
        self.tracker = Sort()

    def track(self, img_path, dets):
        dets = dets[:, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # x, y, w, h to x1, y1, x2, y2
        results = self.tracker.update(dets)  # ID, x1, y1, x2, y2
        results[:, 3:5] -= results[:, 1:3]  # ID, x, y, w, h
        return results
