import os
import contextlib
import numpy as np


class Expert:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def initialize(self):
        self.history = []
        self.frame_idx = -1

    def track(self, img_path, dets):
        self.frame_idx += 1

    def track_seq(self, seq):
        for frame_idx, (img_path, dets) in enumerate(seq):
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                results = self.track(img_path, dets)
            results = np.array(results)
            if len(results) > 0:
                frame_results = np.zeros((results.shape[0], results.shape[1] + 1))
                frame_results[:, 1:] = results
                frame_results[:, 0] = frame_idx + 1
                self.history.append(frame_results)
        self.history = np.concatenate(self.history, axis=0)
        return self.history
