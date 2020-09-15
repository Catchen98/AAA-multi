import numpy as np
from print_manager import do_not_print


class Expert:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def initialize(self, seq_info, *args, **kwargs):
        self.history = []
        self.frame_idx = -1

    def track(self, img_path, dets, *args, **kwargs):
        self.frame_idx += 1

    @do_not_print
    def track_seq(self, seq):
        self.initialize(seq.seq_info)

        for frame_idx, (img_path, dets, _) in enumerate(seq):
            results = self.track(img_path, dets)
            results = np.array(results)
            if len(results) > 0:
                frame_results = np.zeros((results.shape[0], results.shape[1] + 1))
                frame_results[:, 1:] = results
                frame_results[:, 0] = frame_idx + 1
                self.history.append(frame_results)
        self.history = np.concatenate(self.history, axis=0)
        return self.history
