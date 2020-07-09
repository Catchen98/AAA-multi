import sys
import cv2

from experts.expert import Expert

sys.path.append("external/Deep-TAMA")
from tracking.config import config
from tracking.main_tracker import track


class DeepTAMA(Expert):
    def __init__(self):
        super(DeepTAMA, self).__init__("DeepTAMA")

    def initialize(self, seq_info):
        super(DeepTAMA, self).initialize(seq_info)

        dataset_name = seq_info["dataset_name"]
        seq_name = seq_info["seq_name"]

        if dataset_name == "MOT17" and "DPM" in seq_name:
            det_thresh = 0.1
        elif dataset_name == "MOT17" and "FRCNN" in seq_name:
            det_thresh = 0.7
        elif dataset_name == "MOT17" and "SDP" in seq_name:
            det_thresh = 0.8
        elif dataset_name == "MOT16":
            det_thresh = 0.1
        elif seq_name == "MOT20-01" or seq_name == "MOT20-02":
            det_thresh = 0.1
        elif seq_name == "MOT20-03" or seq_name == "MOT20-04" or seq_name == "MOT20-05":
            det_thresh = 0.0
        elif seq_name == "MOT20-07" or seq_name == "MOT20-08":
            det_thresh = 0.2
        else:
            det_thresh = 0.1

        # Get tracking parameters
        _config = config(seq_info["fps"])
        _config.det_thresh = det_thresh

        _seq_info = [seq_info["frame_width"], seq_info["frame_height"], seq_info["fps"]]

        self._track = track(
            None,
            _seq_info,
            None,
            _config,
            semi_on=False,
            fr_delay=0,
            visualization=False,
        )

    def track(self, img_path, dets):
        super(DeepTAMA, self).track(img_path, dets)
        bgr_img, dets = self.preprocess(img_path, dets)
        self._track.track(bgr_img, dets, self.frame_idx)

        results = []
        for trk in self._track.trk_result[-1]:
            results.append([trk[0], trk[1], trk[2], trk[3], trk[4]])

        return results

    def preprocess(self, img_path, dets):
        bgr_img = cv2.imread(img_path)
        if dets is not None:
            dets = dets[:, 1:8]
        else:
            dets = []
        return bgr_img, dets
