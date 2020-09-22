import sys

from experts.expert import Expert

sys.path.append("external/UMA-MOT/UMA-TEST")
from tracker.detection import Detection
from tracker.mot_tracker import MOT_Tracker


class UMA(Expert):
    def __init__(
        self,
        life_span,
        occlusion_thres,
        association_thres,
        checkpoint,
        context_amount,
        iou,
    ):
        super(UMA, self).__init__("UMA")
        self.life_span = life_span
        self.occlusion_thres = occlusion_thres
        self.association_thres = association_thres
        self.checkpoint = checkpoint
        self.context_amount = context_amount
        self.iou = iou

    def initialize(self, seq_info):
        super(UMA, self).initialize(seq_info)

        max_age = int(self.life_span * int(seq_info["fps"]))

        self.tracker = MOT_Tracker(
            max_age, self.occlusion_thres, self.association_thres
        )
        self.tracker.frame_rate = int(seq_info["fps"])

    def track(self, img_path, dets):
        super(UMA, self).track(img_path, dets)

        detections = self.preprocess(dets)

        trackers = self.tracker.update(
            img_path, self.checkpoint, self.context_amount, detections, self.iou
        )

        result = []
        for d in trackers:
            result.append([d[4], d[0], d[1], d[2], d[3]])

        return result

    def preprocess(self, dets):
        if dets is None:
            return []

        detection_list = []
        for row in dets:
            bbox, confidence = row[2:6], row[6]
            detection_list.append(Detection(bbox, confidence))
        return detection_list
