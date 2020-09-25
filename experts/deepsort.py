import sys
import numpy as np
import cv2

from experts.expert import Expert

sys.path.append("external/deep_sort")
from application_util import preprocessing
from tools.generate_detections import create_box_encoder
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


class DeepSORT(Expert):
    def __init__(
        self,
        model,
        min_confidence,
        min_detection_height,
        nms_max_overlap,
        max_cosine_distance,
        nn_budget,
    ):
        super(DeepSORT, self).__init__("DeepSORT")

        self.encoder = create_box_encoder(model, batch_size=32)
        self.min_confidence = min_confidence
        self.min_detection_height = min_detection_height
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

    def initialize(self, seq_info):
        super(DeepSORT, self).initialize(seq_info)

        dataset_name = seq_info["dataset_name"]

        if dataset_name == "MOT16":
            self.min_confidence = 0.3
            self.nn_budget = 100

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )
        self.tracker = Tracker(metric)

    def track(self, img_path, dets):
        super(DeepSORT, self).track(img_path, dets)

        detections = self.preprocess(img_path, dets)

        self.tracker.predict()
        self.tracker.update(detections)

        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return results

    def preprocess(self, img_path, dets):
        if dets is None:
            return []

        detections_out = self.generate_detections(img_path, dets)
        detection_list = self.create_detections(detections_out)
        detections = [d for d in detection_list if d.confidence >= self.min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        return detections

    def generate_detections(self, img_path, dets):
        bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        features = self.encoder(bgr_image, dets[:, 2:6].copy())
        detections_out = [np.r_[(row, feature)] for row, feature in zip(dets, features)]

        return detections_out

    def create_detections(self, detections_out):
        detection_list = []
        for row in detections_out:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < self.min_detection_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        return detection_list
