import sys
import numpy as np
import cv2

from experts.expert import Expert

sys.path.append("external/deep_sort")
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from application_util import preprocessing
from tools.generate_detections import create_box_encoder


class DeepSort(Expert):
    def __init__(
        self,
        model,
        min_confidence=0.8,
        min_detection_height=0,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
    ):
        super(DeepSort, self).__init__("DeepSort")

        self.model = model
        self.encoder = create_box_encoder(self.model, batch_size=32)

        self.min_confidence = min_confidence
        self.min_detection_height = min_detection_height
        self.nms_max_overlap = nms_max_overlap

        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )

    def initialize(self):
        super(DeepSort, self).initialize()
        self.tracker = Tracker(self.metric)

    def track(self, img_path, dets):
        super(DeepSort, self).track(img_path, dets)

        detections = self.preprocess(img_path, dets)

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        # Store results.
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
        bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        features = self.encoder(bgr_image, dets[:, 2:6].copy())
        detections_out = [np.r_[(row, feature)] for row, feature in zip(dets, features)]

        # Load image and generate detections.
        detection_list = []
        for row in detections_out:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < self.min_detection_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        detections = [d for d in detection_list if d.confidence >= self.min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections
