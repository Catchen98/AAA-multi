import sys
import numpy as np
import cv2

sys.path.append("external/deep_sort")
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from application_util import preprocessing
from tools.generate_detections import create_box_encoder


class DeepSort:
    def __init__(self):
        super(DeepSort, self).__init__()
        self.min_confidence = 0.3
        self.nn_budget = 100
        self.min_detection_height = 0
        self.nms_max_overlap = 1.0
        self.max_cosine_distance = 0.2
        self.model = "resources/networks/mars-small128.pb"
        self.encoder = create_box_encoder(self.model, batch_size=32)
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )

    def initialize(self):
        self.tracker = Tracker(self.metric)
        self.frame_idx = -1
        self.history = []

    def track(self, img_path, dets):
        self.frame_idx += 1

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
        self.history.append(results)
        return results
