import sys
import numpy as np

sys.path.append("external/iou-tracker")
from util import iou, nms


class IOU:
    def __init__(self):
        super(IOU, self).__init__()
        self.nms_overlap_thresh = None
        self.visdrone_classes = None
        self.nms_per_class = None
        self.with_classes = False
        self.sigma_l = None
        self.sigma_h = None
        self.t_min = None
        self.sigma_iou = None

    def initialize(self):
        self.tracks_active = []
        self.tracks_finished = []
        self.frame_num = -1

    def track(self, img_path, dets):
        self.frame_num += 1

        # apply low threshold to detections
        dets = self.preprocess(dets, self.with_classes)
        dets = [det for det in dets if det["score"] >= self.sigma_l]

        updated_tracks = []
        for track in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(
                    dets, key=lambda x: iou(track["bboxes"][-1], x["bbox"])
                )
                if iou(track["bboxes"][-1], best_match["bbox"]) >= self.sigma_iou:
                    track["bboxes"].append(best_match["bbox"])
                    track["max_score"] = max(track["max_score"], best_match["score"])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if (
                    track["max_score"] >= self.sigma_h
                    and len(track["bboxes"]) >= self.t_min
                ):
                    self.tracks_finished.append(track)

        # create new tracks
        new_tracks = [
            {
                "bboxes": [det["bbox"]],
                "max_score": det["score"],
                "start_frame": self.frame_num,
            }
            for det in dets
        ]
        self.tracks_active = updated_tracks + new_tracks

        results = [
            [
                id,
                track["bboxes"][0],  # x
                track["bboxes"][1],  # y
                track["bboxes"][2] - track["bboxes"][0],  # w
                track["bboxes"][3] - track["bboxes"][1],  # h
            ]
            for id, track in enumerate(self.tracks_active)
        ]
        return results

    def preprocess(self, dets, with_classes):
        bbox = dets[:, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        bbox -= 1  # correct 1,1 matlab offset
        scores = dets[:, 6]

        if with_classes:
            classes = dets[:, 7]

            bbox_filtered = None
            scores_filtered = None
            classes_filtered = None
            for coi in self.visdrone_classes:
                cids = classes == self.visdrone_classes[coi]
                if self.nms_per_class and self.nms_overlap_thresh:
                    bbox_tmp, scores_tmp = nms(
                        bbox[cids], scores[cids], self.nms_overlap_thresh
                    )
                else:
                    bbox_tmp, scores_tmp = bbox[cids], scores[cids]

                if bbox_filtered is None:
                    bbox_filtered = bbox_tmp
                    scores_filtered = scores_tmp
                    classes_filtered = [coi] * bbox_filtered.shape[0]
                elif len(bbox_tmp) > 0:
                    bbox_filtered = np.vstack((bbox_filtered, bbox_tmp))
                    scores_filtered = np.hstack((scores_filtered, scores_tmp))
                    classes_filtered += [coi] * bbox_tmp.shape[0]

            if bbox_filtered is not None:
                bbox = bbox_filtered
                scores = scores_filtered
                classes = classes_filtered

            if self.nms_per_class is False and self.nms_overlap_thresh:
                bbox, scores, classes = nms(
                    bbox, scores, self.nms_overlap_thresh, np.array(classes)
                )

        else:
            classes = ["pedestrian"] * bbox.shape[0]

        results = []
        for bb, s, c in zip(bbox, scores, classes):
            results.append(
                {"bbox": (bb[0], bb[1], bb[2], bb[3]), "score": s, "class": c}
            )
        return results
