import sys
import numpy as np
import cv2

from experts.expert import Expert

sys.path.append("external/iou-tracker")
from viou_tracker import associate
from vis_tracker import VisTracker
from util import iou, nms


class VIOU(Expert):
    def __init__(
        self,
        nms_overlap_thresh=None,
        nms_per_class=True,
        with_classes=False,
        sigma_l=0,
        sigma_h=0.5,
        sigma_iou=0.5,
        t_min=2,
        ttl=1,
        tracker_type="NONE",
        keep_upper_height_ratio=1.0,
    ):
        super(VIOU, self).__init__("VIOU")
        self.nms_overlap_thresh = nms_overlap_thresh
        self.nms_per_class = nms_per_class
        self.with_classes = with_classes
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.ttl = ttl
        self.tracker_type = tracker_type
        self.keep_upper_height_ratio = keep_upper_height_ratio

        if self.tracker_type not in [
            "BOOSTING",
            "MIL",
            "KCF",
            "KCF2",
            "TLD",
            "MEDIANFLOW",
            "GOTURN",
            "NONE",
        ]:
            raise ValueError("Invalid tracker_type")

        self.visdrone_classes = {
            "car": 4,
            "bus": 9,
            "truck": 6,
            "pedestrian": 1,
            "van": 5,
        }

    def initialize(self):
        super(VIOU, self).initialize()
        self.tracks_active = []
        self.tracks_extendable = []
        self.tracks_finished = []
        self.frame_buffer = []

    def track(self, img_path, dets):
        super(VIOU, self).track(img_path, dets)

        frame = cv2.imread(img_path)
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.ttl + 1:
            self.frame_buffer.pop(0)

        dets = self.preprocess(dets, self.with_classes)
        dets = [det for det in dets if det["score"] >= self.sigma_l]

        track_ids, det_ids = associate(self.tracks_active, dets, self.sigma_iou)
        updated_tracks = []
        for track_id, det_id in zip(track_ids, det_ids):
            self.tracks_active[track_id]["bboxes"].append(dets[det_id]["bbox"])
            self.tracks_active[track_id]["max_score"] = max(
                self.tracks_active[track_id]["max_score"], dets[det_id]["score"]
            )
            self.tracks_active[track_id]["classes"].append(dets[det_id]["class"])
            self.tracks_active[track_id]["det_counter"] += 1

            if self.tracks_active[track_id]["ttl"] != self.ttl:
                # reset visual tracker if active
                self.tracks_active[track_id]["ttl"] = self.ttl
                self.tracks_active[track_id]["visual_tracker"] = None

            updated_tracks.append(self.tracks_active[track_id])

        tracks_not_updated = [
            self.tracks_active[idx]
            for idx in set(range(len(self.tracks_active))).difference(set(track_ids))
        ]

        for track in tracks_not_updated:
            if track["ttl"] > 0:
                if track["ttl"] == self.ttl:
                    # init visual tracker
                    track["visual_tracker"] = VisTracker(
                        self.tracker_type,
                        track["bboxes"][-1],
                        self.frame_buffer[-2],
                        self.keep_upper_height_ratio,
                    )
                # viou forward update
                ok, bbox = track["visual_tracker"].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    self.tracks_extendable.append(track)
                    continue

                track["ttl"] -= 1
                track["bboxes"].append(bbox)
                updated_tracks.append(track)
            else:
                self.tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        tracks_extendable_updated = []
        for track in self.tracks_extendable:
            if (
                track["start_frame"] + len(track["bboxes"]) + self.ttl - track["ttl"]
                >= self.frame_idx
            ):
                tracks_extendable_updated.append(track)
            elif (
                track["max_score"] >= self.sigma_h
                and track["det_counter"] >= self.t_min
            ):
                self.tracks_finished.append(track)
        self.tracks_extendable = tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(
                self.tracker_type, det["bbox"], frame, self.keep_upper_height_ratio
            )

            for f in reversed(self.frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(
                    self.tracks_extendable, key=lambda x: len(x["bboxes"]), reverse=True
                ):

                    offset = (
                        track["start_frame"]
                        + len(track["bboxes"])
                        + len(boxes)
                        - self.frame_idx
                    )
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if (
                        1 <= offset <= self.ttl - track["ttl"]
                        and iou(track["bboxes"][-offset], bbox) >= self.sigma_iou
                    ):
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track["bboxes"] = track["bboxes"][: -offset + 1]
                        track["bboxes"] += list(reversed(boxes))[1:]
                        track["bboxes"].append(det["bbox"])
                        track["max_score"] = max(track["max_score"], det["score"])
                        track["classes"].append(det["class"])
                        track["ttl"] = self.ttl
                        track["visual_tracker"] = None

                        self.tracks_extendable.remove(track)
                        if track in self.tracks_finished:
                            del self.tracks_finished[self.tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                dets_for_new.append(det)

        # create new tracks
        new_tracks = [
            {
                "bboxes": [det["bbox"]],
                "max_score": det["score"],
                "start_frame": self.frame_idx,
                "ttl": self.ttl,
                "classes": [det["class"]],
                "det_counter": 1,
                "visual_tracker": None,
            }
            for det in dets_for_new
        ]
        self.tracks_active = []
        for track in updated_tracks + new_tracks:
            if track["ttl"] == 0:
                self.tracks_extendable.append(track)
            else:
                self.tracks_active.append(track)

        results = [
            [
                id,
                track["bboxes"][-1][0],  # x
                track["bboxes"][-1][1],  # y
                track["bboxes"][-1][2] - track["bboxes"][-1][0],  # w
                track["bboxes"][-1][3] - track["bboxes"][-1][1],  # h
            ]
            for id, track in enumerate(self.tracks_active + self.tracks_extendable)
        ]

        return results

    def preprocess(self, dets, with_classes):
        if dets is None:
            return []

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
