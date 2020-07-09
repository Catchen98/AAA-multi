import sys
import copy

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from experts.expert import Expert

sys.path.append("external/deepMOT")
from utils.tracking_config import tracking_config
from models.siamrpn import SiamRPNvot
from utils.mot_utils import tracking_birth_death
from utils.box_utils import getWarpMatrix, bb_fast_IOU_v1, mix_track_detV2
from models.DAN import build_sst
from utils.DAN_utils import TrackUtil
from utils.sot_utils import SiamRPN_track
from utils.box_utils import xywh2xyxy


class DeepMOT(Expert):
    def __init__(self, sot_model_path, sst_model_path):
        super(DeepMOT, self).__init__("DeepMOT")
        torch.set_grad_enabled(False)
        cudnn.benchmark = False
        cudnn.deterministic = True

        # init sot tracker #
        self.sot_tracker = SiamRPNvot()
        self.sot_tracker.load_state_dict(torch.load(sot_model_path))

        # init appearance model #
        self.sst = build_sst("test", 900)
        self.sst.load_state_dict(torch.load(sst_model_path))

        self.sot_tracker.cuda()
        self.sst.cuda()

        # evaluation mode #
        self.sot_tracker.eval()
        self.sst.eval()

    def initialize(self, seq_info):
        super(DeepMOT, self).initialize(seq_info)
        (
            self.to_refine,
            self.to_combine,
            self.DAN_th,
            self.death_count,
            self.birth_wait,
            self.loose_assignment,
            self.case1_interpolate,
            self.interpolate_flag,
            self.CMC,
            self.birth_iou,
        ) = tracking_config(seq_info["seq_name"], seq_info["dataset_name"])

        self.frames_det = {
            str(frame_id + 1): None for frame_id in range(seq_info["total_length"])
        }

        self.track_init = []

        # previous numpy frame
        self.img_prev = None

        # track id counter
        self.count_ids = 0

        # bbox_track = {frame_id: [[bbox], [bbox], [bbox]..]} dict of torch tensor with shape
        # [num_tracks, 4=(x1,y1,x2,y2)]
        self.bbox_track = dict()

        # id track = ordered [hypo_id1, hypo_id2, hypo_id3...] corresponding to bbox_track
        # of current frame, torch tensor [num_tracks]
        self.id_track = list()

        # states = {track_id: state, ...}
        self.states = dict()

        # previous frame id
        self.prev_frame_id = 0

        # birth candidates, {frames_id:[to birth id(=det_index)...], ...}
        self.birth_candidates = dict()

        # death candidates, {track_id:num_times_track_lost,lost_track ...}
        self.death_candidates = dict()

        # collect_prev_pos = {trackId:[postion_before_lost=[x1,y1,x2,y2], track_appearance_features,
        # matched_count, matched_det_collector(frame, det_id),
        # track_box_collector=[[frameid,[x,y,x,y]],...],'active' or 'inactive', velocity, inactive_pre_pos]}
        self.collect_prev_pos = dict()

        self.bbox_track[self.prev_frame_id] = None

        self.to_interpolate = dict()

        self.pre_warp_matrix = None
        self.w_matrix = None

    def track(self, img_path, dets):
        super(DeepMOT, self).track(img_path, dets)

        img_curr, dets, h, w = self.preprocess(img_path, dets)
        self.frames_det[str(self.frame_idx + 1)] = dets

        # having active tracks
        if len(self.states) > 0:
            self.tmp = []
            self.im_prev_features = TrackUtil.convert_image(self.img_prev.copy())

            # calculate affine transformation for current frame and previous frame
            if self.img_prev is not None and self.CMC:
                self.w_matrix = getWarpMatrix(img_curr, self.img_prev)

            # FOR every track in PREVIOUS frame
            for key, state_curr in self.states.items():
                # center position at frame t-1
                prev_pos = state_curr["target_pos"].copy()
                prev_size = state_curr["target_sz"].copy()

                prev_xyxy = [
                    prev_pos[0] - 0.5 * prev_size[0],
                    prev_pos[1] - 0.5 * prev_size[1],
                    prev_pos[0] + 0.5 * prev_size[0],
                    prev_pos[1] + 0.5 * prev_size[1],
                ]

                if state_curr["gt_id"] not in self.collect_prev_pos.keys():

                    # extract image features by DAN
                    prev_xywh = [
                        prev_pos[0] - 0.5 * prev_size[0],
                        prev_pos[1] - 0.5 * prev_size[1],
                        prev_size[0],
                        prev_size[1],
                    ]
                    prev_xywh = np.array([prev_xywh], dtype=np.float32)
                    prev_xywh[:, [0, 2]] /= float(w)
                    prev_xywh[:, [1, 3]] /= float(h)
                    track_norm_center = TrackUtil.convert_detection(prev_xywh)

                    tracks_features = self.sst.forward_feature_extracter(
                        self.im_prev_features, track_norm_center
                    ).detach_()

                    self.collect_prev_pos[state_curr["gt_id"]] = [
                        [[self.frame_idx - 1, np.array(prev_xyxy)]],
                        [[self.frame_idx - 1, tracks_features.clone()]],
                        0,
                        list(),
                        list(),
                        "active",
                        [0.0, -1.0, -1.0],
                        np.zeros((4)) - 1,
                    ]
                    del tracks_features

                elif self.collect_prev_pos[state_curr["gt_id"]][5] == "active":

                    # extract image features by DAN
                    prev_xywh = [
                        prev_pos[0] - 0.5 * prev_size[0],
                        prev_pos[1] - 0.5 * prev_size[1],
                        prev_size[0],
                        prev_size[1],
                    ]
                    prev_xywh = np.array([prev_xywh], dtype=np.float32)
                    prev_xywh[:, [0, 2]] /= float(w)
                    prev_xywh[:, [1, 3]] /= float(h)
                    track_norm_center = TrackUtil.convert_detection(prev_xywh)
                    tracks_features = self.sst.forward_feature_extracter(
                        self.im_prev_features, track_norm_center
                    ).detach_()

                    # update positions and appearance features of active track
                    self.collect_prev_pos[state_curr["gt_id"]][0].append(
                        [self.frame_idx - 1, np.array(prev_xyxy)]
                    )

                    # only keep the latest 10 active positions (used for estimating velocity for interpolations)
                    if len(self.collect_prev_pos[state_curr["gt_id"]][0]) > 10:
                        self.collect_prev_pos[state_curr["gt_id"]][0].pop(0)

                    # only keep the latest 3 appearance features (used for recovering invisible tracks)
                    self.collect_prev_pos[state_curr["gt_id"]][1].append(
                        [self.frame_idx - 1, tracks_features.clone()]
                    )
                    if len(self.collect_prev_pos[state_curr["gt_id"]][1]) > 3:
                        self.collect_prev_pos[state_curr["gt_id"]][1].pop(0)
                    del tracks_features

                    # remove pre_lost_pos when a track is recovered
                    self.collect_prev_pos[state_curr["gt_id"]][7] = np.zeros((4)) - 1

                    # update velocity during active mode if we have 10 (might be not consecutive) positions
                    if len(self.collect_prev_pos[state_curr["gt_id"]][0]) == 10:
                        avg_h = 0.0
                        avg_w = 0.0
                        for f, pos in self.collect_prev_pos[state_curr["gt_id"]][0]:
                            avg_h += pos[3] - pos[1]
                            avg_w += pos[2] - pos[0]
                        avg_h /= len(self.collect_prev_pos[state_curr["gt_id"]][0])
                        avg_w /= len(self.collect_prev_pos[state_curr["gt_id"]][0])
                        last_t, last_pos = self.collect_prev_pos[state_curr["gt_id"]][
                            0
                        ][-1]
                        first_t, first_pos = self.collect_prev_pos[state_curr["gt_id"]][
                            0
                        ][0]
                        # center point
                        first_pos_center = np.array(
                            [
                                0.5 * (first_pos[0] + first_pos[2]),
                                0.5 * (first_pos[1] + first_pos[3]),
                            ]
                        )
                        last_pos_center = np.array(
                            [
                                0.5 * (last_pos[0] + last_pos[2]),
                                0.5 * (last_pos[1] + last_pos[3]),
                            ]
                        )
                        velocity = (last_pos_center - first_pos_center) / (
                            last_t - first_t
                        )
                        self.collect_prev_pos[state_curr["gt_id"]][6] = [
                            velocity,
                            avg_h,
                            avg_w,
                        ]
                        self.collect_prev_pos[state_curr["gt_id"]][0] = [
                            self.collect_prev_pos[state_curr["gt_id"]][0][-1]
                        ]

                else:
                    # inactive mode, do nothing
                    pass

                target_pos, target_sz, state_curr, _ = SiamRPN_track(
                    state_curr,
                    img_curr.copy(),
                    self.sot_tracker,
                    train=True,
                    CMC=(self.img_prev is not None and self.CMC),
                    prev_xyxy=prev_xyxy,
                    w_matrix=self.w_matrix,
                )
                self.tmp.append(
                    torch.stack(
                        [
                            target_pos[0] - target_sz[0] * 0.5,
                            target_pos[1] - target_sz[1] * 0.5,
                            target_pos[0] + target_sz[0] * 0.5,
                            target_pos[1] + target_sz[1] * 0.5,
                        ],
                        dim=0,
                    )
                    .detach_()
                    .unsqueeze(0)
                )
                del _
                del target_pos
                del target_sz
                torch.cuda.empty_cache()

            self.bbox_track[self.frame_idx] = torch.cat(self.tmp, dim=0).detach_()

            del self.bbox_track[self.prev_frame_id]
            del self.tmp
            torch.cuda.empty_cache()

        else:
            # having no tracks
            self.bbox_track[self.frame_idx] = None

        # refine and calculate "distance" (actually, iou) matrix #
        if len(dets) > 0:
            distance = []
            if self.bbox_track[self.frame_idx] is not None:
                bboxes = self.bbox_track[self.frame_idx].detach().cpu().numpy().tolist()
                for bbox in bboxes:
                    IOU = bb_fast_IOU_v1(bbox, dets)
                    distance.append(IOU.tolist())
                distance = np.vstack(distance)

                # refine tracks bboxes with dets if iou > 0.6
                if self.to_combine:
                    del bboxes
                    # mix dets and tracks boxes
                    self.bbox_track[self.frame_idx] = mix_track_detV2(
                        torch.FloatTensor(distance).cuda(),
                        torch.FloatTensor(dets).cuda(),
                        self.bbox_track[self.frame_idx],
                    )

                    boxes = (
                        self.bbox_track[self.frame_idx].detach().cpu().numpy().tolist()
                    )
                    for idx, [key, state] in enumerate(self.states.items()):
                        # print(idx, key, state['gt_id'])
                        box = boxes[idx]
                        state["target_pos"] = np.array(
                            [0.5 * (box[2] + box[0]), 0.5 * (box[3] + box[1])]
                        )
                        state["target_sz"] = np.array(
                            [box[2] - box[0], box[3] - box[1]]
                        )
                        self.states[key] = state
                    distance = []
                    bboxes = (
                        self.bbox_track[self.frame_idx].detach().cpu().numpy().tolist()
                    )
                    for bbox in bboxes:
                        IOU = bb_fast_IOU_v1(bbox, dets)
                        distance.append(IOU.tolist())
                    distance = np.vstack(distance)

            # no tracks
            else:
                distance = np.array(distance)

            # birth and death process, no need to be differentiable #

            self.bbox_track[self.frame_idx], self.count_ids = tracking_birth_death(
                distance,
                self.bbox_track[self.frame_idx],
                self.frames_det,
                img_curr,
                self.id_track,
                self.count_ids,
                self.frame_idx,
                self.birth_candidates,
                self.track_init,
                self.death_candidates,
                self.states,
                self.sot_tracker,
                self.collect_prev_pos,
                self.sst,
                th=0.5,
                birth_iou=self.birth_iou,
                to_refine=self.to_refine,
                DAN_th=self.DAN_th,
                death_count=self.death_count,
                birth_wait=self.birth_wait,
                to_interpolate=self.to_interpolate,
                interpolate_flag=self.interpolate_flag,
                loose_assignment=self.loose_assignment,
                case1_interpolate=self.case1_interpolate,
            )

            del distance

            result = []
            if self.bbox_track[self.frame_idx] is not None:
                bbox_torecord = (
                    self.bbox_track[self.frame_idx].detach().cpu().numpy().tolist()
                )
                for j in range(len(bbox_torecord)):
                    if self.id_track[j] not in self.death_candidates.keys():
                        # x1,y1,x2,y2 to x1,y1,w,h
                        torecord = copy.deepcopy(bbox_torecord[j])
                        torecord[2] = torecord[2] - torecord[0]
                        torecord[3] = torecord[3] - torecord[1]
                        towrite = [self.id_track[j] + 1] + torecord
                        result.append(towrite)
        else:
            print("no detections! all tracks killed.")
            self.bbox_track[self.frame_idx] = None
            self.id_track = list()
            self.states = dict()
            self.death_candidates = dict()
            self.collect_prev_pos = dict()
            result = []

        self.img_prev = img_curr.copy()
        self.prev_frame_id = self.frame_idx
        torch.cuda.empty_cache()

        return result

    def preprocess(self, img_path, dets):
        img_curr = cv2.imread(img_path)
        h, w, _ = img_curr.shape

        if dets is None:
            return img_curr, [], h, w

        frames = []
        for det in dets:
            bbox = xywh2xyxy(det[2:6]).tolist()
            frames.append(bbox)

        return img_curr, frames, h, w
