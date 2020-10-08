import sys

from PIL import Image

from experts.expert import Expert

import numpy as np

import torch
import yaml

from torchvision.transforms import ToTensor

sys.path.append("external/UMA-MOT/UMA-TEST")
from tracker.detection import Detection
from tracker.mot_tracker import MOT_Tracker

from tracking_wo_bnw.src.tracktor.frcnn_fpn import FRCNN_FPN
from tracking_wo_bnw.src.tracktor.oracle_tracker import OracleTracker
from tracking_wo_bnw.src.tracktor.reid.resnet import resnet50
from tracking_wo_bnw.src.tracktor.tracker import Tracker


class Tracktor(Expert):
    def __init__(
        self,
        reid_network_weights_path,
        obj_detect_model_path,
        tracktor_config_path,
        reid_config_path,
    ):
        super(Tracktor, self).__init__("Tracktor")

        with open(tracktor_config_path) as config_file:
            tracktor = yaml.unsafe_load(config_file)["tracktor"]

        with open(reid_config_path) as config_file:
            reid = yaml.unsafe_load(config_file)["reid"]

        # set all seeds
        torch.manual_seed(tracktor["seed"])
        torch.cuda.manual_seed(tracktor["seed"])
        np.random.seed(tracktor["seed"])
        torch.backends.cudnn.deterministic = True

        ##########################
        # Initialize the modules #
        ##########################

        # object detection
        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(
            torch.load(
                obj_detect_model_path, map_location=lambda storage, loc: storage,
            )
        )

        obj_detect.eval()
        obj_detect.cuda()

        # reid
        reid_network = resnet50(pretrained=False, **reid["cnn"])
        reid_network.load_state_dict(
            torch.load(
                reid_network_weights_path, map_location=lambda storage, loc: storage
            )
        )
        reid_network.eval()
        reid_network.cuda()

        # tracktor
        if "oracle" in tracktor:
            self.tracker = OracleTracker(
                obj_detect, reid_network, tracktor["tracker"], tracktor["oracle"]
            )
        else:
            self.tracker = Tracker(obj_detect, reid_network, tracktor["tracker"])

        self.transforms = ToTensor()

    def initialize(self, seq_info):
        super(Tracktor, self).initialize(seq_info)
        self.tracker.reset()

    def track(self, img_path, dets):
        super(Tracktor, self).track(img_path, dets)

        frame = self.preprocess(img_path, dets)
        with torch.no_grad():
            self.tracker.step(frame)

        results = []
        for i, track in self.tracker.get_results().items():
            if self.frame_idx in track.keys():
                bb = track[self.frame_idx]
                x1 = bb[0]
                y1 = bb[1]
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                score = bb[4]
                results.append([i, x1 + 1, y1 + 1, w + 1, h + 1, score])
        return results

    def preprocess(self, img_path, dets):
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample["img"] = img.unsqueeze(0)
        if dets is not None:
            bb = np.zeros((len(dets), 5), dtype=np.float32)
            bb[:, 0:2] = dets[:, 2:4] - 1
            bb[:, 2:4] = dets[:, 2:4] + dets[:, 4:6] - 1
            bb[:, 4] = dets[:, 6]
            sample["dets"] = torch.FloatTensor([det[:4] for det in bb]).unsqueeze(0)
        else:
            sample["dets"] = torch.FloatTensor([]).unsqueeze(0)
        sample["img_path"] = img_path
        return sample


class UMA(Expert):
    def __init__(
        self,
        life_span,
        occlusion_thres,
        association_thres,
        checkpoint,
        context_amount,
        iou,
        tracktor_config,
    ):
        super(UMA, self).__init__("UMA")
        self.life_span = life_span
        self.occlusion_thres = occlusion_thres
        self.association_thres = association_thres
        self.checkpoint = checkpoint
        self.context_amount = context_amount
        self.iou = iou

        self.tracktor = Tracktor(
            tracktor_config["reid_network_weights_path"],
            tracktor_config["obj_detect_model_path"],
            tracktor_config["tracktor_config_path"],
            tracktor_config["reid_config_path"],
        )

    def initialize(self, seq_info):
        super(UMA, self).initialize(seq_info)

        max_age = int(self.life_span * int(seq_info["fps"]))

        self.tracker = MOT_Tracker(
            max_age, self.occlusion_thres, self.association_thres
        )
        self.tracker.frame_rate = int(seq_info["fps"])

        self.tracktor.initialize(seq_info)

    def track(self, img_path, dets):
        super(UMA, self).track(img_path, dets)

        detections = self.preprocess(img_path, dets)

        trackers = self.tracker.update(
            img_path, self.checkpoint, self.context_amount, detections, self.iou
        )

        result = []
        for d in trackers:
            result.append([d[4], d[0], d[1], d[2], d[3]])

        return result

    def preprocess(self, img_path, dets):
        filtered_dets = self.tracktor.track(img_path, dets)

        detection_list = []
        for row in filtered_dets:
            bbox, confidence = row[1:5], row[5]
            detection_list.append(Detection(bbox, confidence))
        return detection_list
