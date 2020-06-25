import sys
import numpy as np
from PIL import Image
import cv2
import yaml

import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from expert import Expert

sys.path.append("external/tracking_wo_bnw_cuda9")
sys.path.append("external/tracking_wo_bnw_cuda9/src/frcnn")
sys.path.append("external/tracking_wo_bnw_cuda9/src/fpn")
from src.tracktor.resnet import resnet50
from src.tracktor.tracker import Tracker
from src.tracktor.oracle_tracker import OracleTracker
from src.frcnn.frcnn.model import test


class Tracktor(Expert):
    def __init__(
        self,
        reid_network_weights_path,
        obj_detect_model_path,
        tracktor_config_path,
        reid_network_config_path,
        obj_detect_config_path,
    ):
        super(Tracktor, self).__init__("Tracktor_cuda9")

        with open(tracktor_config_path) as config_file:
            tracktor = yaml.load(config_file)["tracktor"]

        with open(reid_network_config_path) as config_file:
            siamese = yaml.load(config_file)["siamese"]

        with open(obj_detect_config_path) as config_file:
            _config = yaml.load(config_file)

        # set all seeds
        torch.manual_seed(tracktor["seed"])
        torch.cuda.manual_seed(tracktor["seed"])
        np.random.seed(tracktor["seed"])
        torch.backends.cudnn.deterministic = True

        ##########################
        # Initialize the modules #
        ##########################

        # object detection
        print("[*] Building object detector")
        if tracktor["network"].startswith("frcnn"):
            # FRCNN
            from src.tracktor.frcnn import FRCNN
            from frcnn.model import config

            if _config["frcnn"]["cfg_file"]:
                config.cfg_from_file(_config["frcnn"]["cfg_file"])
            if _config["frcnn"]["set_cfgs"]:
                config.cfg_from_list(_config["frcnn"]["set_cfgs"])

            obj_detect = FRCNN(num_layers=101)
            obj_detect.create_architecture(
                2,
                tag="default",
                anchor_scales=config.cfg.ANCHOR_SCALES,
                anchor_ratios=config.cfg.ANCHOR_RATIOS,
            )
            obj_detect.load_state_dict(torch.load(obj_detect_model_path))
        elif tracktor["network"].startswith("fpn"):
            # FPN
            from src.tracktor.fpn import FPN
            from fpn.model.utils import config

            config.cfg.TRAIN.USE_FLIPPED = False
            config.cfg.CUDA = True
            config.cfg.TRAIN.USE_FLIPPED = False
            checkpoint = torch.load(obj_detect_model_path)

            if "pooling_mode" in checkpoint.keys():
                config.cfg.POOLING_MODE = checkpoint["pooling_mode"]

            set_cfgs = ["ANCHOR_SCALES", "[4, 8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]
            config.cfg_from_file(obj_detect_config_path)
            config.cfg_from_list(set_cfgs)

            obj_detect = FPN(("__background__", "pedestrian"), 101, pretrained=False)
            obj_detect.create_architecture()

            obj_detect.load_state_dict(checkpoint["model"])
        else:
            raise NotImplementedError(
                f"Object detector type not known: {tracktor['network']}"
            )

        obj_detect.eval()
        obj_detect.cuda()

        # reid
        reid_network = resnet50(pretrained=False, **siamese["cnn"])
        reid_network.load_state_dict(torch.load(reid_network_weights_path))
        reid_network.eval()
        reid_network.cuda()

        # tracktor
        if "oracle" in tracktor:
            self.tracker = OracleTracker(
                obj_detect, reid_network, tracktor["tracker"], tracktor["oracle"]
            )
        else:
            self.tracker = Tracker(obj_detect, reid_network, tracktor["tracker"])

        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        self.transforms = Compose(
            [ToTensor(), Normalize(normalize_mean, normalize_std)]
        )

    def initialize(self, dataset_name, seq_name):
        super(Tracktor, self).initialize()
        self.tracker.reset()

    def track(self, img_path, dets):
        super(Tracktor, self).track(img_path, dets)

        frame = self.preprocess(img_path, dets)
        frame["im_info"] = frame["im_info"].cuda()
        # frame["app_data"] = frame["app_data"].cuda()
        self.tracker.step(frame)

        results = []
        for i, track in self.tracker.get_results().items():
            if self.frame_idx in track.keys():
                bb = track[self.frame_idx]
                x1 = bb[0]
                y1 = bb[1]
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                results.append([i, x1 + 1, y1 + 1, w + 1, h + 1])
        return results

    def preprocess(self, img_path, dets):
        # construct image blob and return new dictionary, so blobs are not saved into this class
        im = cv2.imread(img_path)
        blobs, im_scales = test._get_blobs(im)
        data = blobs["data"]

        sample = {}
        sample["im_path"] = img_path
        sample["data"] = torch.tensor(data).unsqueeze(0)
        sample["im_info"] = torch.tensor(
            np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
        ).unsqueeze(0)

        # convert to siamese input
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = self.transforms(im)
        sample["app_data"] = im.unsqueeze(0).unsqueeze(0)

        sample["dets"] = []
        if dets is not None:
            bb = np.zeros((len(dets), 5), dtype=np.float32)
            bb[:, 0:2] = dets[:, 2:4] - 1
            bb[:, 2:4] = dets[:, 2:4] + dets[:, 4:6] - 1
            bb[:, 4] = dets[:, 6]
            # resize tracks
            for det in bb:
                stackbb = np.hstack([det[:4] * sample["im_info"][0][2], det[4:5]])
                sample["dets"].append(torch.tensor(stackbb).reshape(1, -1))
        return sample
