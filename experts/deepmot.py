import sys
import numpy as np
import cv2
from PIL import Image
import yaml

import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from experts.expert import Expert

sys.path.append("external/deepmot/test_tracktor")
from src.frcnn.frcnn.model import test
from src.tracktor.tracker import Tracker


class DeepMOT(Expert):
    def __init__(
        self, obj_detect_weights_path, tracktor_config_path, obj_detect_config_path
    ):
        super(DeepMOT, self).__init__("DeepMOT")
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        self.transforms = Compose(
            [ToTensor(), Normalize(normalize_mean, normalize_std)]
        )

        with open(tracktor_config_path) as config_file:
            tracktor = yaml.load(config_file)["tracktor"]

        # set all seeds
        torch.manual_seed(tracktor["seed"])
        torch.cuda.manual_seed(tracktor["seed"])
        np.random.seed(tracktor["seed"])
        torch.backends.cudnn.deterministic = True

        ##########################
        # Initialize the modules #
        ##########################

        # object detection
        if tracktor["network"].startswith("fpn"):
            # FPN
            from src.tracktor.fpn import FPN
            from src.fpn.fpn.model.utils import config

            config.cfg.TRAIN.USE_FLIPPED = False
            config.cfg.CUDA = True
            config.cfg.TRAIN.USE_FLIPPED = False
            checkpoint = torch.load(obj_detect_weights_path)

            if "pooling_mode" in checkpoint.keys():
                config.cfg.POOLING_MODE = checkpoint["pooling_mode"]
            else:
                config.cfg.POOLING_MODE = "align"

            set_cfgs = ["ANCHOR_SCALES", "[4, 8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]
            config.cfg_from_file(obj_detect_config_path)
            config.cfg_from_list(set_cfgs)

            if "fpn_1_12.pth" in obj_detect_weights_path:
                classes = (
                    "__background__",
                    "aeroplane",
                    "bicycle",
                    "bird",
                    "boat",
                    "bottle",
                    "bus",
                    "car",
                    "cat",
                    "chair",
                    "cow",
                    "diningtable",
                    "dog",
                    "horse",
                    "motorbike",
                    "person",
                    "pottedplant",
                    "sheep",
                    "sofa",
                    "train",
                    "tvmonitor",
                )
            else:
                classes = ("__background__", "pedestrian")

            obj_detect = FPN(classes, 101, pretrained=False)
            obj_detect.create_architecture()
            if "model" in checkpoint.keys():

                model_dcit = obj_detect.state_dict()
                model_dcit.update(checkpoint["model"])
                obj_detect.load_state_dict(model_dcit)

                # obj_detect.load_state_dict(checkpoint['model'])

                # obj_detect.load_state_dict(checkpoint['model'])
            else:
                # pick the reid branch
                model_dcit = obj_detect.state_dict()
                model_dcit.update(checkpoint)
                obj_detect.load_state_dict(model_dcit)

        else:
            raise NotImplementedError(
                f"Object detector type not known: {tracktor['network']}"
            )

        obj_detect.eval()
        obj_detect.cuda()

        # tracktor
        self.tracker = Tracker(obj_detect, None, tracktor["tracker"])

    def initialize(self, seq_info):
        super(DeepMOT, self).initialize(seq_info)
        self.tracker.reset()

    def track(self, img_path, dets):
        super(DeepMOT, self).track(img_path, dets)

        frame = self.preprocess(img_path, dets)
        frame["im_info"] = frame["im_info"].cuda()
        frame["app_data"] = frame["app_data"].cuda()
        self.tracker.step_pub_reid(frame)

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
        sample["app_data"] = im.unsqueeze(0)

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
