import sys
import yaml
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torchvision.transforms import ToTensor

from feedback.mot_graph_dataset import MOTGraphDataset

sys.path.append("external/mot_neural_solver/src")
from mot_neural_solver.data.preprocessing import FRCNNPreprocessor
from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.tracker.mpn_tracker import MPNTracker
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.data.seq_processing.MOT17loader import (
    MOV_CAMERA_DICT as MOT17_MOV_CAMERA_DICT,
)
from mot_neural_solver.data.seq_processing.MOT15loader import (
    MOV_CAMERA_DICT as MOT15_MOV_CAMERA_DICT,
)
from mot_neural_solver.utils.misc import make_deterministic

sys.path.append("external/mot_neural_solver/tracking_wo_bnw/src")
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.tracker import Tracker

MOV_CAMERA_DICT = {**MOT15_MOV_CAMERA_DICT, **MOT17_MOV_CAMERA_DICT}
TRACKING_OUT_COLS = [
    "frame",
    "ped_id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]


class CustomMOTNeuralSolver(MOTNeuralSolver):
    def __init__(self, *args, **kwargs):
        super(CustomMOTNeuralSolver, self).__init__(*args, **kwargs)

    def load_model(self):
        model = MOTMPNet(self.hparams["graph_model_params"]).cuda()

        cnn_model = resnet50_fc256(10, loss="xent", pretrained=True).cuda()
        load_pretrained_weights(
            cnn_model, self.reid_weights_path,
        )
        cnn_model.return_embeddings = True

        return model, cnn_model

    def track_seq(self, dataset):
        tracker = MPNTracker(
            dataset=dataset,
            graph_model=self.model,
            use_gt=False,
            eval_params=self.hparams["eval_params"],
            dataset_params=self.hparams["dataset_params"],
        )

        seq_name = dataset.seq_names[0]

        tracker.track(seq_name)
        tracker.tracking_out["conf"] = 1
        tracker.tracking_out["x"] = -1
        tracker.tracking_out["y"] = -1
        tracker.tracking_out["z"] = -1

        tracker.tracking_out["bb_left"] += 1  # Indexing is 1-based in the ground truth
        tracker.tracking_out["bb_top"] += 1

        final_out = tracker.tracking_out[TRACKING_OUT_COLS].sort_values(
            by=["frame", "ped_id"]
        )

        return final_out


class NeuralSolver:
    def __init__(
        self,
        ckpt_path,
        frcnn_weights_path,
        reid_weights_path,
        tracking_cfg_path,
        preprocessing_cfg_path,
        prepr_w_tracktor,
    ):
        with open(tracking_cfg_path) as config_file:
            config = yaml.load(config_file)

        with open(preprocessing_cfg_path) as config_file:
            pre_config = yaml.load(config_file)
            frcnn_prepr_params = pre_config["frcnn_prepr_params"]
            tracktor_params = pre_config["tracktor_params"]

        CustomMOTNeuralSolver.reid_weights_path = reid_weights_path

        # Load model from checkpoint and update config entries that may vary from the ones used in training
        self.model = CustomMOTNeuralSolver.load_from_checkpoint(
            checkpoint_path=ckpt_path
        )
        self.model.hparams.update(
            {
                "eval_params": config["eval_params"],
                "data_splits": config["data_splits"],
            }
        )
        self.model.hparams["dataset_params"]["precomputed_embeddings"] = False

        # preprocessor
        obj_detect = FRCNN_FPN(num_classes=2)

        obj_detect.load_state_dict(
            torch.load(frcnn_weights_path, map_location=lambda storage, loc: storage,)
        )
        obj_detect.eval()
        obj_detect.cuda()

        self.prepr_w_tracktor = prepr_w_tracktor
        if self.prepr_w_tracktor:
            self.prepr_params = tracktor_params
        else:
            self.prepr_params = frcnn_prepr_params
        make_deterministic(self.prepr_params["seed"])

        if self.prepr_w_tracktor:
            self.preprocessor = Tracker(obj_detect, None, self.prepr_params["tracker"])
        else:
            self.preprocessor = FRCNNPreprocessor(obj_detect, self.prepr_params)

        self.transforms = ToTensor()

    def track(self, seq_name, img_paths, dets):
        det_df, h, w = self.preprocess(seq_name, img_paths, dets)
        print(det_df)
        dataset = MOTGraphDataset(
            self.model.hparams["dataset_params"],
            seq_name,
            img_paths,
            det_df,
            h,
            w,
            self.model.cnn_model,
        )
        final_out = self.model.track_seq(dataset)
        return final_out.to_numpy()[:, :6]

    def preprocess(self, seq_name, img_paths, dets):
        self.preprocessor.reset()
        if self.prepr_w_tracktor:
            self.preprocessor.do_align = (
                self.prepr_params["tracker"]["do_align"] and MOV_CAMERA_DICT[seq_name]
            )

        for img_path, det in zip(img_paths, dets):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img = self.transforms(img)

            sample = {}
            sample["img"] = img.unsqueeze(0)
            bb = np.zeros((len(det), 5), dtype=np.float32)
            if len(bb) > 0:
                bb[:, 0:2] = det[:, 2:4] - 1
                bb[:, 2:4] = det[:, 2:4] + det[:, 4:6] - 1
            sample["dets"] = torch.FloatTensor([d[:4] for d in bb]).unsqueeze(0)
            sample["img_path"] = img_path

            with torch.no_grad():
                self.preprocessor.step(sample)

        if self.prepr_w_tracktor:
            all_tracks = self.preprocessor.get_results()
            rows = []
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    rows.append(
                        [
                            frame + 1,
                            i + 1,
                            x1 + 1,
                            y1 + 1,
                            x2 - x1 + 1,
                            y2 - y1 + 1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ]
                    )
            det_df = pd.DataFrame(np.array(rows))
        else:
            final_results = pd.concat(self.preprocessor.results_dfs)
            final_results["bb_left"] += 1  # MOT bbox annotations are 1 -based
            final_results["bb_top"] += 1  # MOT bbox annotations are 1 -based
            final_results["id"] = -1
            det_df = final_results[
                ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
            ]

        return det_df, h, w
