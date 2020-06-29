import sys
import yaml

from feedback.preprocess import PreProcess
from feedback.mpn_tracker import MPNTracker

sys.path.append("external/mot_neural_solver/src")
from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.data.seq_processing.MOT17loader import (
    MOV_CAMERA_DICT as MOT17_MOV_CAMERA_DICT,
)
from mot_neural_solver.data.seq_processing.MOT15loader import FPS_DICT as MOT15_FPS_DICT
from mot_neural_solver.data.seq_processing.MOT15loader import (
    MOV_CAMERA_DICT as MOT15_MOV_CAMERA_DICT,
)

MOV_CAMERA_DICT = {**MOT15_MOV_CAMERA_DICT, **MOT17_MOV_CAMERA_DICT}


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
        self.model.hparams["dataset_params"]["precomputed_embeddings"] = True

        self.preprocess = PreProcess(
            self.model.hparams["dataset_params"],
            self.model.cnn_model,
            frcnn_weights_path,
            prepr_w_tracktor,
            frcnn_prepr_params,
            tracktor_params,
        )

    def initialize(self, seq_name):
        if seq_name in MOT15_FPS_DICT.keys():
            self.fps = MOT15_FPS_DICT[seq_name]
        self.seq_type = "moving" if MOV_CAMERA_DICT[seq_name] else "static"

    def track(self, img_paths, dets):

        mot_graph, det_df = self.preprocess.process(
            img_paths, dets, self.fps, self.seq_type
        )
        tracker = MPNTracker(
            self.model,
            self.model.hparams["eval_params"],
            self.model.hparams["dataset_params"],
        )
        final_out = tracker.track(mot_graph, det_df)
        return final_out
