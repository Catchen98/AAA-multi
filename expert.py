import numpy as np
from print_manager import do_not_print


@do_not_print
def get_expert_by_name(name):
    if name == "DAN":
        from experts.dan import DAN as Tracker

        tracker = Tracker("weights/DAN/sst300_0712_83000.pth")
    elif name == "DeepMOT":
        from experts.deepmot import DeepMOT as Tracker

        tracker = Tracker(
            "weights/DeepMOT/trainedSOTtoMOT.pth", "weights/DeepMOT/DAN.pth",
        )
    elif name == "DeepSort":
        from experts.deepsort import DeepSort as Tracker

        tracker = Tracker("weights/DeepSort/mars-small128.pb")
    elif name == "IOU":
        from experts.iou import IOU as Tracker

        tracker = Tracker()
    elif name == "MOTDT":
        from experts.motdt import MOTDT as Tracker

        tracker = Tracker()
    elif name == "Sort":
        from experts.esort import ESort as Tracker

        tracker = Tracker()
    elif name == "Tracktor":
        from experts.tracktor import Tracktor as Tracker

        tracker = Tracker(
            "weights/Tracktor/ResNet_iter_25245.pth",
            "weights/Tracktor/model_epoch_27.model",
            "external/tracking_wo_bnw/experiments/cfgs/tracktor.yaml",
            "weights/Tracktor/sacred_config.yaml",
        )
    elif name == "VIOU":
        from experts.viou import VIOU as Tracker

        tracker = Tracker()
    elif name == "DeepTAMA":
        from experts.deeptama import DeepTAMA as Tracker

        tracker = Tracker()
    else:
        raise ValueError("Invalid name")

    return tracker


class Expert:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def initialize(self, seq_info):
        self.history = []
        self.frame_idx = -1

    def track(self, img_path, dets):
        self.frame_idx += 1

    @do_not_print
    def track_seq(self, seq):
        self.initialize(seq.seq_info)

        for frame_idx, (img_path, dets) in enumerate(seq):
            results = self.track(img_path, dets)
            results = np.array(results)
            if len(results) > 0:
                frame_results = np.zeros((results.shape[0], results.shape[1] + 1))
                frame_results[:, 1:] = results
                frame_results[:, 0] = frame_idx + 1
                self.history.append(frame_results)
        self.history = np.concatenate(self.history, axis=0)
        return self.history
