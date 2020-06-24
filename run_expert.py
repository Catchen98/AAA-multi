from datasets.mot import MOT

# from experts.dan import DAN as Tracker

# tracker = Tracker("models/sst300_0712_83000.pth", (0, 0, 4, 0, 3, 3))

# from experts.deepmot import DeepMOT as Tracker

# tracker = Tracker(
#     "models/deepMOT_Tracktor.pth",
#     "external/deepmot/test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml",
#     "external/deepmot/test_tracktor/output/fpn/res101/mot_2017_train/voc_init_iccv19/config.yaml",
# )

# from experts.deepsort import DeepSort as Tracker

# tracker = Tracker("models/mars-small128.pb", min_confidence=0.3, nn_budget=100)

# from experts.esort import ESort as Tracker

# tracker = Tracker()

# from experts.iou import IOU as Tracker

# tracker = Tracker()

# from experts.viou import VIOU as Tracker

# tracker = Tracker()

# from experts.motdt import MOTDT as Tracker

# tracker = Tracker()

from experts.tracktor import Tracktor as Tracker

tracker = Tracker(
    "data/ResNet_iter_25245.pth",
    "data/model_epoch_27.model",
    "external/tracking_wo_bnw/experiments/cfgs/tracktor.yaml",
    "data/sacred_config.yaml",
)

dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15")

for seq in dataset:
    print(f"Initialize tracker for {seq.name}")
    tracker.initialize()
    results = tracker.track_seq(seq, debug=True)
    seq.write_results(results, "output")
