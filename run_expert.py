from datasets.mot import MOT

# from experts.dan import DAN as Tracker

# tracker = Tracker("models/sst300_0712_83000.pth", (0, 0, 4, 0, 3, 3))

from experts.deepmot import DeepMOT as Tracker

tracker = Tracker(
    "external/deepmot/test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml",
    "models/deepMOT_Tracktor.pth",
    "external/deepmot/test_tracktor/output/fpn/res101/mot_2017_train/voc_init_iccv19/config.yaml",
)

dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15")

for seq in dataset:
    print(f"Initialize tracker for {seq.name}")
    tracker.initialize()
    results = tracker.track_seq(seq)
    seq.write_results(results, "output")
