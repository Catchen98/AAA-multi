from experts.esort import ESort
from datasets.mot import MOT

tracker = ESort()
dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15")

for seq_name, seq in dataset:
    print(f"Initialize tracker for {seq_name}")
    tracker.initialize()
    results = tracker.track_seq(seq)
    print(results)
    print(results.shape)
