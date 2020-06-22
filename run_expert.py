from experts.esort import ESort
from datasets.mot import MOT


dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15")

for seq in dataset:
    print(f"Initialize tracker for {seq.name}")
    tracker = ESort()
    tracker.initialize()
    results = tracker.track_seq(seq)
    seq.write_results(results, "output")
