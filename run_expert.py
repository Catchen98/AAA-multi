from datasets.mot import MOT
from expert import get_expert_by_name


dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT15")
tracker = get_expert_by_name("MOTDT")

for seq in dataset:
    # if seq.name != "ADL-Rundle-1":
    #     continue
    print(f"Start {seq.name}")
    results = tracker.track_seq(seq)
    seq.write_results(results, "output")
