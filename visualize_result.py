import os
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns

from datasets.mot import MOT
from paths import OUTPUT_PATH, DATASET_PATH
from file_manager import ReadResult

sns.set()
sns.set_style("whitegrid")


LINEWIDTH = 3
ANNOT_SIZE = 10
ALL_HEIGHT = 3


def draw_result(
    dataset, tracker_name, result_dir, is_algorithm=True, duration=None,
):
    if is_algorithm:
        show = ["weight", "frame"]
    else:
        show = ["frame"]

    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]
        tracker_reader = ReadResult(dataset_name, tracker_name, seq_name)

        if is_algorithm:
            dataset_dir = OUTPUT_PATH / dataset_name / tracker_name

            weight_path = dataset_dir / f"{seq_name}_weight.txt"
            weights = pd.read_csv(weight_path, header=None).set_index(0)

        save_dir = result_dir / dataset_name / tracker_name / seq_name

        os.makedirs(save_dir, exist_ok=True)

        for frame_idx, (img_path, dets, gts) in enumerate(seq):
            filename = os.path.basename(img_path)
            im = Image.open(img_path).convert("RGB")

            cond = [drawing in show for drawing in ["weight", "frame"]]
            ratios = [1 if i != 1 else 3 for i in range(len(cond)) if cond[i]]

            fig, axes = plt.subplots(
                nrows=sum(cond), ncols=1, gridspec_kw={"height_ratios": ratios}
            )
            fig.add_subplot(111, frameon=False)

            i = 0
            if cond[0]:
                weight_ax = axes[i] if len(ratios) > 1 else axes
                i += 1
            if cond[1]:
                sample_ax = axes[i] if len(ratios) > 1 else axes

            # draw weight graph
            if cond[0]:
                for i in weights.columns:
                    weight = weights.loc[: frame_idx + 1][i]
                    weight_ax.plot(range(len(weight)), weight)
                    weight_ax.set(
                        ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,),
                    )
                weight_ax.set_xticks([])

                # draw anchor line
                for i in range(frame_idx):
                    if i + 1 % duration == 0:
                        weight_ax.axvline(
                            x=i, color="gray", linestyle="--", linewidth=1
                        )

            # draw frame
            if cond[1]:
                sample_ax.imshow(np.asarray(im), aspect="auto")
                bboxes = tracker_reader.get_result_by_frame(frame_idx)

                # draw tracking bbox
                for i in range(len(bboxes)):
                    box = bboxes[i]
                    rect = patches.Rectangle(
                        (box[1], box[2]),
                        box[3],
                        box[4],
                        linewidth=LINEWIDTH,
                        facecolor="none",
                        alpha=1,
                        edgecolor="red",
                    )
                    sample_ax.add_patch(rect)

                    sample_ax.annotate(
                        f"{int(box[0])}",
                        xy=(box[1] + box[3] / 2, box[2]),
                        xycoords="data",
                        weight="bold",
                        xytext=(-5, 5),
                        textcoords="offset points",
                        size=ANNOT_SIZE,
                        color="red",
                    )

                # draw ground truth
                bbxoes = gts[:, 1:6]
                for i in range(len(bboxes)):
                    box = bbxoes[i]
                    rect = patches.Rectangle(
                        (box[1], box[2]),
                        box[3],
                        box[4],
                        linewidth=LINEWIDTH,
                        edgecolor="black",
                        facecolor="none",
                        alpha=1,
                    )
                    sample_ax.add_patch(rect)
                    sample_ax.annotate(
                        f"{int(box[0])}",
                        xy=(box[1] + box[3] / 2, box[2] + box[4]),
                        xycoords="data",
                        weight="bold",
                        xytext=(-5, -10),
                        textcoords="offset points",
                        size=ANNOT_SIZE,
                        color="black",
                    )
                sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.subplots_adjust(wspace=0, hspace=0.1 if len(ratios) > 1 else 0)
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()


def draw_all_result(dataset, trackers_name, result_dir, duration):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]
        trackers_reader = [
            ReadResult(dataset_name, tracker_name, seq_name)
            for tracker_name in trackers_name
        ]

        dataset_dir = OUTPUT_PATH / dataset_name / trackers_name[0]

        weight_path = dataset_dir / f"{seq_name}_weight.txt"
        weights = pd.read_csv(weight_path, header=None).set_index(0)

        selected_path = dataset_dir / f"{seq_name}_selected.txt"
        selected_experts = pd.read_csv(selected_path, header=None).set_index(0)

        save_dir = result_dir / dataset_name / trackers_name[0] / seq_name

        os.makedirs(save_dir, exist_ok=True)

        nexperts_row = len(trackers_name) // 2
        gs = gridspec.GridSpec(1 + ALL_HEIGHT * nexperts_row, 4)

        for frame_idx, (img_path, dets, gts) in enumerate(seq):
            plt.figure(figsize=(15, 20))
            filename = os.path.basename(img_path)
            im = Image.open(img_path).convert("RGB")

            weight_ax = plt.subplot(gs[0, :])

            # draw weight graph
            for i in weights.columns:
                weight = weights.loc[: frame_idx + 1][i]
                weight_ax.plot(range(len(weight)), weight)
                weight_ax.set(
                    ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,),
                )
            weight_ax.set_xticks([])

            # draw anchor line
            for i in range(frame_idx):
                if i + 1 % duration == 0:
                    weight_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1)

            for t_i, tracker_reader in enumerate(trackers_reader):
                row_i = ALL_HEIGHT * (t_i // 2) + 1
                if t_i % 2 == 0:
                    sample_ax = plt.subplot(gs[row_i : row_i + ALL_HEIGHT, :2])
                else:
                    sample_ax = plt.subplot(gs[row_i : row_i + ALL_HEIGHT, 2:])

                if t_i == 0:
                    sample_ax.set_title(
                        f"{trackers_name[t_i].split('_')[0]}-Select:{trackers_name[int(selected_experts.loc[frame_idx+1].values) + 1]}"
                    )
                else:
                    sample_ax.set_title(trackers_name[t_i])

                # draw frame
                sample_ax.imshow(np.asarray(im), aspect="auto")

                # draw tracking bbox
                bboxes = tracker_reader.get_result_by_frame(frame_idx)
                for i in range(len(bboxes)):
                    box = bboxes[i]
                    rect = patches.Rectangle(
                        (box[1], box[2]),
                        box[3],
                        box[4],
                        linewidth=LINEWIDTH,
                        facecolor="none",
                        alpha=1,
                        edgecolor="red",
                    )
                    sample_ax.add_patch(rect)

                    sample_ax.annotate(
                        f"{int(box[0])}",
                        xy=(box[1] + box[3] / 2, box[2]),
                        xycoords="data",
                        weight="bold",
                        xytext=(-5, 5),
                        textcoords="offset points",
                        size=ANNOT_SIZE,
                        color="red",
                    )

                # draw ground truth
                bboxes = gts[:, 1:6]
                for i in range(len(bboxes)):
                    box = bboxes[i]
                    rect = patches.Rectangle(
                        (box[1], box[2]),
                        box[3],
                        box[4],
                        linewidth=LINEWIDTH,
                        edgecolor="black",
                        facecolor="none",
                        alpha=1,
                    )
                    sample_ax.add_patch(rect)
                    sample_ax.annotate(
                        f"{int(box[0])}",
                        xy=(box[1] + box[3] / 2, box[2] + box[4]),
                        xycoords="data",
                        weight="bold",
                        xytext=(-5, -10),
                        textcoords="offset points",
                        size=ANNOT_SIZE,
                        color="black",
                    )
                sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.subplots_adjust(wspace=0.1, hspace=0.3)
            plt.savefig(save_dir / filename, bbox_inches="tight", dpi=300)
            plt.close()


def main():
    datasets = {
        # "MOT15": MOT(DATASET_PATH["MOT15"]),
        # "MOT16": MOT(DATASET_PATH["MOT16"]),
        "MOT17": MOT(DATASET_PATH["MOT17"]),
    }

    algorithm_name = "AAA_{'detector': {'type': 'fixed', 'duration': 30}, 'offline': {'reset': True}, 'matching': {'threshold': 0.3, 'time': 'current'}, 'loss': {'type': 'fmota'}}"
    experts_name = ["DAN", "DeepSort", "DeepTAMA", "Sort", "MOTDT"]
    for dataset_name, dataset in datasets.items():
        # draw_result(
        #     dataset, algorithm_name, Path("visualize"), is_algorithm=True, duration=70
        # )
        draw_all_result(
            dataset, [algorithm_name] + experts_name, Path("visualize"), duration=70
        )


if __name__ == "__main__":
    main()
