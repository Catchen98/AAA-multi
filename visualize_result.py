import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from paths import OUTPUT_PATH
from file_manager import ReadResult

sns.set()
sns.set_style("whitegrid")


def draw_result(
    dataset, tracker_name, result_dir, is_algorithm=True, duration=None,
):
    if is_algorithm:
        show = ["weight", "frame"]
    else:
        show = ["frame"]
    for seq in dataset:
        tracker_reader = ReadResult(
            seq.seq_info["dataset_name"], tracker_name, seq.seq_info["seq_name"]
        )

        if is_algorithm:
            dataset_dir = OUTPUT_PATH / seq.seq_info["dataset_name"] / tracker_name

            weight_path = dataset_dir / f"{seq.seq_info['seq_name']}_weight.txt"
            weights = pd.read_csv(weight_path, header=None).set_index(0)

        save_dir = result_dir / f"{tracker_name}" / seq.name

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
                            x=i, color="gray", linestyle="--", linewidth=0.1
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
                        linewidth=7,
                        facecolor="none",
                        alpha=1,
                    )
                    sample_ax.add_patch(rect)

                    sample_ax.annotate(
                        f"ID:{box[0]}",
                        xy=(box[1] + box[3], box[2]),
                        xycoords="data",
                        weight="bold",
                        xytext=(10, 10),
                        textcoords="offset points",
                        size=10,
                        arrowprops=dict(arrowstyle="->"),
                    )

                # draw ground truth
                bbxoes = gts[:, 1:6]
                for i in range(len(bboxes)):
                    box = bbxoes[i]
                    rect = patches.Rectangle(
                        (box[1], box[2]),
                        box[3],
                        box[4],
                        linewidth=7,
                        facecolor="none",
                        alpha=1,
                    )
                    sample_ax.add_patch(rect)
                    sample_ax.annotate(
                        f"RID:{box[0]}",
                        xy=(box[0], box[1] + box[3]),
                        xycoords="data",
                        weight="bold",
                        xytext=(-50, -20),
                        textcoords="offset points",
                        size=10,
                        arrowprops=dict(arrowstyle="->"),
                    )
                sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.subplots_adjust(wspace=0, hspace=0.1 if len(ratios) > 1 else 0)
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()
