import os
from PIL import Image
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import seaborn as sns

import motmetrics as mm

from algorithms.aaa_util import convert_df
from file_manager import ReadResult

sns.set()
sns.set_style("whitegrid")


LINEWIDTH = 4
ANNOT_SIZE = 12
ALL_HEIGHT = 3


def calc_rank(dataset, trackers_name, scores, reverse=False):
    ranks = []
    for seq_name in dataset.sequence_names["train"]:
        value = np.array(
            [scores[tracker_name][seq_name] for tracker_name in trackers_name]
        )
        temp = value.argsort()
        if reverse:
            temp = temp[::-1]
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(value))
        rank = len(trackers_name) - rank
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks


def draw_detngt(dataset, result_dir):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        if seq_name != "MOT17-09-SDP":
            continue

        save_dir = result_dir / dataset_name / "DetNGT" / seq_name

        os.makedirs(save_dir, exist_ok=True)

        for frame_idx, (img_path, dets, gts) in enumerate(seq):
            filename = os.path.basename(img_path)
            im = Image.open(img_path).convert("RGB")

            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            sample_ax = axes

            # draw frame
            sample_ax.imshow(np.asarray(im), aspect="auto")

            # draw detection results
            for i in range(len(dets)):
                box = dets[i, 2:6]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=LINEWIDTH,
                    edgecolor="purple",
                    facecolor="none",
                    alpha=1,
                )
                sample_ax.add_patch(rect)
                sample_ax.annotate(
                    "Det",
                    xy=(box[0] + box[2] / 2, box[1]),
                    xycoords="data",
                    weight="bold",
                    xytext=(-5, 5),
                    textcoords="offset points",
                    size=ANNOT_SIZE,
                    color="purple",
                )

            # draw ground truth
            for i in range(len(gts)):
                if (dataset_name == "MOT16" or dataset_name == "MOT17") and gts[
                    i, 7
                ] != 1:
                    continue
                box = gts[i, 1:6]
                rect = patches.Rectangle(
                    (box[1], box[2]),
                    box[3],
                    box[4],
                    linewidth=LINEWIDTH,
                    edgecolor="darkorange",
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
                    color="darkorange",
                )
            sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()


def draw_det(dataset, result_dir):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        if seq_name != "MOT17-09-SDP":
            continue

        save_dir = result_dir / dataset_name / "Det" / seq_name

        os.makedirs(save_dir, exist_ok=True)

        for frame_idx, (img_path, dets, gts) in enumerate(seq):
            filename = os.path.basename(img_path)
            im = Image.open(img_path).convert("RGB")

            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            sample_ax = axes

            # draw frame
            sample_ax.imshow(np.asarray(im), aspect="auto")

            # draw detection results
            for i in range(len(dets)):
                box = dets[i, 2:6]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=LINEWIDTH,
                    edgecolor="black",
                    facecolor="none",
                    alpha=1,
                )
                sample_ax.add_patch(rect)
                sample_ax.annotate(
                    "Det",
                    xy=(box[0] + box[2] / 2, box[1]),
                    xycoords="data",
                    weight="bold",
                    xytext=(-5, -5),
                    textcoords="offset points",
                    size=ANNOT_SIZE,
                    color="black",
                )
            sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()


def draw_gt(dataset, result_dir):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        save_dir = result_dir / dataset_name / "GT" / seq_name

        os.makedirs(save_dir, exist_ok=True)

        for frame_idx, (img_path, dets, gts) in enumerate(seq):
            filename = os.path.basename(img_path)
            im = Image.open(img_path).convert("RGB")

            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            sample_ax = axes

            # draw frame
            sample_ax.imshow(np.asarray(im), aspect="auto")

            # draw ground truth
            for i in range(len(gts)):
                if (dataset_name == "MOT16" or dataset_name == "MOT17") and gts[
                    i, 7
                ] != 1:
                    continue
                box = gts[i, 1:6]
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
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()


def draw_result(
    output_dir, dataset, tracker_name, result_dir, is_algorithm=True, duration=None,
):
    if is_algorithm:
        show = ["weight", "frame"]
    else:
        show = ["frame"]

    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]
        tracker_reader = ReadResult(output_dir, dataset_name, tracker_name, seq_name)

        if is_algorithm:
            dataset_dir = output_dir / dataset_name / tracker_name

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
                for i in range(len(gts)):
                    if gts[i, 7] != 1:
                        continue
                    box = gts[i, 1:6]
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


def draw_all_result(output_dir, dataset, trackers_name, result_dir, colors, duration):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]
        trackers_reader = [
            ReadResult(output_dir, dataset_name, tracker_name, seq_name)
            for tracker_name in trackers_name
        ]

        dataset_dir = output_dir / dataset_name / trackers_name[0]

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
                weight_ax.plot(range(len(weight)), weight, color=colors[i])
                weight_ax.set(ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,))
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
                        edgecolor=colors[t_i],
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
                        color=colors[t_i],
                    )

                # draw ground truth
                for i in range(len(gts)):
                    if gts[i, 7] != 1:
                        continue
                    box = gts[i, 1:6]
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
            plt.savefig(save_dir / filename, bbox_inches="tight", dpi=100)
            plt.close()


def draw_weight_graph(
    output_dir, dataset, algorithm, experts_name, result_dir, duration=None,
):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        dataset_dir = output_dir / dataset_name / algorithm

        weight_path = dataset_dir / f"{seq_name}_weight.txt"
        weights = pd.read_csv(weight_path, header=None).set_index(0)

        save_dir = result_dir / dataset_name / algorithm / seq_name
        os.makedirs(save_dir, exist_ok=True)

        fig, weight_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        fig.add_subplot(111, frameon=False)

        # draw weight graph
        for i in weights.columns:
            weight = weights[i]
            weight_ax.plot(range(len(weight)), weight, label=experts_name[i - 1])
            weight_ax.set(
                ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,),
            )
        weight_ax.set_xticks([])

        # draw anchor line
        for i in range(len(seq)):
            if (i + 1) % duration == 0:
                weight_ax.axvline(
                    x=i, color="gray", linestyle="--", linewidth=1, alpha=1
                )

        weight_ax.legend(
            ncol=len(weights.columns),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
        )

        # hide tick and tick label of the big axes
        plt.axis("off")
        plt.grid(False)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_dir / "weight.pdf", bbox_inches="tight")
        plt.close()


def draw_loss_graph(
    output_dir, dataset, algorithm, experts_name, result_dir, duration=None,
):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        experts_reader = [
            ReadResult(output_dir, dataset_name, tracker_name, seq_name)
            for tracker_name in experts_name
        ]

        save_dir = result_dir / dataset_name / algorithm / seq_name
        os.makedirs(save_dir, exist_ok=True)

        fig, loss_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        fig.add_subplot(111, frameon=False)

        # draw loss graph
        fns = {}
        for expert_reader, expert_name in zip(experts_reader, experts_name):
            seq.c = 0
            fns[expert_name] = []
            for frame_idx, (img_path, dets, gts) in enumerate(seq):
                bboxes = expert_reader.get_result_by_frame(frame_idx, with_frame=True)
                df_expert = convert_df(bboxes)
                df_gt = pd.DataFrame(
                    gts,
                    columns=[
                        "FrameId",
                        "Id",
                        "X",
                        "Y",
                        "Width",
                        "Height",
                        "Confidence",
                        "ClassId",
                        "Visibility",
                    ],
                )
                df_gt = df_gt.set_index(["FrameId", "Id"])

                acc, ana = mm.utils.CLEAR_MOT_M(
                    df_gt,
                    df_expert,
                    seq.seq_info["ini_path"],
                    "iou",
                    distth=0.5,
                    vflag="",
                )
                mh = mm.metrics.create()
                summary = mh.compute(acc, ana=ana, metrics=["num_misses"],)
                fn = summary.iloc[0].values[0]
                fns[expert_name].append(fn)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for expert_name, color in zip(experts_name, colors):
            # if expert_name not in ["DeepSort", "MOTDT"]:
            #     continue
            fn = fns[expert_name]
            loss_ax.plot(range(len(fn)), np.cumsum(fn), label=expert_name, color=color)
            loss_ax.set(
                ylabel="FN", xlim=(0, len(seq)),
            )
        loss_ax.set_xticks([])

        # draw anchor line
        for i in range(len(seq)):
            if (i + 1) % duration == 0:
                loss_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=1)

        # loss_ax.set_xlim([200, 420])
        # loss_ax.set_ylim([500, 1500])

        # hide tick and tick label of the big axes
        plt.axis("off")
        plt.grid(False)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_dir / "loss.pdf", bbox_inches="tight")
        plt.close()


def draw_rank(dataset, trackers_name, fn_scores, mota_scores, save_dir):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 3))
    fig.add_subplot(111, frameon=False)

    lines = []
    xs = list(range(1, len(trackers_name) + 1))

    # draw fn
    ax = axes[0]
    fn_ranks = calc_rank(dataset, trackers_name, fn_scores, reverse=True)
    for tracker_name, rank in zip(trackers_name, fn_ranks.T):
        line = ax.plot(
            xs,
            [np.sum(rank == x) / len(dataset) for x in xs],
            label="AAA" if "AAA" in tracker_name else tracker_name,
            linewidth=4 if "AAA" in tracker_name else 2,
        )[0]
        lines.append(line)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([1, len(trackers_name) // 2 + 1, len(trackers_name)])
    ax.set_xticklabels([])
    ax.set_ylabel("FN")
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=len(trackers_name),
        bbox_to_anchor=(0.5, 1.5),
    )

    # draw mota
    ax = axes[1]
    mota_ranks = calc_rank(dataset, trackers_name, mota_scores)
    for tracker_name, rank in zip(trackers_name, mota_ranks.T):
        line = ax.plot(
            xs,
            [np.sum(rank == x) / len(dataset) for x in xs],
            label="AAA" if "AAA" in tracker_name else tracker_name,
            linewidth=4 if "AAA" in tracker_name else 2,
        )[0]
        lines.append(line)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([1, len(trackers_name) // 2 + 1, len(trackers_name)])
    ax.set_xticklabels(["Best", str(len(trackers_name) // 2 + 1), "Worst"])
    ax.set_ylabel("MOTA")

    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)
    plt.xlabel("Rank")
    # plt.ylabel("Frequency of rank")

    plt.subplots_adjust(wspace=None, hspace=0.1)

    plt.savefig(save_dir / "rank.pdf", bbox_inches="tight")
    plt.close()
