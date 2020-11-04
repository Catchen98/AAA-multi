import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from algorithms.aaa_util import frame_loss
from datasets.mot import MOT

sns.set()
sns.set_style("whitegrid")


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


def calc_rank_test(mota_scores, reverse=False):
    ranks = []
    for i in range(len(list(mota_scores.values())[0])):
        value = np.array(
            [mota_scores[tracker_name][i] for tracker_name in mota_scores.keys()]
        )
        temp = value.argsort()
        if reverse:
            temp = temp[::-1]
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(value))
        rank = len(mota_scores.keys()) - rank
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks


def draw_rank(eval_dir, dataset, algorithm, experts_name, result_dir, in_al=True):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))
    fig.add_subplot(111, frameon=False)

    if in_al:
        trackers_name = experts_name + [algorithm]
    else:
        trackers_name = experts_name

    xs = list(range(1, len(trackers_name) + 1))
    seq_names = []

    mota_scores = {tracker_name: {} for tracker_name in trackers_name}
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]
        seq_names.append(seq_name)
        for tracker_name in trackers_name:
            error_path = eval_dir / tracker_name / dataset_name / f"{seq_name}.csv"
            errors = pd.read_csv(error_path, header=0)
            frame_error = frame_loss(errors, range(1, len(seq) + 1))
            frame_error = -frame_error.sum()
            mota_scores[tracker_name][seq_name] = frame_error

    # draw error
    ax = axes[0]
    for i, tracker_name in enumerate(trackers_name):
        ax.plot(
            range(len(seq_names)),
            [-mota_scores[tracker_name][seq_name] for seq_name in seq_names],
            label="Ours" if i == len(trackers_name) - 1 and in_al else tracker_name,
            linewidth=2,
        )
    ax.set_ylabel("Error")
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=len(trackers_name),
        bbox_to_anchor=(0.5, 1.2),
    )

    mota_ranks = calc_rank(dataset, trackers_name, mota_scores)

    # draw rank
    ax = axes[1]
    for i, (tracker_name, rank) in enumerate(zip(trackers_name, mota_ranks.T)):
        ax.plot(
            range(len(seq_names)),
            rank,
            label="Ours" if i == len(trackers_name) - 1 else tracker_name,
            linewidth=2,
        )
    ax.set_ylabel("Rank")

    # draw mota
    ax = axes[2]
    for i, (tracker_name, rank) in enumerate(zip(trackers_name, mota_ranks.T)):
        ax.plot(
            xs,
            [np.sum(rank == x) / len(dataset) for x in xs],
            label="Ours" if i == len(trackers_name) - 1 else tracker_name,
            linewidth=2,
        )

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([1, (len(trackers_name) + 1) // 2, len(trackers_name)])
    ax.set_xticklabels(["Best", str((len(trackers_name) + 1) // 2), "Worst"])
    ax.set_ylabel("Frequency")

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

    plt.subplots_adjust(wspace=None, hspace=0.2)

    save_dir = result_dir / dataset_name / algorithm
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir / "rank.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_test(mota_scores, result_dir):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))
    fig.add_subplot(111, frameon=False)

    trackers_name = mota_scores.keys()
    xs = list(range(1, len(trackers_name) + 1))

    # draw mota
    ax = axes[0]
    for tracker_name in trackers_name:
        ax.plot(
            range(len(mota_scores[tracker_name])),
            mota_scores[tracker_name],
            label=tracker_name,
            linewidth=2,
        )
    ax.set_ylabel("MOTA")
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=len(trackers_name),
        bbox_to_anchor=(0.5, 1.2),
    )

    mota_ranks = calc_rank_test(mota_scores)

    # draw rank
    ax = axes[1]
    for i, (tracker_name, rank) in enumerate(zip(trackers_name, mota_ranks.T)):
        ax.plot(
            range(len(mota_scores[tracker_name])),
            rank,
            label=tracker_name,
            linewidth=2,
        )
    ax.set_ylabel("Rank")

    # draw frequency
    ax = axes[2]
    for i, (tracker_name, rank) in enumerate(zip(trackers_name, mota_ranks.T)):
        ax.plot(
            xs,
            [np.sum(rank == x) / len(mota_scores[tracker_name]) for x in xs],
            label=tracker_name,
            linewidth=2,
        )

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([1, (len(trackers_name) + 1) // 2, len(trackers_name)])
    ax.set_xticklabels(["Best", str((len(trackers_name) + 1) // 2), "Worst"])
    ax.set_ylabel("Frequency")

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

    plt.subplots_adjust(wspace=None, hspace=0.2)

    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(result_dir / "rank.pdf", bbox_inches="tight")
    plt.close()


def draw_weight(
    output_dir, eval_dir, dataset, algorithm, experts_name, result_dir,
):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        dataset_dir = output_dir / dataset_name / algorithm

        weight_path = dataset_dir / f"{seq_name}_weight.txt"
        weights = pd.read_csv(weight_path, header=None).set_index(0)

        loss_path = dataset_dir / f"{seq_name}_loss.txt"
        loss = pd.read_csv(loss_path, header=None).set_index(0)
        cum_loss = loss.cumsum()

        trackers_name = experts_name + [algorithm]
        vis_name = experts_name + ["Ours"]

        frame_errors = []
        for tracker_name in trackers_name:
            error_path = eval_dir / tracker_name / dataset_name / f"{seq_name}.csv"
            errors = pd.read_csv(error_path, header=0)
            frame_error = frame_loss(errors, range(1, len(seq) + 1))
            frame_error = frame_error.sum(axis=1)
            frame_errors.append(frame_error)

        save_dir = result_dir / dataset_name / algorithm / seq_name
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        fig.add_subplot(111, frameon=False)

        error_ax = axes[0]
        cum_error_ax = axes[1]
        weight_ax = axes[2]
        loss_ax = axes[3]

        # draw error graph
        for i in range(len(vis_name)):
            if vis_name[i] in ["DeepSORT", "MOTDT"]:
                continue
            error = frame_errors[i]
            error_ax.plot(range(len(error)), error, label=vis_name[i])
            error_ax.set(
                ylabel="Error", xlim=(0, len(seq)),
            )
        error_ax.set_xticks([])

        # draw cum error graph
        for i in range(len(vis_name)):
            error = np.cumsum(frame_errors[i])
            cum_error_ax.plot(range(len(error)), error, label=vis_name[i])
            cum_error_ax.set(
                ylabel="Cum Error", xlim=(0, len(seq)),
            )
        cum_error_ax.set_xticks([])

        # draw weight graph
        for i in weights.columns:
            weight = weights[i]
            weight_ax.plot(range(len(weight)), weight, label=vis_name[i - 1])
            weight_ax.set(
                ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,),
            )
        weight_ax.set_xticks([])

        # draw loss graph
        for i in cum_loss.columns:
            loss = cum_loss[i]
            loss_ax.plot(cum_loss.index, cum_loss, label=vis_name[i - 1])
            loss_ax.set(
                ylabel="Cum Loss", xlim=(0, len(seq)),
            )
        loss_ax.set_xticks([])

        # draw anchor line
        for i in loss.index:
            weight_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3)
            error_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3)
            cum_error_ax.axvline(
                x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3
            )
            loss_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3)

        error_ax.legend(
            ncol=len(vis_name),
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


def draw_weight_test(
    output_dir, dataset, algorithm, experts_name, result_dir,
):
    for seq in dataset:
        dataset_name = seq.seq_info["dataset_name"]
        seq_name = seq.seq_info["seq_name"]

        dataset_dir = output_dir / dataset_name / algorithm

        weight_path = dataset_dir / f"{seq_name}_weight.txt"
        weights = pd.read_csv(weight_path, header=None).set_index(0)

        loss_path = dataset_dir / f"{seq_name}_loss.txt"
        loss = pd.read_csv(loss_path, header=None).set_index(0)
        cum_loss = loss.cumsum()

        vis_name = experts_name + ["Ours"]

        save_dir = result_dir / dataset_name / algorithm / seq_name
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        fig.add_subplot(111, frameon=False)

        weight_ax = axes[0]
        loss_ax = axes[1]

        # draw weight graph
        for i in weights.columns:
            weight = weights[i]
            weight_ax.plot(range(len(weight)), weight, label=vis_name[i - 1])
            weight_ax.set(
                ylabel="Weight", xlim=(0, len(seq)), ylim=(-0.05, 1.05,),
            )
        weight_ax.set_xticks([])

        # draw loss graph
        for i in cum_loss.columns:
            loss = cum_loss[i]
            loss_ax.plot(cum_loss.index, cum_loss, label=vis_name[i - 1])
            loss_ax.set(
                ylabel="Cum Loss", xlim=(0, len(seq)),
            )
        loss_ax.set_xticks([])

        # draw anchor line
        for i in loss.index:
            weight_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3)
            loss_ax.axvline(x=i, color="gray", linestyle="--", linewidth=1, alpha=0.3)

        weight_ax.legend(
            ncol=len(vis_name),
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


if __name__ == "__main__":
    output_dir = Path("output")
    # eval_dir = Path("eval")
    result_dir = Path("visualize")
    dataset = MOT("/home/heonsong/Disk2/Dataset/MOT/MOT16")
    # algorithm = "{'use_gt': True, 'pre_cnn': True, 'pre_track': 'FRCNN'}, {'method': 'anchor', 'threshold': 0.5, 'score_mode': 'mvote', 'iou_mode': 'giou'}, {'type': 'stable', 'duration': 2, 'threshold': 1}, {'delayed': True, 'type': 'w_id', 'bound': 0.01}"
    # algorithm = "{'use_gt': False, 'pre_cnn': True, 'pre_track': 'FRCNN'}, {'method': 'anchor', 'threshold': 0.5, 'score_mode': 'mvote', 'iou_mode': 'giou'}, {'type': 'stable', 'duration': 70, 'threshold': 0.4}, {'delayed': True, 'type': 'w_id', 'bound': 1.0}"
    # experts_name = ["DeepMOT", "DeepSORT", "MOTDT", "Tracktor", "UMA"]
    # draw_rank(eval_dir, dataset, algorithm, experts_name, result_dir, in_al=False)
    # draw_weightNerror_graph(
    #     output_dir, eval_dir, dataset, algorithm, experts_name, result_dir
    # )

    mota_scores = OrderedDict({
        "AMIR15": [29.1475, 73.6842, 47.0387, 37.5716, 36.6835, 50.4409, 34.5465, 45.0926, 32.9013, 36.15, 40.5863, 24.9302][::-1],
        "AP_HWDPL_p": [39.1409, 61.2523, 38.8549, 38.4863, 34.2879, 40.682, 39.3729, 52.8971, 34.1974, 28.4415, 43.2815, 35.2568][::-1],
        "KCF": [35.7659, 76.0436, 51.8307, 38.8965, 33.7264, 53.0276, 28.1635, 50.4139, 31.5055, 40.0951, 43.0061, 25.5642][::-1],
        "STRN": [39.7765, 67.9673, 47.9203, 38.0615, 26.8389, 38.0952, 33.841, 51.6752, 32.1037, 27.9099, 36.5434, 40.3396][::-1],
        "Tracktor++v2": [45.8032, 77.7677, 46.4993, 46.5983, 47.5201, 51.7931, 49.944, 57.6271, 43.7687, 43.005, 47.0096, 38.3194][::-1],
        "TrctrD15": [36.0728, 77.4955, 47.0698, 44.0934, 49.4853, 49.7354, 49.0929, 57.4695, 43.8684, 34.2753, 45.4948, 34.4509][::-1],
    })
    draw_rank_test(mota_scores, result_dir / "MOT15")
    mota_scores = OrderedDict({
        "GSM_Tracktor": [33.4091, 49.0416, 35.538, 47.0714, 50.0433, 68.4016, 43.4715][::-1],
        "KCF16": [29.2864, 42.8933, 28.7925, 45.8584, 48.2059, 57.1751, 36.9195][::-1],
        "MLT": [30.6714, 47.0886, 42.4688, 46.2505, 50.312, 60.4308, 46.9116][::-1],
        "PV_16": [24.0708, 45.4732, 35.9324, 41.8576, 48.7173, 59.9172, 39.8749][::-1],
        "Tracktor++v2": [32.4893, 48.1977, 33.3931, 43.567, 54.8275, 67.6585, 42.2361][::-1],
        "TrctrD16": [28.2097, 48.1615, 34.6239, 43.0891, 53.6661, 66.2946, 37.7952][::-1],
    })
    draw_rank_test(mota_scores, result_dir / "MOT16")
    mota_scores = OrderedDict({
        "FAMNet": [37.0935, 28.0312, 33.0141, 44.906, 41.7561, 42.2984, 29.0428, 24.4556, 25.6864, 49.7603, 39.3891, 44.5096, 57.6205, 56.11, 49.6775, 72.5197, 58.7265, 56.9152, 50.6667, 41.907, 40.2326][::-1],
        "GSM_Tracktor": [37.034, 34.1503, 33.4091, 45.2983, 44.4098, 47.179, 28.5031, 27.2676, 28.4321, 46.321, 43.3434, 45.7882, 53.4114, 52.8598, 49.72, 72.5856, 68.6124, 68.5111, 43.8295, 43.7674, 43.1008][::-1],
        "SRF17": [35.9465, 33.4145, 32.3919, 45.9905, 45.0098, 46.429, 27.812, 26.5291, 26.9409, 44.8055, 42.1062, 42.5679, 55.7111, 55.5923, 52.6561, 72.364, 67.9131, 67.3332, 43.1783, 43.0543, 41.845][::-1],
        "Tracktor++v2": [36.1576, 34.0313, 32.4893, 45.9444, 45.0213, 46.3136, 27.4806, 26.2782, 26.7232, 44.5096, 42.2305, 42.2838, 58.4097, 58.2315, 54.2855, 73.2658, 68.7624, 67.915, 43.845, 43.7829, 41.876][::-1],
        "TrctrD17": [28.675, 28.4802, 27.4414, 45.1713, 43.406, 45.5406, 28.7351, 27.1113, 27.0593, 41.455, 39.5844, 40.7032, 56.2118, 55.9912, 52.4864, 69.8362, 66.2947, 65.5983, 38.5426, 35.8605, 37.1783][::-1],
        "YOONKJ17": [38.2081, 21.6794, 28.3828, 41.2253, 38.6524, 40.9253, 32.3755, 25.6628, 26.7705, 47.9252, 33.6412, 39.8923, 50.3904, 49.2872, 46.0709, 76.856, 60.2599, 54.4218, 46.0155, 25.2713, 36.8062][::-1],
    })
    draw_rank_test(mota_scores, result_dir / "MOT17")

    # algorithm = "{'use_gt': False, 'pre_cnn': True, 'pre_track': 'FRCNN'}, {'method': 'anchor', 'threshold': 0.5, 'score_mode': 'mvote', 'iou_mode': 'giou'}, {'type': 'stable', 'duration': 100, 'threshold': 0.4}, {'delayed': True, 'type': 'w_id', 'bound': 1.0}"
    # experts_name = ["AMIR15", "AP_HWDPL_p", "KCF", "STRN", "Tracktor++v2", "TrctrD15"]
    # experts_name = ["GSM_Tracktor", "KCF16", "MLT", "PV_16", "Tracktor++v2", "TrctrD16"]
    # experts_name = ["FAMNet", "GSM_Tracktor", "SRF17", "Tracktor++v2", "TrctrD17", "YOONKJ17"]
    # draw_weight_test(output_dir, dataset, algorithm, experts_name, result_dir)
