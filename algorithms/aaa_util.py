import numpy as np
import pandas as pd
import motmetrics as mm


def overlap_ratio(rect1, rect2):
    """
    https://github.com/StrangerZhang/pysot-toolkit
    """
    if rect1.ndim == 1:
        rect1 = rect1[np.newaxis, :]
    if rect2.ndim == 1:
        rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def goverlap_ratio(rect1, rect2):
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    left_min = np.minimum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    right_max = np.maximum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    top_min = np.minimum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])
    bottom_max = np.maximum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    closure = np.maximum(0, right_max - left_min) * np.maximum(0, bottom_max - top_min)
    g_iou = iou - (closure - union) / closure
    g_iou = (1 + g_iou) / 2
    g_iou = np.nan_to_num(g_iou)
    return g_iou


def weighted_random_choice(weights):
    selection_probs = weights / np.sum(weights)
    selected = np.random.choice(len(weights), p=selection_probs)
    return selected


def convert_df(results, is_offline=False):
    if len(results) > 0:
        data = np.zeros((len(results), 10))
        data[:, :6] = results
        data[:, 6:] = 1
        data[:, 7:] = 1 if is_offline else -1
    else:
        data = np.empty((0, 10))
    df = pd.DataFrame(
        data,
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
            "unused",
        ],
    )

    # Account for matlab convention.
    df[["X", "Y"]] -= (1, 1)

    # Removed trailing column
    del df["unused"]

    df = df.set_index(["FrameId", "Id"])

    return df


def frame_loss(df_map, frame_list):
    df = df_map.noraw
    fp = df[df["Type"] == "FP"]
    fn = df[df["Type"] == "MISS"]
    ids = df[df["Type"] == "SWITCH"]

    result = np.zeros((len(frame_list), 3))
    for i, frame in enumerate(frame_list):
        result[i] = [
            len(fp.Type.get(frame, [])),
            len(fn.Type.get(frame, [])),
            len(ids.Type.get(frame, [])),
        ]

    return result


def eval_results(seq_info, gt, pred):
    if (
        seq_info["dataset_name"] == "MOT16"
        or seq_info["dataset_name"] == "MOT17"
        or seq_info["dataset_name"] == "MOT20"
    ):
        acc, ana = mm.utils.CLEAR_MOT_M(
            gt, pred, seq_info["ini_path"], "iou", distth=0.5, vflag="",
        )
    else:
        acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5)
        ana = None

    df_map = mm.metrics.events_to_df_map(acc.events)
    return acc, ana, df_map


def get_summary(acc, ana):
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        ana=ana,
        metrics=["num_false_positives", "num_misses", "num_switches", "mota"],
    )
    return summary.iloc[0].values


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
