import numpy as np
import pandas as pd


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


def loss_function(loss_type, mh, acc, ana):
    if loss_type == "w_id":
        summary = mh.compute(
            acc, ana=ana, metrics=["num_false_positives", "num_misses", "num_switches"],
        )
        loss = sum(summary.iloc[0].values)
    elif loss_type == "wo_id":
        summary = mh.compute(
            acc, ana=ana, metrics=["num_false_positives", "num_misses"],
        )
        loss = sum(summary.iloc[0].values)
    elif loss_type == "fn":
        summary = mh.compute(acc, ana=ana, metrics=["num_misses"],)
        fn = summary.iloc[0].values[0]
        loss = fn
    return loss


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
