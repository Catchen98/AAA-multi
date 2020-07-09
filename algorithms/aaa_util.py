import random

import numpy as np
import pandas as pd

import networkx as nx

SEED = 2
random.seed(SEED)


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


def weighted_random_choice(w):
    pick = random.uniform(0, 1)
    current = 0
    for i, weight in enumerate(w):
        current += weight
        if current >= pick:
            return i
    return len(w) - 1


def match_id(prev_bboxes, curr_bboxes, threshold):
    """
    bbox shoud be [Number of box, 5] which is [id, x, y, w, h]
    """

    if len(curr_bboxes) == 0 or prev_bboxes is None or len(prev_bboxes) == 0:
        return []

    G = nx.DiGraph()
    edges = []
    for prev_bbox in prev_bboxes:
        prev_id = int(prev_bbox[0])
        iou_scores = overlap_ratio(prev_bbox[1:], curr_bboxes[:, 1:])
        valid_idxs = np.where(iou_scores > threshold)[0]

        for valid_idx in valid_idxs:
            edges.append(
                (
                    f"p{prev_id}",
                    f"c{valid_idx}",
                    {"capacity": 1, "weight": int(iou_scores[valid_idx] * 100)},
                )
            )

        edges.append(("s", f"p{prev_id}", {"capacity": 1, "weight": 0}))

    for i in range(len(curr_bboxes)):
        edges.append((f"c{i}", "t", {"weight": 0}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, "s", "t")

    result = []
    for start_point, sflow in mincostFlow["s"].items():
        if sflow == 1:
            for end_point, eflow in mincostFlow[start_point].items():
                if eflow == 1:
                    prev_id = int(start_point[1:])
                    curr_idx = int(end_point[1:])
                    curr_id = curr_bboxes[curr_idx, 0]
                    result.append((prev_id, curr_id))
                    break
    return result


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
    if loss_type == "sum":
        summary = mh.compute(
            acc, ana=ana, metrics=["num_false_positives", "num_misses", "num_switches"],
        )
        loss = sum(summary.iloc[0].values)
    elif loss_type == "sigmoid-sum":
        loss = 0
        for frame in acc.mot_events.index.unique(level=0):
            summary = mh.compute(
                acc.mot_events.loc[frame],
                ana=ana,
                metrics=["num_false_positives", "num_misses", "num_switches"],
            )
            met_sum = sum(summary.iloc[0].values)
            sigmoid = 1 / (1 + np.exp(-met_sum))
            loss += sigmoid
    elif loss_type == "mota":
        summary = mh.compute(acc, ana=ana, metrics=["mota"],)
        mota = summary.iloc[0].values[0]
        loss = 1 - mota / 100
    elif loss_type == "fmota":
        loss = 0
        for frame in acc.mot_events.index.unique(level=0):
            summary = mh.compute(acc.mot_events.loc[frame], ana=ana, metrics=["mota"])
            mota = summary.iloc[0].values[0]
            if np.isinf(mota):
                continue
            loss += 1 - mota / 100
    elif loss_type == "fn":
        summary = mh.compute(acc, ana=ana, metrics=["num_misses"],)
        fn = summary.iloc[0].values[0]
        loss = fn
    return loss
