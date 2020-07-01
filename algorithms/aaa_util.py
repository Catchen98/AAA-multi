import numpy as np
import pandas as pd

import networkx as nx
from networkx.algorithms import bipartite


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


def convert_id(prev_bboxes, curr_bboxes, threshold):
    """
    bbox shoud be [Number of box, 5] which is [id, x, y, w, h]
    """

    if len(curr_bboxes) == 0:
        return []

    # init id
    curr_bboxes[:, 0] = -1

    if prev_bboxes is None or len(prev_bboxes) == 0:
        return curr_bboxes

    # G = {}
    # for prev_bbox in prev_bboxes:
    #     prev_id = int(prev_bbox[0])
    #     iou_scores = overlap_ratio(prev_bbox[1:], curr_bboxes[:, 1:])
    #     valid_idxs = np.where(iou_scores > threshold)[0]

    #     if len(valid_idxs) > 0:
    #         G[f"p{prev_id}"] = {
    #             f"c{valid_idx}": iou_scores[valid_idx] for valid_idx in valid_idxs
    #         }

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

        edges.append(("s", f"p{prev_id}", {"weight": 0}))

    for i in range(len(curr_bboxes)):
        edges.append((f"c{i}", "t", {"weight": 0}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, "s", "t")
    for start_point, sflow in mincostFlow["s"].items():
        if sflow == 1:
            for end_point, eflow in mincostFlow[start_point].items():
                if eflow == 1:
                    prev_id = int(start_point[1:])
                    curr_idx = int(end_point[1:])
                    curr_bboxes[curr_idx, 0] = prev_id
                    break

    return curr_bboxes


def convert_df(results, confidence=1):
    data = np.zeros((len(results), 10))
    data[:, :6] = results
    data[:, 6:] = confidence
    data[:, 7:] = -1
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