import random
import numpy as np
import networkx as nx
from algorithms.aaa_util import overlap_ratio


def hungarian_matching(left_bboxes, right_bboxes, threshold):
    """
    bbox shoud be [Number of box, 5] which is [id, x, y, w, h]
    """

    if len(right_bboxes) == 0 or left_bboxes is None or len(left_bboxes) == 0:
        return []

    G = nx.DiGraph()
    edges = []
    for left_bbox in left_bboxes:
        left_id = int(left_bbox[0])
        iou_scores = overlap_ratio(left_bbox[1:], right_bboxes[:, 1:])
        valid_idxs = np.where(iou_scores > threshold)[0]

        for valid_idx in valid_idxs:
            edges.append(
                (
                    f"p{left_id}",
                    f"c{valid_idx}",
                    {"capacity": 1, "weight": int(iou_scores[valid_idx] * 100)},
                )
            )

        edges.append(("s", f"p{left_id}", {"capacity": 1, "weight": 0}))

    for i in range(len(right_bboxes)):
        edges.append((f"c{i}", "t", {"weight": 0}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, "s", "t")

    result = []
    for start_point, sflow in mincostFlow["s"].items():
        if sflow == 1:
            for end_point, eflow in mincostFlow[start_point].items():
                if eflow == 1:
                    left_id = int(start_point[1:])
                    right_idx = int(end_point[1:])
                    right_id = right_bboxes[right_idx, 0]
                    result.append((left_id, right_id))
                    break
    return result


class COP_KMeans:
    """
    https://github.com/Behrouz-Babaki/COP-Kmeans
    https://github.com/lars76/kmeans-anchor-boxes
    """

    def __init__(self, k, ml, cl, dist_func, max_iterations=300):
        self.k = k
        self.ml = ml
        self.cl = cl
        self.dist_func = dist_func
        self.max_iterations = max_iterations

    def fit(self, data, initial_method="random"):
        # initialize the centroids, the random 'k' elements in the dataset will be our initial centroids
        self.centroids = self.initialize_centers(data, self.k, initial_method)

        # begin iterations
        for i in range(self.max_iterations):
            self.is_clustered = [-1] * len(data)

            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = set()

            # find the distance between the point and cluster; choose the nearest centroid
            for x_index in range(len(data)):
                distances = self.dist_func(data[x_index], self.centroids)

                sorted_distances = np.argsort(distances)
                empty_flag = True

                for center_index in sorted_distances:
                    vc_result = self.violate_constraints(
                        x_index, center_index, self.ml, self.cl
                    )

                    if not vc_result:
                        self.clusters[center_index].add(x_index)
                        self.is_clustered[x_index] = center_index

                        for j in self.ml[x_index]:
                            self.is_clustered[j] = center_index

                        empty_flag = False
                        break

                # Unfeasible
                if empty_flag:
                    return -1

            previous = self.centroids.copy()

            # average the cluster data points to re-calculate the centroids
            for _center in self.clusters:
                lst = []
                for index_value in self.clusters[_center]:
                    lst.append(data[index_value])

                if len(lst) == 0:
                    continue

                if len(lst) == 1:
                    self.centroids[_center] = lst[0]
                    continue

                min_idx = -1
                min_sum_dist = np.inf
                for i in range(len(lst)):
                    others = np.array(lst[:i] + lst[i + 1 :])
                    distances = self.dist_func(lst[i], others)
                    sum_dist = np.sum(distances)
                    if sum_dist < min_sum_dist:
                        min_sum_dist = sum_dist
                        min_idx = i

                self.centroids[_center] = lst[min_idx]

            # Converged
            if (previous == self.centroids).all():
                return 1

        # Not converged
        return 0

    def violate_constraints(self, data_index, cluster_index, ml, cl):
        for i in ml[data_index]:
            if (
                data_index != i
                and self.is_clustered[i] != -1
                and self.is_clustered[i] != cluster_index
            ):
                return True

        for i in cl[data_index]:
            if (
                data_index != i
                and self.is_clustered[i] != -1
                and self.is_clustered[i] == cluster_index
            ):
                return True

        return False

    def initialize_centers(self, dataset, k, method):
        if method == "random":
            c = np.zeros((k, 4))
            index_list = list(range(len(dataset)))
            random.shuffle(index_list)
            for i in range(k):
                c[i] = dataset[index_list[i]]

        elif method == "km++":
            chances = [1] * len(dataset)
            centers = []
            c = np.zeros((k, 4))

            for i in range(k):
                chances = [x / sum(chances) for x in chances]
                r = random.random()
                acc = 0.0
                for index, chance in enumerate(chances):
                    if acc + chance >= r:
                        break
                    acc += chance
                centers.append(dataset[index])
                c[i] = dataset[index]
                for index, point in enumerate(dataset):
                    distances = self.dist_func(point, c)
                    chances[index] = np.min(distances)

        return c


class IDMatcher:
    def __init__(self):
        pass

    def initialize(self, n_experts):
        self.last_id = 0
        self.id_table = {i: {} for i in range(n_experts)}

    def previous_match(self, prev_bboxes, selected_expert, results, threshold):
        curr_expert_bboxes = results[selected_expert].copy()

        matched_id = hungarian_matching(prev_bboxes, curr_expert_bboxes, threshold)
        modified_bboxes = curr_expert_bboxes.copy()

        # get target idx
        target_idxs = {}
        for prev_id, curr_id in matched_id:
            curr_idx = np.where(modified_bboxes[:, 0] == curr_id)[0]
            if len(curr_idx) > 0:
                target_idxs[curr_idx[0]] = prev_id

        # assign id
        for target_idx, prev_id in target_idxs.items():
            modified_bboxes[target_idx, 0] = prev_id

        # create new id
        for i in range(len(modified_bboxes)):
            if i not in target_idxs.keys():
                modified_bboxes[i, 0] = self.last_id
                self.last_id += 1

        return modified_bboxes

    def kmeans_match(
        self, experts_w, selected_expert, results, threshold, cl_mode="all"
    ):
        def iou_distance(x, y):
            return 1 - overlap_ratio(x, y)

        # flatten results
        flatid2originid = []
        originid2flatid = []
        flat_bboxes = []
        for e_i, result in enumerate(results):
            flatids = []
            for box in result:
                flat_bboxes.append(box[1:])
                flatids.append(len(flatid2originid))
                flatid2originid.append((e_i, box[0]))
            originid2flatid.append(flatids)
        flat_bboxes = np.array(flat_bboxes)

        if len(flat_bboxes) == 0:
            return []

        # make can not link
        min_k = 0
        cl = [[] for i in range(len(flat_bboxes))]
        for i in range(len(flat_bboxes)):
            expert_idx = flatid2originid[i][0]

            # the boxes from an expert can not be in a cluster
            if cl_mode == "all":
                cl[i] += originid2flatid[expert_idx]

                # get the maximum number of elements
                if len(originid2flatid[expert_idx]) > min_k:
                    min_k = len(originid2flatid[expert_idx])

            elif cl_mode == "selected":
                # only if the expert is selected expert
                if expert_idx == selected_expert:
                    cl[i] += originid2flatid[expert_idx]

                # get the number of elements
                min_k = len(originid2flatid[selected_expert])

            # when the score is lower than the threshold, the boxes can not be connected
            scores = overlap_ratio(flat_bboxes[i], flat_bboxes[i + 1 :])
            for j, score in enumerate(scores):
                if score < threshold:
                    cl[i].append(i + 1 + j)
                    cl[i + 1 + j].append(i)

        # make must link
        ml = [[] for i in range(len(flat_bboxes))]

        # cluster boxes
        for k in range(min_k, len(flat_bboxes) + 1):
            kmeans = COP_KMeans(k, ml, cl, iou_distance)
            fit_result = kmeans.fit(flat_bboxes)
            if fit_result == 1:
                break

        assigned_cluster_idx = set()
        assigned_cluster_ids = set()

        # dorminant order
        weight_order = list(np.argsort(experts_w)[::-1])
        weight_order.remove(selected_expert)
        expert_order = [selected_expert] + weight_order
        cluster_idxs = sorted(kmeans.clusters.keys())
        for expert_idx in expert_order:
            for cluster_idx in cluster_idxs:
                flat_ids = kmeans.clusters[cluster_idx]
                expert_idxs = [flatid2originid[flat_id][0] for flat_id in flat_ids]
                box_ids = [flatid2originid[flat_id][1] for flat_id in flat_ids]

                # check the cluster is never assigned and the expert is in the cluster
                if (
                    cluster_idx not in assigned_cluster_idx
                    and expert_idx in expert_idxs
                ):
                    box_id = box_ids[expert_idxs.index(expert_idx)]

                    # if the box is already assigned, give the cluster the same id as the box.
                    if box_id in self.id_table[expert_idx].keys():
                        cluster_id = self.id_table[expert_idx][box_id]

                        # if the cluster id is already used, pass the cluster
                        if cluster_id in assigned_cluster_ids:
                            continue
                    # or, give the cluster new id
                    else:
                        cluster_id = self.last_id
                        self.last_id += 1

                    # save the cluster id to id pool
                    for expert_idx, box_id in zip(expert_idxs, box_ids):
                        self.id_table[expert_idx][box_id] = cluster_id

                    # save the cluster idx and id
                    assigned_cluster_idx.add(cluster_idx)
                    assigned_cluster_ids.add(cluster_id)

        # for not assigned cluster, assign new id
        for cluster_idx in cluster_idxs:
            if cluster_idx not in assigned_cluster_idx:
                flat_ids = kmeans.clusters[cluster_idx]
                expert_idxs = [flatid2originid[flat_id][0] for flat_id in flat_ids]
                box_ids = [flatid2originid[flat_id][1] for flat_id in flat_ids]

                cluster_id = self.last_id
                self.last_id += 1

                # save the cluster id to id pool
                for expert_idx, box_id in zip(expert_idxs, box_ids):
                    self.id_table[expert_idx][box_id] = cluster_id

        # assign cluster id to box
        modified_bboxes = results[selected_expert].copy()
        for i in range(len(modified_bboxes)):
            modified_bboxes[i, 0] = self.id_table[selected_expert][
                modified_bboxes[i, 0]
            ]

        return modified_bboxes
