import random
from collections.abc import Iterable
import numpy as np
import networkx as nx
from algorithms.aaa_util import overlap_ratio, goverlap_ratio


def proper_overlap(x, y, mode):
    if mode == "iou":
        score = overlap_ratio(x, y)
    elif mode == "giou":
        score = goverlap_ratio(x, y)
    else:
        raise NameError("Please enter a valid iou method")

    return score


def overlap_distance(iou_mode):
    def distance(x, y):
        valid_x = x.copy()
        valid_y = y.copy()
        if len(valid_x) == 5:
            valid_x = valid_x[1:]
            valid_y = valid_y[:, 1:]

        score = proper_overlap(valid_x, valid_y, iou_mode)
        return 1 - score

    return distance


def hungarian_matching(left_nodes, right_nodes, threshold, dist_fn):
    """
    node shoud be [id, features]
    """

    if len(right_nodes) == 0 or left_nodes is None or len(left_nodes) == 0:
        return []

    G = nx.DiGraph()
    edges = []
    for left_node in left_nodes:
        if isinstance(left_node, Iterable):
            left_id = int(left_node[0])
        else:
            left_id = int(left_node)
        costs = dist_fn(left_node, right_nodes)
        valid_idxs = np.where(costs < threshold)[0]

        for valid_idx in valid_idxs:
            edges.append(
                (
                    f"p{left_id}",
                    f"c{valid_idx}",
                    {"capacity": 1, "weight": int(costs[valid_idx] * 100)},
                )
            )

        edges.append(("s", f"p{left_id}", {"capacity": 1, "weight": 0}))

    for i in range(len(right_nodes)):
        right_node = right_nodes[i]
        if isinstance(right_node, Iterable):
            right_id = right_node[0]
        else:
            right_id = right_node
        if right_id == -1:
            edges.append((f"c{i}", "t", {"capacity": np.inf, "weight": 0}))
        else:
            edges.append((f"c{i}", "t", {"capacity": 1, "weight": 0}))

    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, "s", "t")

    result = []
    for start_point, sflow in mincostFlow["s"].items():
        if sflow == 1:
            for end_point, eflow in mincostFlow[start_point].items():
                if eflow == 1:
                    left_id = int(start_point[1:])
                    right_idx = int(end_point[1:])
                    right_node = right_nodes[right_idx]
                    if isinstance(right_node, Iterable):
                        right_id = right_node[0]
                    else:
                        right_id = right_node
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
    def __init__(self, config):
        self.config = config
        self.overlap_fn = overlap_distance(self.config["MATCHING"]["iou_mode"])

    def initialize(self, n_experts):
        self.last_id = 0
        self.id_table = {i: {} for i in range(n_experts)}

    def get_id(self, expert_id, box_id):
        if box_id not in self.id_table[expert_id].keys():
            self.id_table[expert_id][box_id] = self.last_id
            self.last_id += 1
        return self.id_table[expert_id][box_id]

    def default_match(self, selected_expert, results):
        curr_expert_bboxes = results[selected_expert].copy()
        for i in range(len(curr_expert_bboxes)):
            box_id = curr_expert_bboxes[i, 0]
            curr_expert_bboxes[i, 0] = self.get_id(selected_expert, box_id)
        return curr_expert_bboxes

    def anchor_match(self, prev_selected_expert, selected_expert, results):
        if prev_selected_expert != selected_expert and prev_selected_expert is not None:
            curr_expert_bboxes = results[selected_expert].copy()
            prev_expert_bboxes = results[prev_selected_expert].copy()
            matched_id = hungarian_matching(
                prev_expert_bboxes,
                curr_expert_bboxes,
                self.config["MATCHING"]["threshold"],
                self.overlap_fn,
            )

            # match id
            for prev_id, curr_id in matched_id:
                target_id = self.get_id(prev_selected_expert, prev_id)

                # remove id which is already used
                remove_key = None
                for key, value in self.id_table[selected_expert].items():
                    if value == target_id:
                        remove_key = key
                        break
                if remove_key is not None:
                    self.id_table[selected_expert].pop(remove_key)

                self.id_table[selected_expert][curr_id] = target_id

        # assing id
        curr_expert_bboxes = self.default_match(selected_expert, results)

        return curr_expert_bboxes

    def kmeans_match(self, experts_w, selected_expert, results):
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
            cl[i] += originid2flatid[expert_idx]

            # get the maximum number of elements
            if len(originid2flatid[expert_idx]) > min_k:
                min_k = len(originid2flatid[expert_idx])

            scores = proper_overlap(
                flat_bboxes[i],
                flat_bboxes[i + 1 :],
                self.config["MATCHING"]["iou_mode"],
            )

            # when the score is lower than the threshold, the boxes can not be connected
            for j, score in enumerate(scores):
                if score <= self.config["MATCHING"]["threshold"]:
                    cl[i].append(i + 1 + j)
                    cl[i + 1 + j].append(i)

        # make must link
        ml = [[] for i in range(len(flat_bboxes))]

        # cluster boxes
        for k in range(min_k, len(flat_bboxes) + 1):
            kmeans = COP_KMeans(k, ml, cl, self.overlap_fn)
            fit_result = kmeans.fit(flat_bboxes)
            if fit_result == 1:
                break

        target_ids = set()
        scores = {}
        cluster_idxs = sorted(kmeans.clusters.keys())
        for cluster_idx in cluster_idxs:
            candidates = {-1: 0}
            flat_ids = kmeans.clusters[cluster_idx]
            for flat_id in flat_ids:
                expert_idx, box_id = flatid2originid[flat_id]
                candidate = self.id_table[expert_idx].get(box_id, -1)
                if self.config["MATCHING"]["score_mode"].endswith("vote"):
                    if "m" in self.config["MATCHING"]["score_mode"]:
                        ballot = experts_w[expert_idx]
                    else:
                        ballot = 1

                    if "n" in self.config["MATCHING"]["score_mode"]:
                        if candidate != -1:
                            candidates[candidate] = (
                                candidates.get(candidate, 0) + ballot
                            )
                    else:
                        candidates[candidate] = candidates.get(candidate, 0) + ballot
            target_ids.update(candidates.keys())
            scores[cluster_idx] = candidates
        target_ids = list(target_ids)

        def bullet_dist(left_id, right_ids):
            dists = np.ones((len(right_ids))) * np.inf
            total = sum(scores[left_id].values())
            for i in range(len(right_ids)):
                right_id = right_ids[i]
                if right_id in scores[left_id].keys():
                    if total > 0:
                        score = scores[left_id][right_id] / total
                        dists[i] = 1 - score
                    else:
                        dists[i] = 0
            return dists

        # find best matching
        matched_id = hungarian_matching(cluster_idxs, target_ids, np.inf, bullet_dist)

        # match the cluster id to id pool
        for cluster_idx, target_id in matched_id:
            flat_ids = kmeans.clusters[cluster_idx]

            if target_id == -1:
                cluster_id = self.last_id
                self.last_id += 1
            else:
                cluster_id = target_id

            for flat_id in flat_ids:
                expert_idx, box_id = flatid2originid[flat_id]
                self.id_table[expert_idx][box_id] = cluster_id

        # assign id
        curr_expert_bboxes = self.default_match(selected_expert, results)

        return curr_expert_bboxes
