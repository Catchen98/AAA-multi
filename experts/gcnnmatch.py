import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard as tb
from torchvision.transforms import ToTensor
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataListLoader
from torch_geometric.nn import DataParallel

from PIL import Image
import yaml

from experts.expert import Expert

sys.path.append("external/GCNNMatch")
from utils import hungarian
from network.complete_net import completeNet

from tracking_wo_bnw.src.tracktor.frcnn_fpn import FRCNN_FPN
from tracking_wo_bnw.src.tracktor.oracle_tracker import OracleTracker
from tracking_wo_bnw.src.tracktor.reid.resnet import resnet50
from tracking_wo_bnw.src.tracktor.tracker import Tracker


def build_graph(
    tracklets,
    current_detections,
    images_path,
    current_frame,
    distance_limit,
    fps,
    test=True,
):

    if len(tracklets):
        edges_first_row = []
        edges_second_row = []
        edges_complete_first_row = []
        edges_complete_second_row = []
        edge_attr = []
        ground_truth = []
        idx = []
        node_attr = []
        coords = []
        frame = []
        coords_original = []
        transform = ToTensor()
        # tracklet graphs
        for tracklet in tracklets:
            tracklet1 = tracklet[-1]
            xmin, ymin, width, height = (
                int(round(tracklet1[2])),
                int(round(tracklet1[3])),
                int(round(tracklet1[4])),
                int(round(tracklet1[5])),
            )
            image_name = images_path[int(tracklet1[0]) - 1]
            image = plt.imread(image_name)
            frame_width, frame_height, channels = image.shape
            coords.append(
                [
                    xmin / frame_width,
                    ymin / frame_height,
                    width / frame_width,
                    height / frame_height,
                ]
            )
            coords_original.append([xmin, ymin, xmin + width / 2, ymin + height / 2])
            image_cropped = image[ymin : ymin + height, xmin : xmin + width]
            image_resized = cv2.resize(
                image_cropped, (90, 150), interpolation=cv2.INTER_AREA
            )
            image_resized = image_resized / 255
            image_resized = image_resized.astype(np.float32)
            image_resized -= [0.485, 0.456, 0.406]
            image_resized /= [0.229, 0.224, 0.225]
            image_resized = transform(image_resized)
            node_attr.append(image_resized)
            frame.append([tracklet1[0] / fps])  # the frame it is observed
        # new detections graph
        for detection in current_detections:
            xmin, ymin, width, height = (
                int(round(detection[2])),
                int(round(detection[3])),
                int(round(detection[4])),
                int(round(detection[5])),
            )
            image_name = images_path[int(detection[0]) - 1]
            image = plt.imread(image_name)
            frame_width, frame_height, channels = image.shape
            coords.append(
                [
                    xmin / frame_width,
                    ymin / frame_height,
                    width / frame_width,
                    height / frame_height,
                ]
            )
            coords_original.append([xmin, ymin, xmin + width / 2, ymin + height / 2])
            image_cropped = image[ymin : ymin + height, xmin : xmin + width]
            image_resized = cv2.resize(
                image_cropped, (90, 150), interpolation=cv2.INTER_AREA
            )
            image_resized = image_resized / 255
            image_resized = image_resized.astype(np.float32)
            image_resized -= [0.485, 0.456, 0.406]
            image_resized /= [0.229, 0.224, 0.225]
            image_resized = transform(image_resized)
            node_attr.append(image_resized)
            frame.append([detection[0] / fps])  # the frame it is observed
        # construct connections between tracklets and detections
        k = 0
        for i in range(len(tracklets) + len(current_detections)):
            for j in range(len(tracklets) + len(current_detections)):
                distance = (
                    (coords_original[i][0] - coords_original[j][0]) ** 2
                    + (coords_original[i][1] - coords_original[j][1]) ** 2
                ) ** 0.5
                if i < len(tracklets) and j >= len(
                    tracklets
                ):  # i is tracklet j is detection
                    # adjacency matrix
                    if distance < distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])
                    if test:
                        edges_complete_first_row.append(i)
                        edges_complete_second_row.append(j)
                    if tracklets[i][-1][1] == current_detections[j - len(tracklets)][1]:
                        ground_truth.append(1.0)
                    else:
                        ground_truth.append(0.0)
                    k += 1
                elif i >= len(tracklets) and j < len(
                    tracklets
                ):  # j is tracklet i is detection
                    # adjacency matrix
                    if distance < distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])
                    k += 1
        idx.append(current_frame - 2)
        frame_node_attr = torch.stack(node_attr)
        frame_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        frame_edges_index = torch.tensor(
            [edges_first_row, edges_second_row], dtype=torch.long
        )
        frame_coords = torch.tensor(coords, dtype=torch.float)
        frame_ground_truth = torch.tensor(ground_truth, dtype=torch.float)
        frame_idx = torch.tensor(idx, dtype=torch.float)
        frame_edges_number = torch.tensor(
            len(edges_first_row), dtype=torch.int
        ).reshape(1)
        frame_frame = torch.tensor(frame, dtype=torch.float)
        tracklets_frame = torch.tensor(len(tracklets), dtype=torch.float).reshape(1)
        detections_frame = torch.tensor(
            len(current_detections), dtype=torch.float
        ).reshape(1)
        coords_original = torch.tensor(coords_original, dtype=torch.float)
        edges_complete = torch.tensor(
            [edges_complete_first_row, edges_complete_second_row], dtype=torch.long
        )
        data = Data(
            x=frame_node_attr,
            edge_index=frame_edges_index,
            edge_attr=frame_edge_attr,
            coords=frame_coords,
            coords_original=coords_original,
            ground_truth=frame_ground_truth,
            idx=frame_idx,
            edges_number=frame_edges_number,
            frame=frame_frame,
            det_num=detections_frame,
            track_num=tracklets_frame,
            edges_complete=edges_complete,
        )
        return data


class Tracktor(Expert):
    def __init__(
        self,
        reid_network_weights_path,
        obj_detect_model_path,
        tracktor_config_path,
        reid_config_path,
    ):
        super(Tracktor, self).__init__("Tracktor")

        with open(tracktor_config_path) as config_file:
            tracktor = yaml.unsafe_load(config_file)["tracktor"]

        with open(reid_config_path) as config_file:
            reid = yaml.unsafe_load(config_file)["reid"]

        # set all seeds
        torch.manual_seed(tracktor["seed"])
        torch.cuda.manual_seed(tracktor["seed"])
        np.random.seed(tracktor["seed"])
        torch.backends.cudnn.deterministic = True

        ##########################
        # Initialize the modules #
        ##########################

        # object detection
        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(
            torch.load(
                obj_detect_model_path, map_location=lambda storage, loc: storage,
            )
        )

        obj_detect.eval()
        obj_detect.cuda()

        # reid
        reid_network = resnet50(pretrained=False, **reid["cnn"])
        reid_network.load_state_dict(
            torch.load(
                reid_network_weights_path, map_location=lambda storage, loc: storage
            )
        )
        reid_network.eval()
        reid_network.cuda()

        # tracktor
        if "oracle" in tracktor:
            self.tracker = OracleTracker(
                obj_detect, reid_network, tracktor["tracker"], tracktor["oracle"]
            )
        else:
            self.tracker = Tracker(obj_detect, reid_network, tracktor["tracker"])

        self.transforms = ToTensor()

    def initialize(self, seq_info):
        super(Tracktor, self).initialize(seq_info)
        self.tracker.reset()

    def track(self, img_path, dets):
        super(Tracktor, self).track(img_path, dets)

        frame = self.preprocess(img_path, dets)
        with torch.no_grad():
            self.tracker.step(frame)

        results = []
        for i, track in self.tracker.get_results().items():
            if self.frame_idx in track.keys():
                bb = track[self.frame_idx]
                x1 = bb[0]
                y1 = bb[1]
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                score = bb[4]
                results.append([i, x1 + 1, y1 + 1, w + 1, h + 1, score])
        return results

    def preprocess(self, img_path, dets):
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample["img"] = img.unsqueeze(0)
        if dets is not None:
            bb = np.zeros((len(dets), 5), dtype=np.float32)
            bb[:, 0:2] = dets[:, 2:4] - 1
            bb[:, 2:4] = dets[:, 2:4] + dets[:, 4:6] - 1
            bb[:, 4] = dets[:, 6]
            sample["dets"] = torch.FloatTensor([det[:4] for det in bb]).unsqueeze(0)
        else:
            sample["dets"] = torch.FloatTensor([]).unsqueeze(0)
        sample["img_path"] = img_path
        return sample


class GCNNMatch(Expert):
    def __init__(self, config):
        super(GCNNMatch, self).__init__("GCNNMatch")

        # load model
        self.model = completeNet()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.model = DataParallel(self.model)
        self.model.load_state_dict(torch.load(config["model_path"])["model_state_dict"])
        self.model.eval()

        self.frames_look_back = config["frames_look_back"]
        self.match_thres = config["match_thres"]
        self.det_conf_thres = config["det_conf_thres"]
        self.distance_limit = config["distance_limit"]
        self.min_height = config["min_height"]
        self.fp_look_back = config["fp_look_back"]
        self.fp_recent_frame_limit = config["fp_recent_frame_limit"]
        self.fp_min_times_seen = config["fp_min_times_seen"]

        self.tracktor = Tracktor(
            config["TRACKTOR"]["reid_network_weights_path"],
            config["TRACKTOR"]["obj_detect_model_path"],
            config["TRACKTOR"]["tracktor_config_path"],
            config["TRACKTOR"]["reid_config_path"],
        )

    def initialize(self, seq_info):
        super(GCNNMatch, self).initialize(seq_info)

        self.device = torch.device("cuda")
        # pick one frame and load previous results
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        self.id_num = 0
        self.tracking_output = []
        self.checked_ids = []
        self.images_path = []

        self.fps = int(seq_info["fps"])

        self.transform = ToTensor()

        self.tracktor.initialize(seq_info)

    def track(self, img_path, dets):
        super(GCNNMatch, self).track(img_path, dets)

        self.images_path.append(img_path)

        dets = self.preprocess(img_path, dets)

        data_list = []
        # Give IDs to the first frame
        tracklets = []
        if not self.tracking_output:
            for i, detection in enumerate(dets):
                if detection[0] == 1:
                    frame = detection[0]
                    xmin, ymin, width, height = (
                        int(round(detection[2])),
                        int(round(detection[3])),
                        int(round(detection[4])),
                        int(round(detection[5])),
                    )
                    confidence = detection[6]
                    if (
                        xmin > 0
                        and ymin > 0
                        and width > 0
                        and height > self.min_height
                        and confidence > self.det_conf_thres
                    ):
                        self.id_num += 1
                        ID = int(self.id_num)
                        self.tracking_output.append(
                            [
                                frame,
                                ID,
                                xmin,
                                ymin,
                                width,
                                height,
                                int(detection[6]),
                                1,
                                1,
                            ]
                        )
                        tracklets.append(
                            [
                                [
                                    frame,
                                    ID,
                                    xmin,
                                    ymin,
                                    width,
                                    height,
                                    int(detection[6]),
                                    1,
                                    1,
                                ]
                            ]
                        )
        else:
            # Get all tracklets
            tracklet_IDs = []
            for j, tracklet in enumerate(self.tracking_output):
                xmin, ymin, width, height = (
                    int(round(tracklet[2])),
                    int(round(tracklet[3])),
                    int(round(tracklet[4])),
                    int(round(tracklet[5])),
                )
                if xmin > 0 and ymin > 0 and width > 0 and height > 0:
                    if (tracklet[0] < self.frame_idx + 1) and tracklet[
                        0
                    ] >= self.frame_idx + 1 - self.frames_look_back:
                        new_tracklet = True
                        for k, i in enumerate(tracklet_IDs):
                            if tracklet[1] == i:
                                new_tracklet = False
                                tracklets[k].append(tracklet)
                                break
                        if new_tracklet:
                            tracklet_IDs.append(int(tracklet[1]))
                            tracklets.append([tracklet])
        # Get new detections
        current_detections = []
        for i, detection in enumerate(dets):
            if detection[0] != 1 and detection[0] == self.frame_idx + 1:
                frame = detection[0]
                xmin, ymin, width, height = (
                    int(round(detection[2])),
                    int(round(detection[3])),
                    int(round(detection[4])),
                    int(round(detection[5])),
                )
                confidence = detection[6]
                if (
                    xmin > 0
                    and ymin > 0
                    and width > 0
                    and height > self.min_height
                    and confidence > self.det_conf_thres
                ):
                    current_detections.append(
                        [frame, -1, xmin, ymin, width, height, int(detection[6]), 1, 1]
                    )
        # build graph and run model
        data = build_graph(
            tracklets,
            current_detections,
            self.images_path,
            self.frame_idx + 1,
            self.distance_limit,
            self.fps,
            test=True,
        )
        if data:
            if current_detections and data.edge_attr.size()[0] != 0:
                data_list.append(data)

                loader = DataListLoader(data_list)
                for graph_num, batch in enumerate(loader):
                    # MODEL FORWARD
                    (
                        output,
                        output2,
                        ground_truth,
                        ground_truth2,
                        det_num,
                        tracklet_num,
                    ) = self.model(batch)
                    # FEATURE MAPS on tensorboard
                    # embedding
                    images = batch[0].x
                    images = F.interpolate(images, size=250)
                    edge_index = data_list[graph_num].edges_complete
                    # THRESHOLDS
                    temp = []
                    for i in output2:
                        if i > self.match_thres:
                            temp.append(i)
                        else:
                            temp.append(i - i)
                    output2 = torch.stack(temp)
                    # HUNGARIAN
                    cleaned_output = hungarian(
                        output2, ground_truth2, det_num, tracklet_num
                    )
                    # Give Ids to current frame
                    for i, detection in enumerate(current_detections):
                        match_found = False
                        for k, m in enumerate(cleaned_output):  # cleaned_output):
                            if m == 1 and edge_index[1, k] == i + len(
                                tracklets
                            ):  # match found
                                ID = tracklets[edge_index[0, k]][-1][1]
                                frame = detection[0]
                                xmin, ymin, width, height = (
                                    int(round(detection[2])),
                                    int(round(detection[3])),
                                    int(round(detection[4])),
                                    int(round(detection[5])),
                                )
                                self.tracking_output.append(
                                    [
                                        frame,
                                        ID,
                                        xmin,
                                        ymin,
                                        width,
                                        height,
                                        int(detection[6]),
                                        1,
                                        1,
                                    ]
                                )
                                match_found = True
                                break
                        if not match_found:  # give new ID
                            # print("no match")
                            self.id_num += 1
                            ID = self.id_num
                            frame = detection[0]
                            xmin, ymin, width, height = (
                                int(round(detection[2])),
                                int(round(detection[3])),
                                int(round(detection[4])),
                                int(round(detection[5])),
                            )
                            self.tracking_output.append(
                                [
                                    frame,
                                    ID,
                                    xmin,
                                    ymin,
                                    width,
                                    height,
                                    int(detection[6]),
                                    1,
                                    1,
                                ]
                            )
                    # Clean output for false positives
                    if self.frame_idx + 1 >= self.fp_look_back:
                        # reduce to recent objects
                        recent_tracks = [
                            i
                            for i in self.tracking_output
                            if i[0] >= self.frame_idx + 1 - self.fp_look_back
                        ]
                        # find the different IDs
                        candidate_ids = []
                        times_seen = []
                        first_frame_seen = []
                        for i in recent_tracks:
                            if i[1] not in self.checked_ids:
                                if i[1] not in candidate_ids:
                                    candidate_ids.append(i[1])
                                    times_seen.append(1)
                                    first_frame_seen.append(i[0])
                                else:
                                    index = candidate_ids.index(i[1])
                                    times_seen[index] = times_seen[index] + 1
                        # find which IDs to remove
                        remove_ids = []
                        for i, j in enumerate(candidate_ids):
                            if (
                                times_seen[i] < self.fp_min_times_seen
                                and self.frame_idx + 1 - first_frame_seen[i]
                                >= self.fp_look_back
                            ):
                                remove_ids.append(j)
                            elif times_seen[i] > self.fp_min_times_seen:
                                self.checked_ids.append(j)
                        # keep only those IDs that are seen enough times
                        self.tracking_output = [
                            j for j in self.tracking_output if j[1] not in remove_ids
                        ]

        result = []
        for t in self.tracking_output:
            frame = t[0]
            if frame == self.frame_idx + 1:
                result.append([t[1], t[2], t[3], t[4], t[5]])

        return result

    def preprocess(self, img_path, dets):
        filtered_dets = self.tracktor.track(img_path, dets)

        detections = []
        for det in filtered_dets:
            detections.append(
                [self.frame_idx + 1, det[0], det[1], det[2], det[3], det[4], det[5]]
            )
        return detections
