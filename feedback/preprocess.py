import sys
from PIL import Image
import pandas as pd
import numpy as np
from skimage.io import imread
from numpy import pad

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

sys.path.append("external/mot_neural_solver/src")
from mot_neural_solver.data.preprocessing import FRCNNPreprocessor
from mot_neural_solver.utils.graph import (
    get_time_valid_conn_ixs,
    get_knn_mask,
    compute_edge_feats_dict,
)
from mot_neural_solver.data.mot_graph import Graph

sys.path.append("external/mot_neural_solver/tracking_wo_bnw/src")
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.tracker import Tracker


DET_COL_NAMES = ("frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf")


class PreProcess:
    def __init__(
        self,
        dataset_params,
        cnn_model,
        frcnn_weights_path,
        prepr_w_tracktor,
        frcnn_prepr_params,
        tracktor_params,
    ):
        self.dataset_params = dataset_params
        self.cnn_model = cnn_model
        self.prepr_w_tracktor = prepr_w_tracktor

        if prepr_w_tracktor:
            prepr_params = tracktor_params

        else:
            prepr_params = frcnn_prepr_params

        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(
            torch.load(frcnn_weights_path, map_location=lambda storage, loc: storage,)
        )
        obj_detect.eval()
        obj_detect.cuda()

        if prepr_w_tracktor:
            self.preprocessor = Tracker(obj_detect, None, prepr_params["tracker"])
        else:
            self.preprocessor = FRCNNPreprocessor(obj_detect, prepr_params)

        self.transforms = ToTensor()

    def process(self, img_paths, dets, fps, seq_type):
        det_df, frame_height, frame_width = self.get_det_df(img_paths, dets)
        node_embeds, reid_embeds = self.store_embeddings(
            det_df, frame_height, frame_width
        )
        target_fps = self.dataset_params["target_fps_dict"][seq_type]
        if fps <= target_fps:
            step_size = 1

        else:
            step_size = round(fps / target_fps)

        mot_graph = MOTGraph(
            det_df,
            fps=fps,
            start_frame=0,
            end_frame=len(img_paths) - 1,
            step_size=step_size,
            cnn_model=self.cnn_model,
            ensure_end_is_in=True,
            dataset_params=self.dataset_params,
            max_frame_dist=self.dataset_params["max_frame_dist"],
            inference_mode=True,
        )
        mot_graph.construct_graph_object(reid_embeds, node_embeds)

        return mot_graph, det_df

    def store_embeddings(self, det_df, frame_height, frame_width):
        bbox_dataset = BoundingBoxDataset(
            det_df, frame_height, frame_width, return_det_ids_and_frame=True,
        )
        bbox_loader = DataLoader(
            bbox_dataset,
            batch_size=self.dataset_params["img_batch_size"] // 2,
            pin_memory=True,
            num_workers=4,
        )

        # Feed all bboxes to the CNN to obtain node and reid embeddings
        self.cnn_model.eval()
        node_embeds, reid_embeds = [], []
        frame_nums, det_ids = [], []
        with torch.no_grad():
            for frame_num, det_id, bboxes in bbox_loader:
                node_out, reid_out = self.cnn_model(bboxes.cuda())
                node_embeds.append(node_out.cpu())
                reid_embeds.append(reid_out.cpu())
                frame_nums.append(frame_num)
                det_ids.append(det_id)

        det_ids = torch.cat(det_ids, dim=0)
        frame_nums = torch.cat(frame_nums, dim=0)

        node_embeds = torch.cat(node_embeds, dim=0)
        reid_embeds = torch.cat(reid_embeds, dim=0)

        # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
        node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
        reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

        return node_embeds, reid_embeds

    def preprocess_det(self, img_paths, dets):

        self.preprocessor.reset()

        for img_path, det in zip(img_paths, dets):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img = self.transforms(img)

            sample = {}
            sample["img"] = img.unsqueeze(0)
            bb = np.zeros((len(det), 5), dtype=np.float32)
            bb[:, 0:2] = det[:, 2:4] - 1
            bb[:, 2:4] = det[:, 2:4] + det[:, 4:6] - 1
            sample["dets"] = torch.FloatTensor([d[:4] for d in bb]).unsqueeze(0)
            sample["img_path"] = img_path

            with torch.no_grad():
                self.preprocessor.step(sample)

        if self.prepr_w_tracktor:
            all_tracks = self.preprocessor.get_results()
            rows = []
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    rows.append(
                        [
                            frame + 1,
                            i + 1,
                            x1 + 1,
                            y1 + 1,
                            x2 - x1 + 1,
                            y2 - y1 + 1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ]
                    )
            det_df = pd.DataFrame(np.array(rows))
        else:
            final_results = pd.concat(self.preprocessor.results_dfs)
            final_results["bb_left"] += 1  # MOT bbox annotations are 1 -based
            final_results["bb_top"] += 1  # MOT bbox annotations are 1 -based
            final_results["id"] = -1
            det_df = final_results[
                ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
            ]

        return det_df, h, w

    def ensure_boxes_in_frame(self, det_df, frame_height, frame_width):
        """
        Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside
        the frame hence we crop its detections to also be inside it)
        """
        initial_bb_top = det_df["bb_top"].values.copy()
        initial_bb_left = det_df["bb_left"].values.copy()

        det_df["bb_top"] = np.maximum(det_df["bb_top"].values, 0).astype(int)
        det_df["bb_left"] = np.maximum(det_df["bb_left"].values, 0).astype(int)

        bb_top_diff = det_df["bb_top"].values - initial_bb_top
        bb_left_diff = det_df["bb_left"].values - initial_bb_left

        det_df["bb_height"] -= bb_top_diff
        det_df["bb_width"] -= bb_left_diff

        img_height, img_width = (
            frame_height,
            frame_width,
        )
        det_df["bb_height"] = np.minimum(
            img_height - det_df["bb_top"], det_df["bb_height"]
        ).astype(int)
        det_df["bb_width"] = np.minimum(
            img_width - det_df["bb_left"], det_df["bb_width"]
        ).astype(int)

        return det_df

    def get_det_df(self, img_paths, dets):
        det_df, frame_height, frame_width = self.preprocess_det(img_paths, dets)

        # Number and order of columns is always assumed to be the same
        det_df = det_df[det_df.columns[: len(DET_COL_NAMES)]]
        det_df.columns = DET_COL_NAMES

        det_df["bb_left"] -= 1  # Coordinates are 1 based
        det_df["bb_top"] -= 1

        # If id already contains an ID assignment (e.g. using tracktor output), keep it
        if len(det_df["id"].unique()) > 1:
            det_df["tracktor_id"] = det_df["id"]

        det_df["frame_path"] = det_df.apply(
            lambda row: img_paths[int(row.frame) - 1], axis=1
        )

        det_df = self.ensure_boxes_in_frame(det_df, frame_height, frame_width)

        # Add some additional box measurements that might be used for graph construction
        det_df["bb_bot"] = (det_df["bb_top"] + det_df["bb_height"]).values
        det_df["bb_right"] = (det_df["bb_left"] + det_df["bb_width"]).values
        det_df["feet_x"] = det_df["bb_left"] + 0.5 * det_df["bb_width"]
        det_df["feet_y"] = det_df["bb_top"] + det_df["bb_height"]

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        conds = (det_df["bb_width"] > 0) & (det_df["bb_height"] > 0)
        conds = conds & (det_df["bb_right"] > 0) & (det_df["bb_bot"] > 0)
        conds = (
            conds
            & (det_df["bb_left"] < frame_width)
            & (det_df["bb_top"] < frame_height)
        )
        det_df = det_df[conds].copy()

        det_df.sort_values(by="frame", inplace=True)
        det_df["detection_id"] = np.arange(
            det_df.shape[0]
        )  # This id is used for future tastks

        return det_df, frame_height, frame_width


class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """

    def __init__(
        self,
        det_df,
        frame_height,
        frame_width,
        pad_=True,
        pad_mode="mean",
        output_size=(128, 64),
        return_det_ids_and_frame=False,
    ):
        self.det_df = det_df
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.pad = pad_
        self.pad_mode = pad_mode
        self.transforms = Compose(
            (
                Resize(output_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        )

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_img_path = None

        self.return_det_ids_and_frame = return_det_ids_and_frame

    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if row["frame_path"] != self.curr_img_path:
            self.curr_img = imread(row["frame_path"])
            self.curr_img_path = row["frame_path"]

        frame_img = self.curr_img

        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[
            int(max(0, row["bb_top"])) : int(max(0, row["bb_bot"])),
            int(max(0, row["bb_left"])) : int(max(0, row["bb_right"])),
        ]
        if self.pad:
            x_height_pad = np.abs(row["bb_top"] - max(row["bb_top"], 0)).astype(int)
            y_height_pad = np.abs(
                row["bb_bot"] - min(row["bb_bot"], self.frame_height)
            ).astype(int)

            x_width_pad = np.abs(row["bb_left"] - max(row["bb_left"], 0)).astype(int)
            y_width_pad = np.abs(
                row["bb_right"] - min(row["bb_right"], self.frame_width)
            ).astype(int)

            bb_img = pad(
                bb_img,
                ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)),
                mode=self.pad_mode,
            )

        bb_img = Image.fromarray(bb_img)
        if self.transforms is not None:
            bb_img = self.transforms(bb_img)

        if self.return_det_ids_and_frame:
            return row["frame"], row["detection_id"], bb_img
        else:
            return bb_img


class MOTGraph(object):
    """
    This the main class we use to create MOT graphs from detection (and possibly ground truth) files. Its main attribute
    is 'graph_obj', which is an instance of the class 'Graph' and serves as input to the tracking model.
    Moreover, each 'MOTGraph' has several additional attributes that provide further information about the detections in
    the subset of frames from which the graph is constructed.
    """

    def __init__(
        self,
        seq_det_df=None,
        fps=None,
        start_frame=None,
        end_frame=None,
        ensure_end_is_in=False,
        step_size=None,
        dataset_params=None,
        inference_mode=False,
        cnn_model=None,
        max_frame_dist=None,
    ):
        self.dataset_params = dataset_params
        self.step_size = step_size
        self.fps = fps
        self.inference_mode = inference_mode
        self.max_frame_dist = max_frame_dist

        self.cnn_model = cnn_model

        if seq_det_df is not None:
            self.graph_df, self.frames = self._construct_graph_df(
                seq_det_df=seq_det_df.copy(),
                start_frame=start_frame,
                end_frame=end_frame,
                ensure_end_is_in=ensure_end_is_in,
            )

    def _construct_graph_df(
        self, seq_det_df, start_frame, end_frame=None, ensure_end_is_in=False
    ):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.
        Args:
            seq_det_df: DataFrame with scene detections information
            start_frame: frame at which the graph starts
            end_frame: (optional) frame at which the graph ends
            ensure_end_is_in: (only if end_frame is given). Bool indicating whether end_frame must be in the graph.
        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
            valid_frames: list of selected frames
        """
        if end_frame is not None:
            # Just load all frames between start_frame and end_frame at the desired step size
            valid_frames = np.arange(start_frame, end_frame + 1, self.step_size)

            if ensure_end_is_in and (end_frame not in valid_frames):
                valid_frames = valid_frames.tolist() + [end_frame]

        else:
            # Consider all posible future frames (at distance step_size)
            valid_frames = np.arange(
                start_frame, seq_det_df.frame.max(), self.step_size
            )

            # We cannot have more than dataset_params['frames_per_graph'] frames
            if self.dataset_params["frames_per_graph"] != "max":
                valid_frames = valid_frames[: self.dataset_params["frames_per_graph"]]

            # We cannot have more than dataset_params['max_detects'] detections
            if self.dataset_params["max_detects"] is not None:
                scene_df_ = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
                frames_cumsum = scene_df_.groupby("frame")["bb_left"].count().cumsum()
                valid_frames = frames_cumsum[
                    frames_cumsum <= self.dataset_params["max_detects"]
                ].index

        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
        graph_df = graph_df.sort_values(by=["frame", "detection_id"]).reset_index(
            drop=True
        )

        return graph_df, sorted(graph_df.frame.unique())

    def _get_edge_ixs(self, reid_embeddings):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            reid_embeddings: torch.tensor with shape (num_nodes, reid_embeds_dim)
        Returns:
            torch.tensor withs shape (2, num_edges)
        """

        edge_ixs = get_time_valid_conn_ixs(
            frame_num=torch.from_numpy(self.graph_df.frame.values),
            max_frame_dist=self.max_frame_dist,
            use_cuda=self.inference_mode
            and self.graph_df["frame_path"].iloc[0].find("MOT17-03") == -1,
        )

        # During inference, top k nns must not be done here, as it is computed independently for sequence chunks
        if not self.inference_mode and self.dataset_params["top_k_nns"] is not None:
            reid_pwise_dist = F.pairwise_distance(
                reid_embeddings[edge_ixs[0]], reid_embeddings[edge_ixs[1]]
            )
            k_nns_mask = get_knn_mask(
                pwise_dist=reid_pwise_dist,
                edge_ixs=edge_ixs,
                num_nodes=self.graph_df.shape[0],
                top_k_nns=self.dataset_params["top_k_nns"],
                reciprocal_k_nns=self.dataset_params["reciprocal_k_nns"],
                symmetric_edges=False,
                use_cuda=self.inference_mode,
            )
            edge_ixs = edge_ixs.T[k_nns_mask].T

        return edge_ixs

    def construct_graph_object(self, reid_embeddings, node_feats):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj,
        """
        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs = self._get_edge_ixs(reid_embeddings)
        edge_feats_dict = compute_edge_feats_dict(
            edge_ixs=edge_ixs,
            det_df=self.graph_df,
            fps=self.fps,
            use_cuda=self.inference_mode,
        )
        edge_feats = [
            edge_feats_dict[feat_names]
            for feat_names in self.dataset_params["edge_feats_to_use"]
            if feat_names in edge_feats_dict
        ]
        edge_feats = torch.stack(edge_feats).T

        # Compute embeddings distances. Pairwise distance computation might create out of memmory errors, hence we batch it
        emb_dists = []
        for i in range(0, edge_ixs[0].shape[0], 50000):
            emb_dists.append(
                F.pairwise_distance(
                    reid_embeddings[edge_ixs[0][i : i + 50000]],
                    reid_embeddings[edge_ixs[1][i : i + 50000]],
                ).view(-1, 1)
            )
        emb_dists = torch.cat(emb_dists, dim=0).to(edge_feats.device)

        # Add embedding distances to edge features if needed
        if "emb_dist" in self.dataset_params["edge_feats_to_use"]:
            edge_feats = torch.cat((edge_feats, emb_dists), dim=1)

        self.graph_obj = Graph(
            x=node_feats,
            edge_attr=torch.cat((edge_feats, edge_feats), dim=0),
            edge_index=torch.cat(
                (edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1
            ),
        )

        if self.inference_mode:
            self.graph_obj.reid_emb_dists = torch.cat((emb_dists, emb_dists))

        self.graph_obj.to(
            torch.device(
                "cuda" if torch.cuda.is_available() and self.inference_mode else "cpu"
            )
        )
