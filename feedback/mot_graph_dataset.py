import sys
from feedback.seq_process import MOTSeqProcessor

sys.path.append("external/mot_neural_solver/src")
from mot_neural_solver.data.mot_graph import MOTGraph


class MOTGraphDataset:
    def __init__(self, dataset_params, img_paths, det_df, seq_info, cnn_model=None):
        self.dataset_params = dataset_params
        self.cnn_model = cnn_model
        self.seq_info = seq_info

        self.seq = MOTSeqProcessor(img_paths, det_df, seq_info)
        self.seq_det_df = self.seq.load_or_process_detections()
        self.seq_det_dfs = {seq_info["seq_name"]: self.seq_det_df}
        self.seq_info_dicts = {seq_info["seq_name"]: self.seq_det_df.seq_info_dict}
        self.seq_names = [seq_info["seq_name"]]

        # Update each sequence's meatinfo with step sizes
        self._compute_seq_step_sizes()

    def _compute_seq_step_sizes(self):
        for seq_name, seq_info_dict in self.seq_info_dicts.items():
            seq_type = "moving" if seq_info_dict["mov_camera"] else "static"
            target_fps = self.dataset_params["target_fps_dict"][seq_type]
            scene_fps = seq_info_dict["fps"]
            if scene_fps <= target_fps:
                step_size = 1

            else:
                step_size = round(scene_fps / target_fps)

            self.seq_info_dicts[seq_name]["step_size"] = step_size

    def get_from_frame_and_seq(
        self,
        seq_name,
        start_frame,
        max_frame_dist,
        end_frame=None,
        ensure_end_is_in=False,
        return_full_object=False,
        inference_mode=False,
    ):
        """
        Method behind __getitem__ method. We load a graph object of the given sequence name, starting at 'start_frame'.

        Args:
            seq_name: string indicating which scene to get the graph from
            start_frame: int indicating frame num at which the graph should start
            end_frame: int indicating frame num at which the graph should end (optional)
            ensure_end_is_in: bool indicating whether end_frame needs to be in the graph
            return_full_object: bool indicating whether we need the whole MOTGraph object or only its Graph object
                                (Graph Network's input)

        Returns:
            mot_graph: output MOTGraph object or Graph object, depending on whethter return full_object == True or not

        """
        seq_det_df = self.seq_det_dfs[seq_name]
        seq_info_dict = self.seq_info_dicts[seq_name]
        seq_step_size = self.seq_info_dicts[seq_name]["step_size"]

        mot_graph = MOTGraph(
            dataset_params=self.dataset_params,
            seq_info_dict=seq_info_dict,
            seq_det_df=seq_det_df,
            step_size=seq_step_size,
            start_frame=start_frame,
            end_frame=end_frame,
            ensure_end_is_in=ensure_end_is_in,
            max_frame_dist=max_frame_dist,
            cnn_model=self.cnn_model,
            inference_mode=inference_mode,
        )

        # Construct the Graph Network's input
        mot_graph.construct_graph_object()

        if return_full_object:
            return mot_graph

        else:
            return mot_graph.graph_obj

    def __len__(self):
        return 1

    def __getitem__(self, ix):
        seq_name = self.seq_names[ix]
        return self.get_from_frame_and_seq(
            seq_name,
            start_frame=0,
            end_frame=None,
            ensure_end_is_in=False,
            return_full_object=False,
            inference_mode=False,
            max_frame_dist=self.dataset_params["max_frame_dist"],
        )
