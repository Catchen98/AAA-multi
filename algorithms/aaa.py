import sys
import copy

import numpy as np
import pandas as pd
import scipy.special as sc
import motmetrics as mm

from algorithms.anchor_detector import FixedDetector
from algorithms.id_matcher import IDMatcher
from feedback.neural_solver import NeuralSolver
from algorithms.aaa_util import (
    convert_df,
    weighted_random_choice,
    loss_function,
)


class WAADelayed:
    def __init__(self):
        pass

    def initialize(self, n):
        self.w = np.ones(n) / n
        self.est_D = 1
        self.real_D = 0

    """
    gradient_losses should be n
    """

    def update(self, gradient_losses, dt):
        # check the number of element
        assert len(gradient_losses) == len(self.w)

        for i in range(1, dt + 1):
            self.real_D += i
            if self.est_D < self.real_D:
                self.est_D *= 2

        lr = np.sqrt(self.est_D * np.log(len(self.w)))

        changes = lr * gradient_losses
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class AAA:
    def __init__(self, config):
        self.name = f"{config['MATCHING']}, {config['LOSS']}"
        self.n_experts = len(config["EXPERTS"])
        self.config = config

        if self.config["DETECTOR"]["type"] == "fixed":
            self.detector = FixedDetector(self.config["DETECTOR"]["duration"])

        self.learner = WAADelayed()
        self.matcher = IDMatcher(config)

        self.offline = NeuralSolver(
            self.config["FEEDBACK"]["ckpt_path"],
            self.config["FEEDBACK"]["frcnn_weights_path"],
            self.config["FEEDBACK"]["reid_weights_path"],
            self.config["FEEDBACK"]["tracking_cfg_path"],
            self.config["FEEDBACK"]["preprocessing_cfg_path"],
            self.config["FEEDBACK"]["prepr_w_tracktor"],
        )
        self.is_reset_offline = self.config["OFFLINE"]["reset"]

        self.acc = mm.MOTAccumulator(auto_id=True)

    def initialize(self, seq_info):
        self.frame_idx = -1

        self.seq_info = seq_info
        self.detector.initialize(seq_info)
        self.learner.initialize(self.n_experts)
        self.matcher.initialize(self.n_experts)
        self.prev_expert = None
        self.prev_bboxes = None

        self.reset_offline()
        self.reset_history()

    def reset_offline(self):
        self.img_paths = []
        self.dets = []

    def reset_history(self):
        self.timer = -1
        self.experts_results = [[] for _ in range(self.n_experts)]

    def track(self, img_path, dets, results):
        self.frame_idx += 1
        self.timer += 1

        self.img_paths.append(img_path)
        self.dets.append(dets)

        # save experts' result
        for i, result in enumerate(results):
            if len(result) > 0:
                frame_result = np.zeros((result.shape[0], result.shape[1] + 1))
                frame_result[:, 1:] = result
                frame_result[:, 0] = self.timer + 1
                if len(self.experts_results[i]) > 0:
                    self.experts_results[i] = np.concatenate(
                        [self.experts_results[i], frame_result], axis=0
                    )
                else:
                    self.experts_results[i] = frame_result

        # detect anchor frame
        if self.detector.detect(img_path, dets, results):

            # try to receive feedback
            try:
                feedback = self.offline.track(self.seq_info, self.img_paths, self.dets)
            except Exception:
                feedback = None

            # update weight
            if feedback is not None:
                df_feedback = convert_df(feedback, is_offline=True)

                smallest_frame = len(self.dets) - self.timer - 1
                df_cond = df_feedback.index.get_level_values(0) > smallest_frame
                df_feedback = df_feedback[df_cond]

                df_feedback.index = pd.MultiIndex.from_tuples(
                    [(x[0] - smallest_frame, x[1]) for x in df_feedback.index]
                )

                # calculate loss
                gradient_losses = np.zeros((self.n_experts))
                for i, expert_results in enumerate(self.experts_results):
                    df_expert_results = convert_df(expert_results)

                    if (
                        self.seq_info["dataset_name"] == "MOT16"
                        or self.seq_info["dataset_name"] == "MOT17"
                    ):
                        acc, ana = mm.utils.CLEAR_MOT_M(
                            df_feedback,
                            df_expert_results,
                            self.seq_info["ini_path"],
                            "iou",
                            distth=0.5,
                            vflag="",
                        )
                    else:
                        acc = mm.utils.compare_to_groundtruth(
                            df_feedback, df_expert_results, "iou", distth=0.5
                        )
                        ana = None
                    mh = mm.metrics.create()
                    loss = loss_function(self.config["LOSS"]["type"], mh, acc, ana)
                    gradient_losses[i] = loss
                self.learner.update(gradient_losses, self.timer + 1)

                self.reset_history()
                if self.is_reset_offline:
                    self.reset_offline()

            else:
                gradient_losses = None

        else:
            feedback = None
            gradient_losses = None

        # select expert
        selected_expert = weighted_random_choice(self.learner.w)
        curr_expert_bboxes = results[selected_expert]

        # match id
        if self.config["MATCHING"]["method"] == "previous":
            curr_expert_bboxes = self.matcher.previous_match(
                self.prev_bboxes, selected_expert, results
            )
        elif self.config["MATCHING"]["method"] == "kmeans":
            curr_expert_bboxes = self.matcher.kmeans_match(
                self.learner.w, selected_expert, results
            )
        else:
            raise NameError("Please enter a valid matching method")

        self.prev_expert = selected_expert
        self.prev_bboxes = copy.deepcopy(curr_expert_bboxes)

        return (
            curr_expert_bboxes,
            self.learner.w,
            gradient_losses,
            feedback,
            selected_expert,
        )
