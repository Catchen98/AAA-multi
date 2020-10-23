import sys

import numpy as np

import scipy.special as sc

from algorithms.anchor_detector import AnchorDetector
from algorithms.id_matcher import IDMatcher
from feedback.neural_solver import NeuralSolver
from algorithms.aaa_util import (
    convert_df,
    weighted_random_choice,
    eval_results,
    frame_loss,
)


class WAADelayed:
    def __init__(self):
        pass

    def initialize(self, n):
        self.w = np.ones(n) / n
        self.est_D = 1

    """
    gradient_losses should be n
    """

    def update(self, gradient_losses, delays):
        # check the number of element
        assert len(gradient_losses) == len(self.w)

        total_delayed = sum(delays)
        while self.est_D < total_delayed:
            self.est_D *= 2

        lr = np.sqrt(self.est_D * np.log(len(self.w)))

        changes = lr * gradient_losses
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class AAA:
    def __init__(self, config):
        self.name = f"{config['OFFLINE']}, {config['MATCHING']}, {config['DETECTOR']}, {config['LOSS']}"
        self.n_experts = len(config["EXPERTS"])
        self.config = config

        self.offline = NeuralSolver(
            self.config["FEEDBACK"]["ckpt_path"],
            self.config["FEEDBACK"]["frcnn_weights_path"],
            self.config["FEEDBACK"]["reid_weights_path"],
            self.config["FEEDBACK"]["tracking_cfg_path"],
            self.config["FEEDBACK"]["preprocessing_cfg_path"],
            self.config["OFFLINE"]["use_gt"],
            self.config["OFFLINE"]["pre_cnn"],
            self.config["OFFLINE"]["pre_track"],
        )

        self.learner = WAADelayed()
        self.detector = AnchorDetector(self.offline)
        self.matcher = IDMatcher(config)

    def initialize(self, seq_info):
        self.frame_idx = -1

        self.seq_info = seq_info
        self.detector.initialize(seq_info)
        self.learner.initialize(self.n_experts)
        self.matcher.initialize(self.n_experts)
        self.offline.initialize(seq_info)

        self.experts_results = [[] for _ in range(self.n_experts)]
        self.delay = []
        self.evaluated = []

    def track(self, img_path, dets, gts, results):
        self.frame_idx += 1

        # save experts' result
        for i, result in enumerate(results):
            if len(result) > 0:
                frame_result = np.zeros((result.shape[0], result.shape[1] + 1))
                frame_result[:, 1:] = result.copy()
                frame_result[:, 0] = self.frame_idx + 1
                if len(self.experts_results[i]) > 0:
                    self.experts_results[i] = np.concatenate(
                        [self.experts_results[i], frame_result], axis=0
                    )
                else:
                    self.experts_results[i] = frame_result

        if self.config["LOSS"]["delayed"]:
            self.delay = [
                self.delay[i] if self.evaluated[i] else self.delay[i] + 1
                for i in range(len(self.delay))
            ]
        self.delay.append(1)
        self.evaluated.append(False)

        self.offline.step(img_path, dets, gts, results, self.learner.w)

        # detect anchor frame
        if self.config["DETECTOR"]["type"] == "fixed":
            is_anchor, feedback, feedback_length = self.detector.fixed_detect(
                self.frame_idx, self.config["DETECTOR"]["duration"]
            )
        elif self.config["DETECTOR"]["type"] == "stable":
            is_anchor, feedback, feedback_length = self.detector.stable_detect(
                self.seq_info,
                self.frame_idx,
                self.config["DETECTOR"]["duration"],
                self.config["DETECTOR"]["threshold"],
            )

        # update weight
        if is_anchor and feedback is not None:
            df_feedback = convert_df(feedback, is_offline=True)

            # calculate loss
            gradient_losses = np.zeros((self.n_experts))
            dt = (
                self.frame_idx + 1 - feedback_length
                if self.frame_idx + 1 > feedback_length
                else 0
            )
            for i, expert_results in enumerate(self.experts_results):
                if len(expert_results) > 0:
                    evaluate_idx = expert_results[:, 0] > dt
                    evaluate_results = expert_results[evaluate_idx]
                    evaluate_results[:, 0] -= dt
                df_expert_results = convert_df(evaluate_results)

                acc, ana, df_map = eval_results(
                    self.seq_info, df_feedback, df_expert_results,
                )
                loss = frame_loss(df_map, range(1, feedback_length + 1))
                if self.config["LOSS"]["type"] == "w_id":
                    loss = loss.sum(axis=1)
                elif self.config["LOSS"]["type"] == "wo_id":
                    loss = loss[:, :2].sum(axis=1)
                elif self.config["LOSS"]["type"] == "fn":
                    loss = loss[:, 1]
                loss = (
                    loss
                    * ~np.array(self.evaluated[-feedback_length:])
                    / self.config["LOSS"]["bound"]
                )

                gradient_losses[i] = loss.sum()

            self.learner.update(gradient_losses, self.delay)

            self.evaluated[-feedback_length:] = [True] * len(
                self.evaluated[-feedback_length:]
            )
        else:
            feedback = None
            gradient_losses = None

        if self.frame_idx > 0:
            prev_selected_expert = self.selected_expert
        else:
            prev_selected_expert = None

        # select expert
        if self.frame_idx == 0 or is_anchor or self.config["LOSS"]["delayed"]:
            self.selected_expert = weighted_random_choice(self.learner.w)

        # match id
        if self.config["MATCHING"]["method"] == "anchor":
            curr_expert_bboxes = self.matcher.anchor_match(
                prev_selected_expert, self.selected_expert, results
            )
        elif self.config["MATCHING"]["method"] == "kmeans":
            curr_expert_bboxes = self.matcher.kmeans_match(
                self.learner.w, self.selected_expert, results
            )
        else:
            raise NameError("Please enter a valid matching method")

        if len(curr_expert_bboxes) > 0:
            u, c = np.unique(curr_expert_bboxes[:, 0], return_counts=True)
            assert (c == 1).all(), f"Duplicated ID in frame {self.frame_idx}"

        # print(f"Frame {self.frame_idx}, Selected {self.selected_expert}")
        return (
            curr_expert_bboxes,
            self.learner.w,
            gradient_losses,
            feedback,
            self.selected_expert,
        )
