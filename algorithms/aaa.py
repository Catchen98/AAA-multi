import sys
import random
import copy

import numpy as np
import pandas as pd
import scipy.special as sc
import motmetrics as mm

from algorithms.anchor_detector import FixedDetector
from feedback.neural_solver import NeuralSolver
from algorithms.aaa_util import convert_id, convert_df


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
    def __init__(self, n_experts):
        self.name = "AAA"
        self.n_experts = n_experts

        self.detector = FixedDetector(30)

        self.learner = WAADelayed()

        self.offline = NeuralSolver(
            "weights/NeuralSolver/mot_mpnet_epoch_006.ckpt",
            "weights/NeuralSolver/frcnn_epoch_27.pt.tar",
            "weights/NeuralSolver/resnet50_market_cuhk_duke.tar-232",
            "external/mot_neural_solver/configs/tracking_cfg.yaml",
            "external/mot_neural_solver/configs/preprocessing_cfg.yaml",
            True,
        )
        self.is_reset_offline = True

        self.acc = mm.MOTAccumulator(auto_id=True)

    def initialize(self, seq_info):
        self.frame_idx = -1
        self.last_id = 0

        self.seq_info = seq_info
        self.detector.initialize(seq_info)
        self.learner.initialize(self.n_experts)
        self.prev_bboxes = None

        self.reset_offline()
        self.reset_history()

    def reset_offline(self):
        self.img_paths = []
        self.dets = []

    def reset_history(self):
        self.timer = -1
        self.experts_results = [[] for _ in range(self.n_experts)]

    def _weighted_random_choice(self):
        pick = random.uniform(0, sum(self.learner.w))
        current = 0
        for i, weight in enumerate(self.learner.w):
            current += weight
            if current >= pick:
                return i

    def track(self, img_path, dets, results):
        self.frame_idx += 1
        self.timer += 1

        self.img_paths.append(img_path)
        self.dets.append(dets)

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

        if self.detector.detect(img_path, dets, results):
            try:
                feedback = self.offline.track(self.seq_info, self.img_paths, self.dets)
            except Exception:
                feedback = None

            if feedback is not None:
                df_feedback = convert_df(feedback)

                smallest_frame = len(self.dets) - self.timer - 1
                df_cond = df_feedback.index.get_level_values(0) > smallest_frame
                df_feedback = df_feedback[df_cond]

                df_feedback.index = pd.MultiIndex.from_tuples(
                    [(x[0] - smallest_frame, x[1]) for x in df_feedback.index]
                )

                gradient_losses = np.zeros((self.n_experts))
                for i, expert_results in enumerate(self.experts_results):
                    df_expert_results = convert_df(expert_results)
                    acc = mm.utils.compare_to_groundtruth(
                        df_feedback, df_expert_results, "iou", distth=0.5
                    )
                    mh = mm.metrics.create()
                    summary = mh.compute(
                        acc,
                        metrics=["num_false_positives", "num_misses", "num_switches"],
                    )
                    loss = sum(summary.iloc[0].values)
                    gradient_losses[i] = loss
                self.learner.update(gradient_losses, self.timer + 1)

                self.curr_bboxes = feedback[feedback[:, 0] == self.timer + 1, 1:]

                self.reset_history()
                if self.is_reset_offline:
                    self.reset_offline()

            else:
                selected_expert = self._weighted_random_choice()
                self.curr_bboxes = results[selected_expert]
                gradient_losses = None

        else:
            selected_expert = self._weighted_random_choice()
            self.curr_bboxes = results[selected_expert]
            feedback = None
            gradient_losses = None

        self.curr_bboxes = convert_id(self.prev_bboxes, self.curr_bboxes, 0.3)
        for i in range(len(self.curr_bboxes)):
            if self.curr_bboxes[i, 0] == -1:
                self.curr_bboxes[i, 0] = self.last_id
                self.last_id += 1

        self.prev_bboxes = copy.deepcopy(self.curr_bboxes)

        return self.curr_bboxes, self.learner.w, gradient_losses, feedback
