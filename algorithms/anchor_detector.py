from .aaa_util import eval_results, get_summary, convert_df


class AnchorDetector:
    def __init__(self, offline):
        self.offline = offline

    def initialize(self, seq_info):
        self.seq_info = seq_info
        self.previous_offline = None

    def fixed_detect(self, frame_idx, duration):
        feedback_length = duration
        if (frame_idx + 1) % duration == 0:
            is_anchor, feedback = (
                True,
                self._get_feedback(frame_idx - duration + 1, frame_idx),
            )
        else:
            is_anchor, feedback = False, None

        return is_anchor, feedback, feedback_length

    def stable_detect(self, seq_info, frame_idx, duration, threshold):
        if frame_idx + 1 > duration:
            current_offline = self._get_feedback(frame_idx - duration + 1, frame_idx)

            if self.previous_offline is not None and current_offline is not None:
                overlap_previous = self.previous_offline[
                    self.previous_offline[:, 0] > 1
                ]
                overlap_previous[:, 0] -= 1
                overlap_previous = convert_df(overlap_previous, is_offline=True)

                overlap_current = current_offline[current_offline[:, 0] < duration]
                overlap_current = convert_df(overlap_current, is_offline=True)

            feedback_length = duration
        else:
            current_offline = self._get_feedback(0, frame_idx)

            if self.previous_offline is not None and current_offline is not None:
                overlap_previous = convert_df(self.previous_offline, is_offline=True)

                overlap_current = current_offline[current_offline[:, 0] <= frame_idx]
                overlap_current = convert_df(overlap_current, is_offline=True)

            feedback_length = frame_idx + 1

        if self.previous_offline is not None and current_offline is not None:
            prev_acc, prev_ana, _ = eval_results(
                seq_info, overlap_previous, overlap_current
            )
            prev_sum = get_summary(prev_acc, prev_ana)

            curr_acc, curr_ana, _ = eval_results(
                seq_info, overlap_current, overlap_previous
            )
            curr_sum = get_summary(curr_acc, curr_ana)

            mean_mota = (prev_sum[3] + curr_sum[3]) / 2
            if mean_mota >= threshold:
                is_anchor = True
                feedback = current_offline
            else:
                is_anchor = False
                feedback = None

            print(f"Frame {frame_idx}, MOTA {mean_mota}")
        else:
            is_anchor = False
            feedback = None
        self.previous_offline = current_offline

        return is_anchor, feedback, feedback_length

    def _get_feedback(self, start_frame, end_frame):
        # feedback = self.offline.track(start_frame, end_frame)
        try:
            feedback = self.offline.track(start_frame, end_frame)
        except (RuntimeError, ValueError):
            feedback = None

        return feedback
