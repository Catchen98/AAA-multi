class FixedDetector:
    def __init__(self, duration):
        self.duration = duration

    def initialize(self, seq_info):
        self.seq_info = seq_info
        self.frame_idx = -1

    def detect(self, img_path, dets, results):
        self.frame_idx += 1

        if (self.frame_idx + 1) % self.duration == 0:
            return True

        else:
            return False
