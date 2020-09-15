from experts.expert import Expert
from file_manager import ReadResult


class PreExpert(Expert):
    def __init__(self, name):
        super(PreExpert, self).__init__(name)

    def initialize(self, seq_info, result_path):
        super(PreExpert, self).initialize(seq_info)
        self.reader = ReadResult(
            seq_info["dataset_name"], self.name, seq_info["seq_name"]
        )

    def track(self, img_path, dets):
        super(PreExpert, self).track(img_path, dets)
        results = self.reader.get_result_by_frame(self.frame_idx)
        return results
