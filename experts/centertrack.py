import sys
import cv2
import numpy as np
from types import SimpleNamespace

from experts.expert import Expert

sys.path.append("external/CenterTrack/src/lib")
from detector import Detector
from opts import opts


def get_default_calib(width, height):
    rest_focal_length = 1200
    calib = np.array(
        [
            [rest_focal_length, 0, width / 2, 0],
            [0, rest_focal_length, height / 2, 0],
            [0, 0, 1, 0],
        ]
    )
    return calib


def parse_opt(opt):
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(",")]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(",")]
    opt.save_point = [int(i) for i in opt.save_point.split(",")]
    opt.test_scales = [float(i) for i in opt.test_scales.split(",")]
    opt.save_imgs = [i for i in opt.save_imgs.split(",")] if opt.save_imgs != "" else []
    opt.ignore_loaded_cats = (
        [int(i) for i in opt.ignore_loaded_cats.split(",")]
        if opt.ignore_loaded_cats != ""
        else []
    )

    opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus))
    opt.pre_img = False
    if "tracking" in opt.task:
        print("Running tracking")
        opt.tracking = True
        opt.out_thresh = max(opt.track_thresh, opt.out_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
        opt.pre_img = not opt.no_pre_img
        if "ddd" in opt.task:
            opt.show_track_color = True

    opt.fix_res = not opt.keep_res

    if opt.head_conv == -1:  # init default head_conv
        opt.head_conv = 256 if "dla" in opt.arch else 64

    opt.pad = 127 if "hourglass" in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == "hourglass" else 1

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = opt.batch_size - opt.master_batch_size
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)

    if opt.debug > 0:
        opt.num_workers = 0
        opt.batch_size = 1
        opt.gpus = [opt.gpus[0]]
        opt.master_batch_size = -1

    return opt


def update_dataset_info_and_set_heads(opt, num_classes, default_resolution, num_joints):
    opt.num_classes = num_classes
    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h, input_w = default_resolution
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)

    opt.heads = {"hm": opt.num_classes, "reg": 2, "wh": 2}

    if "tracking" in opt.task:
        opt.heads.update({"tracking": 2})

    if "ddd" in opt.task:
        opt.heads.update({"dep": 1, "rot": 8, "dim": 3, "amodel_offset": 2})

    if "multi_pose" in opt.task:
        opt.heads.update({"hps": num_joints * 2, "hm_hp": num_joints, "hp_offset": 2})

    if opt.ltrb:
        opt.heads.update({"ltrb": 4})
    if opt.ltrb_amodal:
        opt.heads.update({"ltrb_amodal": 4})
    if opt.nuscenes_att:
        opt.heads.update({"nuscenes_att": 8})
    if opt.velocity:
        opt.heads.update({"velocity": 3})

    weight_dict = {
        "hm": opt.hm_weight,
        "wh": opt.wh_weight,
        "reg": opt.off_weight,
        "hps": opt.hp_weight,
        "hm_hp": opt.hm_hp_weight,
        "hp_offset": opt.off_weight,
        "dep": opt.dep_weight,
        "rot": opt.rot_weight,
        "dim": opt.dim_weight,
        "amodel_offset": opt.amodel_offset_weight,
        "ltrb": opt.ltrb_weight,
        "tracking": opt.tracking_weight,
        "ltrb_amodal": opt.ltrb_amodal_weight,
        "nuscenes_att": opt.nuscenes_att_weight,
        "velocity": opt.velocity_weight,
    }
    opt.weights = {head: weight_dict[head] for head in opt.heads}
    for head in opt.weights:
        if opt.weights[head] == 0:
            del opt.heads[head]
    opt.head_conv = {
        head: [opt.head_conv for i in range(opt.num_head_conv if head != "reg" else 1)]
        for head in opt.heads
    }

    return opt


class CenterTrack(Expert):
    def __init__(self, load_model, track_thresh, pre_thresh, private):
        super(CenterTrack, self).__init__("CenterTrack")
        parser = opts().parser
        opt = {}
        for action in parser._actions:
            if not action.required and action.dest != "help":
                opt[action.dest] = action.default
        opt = SimpleNamespace(**opt)
        opt.task = "tracking"
        opt.load_model = load_model
        opt.track_thresh = track_thresh
        opt.pre_thresh = pre_thresh
        opt.pre_hm = True
        opt.ltrb_amodal = True
        opt.public_det = private

        self.opt = parse_opt(opt)
        self.opt = update_dataset_info_and_set_heads(self.opt, 1, [544, 960], 17)
        self.private = private

    def initialize(self, seq_info):
        super(CenterTrack, self).initialize(seq_info)

        self.tracker = Detector(self.opt)
        self.tracker.reset_tracking()

    def track(self, img_path, dets):
        super(CenterTrack, self).track(img_path, dets)

        input_meta = self.preprocess(img_path, dets)

        ret = self.tracker.run(img_path, input_meta)

        result = []
        for t in ret["results"]:
            bbox = t["bbox"]
            tracking_id = t["tracking_id"]
            result.append(
                [tracking_id, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            )

        return result

    def preprocess(self, img_path, dets):
        img = cv2.imread(img_path)
        input_meta = {}
        input_meta["calib"] = get_default_calib(img.shape[1], img.shape[0])

        detections = []
        if dets is not None:
            for det in dets:
                bbox = [
                    float(det[1]),
                    float(det[2]),
                    float(det[1] + det[3]),
                    float(det[2] + det[4]),
                ]
                ct = [(det[1] + det[3]) / 2, (det[2] + det[4]) / 2]
                detections.append(
                    {"bbox": bbox, "score": float(det[5]), "class": 1, "ct": ct}
                )

        if self.frame_idx == 0:
            if self.private:
                input_meta["pre_dets"] = []
            else:
                input_meta["pre_dets"] = detections

        if self.private:
            input_meta["cur_dets"] = []
        else:
            input_meta["cur_dets"] = detections

        return input_meta
