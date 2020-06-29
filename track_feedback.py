from datasets.mot import MOT
from paths import DATASET_PATH, OUTPUT_PATH
from feedback.neural_solver import NeuralSolver


def main(expert_name):
    datasets = {
        "MOT15": MOT(DATASET_PATH["MOT15"]),
        # "MOT16": MOT(DATASET_PATH["MOT16"]),
        # "MOT17": MOT(DATASET_PATH["MOT17"]),
    }
    tracker = NeuralSolver(
        "weights/NeuralSolver/mot_mpnet_epoch_006.ckpt",
        "weights/NeuralSolver/frcnn_epoch_27.pt.tar",
        "weights/NeuralSolver/resnet50_market_cuhk_duke.tar-232",
        "external/mot_neural_solver/configs/tracking_cfg.yaml",
        "external/mot_neural_solver/configs/preprocessing_cfg.yaml",
        True,
    )

    for dataset_name, dataset in datasets.items():
        dataset_dir = OUTPUT_PATH / f"{dataset_name}/{expert_name}"
        for seq in dataset:
            if (dataset_dir / f"{seq.name}.txt").exists():
                print(f"Pass {seq.name}")
            else:
                print(f"Start {seq.name}")
                tracker.initialize(seq.name)

                img_paths = []
                dets = []
                for frame_idx, (img_path, det) in enumerate(seq):
                    img_paths.append(img_path)
                    if det is None:
                        dets.append([])
                    else:
                        dets.append(det)

                results = tracker.track(img_paths, dets)
                seq.write_results(results, dataset_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experts")
    parser.add_argument(
        "-n", "--name", type=str, help="The name of the expert",
    )

    args = parser.parse_args()
    main(args.name)
