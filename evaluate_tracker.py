from collections import OrderedDict
import glob
import os
import pickle
from pathlib import Path

import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes as compare_mot15
from motmetrics.apps.evaluateTracking import compare_dataframes as compare_mot

from paths import DATASET_PATH, OUTPUT_PATH
from print_manager import do_not_print


@do_not_print
def eval_tracker(tracker, dataset, result_dir):
    tracker_dir = result_dir / tracker
    os.makedirs(tracker_dir, exist_ok=True)

    result_path = tracker_dir / f"{dataset}.pkl"
    if os.path.exists(result_path):
        accs, analysis, names = pickle.loads(result_path.read_bytes())
    else:
        gtfiles = glob.glob(
            os.path.join(DATASET_PATH[dataset] / "train", "*/gt/gt.txt")
        )
        tsfiles = [
            f
            for f in glob.glob(os.path.join(OUTPUT_PATH / dataset / tracker, "*.txt"))
            if not os.path.basename(f).startswith("eval")
        ]

        if dataset == "MOT16" or dataset == "MOT17":
            gt = OrderedDict(
                [
                    (
                        Path(f).parts[-3],
                        (
                            mm.io.load_motchallenge(f),
                            DATASET_PATH[dataset]
                            / "train"
                            / Path(f).parts[-3]
                            / "seqinfo.ini",
                        ),
                    )
                    for f in gtfiles
                ]
            )
            ts = OrderedDict(
                [
                    (
                        os.path.splitext(Path(f).parts[-1])[0],
                        mm.io.load_motchallenge(f),
                    )
                    for f in tsfiles
                ]
            )

            accs, analysis, names = compare_mot(gt, ts)

        else:
            gt = OrderedDict(
                [
                    (Path(f).parts[-3], mm.io.load_motchallenge(f, min_confidence=1),)
                    for f in gtfiles
                ]
            )
            ts = OrderedDict(
                [
                    (
                        os.path.splitext(Path(f).parts[-1])[0],
                        mm.io.load_motchallenge(f),
                    )
                    for f in tsfiles
                ]
            )

            accs, names = compare_mot15(gt, ts)
            analysis = None

        result_path.write_bytes(pickle.dumps((accs, analysis, names)))

    metrics = list(mm.metrics.motchallenge_metrics)

    mh = mm.metrics.create()

    summary = mh.compute_many(
        accs, anas=analysis, names=names, metrics=metrics, generate_overall=True
    )

    summary_text = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names,
    )

    summary_path = tracker_dir / f"{dataset}.txt"
    summary_path.write_text(summary_text)

    return summary_text


@do_not_print
def eval_trackers(trackers, datasets, result_dir):
    results = {}
    for tracker in trackers:
        results[tracker] = {}
        for dataset in datasets:
            summary_text = eval_tracker(tracker, dataset, result_dir)
            results[tracker][dataset] = summary_text
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trackers", default=list(), nargs="+")
    parser.add_argument("-d", "--datasets", default=list(), nargs="+")
    args = parser.parse_args()

    result_dir = Path("eval")

    results = eval_trackers(args.trackers, args.datasets, result_dir)
    for tracker, tracker_result in results.items():
        for dataset, dataset_result in tracker_result.items():
            print(f"{tracker}: {dataset}")
            print(dataset_result)
