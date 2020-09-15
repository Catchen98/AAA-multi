from collections import OrderedDict
import glob
import os
import pickle
from pathlib import Path
import pandas as pd

import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes as compare_mot15
from motmetrics.apps.evaluateTracking import compare_dataframes as compare_mot

from paths import DATASET_PATH, OUTPUT_PATH, EVAL_PATH
from print_manager import do_not_print


@do_not_print
def eval_tracker(tracker, dataset_name, result_dir):
    tracker_dir = result_dir / tracker
    os.makedirs(tracker_dir, exist_ok=True)

    acc_path = tracker_dir / f"{dataset_name}_acc.pkl"
    if os.path.exists(acc_path):
        accs, analysis, names = pickle.loads(acc_path.read_bytes())
    else:
        gtfiles = glob.glob(
            os.path.join(DATASET_PATH[dataset_name] / "train", "*/gt/gt.txt")
        )

        tsfiles = [
            f
            for f in glob.glob(
                os.path.join(OUTPUT_PATH / dataset_name / tracker, "*.txt")
            )
            if not os.path.basename(f).startswith("eval")
        ]

        if dataset_name == "MOT16" or dataset_name == "MOT17":
            gt = OrderedDict(
                [
                    (
                        Path(f).parts[-3],
                        (
                            mm.io.load_motchallenge(f),
                            DATASET_PATH[dataset_name]
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

        acc_path.write_bytes(pickle.dumps((accs, analysis, names)))

    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()

    summary_path = tracker_dir / f"{dataset_name}_summary.txt"
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, index_col=0)
    else:
        summary = mh.compute_many(
            accs, anas=analysis, names=names, metrics=metrics, generate_overall=True
        )
        summary.to_csv(summary_path)

    summary_text_path = tracker_dir / f"{dataset_name}_summary_text.txt"
    if os.path.exists(summary_text_path):
        summary_text = summary_text_path.read_text()
    else:
        summary_text = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names,
        )
        summary_text_path.write_text(summary_text)

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

    results = eval_trackers(args.trackers, args.datasets, EVAL_PATH)
    for tracker, tracker_result in results.items():
        for dataset, dataset_result in tracker_result.items():
            print(f"{tracker}: {dataset}")
            print(dataset_result)
