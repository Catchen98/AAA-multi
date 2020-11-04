from collections import OrderedDict
import glob
import os
import pickle
from pathlib import Path
import pandas as pd

import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes as compare_mot15
from motmetrics.apps.evaluateTracking import compare_dataframes as compare_mot

from algorithms.aaa_util import eval_results

from print_manager import do_not_print


@do_not_print
def eval_tracker(
    dataset_dir, output_dir, tracker, dataset_name, result_dir, tracker_dir=None
):
    if tracker_dir is None:
        tracker_dir = os.path.join(result_dir, tracker, dataset_name)
    os.makedirs(tracker_dir, exist_ok=True)

    acc_path = Path(os.path.join(tracker_dir, "acc.pkl"))
    if os.path.exists(acc_path):
        accs, analysis, names = pickle.loads(acc_path.read_bytes())
    else:
        gtfiles = glob.glob(
            os.path.join(dataset_dir[dataset_name], "train", "*/gt/gt.txt")
        )

        tsfiles = [
            os.path.join(output_dir, dataset_name, tracker, f)
            for f in os.listdir(os.path.join(output_dir, dataset_name, tracker))
            if f.endswith(".txt")
        ]

        if dataset_name == "MOT15":
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

        elif dataset_name.startswith("MOT"):
            gt = OrderedDict(
                [
                    (
                        Path(f).parts[-3],
                        (
                            mm.io.load_motchallenge(f),
                            os.path.join(
                                dataset_dir[dataset_name],
                                "train",
                                Path(f).parts[-3],
                                "seqinfo.ini",
                            ),
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
            raise NameError("Invalid dataset")

        for k, tsacc in ts.items():
            if k in gt:
                eval_path = Path(os.path.join(tracker_dir, f"{k}.csv"))
                seq_info = {"dataset_name": dataset_name}
                if dataset_name == "MOT15":
                    _, _, df_map = eval_results(seq_info, gt[k], tsacc)
                elif dataset_name.startswith("MOT"):
                    seq_info["ini_path"] = gt[k][1]
                    _, _, df_map = eval_results(seq_info, gt[k][0], tsacc)
                df_map.noraw.to_csv(eval_path)

        acc_path.write_bytes(pickle.dumps((accs, analysis, names)))

    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()

    summary_path = Path(os.path.join(tracker_dir, "summary.csv"))
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, index_col=0)
    else:
        summary = mh.compute_many(
            accs, anas=analysis, names=names, metrics=metrics, generate_overall=True
        )
        summary.to_csv(summary_path)

    summary_text_path = Path(os.path.join(tracker_dir, "summary_text.txt"))
    if os.path.exists(summary_text_path):
        summary_text = summary_text_path.read_text()
    else:
        summary_text = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names,
        )
        summary_text_path.write_text(summary_text)

    return summary_text


@do_not_print
def eval_trackers(dataset_dir, output_dir, trackers, datasets, result_dir):
    results = {}
    for tracker in trackers:
        results[tracker] = {}
        for dataset in datasets:
            summary_text = eval_tracker(
                dataset_dir, output_dir, tracker, dataset, result_dir
            )
            results[tracker][dataset] = summary_text
    return results
