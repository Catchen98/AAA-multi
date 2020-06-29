from collections import OrderedDict
import glob
import os
from pathlib import Path

import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes

from paths import DATASET_PATH, OUTPUT_PATH
from print_manager import do_not_print


@do_not_print
def eval_trackers(trackers, datasets):
    results = {}
    for tracker in trackers:
        results[tracker] = {}
        for dataset in datasets:
            gtfiles = glob.glob(
                os.path.join(DATASET_PATH[dataset] / "train", "*/gt/gt.txt")
            )
            tsfiles = [
                f
                for f in glob.glob(
                    os.path.join(OUTPUT_PATH / f"{dataset}/{tracker}", "*.txt")
                )
                if not os.path.basename(f).startswith("eval")
            ]

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

            mh = mm.metrics.create()
            accs, names = compare_dataframes(gt, ts)

            metrics = list(mm.metrics.motchallenge_metrics)

            summary = mh.compute_many(
                accs, names=names, metrics=metrics, generate_overall=True
            )
            results[tracker][dataset] = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names,
            )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trackers", default=list(), nargs="+")
    parser.add_argument("-d", "--datasets", default=list(), nargs="+")
    args = parser.parse_args()

    results = eval_trackers(args.trackers, args.datasets)
    for tracker, result in results.items():
        print(tracker)
        print(result["MOT15"])
