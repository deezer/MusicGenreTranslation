import os
import pandas as pd

from tag_translation.translators.baseline_translators import LevenshteinTranslator, NormDirectTranslator
from tag_translation.utils import TagManager, DataHelper, get_default_fold_split_file_for_target
from tag_translation.judge import Judge

"""
This script computes the scores that are reported as the baseline in the article
"""


opj = os.path.join
opd = os.path.dirname
opa = os.path.abspath
ope = os.path.exists

tag_rep_dir = opj(opd(opd(opa(__file__))), "data", "tag_representation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("-o", "--out", required=True, help="Where to write the results.")
    args = parser.parse_args()
    target = args.target
    sources = list({"lastfm", "discogs", "tagtraum"} - {target})
    tm = TagManager.get(sources, target)
    dhelper = DataHelper(tm, dataset_path=get_default_fold_split_file_for_target(target))
    judge = Judge(tm)
    baseline_translators = [LevenshteinTranslator(tm)]
    base_results_dir = opj(args.out, target)
    if not ope(args.out):
        os.mkdir(args.out)
    if not ope(base_results_dir):
        os.mkdir(base_results_dir)
    for tr in baseline_translators:
        tname = type(tr).__name__
        for fold in range(4):
            print("Computing KB results for fold {}".format(fold))
            eval_data, eval_target = dhelper.get_test_data(fold=fold)
            eval_target = eval_target.toarray().astype("float32")
            res = judge.compute_macro_metrics(eval_target, tr.predict_scores(eval_data))
            print(res)
            base_dir = os.path.join(base_results_dir, tname)
            if not ope(base_dir):
                os.mkdir(base_dir)
            out_file = os.path.join(base_dir, "results_fold_{}.csv".format(fold))
            print("Writing results to {}".format(out_file))
            pd.DataFrame(res, index=["mean"]).to_csv(out_file)
