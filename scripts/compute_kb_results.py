import os
import pandas as pd

from tag_translation.translators.graph_translators import DbpMappingTranslator
from tag_translation.utils import TagManager, DataHelper, get_default_fold_split_file_for_target
from tag_translation.judge import Judge


"""
This script evaluates the results for the Knowledge-Based approach, using a translation table computed previously.
The annotation scores are obtained using the translation table as the weight matrix when doing logistic regression,
and taking 0 for the biases.
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("--tr-table", required=True, help="The path to the translation table")
    parser.add_argument("-o", "--out", required=True, help="Where to write the results.")
    args = parser.parse_args()
    target = args.target
    sources = list({"lastfm", "discogs", "tagtraum"} - {target})
    tm = TagManager.get(sources, target)
    dhelper = DataHelper(tm, dataset_path=get_default_fold_split_file_for_target(target))
    judge = Judge(tm)
    tr = DbpMappingTranslator(tm, args.tr_table)
    for fold in range(4):
        print("Computing KB results for fold {}".format(fold))
        eval_data, eval_target = dhelper.get_test_data(fold=fold)
        eval_target = eval_target.toarray().astype("float32")
        res = judge.compute_macro_metrics(eval_target, tr.predict_scores(eval_data))
        print(res)
        out_file = os.path.join(args.out, "results_fold_{}.csv".format(fold))
        print("Writing results to {}".format(out_file))
        pd.DataFrame(res, index=["mean"]).to_csv(out_file)
