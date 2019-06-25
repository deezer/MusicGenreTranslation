import os
import pandas as pd
from tag_translation.translators.sklearn_logreg import SKLearnTranslator
from tag_translation.judge import Judge
from tag_translation.utils import TagManager, DataHelper, get_default_fold_split_file_for_target

"""
This is the script used to train Maximum Likelihood logistic regression models.
"""


opd = os.path.dirname


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sources", nargs='+',
                        required=False, default=["lastfm", "tagtraum"],
                        help="List of source names")
    parser.add_argument("-t", "--target", help="name of the target",
                        required=False, default="discogs")
    parser.add_argument("-i", "--input", required=False,
                        help="Path to the training data", default=None)
    parser.add_argument("-o", "--out", required=True,
                        help="Directory used for logging and saving the model")
    parser.add_argument("--fold", type=int, required=False, default=None,
                        help="Index of the fold corresponding to the training data")
    parser.add_argument("-f", "--frac", type=int, required=False, default=None,
                        help="log2 of the fraction to keep in the fold")
    args = parser.parse_args()

    assert args.frac is None or args.input is None
    frac = 0 if args.frac is None else args.frac
    assert isinstance(frac, float) or isinstance(frac, int) and frac <= 0
    frac = 2 ** frac

    sources = args.sources
    target = args.target
    tm = TagManager.get(sources, target)
    judge = Judge(tm)
    dhelper = DataHelper(tm, train_data_path=args.input, dataset_path=get_default_fold_split_file_for_target(target))
    model_dir = args.out

    if args.fold is None:
        folds = range(4)
    else:
        folds = [args.fold]

    def _score_function(gt, probas, **kwargs):
        res = judge.compute_macro_metrics(gt, probas)
        print("================= Results Report ====================")
        print("MAP macro: {}, AUC macro".format(res["map_macro"], res["auc_macro"]))
        print("=====================================================")
        out = kwargs["out"]
        print("Writing results to {}".format(out))
        pd.DataFrame(res, index=["mean"]).to_csv(out)

    for fold in folds:
        train_data, target_data = dhelper.get_train_data(frac=frac, fold=fold, store_train_df=True)
        fold_dir = os.path.join(model_dir, "fold_{}".format(fold))
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        dhelper.train_df.to_csv(os.path.join(fold_dir, "train_data"))
        eval_data, eval_target = dhelper.get_test_data(fold=fold)
        eval_target = eval_target.toarray().astype("float32")
        model = SKLearnTranslator()
        model.train_and_evaluate(train_data, target_data, eval_data, eval_target,
                                 lambda x, y, **kwargs: _score_function(
                                     x, y, out=os.path.join(fold_dir, "results_ml_logreg.csv"), **kwargs))
