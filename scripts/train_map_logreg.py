import os
import numpy as np
import tensorflow as tf
import pandas as pd

from tag_translation.judge import Judge
from tag_translation.utils import DataHelper, TagManager, get_default_fold_split_file_for_target, \
    read_translation_table
from tag_translation.translators.map_logreg import MapLogReg


def check_args(args):
    assert(args.eval_every > 0)
    assert(args.lr > 0)
    assert(args.epochs > 0)
    assert(args.frac <= 0)
    assert(args.sigma is None or args.sigma > 0)
    assert(args.batch_size > 0)
    assert(args.fold is None or args.fold >= 0)
    assert(args.bias_reg is None or args.bias_reg > 0)


def save_train_data_and_missing_tags(model_dir, train_data, tag_manager):
    train_data.to_csv(os.path.join(model_dir, "train_data"))
    print("{} samples in the training data".format(len(train_data)))
    target_tag_counts = {}
    for tags in train_data[target]:
        for tag in tags:
            target_tag_counts.setdefault(tag, 0)
            target_tag_counts[tag] += 1
    missing_tags = set(tag_manager.mlb_target.classes_) - set(target_tag_counts.keys())
    print("{} missing tags".format(len(missing_tags)))

    with open(os.path.join(model_dir, "missing_target_tags"), "w") as _:
        _.write("\n".join(missing_tags))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sources", nargs='+',
                        required=False, default=["lastfm", "tagtraum"],
                        help="List of source names")
    parser.add_argument("-t", "--target", help="name of the target",
                        required=False, default="discogs")
    parser.add_argument("-f", "--frac", type=int,
                        help="log2 of the fraction to keep in the fold")
    parser.add_argument("--sigma", type=float, required=False, default=None,
                        help="Covariance value used to define prior distribution")
    parser.add_argument("--bias-reg", type=float, required=False, default=None,
                        help="Bias regularization parameter")
    parser.add_argument("--lr", type=float, required=False, default=0.05,
                        help="The learning rate to use")
    parser.add_argument("--epochs", type=int, required=False, default=50,
                        help="Maximum number of epochs for which to train the model")
    parser.add_argument("--eval-every", type=int, required=False, default=2,
                        help="How many training epochs should there be between "
                        "two evaluations")
    parser.add_argument("--batch-size", type=int, required=False, default=128,
                        help="The size of batch to use in training and eval")
    parser.add_argument("-o", "--out", required=True,
                        help="Directory used for logging and saving the model")
    parser.add_argument("--tr-table", required=False, default=None,
                        help="If passed, this is the pass of the translation"
                        " table. This will be used to define the prior "
                        "distribution when training the MAP objective.")
    parser.add_argument("--fold", required=False, type=int, default=None,
                        help="The index of the fold to use for training")

    args = parser.parse_args()
    check_args(args)
    tf.logging.set_verbosity(tf.logging.INFO)

    sources = args.sources
    target = args.target
    print(f"sources are {sources} target: {target}")
    frac = 2**args.frac

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    tag_manager = TagManager(sources, target)
    judge = Judge(tag_manager)
    dhelper = DataHelper(tag_manager, dataset_path=get_default_fold_split_file_for_target(target))
    if args.tr_table is not None:
        kb_tr_table = read_translation_table(args.tr_table, tag_manager)
        map0 = kb_tr_table.values.T
    else:
        kb_tr_table = None
        map0 = None

    if args.fold is None:
        folds = range(4)
    else:
        folds = [args.fold]
    for fold in folds:
        print("Training on fold {}".format(fold))
        model_dir = os.path.join(args.out, "fold_{}".format(fold))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        train_data, target_data = dhelper.get_train_data(fold=fold, frac=frac, store_train_df=True)
        eval_data, eval_target = dhelper.get_test_data(fold=fold)
        eval_target = eval_target.toarray().astype("float32")

        save_train_data_and_missing_tags(model_dir, dhelper.train_df, tag_manager)

        def _score_function(gt, probas, **kwargs):
            epoch = kwargs["epoch"]
            res = judge.compute_macro_metrics(gt, probas)
            print("================= Results Report ====================")
            print("MAP macro: {}, AUC macro".format(res["map_macro"], res["auc_macro"]))
            print("=====================================================")
            pd.DataFrame(res, index=["mean"]).to_csv(
                os.path.join(model_dir, "results_epoch_{}.csv".format(epoch)))

        model = MapLogReg(model_dir, map0=map0, epochs=args.epochs, sigma=args.sigma, batch_size=args.batch_size,
                          bias_reg=args.bias_reg, lr=args.lr, eval_every=args.eval_every)
        model.train_and_evaluate(train_data, target_data, eval_data, eval_target,  _score_function)
