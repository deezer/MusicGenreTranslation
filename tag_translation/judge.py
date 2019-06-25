import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score
from joblib import Parallel, delayed

from tag_translation.utils import get_default_fold_split_file_for_target, TagManager, DataHelper
from tag_translation.conf import N_JOBS


class Judge():

    def __init__(self, tag_manager):

        self.sources = tag_manager.sources
        self.target = tag_manager.target
        self.tag_manager = tag_manager

    def compute_metrics(self, ground_truth, pred):
        print("Computing metrics")
        out = {}
        per_tag_map = average_precision_score(ground_truth, pred, average=None)
        for k, tag in enumerate(self.tag_manager.target_tags):
            out[f"map_{tag}"] = per_tag_map[k]

        per_tag_map = roc_auc_score(ground_truth, pred, average=None)
        for k, tag in enumerate(self.tag_manager.target_tags):
            out[f"auc_{tag}"] = per_tag_map[k]
        out.update(self.compute_high_level_metrics(ground_truth, pred))
        return out

    def compute_high_level_metrics(self, ground_truth, pred):
        out = {}
        out["map_sample"] = self.compute_map(ground_truth, pred)
        out["map_macro"] = self.compute_map(ground_truth, pred, "macro")
        out["map_micro"] = self.compute_map(ground_truth, pred, "micro")
        out["auc_sample"] = self.compute_auc(ground_truth, pred)
        out["auc_macro"] = self.compute_auc(ground_truth, pred, "macro")
        out["auc_micro"] = self.compute_auc(ground_truth, pred, "micro")
        return out

    def compute_macro_metrics(self, ground_truth, pred):
        return {"map_macro": self.compute_map(ground_truth, pred, "macro"),
                "auc_macro": self.compute_auc(ground_truth, pred, "macro")}

    def compute_map(self, ground_truth, pred, average="samples"):
        return fast_average_precision_score(ground_truth, pred, average=average)

    def compute_auc(self, ground_truth, pred, average="samples"):
        return roc_auc_score(ground_truth, pred, average=average)


def fast_average_precision_score(y_true, y_pred, average):
    """
        compute average precision in a parallelized way (for "samples" and "macro" averaging only),
        resulting in much faster computation.
    """
    if average=="samples":
        return fast_average_precision_score_samples(y_true, y_pred, n_jobs=N_JOBS, block_size=2_000)
    elif average=="macro":
        return fast_average_precision_score_samples(y_true.T, y_pred.T, n_jobs=N_JOBS, block_size=5)
    else:
        return average_precision_score(y_true, y_pred, average=average)


def fast_average_precision_score_samples(y_true, y_pred, n_jobs=-1, block_size=None):
    """
        compute average precision with "samples" averagin in a parallelized way for "samples" and
        "macro" averaging, resulting in much faster computation.
    """
    parallel = Parallel(n_jobs=n_jobs)
    N = y_true.shape[0]
    m = parallel((delayed(label_ranking_average_precision_score)(
        y_true[k:k+block_size, :], y_pred[k:k+block_size, :])) for k in range(0, N, block_size))
    return np.sum([el*y_true[k:k+block_size, :].shape[0] for k, el in zip(range(0, N, block_size), m)])/N


def get_default_judge(sources, target):
    return Judge(TagManager.get(sources, target))
