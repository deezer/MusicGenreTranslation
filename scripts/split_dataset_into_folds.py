"""
Stratified sampling implementation to split the total corpus in folds. It enforces some additional
constraints like ensuring that items corresponding to the same artist belong to the same fold. This is aimed at
avoiding to overfit on the tags of a specific artist.
"""

import os
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_random_state


def filter_data(df, target):
    """
        Filter out the samples which do not contain tags from both the source and target
    """
    filtered_df = pd.DataFrame(df.dropna(subset=[target]))
    filtered_df[target] = filtered_df[target].apply(literal_eval)
    return filtered_df.reset_index(drop=True)


def mark_groups_for_samples(df, n_samples, extra_criterion):
    """
        Return groups, an array of size n_samples, marking the group to which each sample belongs
        The default group is -1 if extra_criterion is None
        If a criterion is given (artist or album), then this information is taken into account
    """
    groups = np.array([-1 for _ in range(n_samples)])
    if extra_criterion is None:
        return groups

    if extra_criterion == "artist":
        crit_col = "artistmbid"
    elif extra_criterion == "album":
        crit_col = "releasegroupmbid"
    else:
        return groups

    gp = df.groupby(crit_col)
    i_key = 0
    for g_key in gp.groups:
        samples_idx_per_group = gp.groups[g_key].tolist()
        groups[samples_idx_per_group] = i_key
        i_key += 1
    return groups


def select_fold(index_label, desired_samples_per_label_per_fold, desired_samples_per_fold, random_state):
    """
        For a label, finds the fold where the next sample should be distributed
    """
    # Find the folds with the largest number of desired samples for this label
    largest_desired_label_samples = max(desired_samples_per_label_per_fold[:, index_label])
    folds_targeted = np.where(desired_samples_per_label_per_fold[:, index_label] == largest_desired_label_samples)[0]

    if len(folds_targeted) == 1:
        selected_fold = folds_targeted[0]
    else:
        # Break ties by considering the largest number of desired samples
        largest_desired_samples = max(desired_samples_per_fold[folds_targeted])
        folds_re_targeted = np.intersect1d(np.where(
            desired_samples_per_fold == largest_desired_samples)[0], folds_targeted)

        # If there is still a tie break it picking a random index
        if len(folds_re_targeted) == 1:
            selected_fold = folds_re_targeted[0]
        else:
            selected_fold = random_state.choice(folds_re_targeted)
    return selected_fold


def iterative_split(df, out_file, target, n_splits, extra_criterion=None, seed=None):
    """
        Implement iterative split algorithm
        df is the input data
        out_file is the output file containing the same data as the input plus a column about the fold
        n_splits the number of folds
        target is the target source for which the files are generated
        extra_criterion, an extra condition to be taken into account in the split such as the artist
    """
    print("Starting the iterative split")
    random_state = check_random_state(seed)

    mlb_target = MultiLabelBinarizer(sparse_output=True)
    M = mlb_target.fit_transform(df[target])

    n_samples = len(df)
    n_labels = len(mlb_target.classes_)

    # If the extra criterion is given create "groups", which shows to which group each sample belongs
    groups = mark_groups_for_samples(df, n_samples, extra_criterion)

    ratios = np.ones((1, n_splits))/n_splits
    # Calculate the desired number of samples for each fold
    desired_samples_per_fold = ratios.T * n_samples

    # Calculate the desired number of samples of each label for each fold
    number_samples_per_label = np.asarray(M.sum(axis=0)).reshape((n_labels, 1))
    desired_samples_per_label_per_fold = np.dot(ratios.T, number_samples_per_label.T)  # shape: n_splits, n_samples

    seen = set()
    out_folds = np.array([-1 for _ in range(n_samples)])

    count_seen = 0
    print("Going through the samples")
    while n_samples > 0:
        # Find the index of the label with the fewest remaining examples
        valid_idx = np.where(number_samples_per_label > 0)[0]
        index_label = valid_idx[number_samples_per_label[valid_idx].argmin()]
        label = mlb_target.classes_[index_label]

        # Find the samples belonging to the label with the fewest remaining examples
        # second select all samples belonging to the selected label and remove the indices
        # of the samples which have been already seen
        all_label_indices = set(M[:, index_label].nonzero()[0])
        indices = all_label_indices - seen
        assert(len(indices) > 0)

        print(label, index_label, number_samples_per_label[index_label], len(indices))

        for i in indices:
            if i in seen:
                continue

            # Find the folds with the largest number of desired samples for this label
            selected_fold = select_fold(index_label, desired_samples_per_label_per_fold,
                                        desired_samples_per_fold, random_state)

            # put in this fold all the samples which belong to the same group
            idx_same_group = np.array([i])
            if groups[i] != -1:
                idx_same_group = np.where(groups == groups[i])[0]

            # Update the folds, the seen, the number of samples and desired_samples_per_fold
            out_folds[idx_same_group] = selected_fold
            seen.update(idx_same_group)
            count_seen += idx_same_group.size
            n_samples -= idx_same_group.size
            desired_samples_per_fold[selected_fold] -= idx_same_group.size

            # The sample may have multiple labels so update for all
            for idx in idx_same_group:
                _, all_labels = M[idx].nonzero()
                desired_samples_per_label_per_fold[selected_fold, all_labels] -= 1
                number_samples_per_label[all_labels] -= 1

    df['fold'] = out_folds
    print(count_seen, len(df))
    df.to_csv(out_file, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Splits the common recordings dataset into folds, with a constraint on artist"
                                     " or albums to belong to the same fold.")
    parser.add_argument("dataset", type=str, help="Path to the common recordings data")
    parser.add_argument("output", type=str, help="Path where to write the splitted data")

    parser.add_argument('--folds',
                        required=True,
                        type=int,
                        help='The number of folds')

    parser.add_argument('--by',
                        default=None,
                        help='Extra criterion to consider in the sampling. Possible values: artist, album')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        assert os.path.isdir(args.output)
    folds = args.folds
    assert isinstance(folds, int) and folds > 0
    extra_criterion = args.by
    crit_str = "" if extra_criterion is None else "_by_" + extra_criterion
    assert os.path.exists(args.dataset)
    df = pd.read_csv(args.dataset, sep='\t')
    for target in ["lastfm", "discogs", "tagtraum"]:
        print("Creating split on {} folds for target {}".format(folds, target))
        df_target = filter_data(df, target)
        out_file = os.path.join(args.output, target + "_" + str(folds) + "-fold" + crit_str + ".tsv")
        iterative_split(df_target, out_file, target, folds, extra_criterion)
