import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# sns.set_style("darkgrid")

mpl.rcParams["legend.markerscale"] = 2
mpl.rcParams["legend.handlelength"] = 5
graycolors = sns.cubehelix_palette(5)
# graycolors = sns.dark_palette("gray", 5)

count_per_target = {"tagtraum": 532738, "lastfm": 558820, "discogs": 651715}

x_key = "Amount of training samples"
results_dir = sys.argv[1]
plot_dir = sys.argv[2]
opj = os.path.join

def read_results_file(path, scores, tot_amount_data, method_name, m_idx):
    try:
        res = pd.read_csv(path, index_col=0)
        for metric in ["auc_macro", "map_macro"]:
            scores.append({"metric": metric, "value": res[metric]["mean"], x_key: tot_amount_data,
                           "method": method_name, "method_idx": m_idx})
    except Exception as e:
        print(e)


def read_all_files(base_path_tpl, scores, method, m_idx, target, folds=range(4)):
    for frac in range(-13, 1):
        for fold in folds:
            read_results_file(base_path_tpl.format(frac=-frac, fold=fold, target=target),
                              scores, count_per_target[target]*2**frac, method, m_idx)


def read_map_logreg_with_bias(scores, target):
    method = r"MAP logistic regression $\nu = 0.1$"
    path_tpl = opj(results_dir, "map_results/{target}/results_frac_{frac}/fold_{fold}/results_epoch_500.csv")
    read_all_files(path_tpl, scores, method, 0, target)


def read_map_logreg_no_bias(scores, target):
    method = r"MAP logistic regression no bias reg $\nu = 0$"
    path_tpl = opj(results_dir, "map_results_no_bias/{target}/results_frac_{frac}/fold_{fold}/results_epoch_500.csv")
    read_all_files(path_tpl, scores, method, 1, target)


def read_ml_logreg_no_bias(scores, target):
    method = "ML logistic regression"
    path_tpl = opj(results_dir, "ml_results/{target}/results_frac_{frac}/fold_{fold}/results_ml_logreg.csv")
    read_all_files(path_tpl, scores, method, 2, target)


def read_kb_results(scores, target):
    method = "Knowledge-based"
    path_tpl = opj(results_dir, "kb_results/{target}/results_fold_{fold}.csv")
    read_all_files(path_tpl, scores, method, 4, target)


def read_bs_results(scores, target):
    method = "Levenshtein Baseline"
    path_tpl = opj(results_dir, "bs_results/{target}/LevenshteinTranslator/results_fold_{fold}.csv")
    read_all_files(path_tpl, scores, method, 5, target)


for target in ["lastfm", "discogs", "tagtraum"]:
    scores = []
    read_kb_results(scores, target)
    read_map_logreg_with_bias(scores, target)
    read_map_logreg_no_bias(scores, target)
    read_ml_logreg_no_bias(scores, target)
    read_bs_results(scores, target)
    df = pd.DataFrame(scores, columns=["value", x_key, "method", "metric"])
    # print(df)
    for metric in ["map_macro", "auc_macro"]:
        fig = plt.figure(figsize=(15, 10))
        metric_name = "MeanAP macro" if metric == "map_macro" else "AUC macro"
        df_for_metric = df[df.metric == metric]
        ax = sns.lineplot(x=x_key, y="value", hue="method", style="method", data=df_for_metric, ci="sd",
                     markers=["o", "v", "s", "X", "^"], color="black", palette=graycolors, size="method", sizes=[3] * 5,
                     markersize=10)
        ax.set_xscale('log')

        ax.set_xticks([10 ** i for i in range(2, 7)])
        ax.set_xticklabels([("%2.E" % int(10 ** i)).replace("E+", "e") for i in range(2, 7)])

        plt.tick_params(axis='both', which='major', labelsize=25, length=7, width=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='25')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='35')  # for legend title
        plt.xlabel(x_key, fontsize=32)
        plt.ylabel(metric_name, fontsize=32)

        plt.ylim(0 if metric == "map_macro" else 0.5, max(df_for_metric["value"]) + 0.05)

        plt.savefig(os.path.join(plot_dir, "{}_{}.png".format(metric, target)), bbox_inches='tight')
        plt.close(fig)