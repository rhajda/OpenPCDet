import os
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import statsmodels.api as sm
from matplotlib.text import Text

plt.rcParams["font.size"] = "16"
rc('axes', linewidth=2)
rc('font', weight='bold', family='serif')
plt.show()


def read_csv(training_dataset_run, testing_dataset):
    return pd.read_csv(os.path.join(network_path, training_dataset_run, "eval/eval_all_default", testing_dataset,
                                    "results.csv"), sep=",", index_col="epoch_id")


def create_result_dict(training_dataset_runs):
    result_dict = {}
    testing_datasets = []
    for training_dataset_run in training_dataset_runs:
        testing_datasets.append(training_dataset_run[:-2])
    testing_datasets = sorted(list(set(testing_datasets)))
    for training_dataset_run in training_dataset_runs:
        for testing_dataset in testing_datasets:
            result_dict[f"{testing_dataset}_{training_dataset_run}"] = read_csv(training_dataset_run, testing_dataset)
    return result_dict, testing_datasets


def merge_runs(result_dict, testing_datasets):
    metrics = ["AP_Car_3d/0.5_R11", "AP_Car_3d/0.5_R40", "AP_Car_3d/0.7_R11", "AP_Car_3d/0.7_R40", "recall/rcnn_0.3",
               "recall/rcnn_0.5", "recall/rcnn_0.7", "recall/roi_0.3", "recall/roi_0.5", "recall/roi_0.7",
               "avg_pred_obj"]
    relevant_metrics = {"AP_Car_3d/0.5_R40": 0, "AP_Car_3d/0.7_R40": 0, "recall/rcnn_0.5": 0, "recall/rcnn_0.7": 0}
    y_scales = {"AP_Car": 100, "recall": 100, "avg_pr": 1}
    means = np.zeros((len(metrics), len(testing_datasets), len(testing_datasets)))
    stds = np.zeros((len(metrics), len(testing_datasets), len(testing_datasets)))
    full_csv = os.path.join(sys.argv[1], f"{sys.argv[1].split('/')[-1]}.csv")
    if os.path.isfile(full_csv):
        os.remove(full_csv)
    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(4, 1)
        fig.canvas.manager.set_window_title(metric)

        fig2, ax2 = plt.subplots(1, 1)
        fig2.canvas.manager.set_window_title(metric)

        facecolor = {0: "red", 1: "blue", 2: "dodgerblue", 3: "deepskyblue"}
        positions = {0: -0.3, 1: -0.1, 2: 0.1, 3: 0.3}

        means_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_means.csv")
        stds_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_stds.csv")
        if os.path.isfile(means_csv):
            os.remove(means_csv)
        if os.path.isfile(stds_csv):
            os.remove(stds_csv)

        if metric in relevant_metrics.keys():
            relevant_metrics[metric] = metric_idx

        for testing_idx, testing_dataset in enumerate(testing_datasets):
            boxplot_list = []
            beanplot_single_list = []
            for training_idx, training_dataset in enumerate(testing_datasets):
                result_dict_similar_runs = {k: v for k, v in result_dict.items() if
                                            f"{testing_dataset}_{training_dataset}" in k}
                metric_data_df = pd.concat([df.loc[75:, metric] for df in result_dict_similar_runs.values()]
                                           , axis=1).transpose()
                # Convert recall values to percent
                if "recall" in metric:
                    metric_data_df *= 100

                means[metric_idx, testing_idx, training_idx] = np.mean(metric_data_df.values)
                stds[metric_idx, testing_idx, training_idx] = np.std(metric_data_df.values)

                boxprops = dict(linestyle='--', linewidth=2, color=facecolor[training_idx])
                boxplot_list.append(ax[testing_idx].boxplot(metric_data_df,
                                                            labels=metric_data_df.columns,
                                                            boxprops=boxprops,
                                                            positions=metric_data_df.columns.values +
                                                                      positions[training_idx],
                                                            widths=0.1,
                                                            showmeans=True))

                plot_opts = dict(violin_width=0.1, bean_show_median=False, bean_color='black', jitter_fc='black',
                                 bean_mean_color='black', bean_mean_size=0.15,
                                 violin_fc=facecolor[training_idx])
                if sys.argv[1].split('/')[-1] == "pointrcnn":
                    beanplot_single_list.append(sm.graphics.beanplot(data=np.swapaxes(metric_data_df.values, 0, 1), ax=ax2,
                                                                     labels=[
                                                                         ["Real", "Sim", "Sim Noise", "Sim downsampled"][
                                                                             training_idx]],
                                                                     plot_opts=plot_opts,
                                                                     positions=[1 + testing_idx + positions[training_idx]],
                                                                     jitter=True))
            ticks = metric_data_df.columns
            ax[testing_idx].set_title(f"Testing dataset: {testing_dataset}")
            ax[testing_idx].set_xticks(ticks)
            ax[testing_idx].legend([boxplot["boxes"][0] for boxplot in boxplot_list], testing_datasets,
                                   title="Training dataset")
            ax[testing_idx].set_ylim([0, y_scales[metric[:6]]])
            ax[testing_idx].grid(axis='y')
            with open(means_csv, "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(means[metric_idx, testing_idx, :])
            with open(stds_csv, "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(stds[metric_idx, testing_idx, :])

        # plot boxplots in single plot for each of the 11 metrics [4(training)*4(testing) boxes per plot]
        font = {'family': 'serif',
                'color': 'black',
                'weight': 'bold',
                'size': 16,
                }

        tick_labels = ["Real", "Sim", "Sim Noise", "Sim downsampled"]
        ticks = list(range(1, 5))
        ax2.set_xticks(ticks, labels=tick_labels)
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), facecolor.values()):
            ticklabel.set_color(tickcolor)

        # ax2.legend([boxplot["boxes"][0] for boxplot in beanplot_single_list], tick_labela,
        #                       title="Training dataset")
        ax2.legend(ax2.collections[::2], tick_labels, title="Training dataset", loc="lower right")
        ax2.set_ylim([0, y_scales[metric[:6]]])
        ax2.set_xlabel("Testing dataset", labelpad=20, fontdict=font)
        ax2.set_ylabel("AP in %", fontdict=font)
        ax2.grid(axis='y')
        ax2.set_xlim([0.5, 4.5])
        [ax2.axvline(x, color='k', linestyle='--') for x in [1.5, 2.5, 3.5]]

    relevant_means = means[list(relevant_metrics.values())]
    relevant_stds = stds[list(relevant_metrics.values())]
    with open(full_csv, "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        for training_idx, training_dataset in enumerate(testing_datasets):
            for testing_idx, testing_dataset in enumerate(testing_datasets):
                # csv_writer.writerow(",".join([f"{round(el[0], 2)} ({round(el[1], 2)})" for el in
                #                              zip(relevant_means[:, testing_idx, training_idx],
                #                                  relevant_stds[:, testing_idx, training_idx])]))
                csv_writer.writerow([f"{round(el[0], 2)} ({round(el[1], 2)})" for el in
                                     zip(relevant_means[:, testing_idx, training_idx],
                                         relevant_stds[:, testing_idx, training_idx])])
    plt.show()
    print()


def main(training_dataset_runs):
    result_dict, testing_datasets = create_result_dict(training_dataset_runs)
    merge_runs(result_dict, testing_datasets)
    print()


if __name__ == "__main__":
    network_path = sys.argv[1]
    training_dataset_runs = sorted(next(os.walk(network_path))[1])

    main(training_dataset_runs)
