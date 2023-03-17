import os
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import statsmodels.api as sm

plt.rcParams["font.size"] = "28"
rc('axes', linewidth=2)
plt.rcParams["font.family"] = ["Times New Roman"]

plt.show()


def read_csv(training_dataset_run, testing_dataset):
    return pd.read_csv(os.path.join(network_path, training_dataset_run, "eval/eval_all_default", testing_dataset,
                                    "results.csv"), sep=",", index_col="epoch_id")


def create_result_dict(training_dataset_runs, ranges):
    result_dict = {}
    testing_datasets = []
    training_datasets = []
    for training_dataset_run in training_dataset_runs:
        training_datasets.append(training_dataset_run[:-2])
        if ranges:
            testing_datasets.append(training_dataset_run[:-2] + "_" + str(ranges))
        else:
            testing_datasets.append(training_dataset_run[:-2])

    training_datasets = sorted(list(set(training_datasets)))
    testing_datasets = sorted(list(set(testing_datasets)))

    for training_dataset_run in training_dataset_runs:
        for testing_dataset in testing_datasets:
            result_dict[f"{testing_dataset}_{training_dataset_run}"] = read_csv(training_dataset_run, testing_dataset)
    return result_dict, testing_datasets, training_datasets


def merge_runs(result_dict, testing_datasets, training_datasets, ranges, plot, sim_only):
    metrics = ["AP_Car_3d/0.5_R11", "AP_Car_3d/0.5_R40", "AP_Car_3d/0.7_R11", "AP_Car_3d/0.7_R40", "recall/rcnn_0.3",
               "recall/rcnn_0.5", "recall/rcnn_0.7", "recall/roi_0.3", "recall/roi_0.5", "recall/roi_0.7",
               "avg_pred_obj"]
    relevant_metrics = {"AP_Car_3d/0.5_R40": 0, "AP_Car_3d/0.7_R40": 0, "recall/rcnn_0.5": 0, "recall/rcnn_0.7": 0}
    y_scales = {"AP_Car": 100, "recall": 100, "avg_pr": 1}
    means = np.zeros((len(metrics), len(testing_datasets), len(testing_datasets)))
    stds = np.zeros((len(metrics), len(testing_datasets), len(testing_datasets)))
    if ranges:
        full_csv = os.path.join(sys.argv[1], f"{sys.argv[1].split('/')[-1]}_{str(ranges)}.csv")
    else:
        if sim_only:
            full_csv = os.path.join(sys.argv[1], f"{sys.argv[1].split('/')[-1]}.csv")
        else:
            full_csv = os.path.join(sys.argv[1], f"{sys.argv[1].split('/')[-1]}_all.csv")
    if os.path.isfile(full_csv):
        os.remove(full_csv)
    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(4, 1, figsize=(7.5, 9))
        fig.canvas.manager.set_window_title(metric)

        if sim_only:
            fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 9))
            ax2.set_position((0.15, 0.125, 0.8, 0.85))
            fontsize = 28
            plt.rcParams["font.size"] = str(fontsize)
        else:
            fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
            ax2.set_position((0.05, 0.15, 0.925, 0.825))
            fontsize = 16
            plt.rcParams["font.size"] = str(fontsize)
        fig2.canvas.manager.set_window_title(metric)

        facecolor = {0: tuple(np.asarray((190,0,0))/255),
                     1: tuple(np.asarray((49,130,189))/255),
                     2: tuple(np.asarray((158,202,225))/255),
                     3: tuple(np.asarray((222,235,247))/255)}

        if sim_only:
            labels = ["Real", "Sim"]
            positions = {0: -0.3, 1: 0.3}
        else:
            labels = ["Real", "Sim", "Sim Noise", "Sim Downsampled"]
            positions = {0: -0.3, 1: -0.1, 2: 0.1, 3: 0.3}

        if ranges:
            means_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_means_{str(ranges)}.csv")
            stds_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_stds_{str(ranges)}.csv")
        else:
            if sim_only:
                means_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_means.csv")
                stds_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_stds.csv")
            else:
                means_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_means_all.csv")
                stds_csv = os.path.join(sys.argv[1], f"{metric.replace('/', '_').replace('.', '_')}_stds_all.csv")
        if os.path.isfile(means_csv):
            os.remove(means_csv)
        if os.path.isfile(stds_csv):
            os.remove(stds_csv)

        if metric in relevant_metrics.keys():
            relevant_metrics[metric] = metric_idx

        for testing_idx, testing_dataset in enumerate(testing_datasets):
            boxplot_list = []
            beanplot_single_list = []
            for training_idx, training_dataset in enumerate(training_datasets):
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
                                 violin_fc=facecolor[training_idx], violin_alpha=1.0)
                if sys.argv[1].split('/')[-1] == "pointrcnn":
                    beanplot_single_list.append(sm.graphics.beanplot(data=np.swapaxes(metric_data_df.values, 0, 1), ax=ax2,
                                                                     labels=[labels[training_idx]],
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
        font = {'family': ['Times New Roman'],
                'color': 'black',
                'size': fontsize,
                }

        tick_labels = labels
        ticks = list(range(1, len(labels)+1))
        ax2.set_xticks(ticks, labels=tick_labels)
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), facecolor.values()):
            if ticklabel._text == "Sim" or ticklabel._text == "Sim Noise" or ticklabel._text == "Sim Downsampled":
                ticklabel.set_color(facecolor[1])
            else:
                ticklabel.set_color(tickcolor)

        ax2.tick_params(axis='x', which='major', pad=10)

        # ax2.legend([boxplot["boxes"][0] for boxplot in beanplot_single_list], tick_labela,
        #                       title="Training dataset")

        if ranges == "066_100":
            legend_loc = "upper left"
        else:
            legend_loc = "lower right"

        if sim_only:
            ax2.legend(ax2.collections[::2], tick_labels, title="Training\n dataset", loc=legend_loc, frameon=True)
        else:
            ax2.legend(ax2.collections[::2], tick_labels, title="Training dataset", loc=legend_loc, frameon=True)
        ax2.set_ylim([0, y_scales[metric[:6]]])
        ax2.set_xlabel("Testing dataset", labelpad=15, fontdict=font)
        ax2.set_ylabel("3D AP (0.7) in %", fontdict=font)
        ax2.grid(axis='y')
        if sim_only:
            ax2.set_xlim([0.5, 2.5])
        else:
            ax2.set_xlim([0.5, 4.5])
        [ax2.axvline(x, color='k', linestyle='--') for x in [1.5, 2.5, 3.5]]

        if ranges:
            fig_name = f"{metric.replace('/', '_').replace('.', '_')}_{str(ranges)}.pdf"
        else:
            if sim_only:
                fig_name = f"{metric.replace('/', '_').replace('.', '_')}.pdf"
            else:
                fig_name = f"{metric.replace('/', '_').replace('.', '_')}_all.pdf"
        fig_path = os.path.join(sys.argv[1], fig_name)
        fig2.savefig(fig_path)

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
    if plot:
        plt.show()


def main(training_dataset_runs, ranges, plot, sim_only):
    result_dict, testing_datasets, training_datasets = create_result_dict(training_dataset_runs, ranges)
    merge_runs(result_dict, testing_datasets, training_datasets, ranges, plot, sim_only)
    print()


if __name__ == "__main__":
    network_path = sys.argv[1]
    plot = False  # activate or suppress plot
    sim_only = False  # True: real-sim, False: real-sim-sim_noise-sim_downsampled

    if sim_only:
        range_secs = ["", "000_033", "033_066", "066_100"]
    else:
        range_secs = [""]

    for ranges in range_secs:  # "", "000_033", "033_066", "066_100"
        training_dataset_runs = sorted(next(os.walk(network_path))[1])
        if sim_only:
            training_dataset_runs = training_dataset_runs[:10]

        main(training_dataset_runs, ranges, plot, sim_only)
