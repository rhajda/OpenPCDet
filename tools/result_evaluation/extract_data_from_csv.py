import os
import csv
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Import mpatches
from sklearn.manifold import TSNE


def extract_data(runs):
    # Create a directory to store CSV files
    output_dir = "output_csv"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a dict to store max epochs
    epochs = {}

    # Iterate over training datasets and runs
    for run in runs:
        for training_dataset in training_datasets:
            # Iterate over evaluation ranges
            for eval_range in evaluation_ranges:
                # Iterate over evaluation datasets
                for eval_dataset in evaluation_datasets:
                    # Construct the evaluation range folder name
                    eval_range_folder = f"{eval_dataset}_{eval_range}"

                    # Define the path to the evaluation folder for the current evaluation dataset
                    eval_folder = os.path.join(root_dir, network, f"{training_dataset}_{run}", "eval", "eval_all_default", eval_range_folder)

                    # Check if it's a directory
                    if os.path.isdir(eval_folder):
                        # Find the results.csv file
                        results_csv_path = os.path.join(eval_folder, "results.csv")

                        # Check if results.csv exists
                        if os.path.exists(results_csv_path):
                            # Read the CSV file and extract the desired value and epoch
                            with open(results_csv_path, "r") as csv_file:
                                csv_reader = csv.reader(csv_file)

                                # Skip the header line
                                next(csv_reader)

                                values_list = []
                                epochs_list = []

                                for row in csv_reader:
                                    epoch, values, _ = row
                                    epoch = int(epoch)
                                    value = float(values.split(",")[3])  # Extract the desired value

                                    # Append the value and epoch to the lists
                                    values_list.append(value)
                                    epochs_list.append(epoch)

                                # Calculate the average, last_value, max_value, and median_value based on last_n_epochs
                                values_list_last_n = values_list[-last_n_epochs:]

                                average_value = sum(values_list_last_n) / len(values_list_last_n)
                                last_value = values_list_last_n[-1]
                                max_value = max(values_list_last_n)
                                median_value = np.median(values_list_last_n)

                                values = {
                                    "avg": average_value,
                                    "max": max_value,
                                    "last": last_value,
                                    "median": median_value
                                }

                                # Store the extracted value along with information about the dataset, eval range folder, and network
                                key = f"{network}_{training_dataset}_{run}_{eval_range_folder}"
                                extracted_data[key] = values[metric]
                                extracted_data_list[key] = values_list_last_n  # Store the last_n_epochs values list

                                # Store the epoch separately
                                epochs[key] = epochs_list[-last_n_epochs:][np.asarray(extracted_data_list[key][-last_n_epochs:]).argmax()] if metric == "max" else int(epoch)

        # Iterate over evaluation ranges
        for eval_range in evaluation_ranges:
            # Create a CSV file for the current evaluation range (extracted data)
            output_csv_path = os.path.join(output_dir, f"extracted_data_{eval_range}_{run}.csv")

            # Create a CSV file for the current evaluation range (epochs)
            epochs_output_csv_path = os.path.join(output_dir, f"epochs_{eval_range}_{run}.csv")

            # Open the CSV files for writing
            with open(output_csv_path, "w", newline="") as csv_file, open(epochs_output_csv_path, "w",
                                                                          newline="") as epochs_csv_file:
                csv_writer = csv.writer(csv_file)
                epochs_csv_writer = csv.writer(epochs_csv_file)

                # Write the data rows without the first column (training dataset)
                for training_dataset in training_datasets:
                    data_row = []
                    epochs_row = []  # Create a row for epochs

                    for eval_dataset in evaluation_datasets:
                        key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"
                        data_row.append(extracted_data.get(key, ""))
                        epoch = epochs.get(key, "")
                        epochs_row.append(epoch)

                    # Write the rows to the respective CSV files
                    csv_writer.writerow(data_row)
                    epochs_csv_writer.writerow(epochs_row)

    print(f"Extracted data and epoch data saved to {output_dir}")


# Define a function to extract data based on epoch information
def extract_glob_feats(runs):
    # Initialize a dictionary to store extracted data
    extracted_glob_feats = {}

    # Define the output directory for CSVs
    output_dir = "output_csv"

    # Iterate over training datasets
    for training_dataset in training_datasets:
        # Iterate over evaluation ranges
        for eval_range in evaluation_ranges:
            # Iterate over evaluation datasets
            for eval_dataset in evaluation_datasets:
                # Construct the evaluation range folder name
                eval_range_folder = f"{eval_dataset}_{eval_range}"

                # Iterate over the specified runs
                for run in runs:
                    # Define the path to the epochs CSV file for the current run
                    epochs_csv_path = os.path.join(output_dir, f"epochs_{eval_range}_{run}.csv")

                    # Check if the epochs CSV file exists
                    if os.path.exists(epochs_csv_path):
                        # Read the epochs CSV file
                        with open(epochs_csv_path, "r") as epochs_csv_file:
                            epochs_csv_reader = csv.reader(epochs_csv_file)

                            for idx, row in enumerate(epochs_csv_reader):
                                if idx == training_datasets.index(training_dataset):
                                    epoch = row[evaluation_datasets.index(eval_dataset)]  # Get the epoch for this dataset
                                    break

                            # Define the folder for the current epoch and run
                            epoch_folder = os.path.join(root_dir, network, f"{training_dataset}_{run}", "eval", "eval_all_default", eval_range_folder, f"epoch_{epoch}")

                            # Check if the epoch folder exists
                            if os.path.exists(epoch_folder):
                                # Define the path to the pkl file
                                pkl_file_path = os.path.join(epoch_folder, "test", f"glob_feat_{epoch}.pkl")

                                # Check if the pkl file exists
                                if os.path.exists(pkl_file_path):
                                    # Load data from the pkl file
                                    with open(pkl_file_path, "rb") as pkl_file:
                                        data = pickle.load(pkl_file)

                                    # Create a key based on dataset, training_dataset, run, and eval_range
                                    key = f"{network}_{training_dataset}_{run}_{eval_range_folder}"

                                    extracted_glob_feats[key] = data

    return extracted_glob_feats


def define_colors():
    # Define the specified colors
    colors = dict()
    for i in range(num_real):
        colors[i] = tuple(np.asarray((190, (255/num_real) * i / 2, 0)) / 255)
    for i in range(num_sim):
        colors[num_sim - i + num_real - 1] = tuple(np.asarray(((255/num_sim) * i, 202, 225)) / 255)
    # sort dict by key
    colors = dict(sorted(colors.items()))
    return colors


def plot_results(colors, runs):
    # Create a directory to store figures
    figs_dir = "output_figs"
    os.makedirs(figs_dir, exist_ok=True)

    for eval_range in evaluation_ranges:
        # Load data from CSV files for the specified runs
        data_list = []
        for run in runs:
            data = pd.read_csv(f"output_csv/extracted_data_{eval_range}_{run}.csv", header=None)
            data_list.append(data)

        # Concatenate data from multiple runs
        combined_data = np.asarray(data_list)

        # Calculate mean and standard deviation for each evaluation dataset
        means = combined_data.mean(axis=0)
        stds = combined_data.std(axis=0)

        # Create a DataFrame for mean and standard deviation data
        mean_df = pd.DataFrame(means)
        std_df = pd.DataFrame(stds)

        # Transpose DataFrames for plotting
        mean_df = mean_df.transpose()
        std_df = std_df.transpose()

        # Set the column names of mean_df and std_df to the training dataset names
        mean_df.columns = training_datasets
        std_df.columns = training_datasets

        # Create a grouped bar plot with adjusted bar width and spacing
        plt.figure(figsize=(24, 12))
        bar_width = 0.2
        num_training_datasets = len(training_datasets)
        index = np.arange(len(evaluation_datasets)) * (num_training_datasets * bar_width + bar_width)

        for i, training_dataset in enumerate(mean_df.columns):
            plt.bar(
                index + i * bar_width,
                mean_df[training_dataset],
                bar_width,
                yerr=std_df[training_dataset],  # Include error bars for standard deviation
                label=f'{training_dataset} (Evaluation)',
                color=colors[i],  # Set the specified bar color
                zorder=2
            )

            # Add the values above each bar
            for j, value in enumerate(mean_df[training_dataset]):
                plt.text(
                    index[j] + i * bar_width,
                    value + 1,  # Adjust the vertical position as needed
                    f"{value:.2f}",  # Format the value with 2 decimal places
                    ha='center',  # Center the text
                    fontsize=8  # Adjust the font size as needed
                )

                # Add the name of each training dataset vertically inside each bar
                plt.text(
                    index[j] + i * bar_width,
                    1,  # Place vertically at the bottom of the bar
                    f'{training_dataset}',  # Training dataset name
                    va='bottom',  # Align the text at the bottom
                    ha='center',  # Center the text
                    fontsize=8,  # Adjust the font size as needed
                    rotation='vertical'  # Rotate the text vertically
                )

        plt.xlabel("Evaluation Dataset")
        plt.ylabel("3D AP (0.7) in %")
        plt.title(f"3D AP (0.7) for Evaluation Range {eval_range} - Metric: {metric.capitalize()} of last {last_n_epochs} epochs - Runs: {runs}")
        plt.xticks(index + bar_width * (len(mean_df.columns) - 1) / 2, evaluation_datasets, rotation=45)

        # Add a legend that includes both evaluation and training dataset names
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), labels=[f'{training_dataset}' for training_dataset in mean_df.columns], title="Training Dataset")

        # Insert horizontal grid lines at every 10th value on the y-axis
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(axis='y', linestyle='--', which='major', color='gray', linewidth=0.5)

        # Add minor horizontal grid lines without labels every 1 value
        plt.minorticks_on()
        plt.grid(axis='y', linestyle=':', which='minor', color='gray', linewidth=0.5, alpha=0.4)

        # Set y-axis limit from 0 to 100
        plt.ylim(0, 100)

        plt.tight_layout()  # Ensure tight layout to prevent overlapping

        # Save the plot as an image (optional)
        plt.savefig(os.path.join(figs_dir, f"plot_barplot_{eval_range}.png"))

        ################################################################################################################
        ################################################## BOX PLOTS ###################################################
        ################################################################################################################

        # Create a new figure with a single subplot for both boxplots
        fig, axs = plt.subplots(1, len(evaluation_datasets), figsize=(24, 12))

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2)  # Adjust the value as needed

        # Iterate over evaluation datasets
        for i, eval_dataset in enumerate(evaluation_datasets):
            # Get the data for the current evaluation dataset from combined_data
            data_by_eval_dataset = combined_data[:, :, i]

            # Create boxplots for the current evaluation dataset on the appropriate subplot
            boxplot = axs[i].boxplot(data_by_eval_dataset, vert=True, patch_artist=True)

            # Set the x-ticks positions and labels
            x_ticks_positions = np.arange(1, len(training_datasets) + 1)
            x_ticks_labels = training_datasets

            axs[i].set_xticks(x_ticks_positions)
            axs[i].set_xticklabels(x_ticks_labels, rotation=30)
            axs[i].tick_params(axis='x', which='both', bottom=True, top=False)
            axs[i].tick_params(axis='x', which='minor', bottom=False)

            # Set the x-axis label
            axs[i].set_xlabel("Training Dataset", labelpad=10)

            axs[i].set_ylabel("3D AP (0.7) in %")
            axs[i].set_title(f"Evaluation Dataset: {eval_dataset}")
            axs[i].set_ylim(0, 100)

            # Apply the same face colors as the bar plot
            for patch, color in zip(boxplot['boxes'], colors.values()):
                patch.set_facecolor(color)

            # Add horizontal lines at every 10th value on the y-axis
            axs[i].set_yticks(np.arange(0, 101, 10))
            axs[i].grid(axis='y', linestyle='--', which='major', color='gray', linewidth=0.5)

            # Add minor horizontal grid lines without labels every 1 value
            axs[i].minorticks_on()
            axs[i].grid(axis='y', linestyle=':', which='minor', color='gray', linewidth=0.5, alpha=0.4)

            # Calculate and display the median next to each boxplot
            for j, box in enumerate(boxplot['boxes']):
                data = data_by_eval_dataset[:, j]
                median_value = np.median(data)
                axs[i].text(j + 1, median_value + 0.1, f"{median_value:.2f}", ha='center', fontsize=8)

        # Create a custom legend for training datasets outside of the subplots
        legend_patches = [mpatches.Patch(color=color, label=training_dataset) for training_dataset, color in
                          colors.items()]

        # Include the evaluation range in the title of the figure
        fig.suptitle(f"Boxplots for runs {runs} in evaluation range {eval_range} for metric '{metric}' calculated from last {last_n_epochs} epochs", fontsize=16)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Add the custom legend for training datasets
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 0.5), title="Training Dataset")

        # Save the plot as an image (optional)
        plt.savefig(os.path.join(figs_dir, f"plot_boxplot_{eval_range}.png"))

    # Show the plot (optional)
    plt.show()


def plot_tsne(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=30):
    # Create a directory to store figures
    figs_dir = "output_figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Iterate over evaluation datasets
        for eval_dataset in evaluation_datasets:
            # Create a sub-plot for the current evaluation range and dataset
            plt.figure(figsize=(12, 8))
            plt.title(f"T-SNE Plot for Evaluation Range {eval_range} - Dataset {eval_dataset}")

            # Create empty arrays to store t-SNE points and labels
            tsne_points = np.array([])
            tsne_labels = []

            # Iterate over training datasets
            for training_dataset in training_datasets:
                for run in runs:
                    # Define the key for extracted_glob_feats
                    key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"

                    # Check if the key exists in extracted_glob_feats
                    if key in extracted_glob_feats:
                        data = extracted_glob_feats[key]

                        # Convert data to a NumPy array
                        data = np.array(data)

                        # Append the t-SNE points to the array
                        if tsne_points.size == 0:
                            tsne_points = data
                        else:
                            tsne_points = np.vstack((tsne_points, data))

                        # Create labels for the current training dataset
                        labels = np.full(data.shape[0], training_dataset)

                        # Append the labels to tsne_labels
                        tsne_labels.extend(labels)

            # Use t-SNE to reduce the dimensionality to 2
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity, method="barnes_hut", verbose=True, n_iter=1000)
            tsne_result = tsne.fit_transform(tsne_points)

            # Scatter plot the t-SNE points with labels
            unique_labels = np.asarray(training_datasets)

            for i, label in enumerate(unique_labels):
                x = tsne_result[np.array(tsne_labels) == label][:, 0]
                y = tsne_result[np.array(tsne_labels) == label][:, 1]
                scat = plt.scatter(x, y, color=colors[i], label=label)
                scat.set_rasterized(True)

                for run_idx, run in enumerate(runs):
                    # Calculate the center of the cluster
                    cluster_center_x = np.median(x[int(run_idx * len(x) / len(runs)) : int((run_idx + 1) * len(x) / len(runs))])
                    cluster_center_y = np.median(y[int(run_idx * len(y) / len(runs)) : int((run_idx + 1) * len(y) / len(runs))])

                    # Add label as text at the cluster center
                    plt.text(cluster_center_x, cluster_center_y, f"{label}_{run}", fontsize=12, ha='center', va='center')

            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend()
            plt.grid(True)

            # Save the plot as an image (optional)
            plt.savefig(os.path.join(figs_dir, f"plot_tsne_{eval_dataset}_{eval_range}_{perplexity}.png"))

    # Show or save the plot
    plt.show()  # You can save the plot to a file using plt.savefig() if needed

def main():
    extract_data(runs)

    extracted_glob_feats = extract_glob_feats(runs)

    colors = define_colors()

    # Plot bar charts and box plots
    plot_results(colors, runs)

    # Call the function with your extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, and optional perplexity
    plot_tsne(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=3)

    print()


if __name__ == "__main__":
    # Define the list of training dataset names
    training_datasets = ["real_", "real2sim20_", "sim_noise_", "sim_noise_obj_", "sim2real20_", "sim2real21_", "sim_"]
    num_real = 2
    num_sim = 5

    runs = [1, 2, 3]

    # Define the list of evaluation ranges
    evaluation_ranges = ["0_33", "33_66", "66_100", "0_100"]

    # Define the list of evaluation datasets
    evaluation_datasets = ["real", "sim"]

    # Select the network (pointpillar or pointrcnn)
    network = "pointrcnn"

    # Create a dictionary to store the extracted values
    extracted_data = {}
    extracted_data_list = {}

    # Define the root directory
    root_dir = "/mnt/ge75huw/01_Trainings/01_Sim2Real_OpenPCDet/indy/indy_models"

    # Number of last epochs to consider for averaging/maxing
    last_n_epochs = 10  # Change this as needed

    # avg, max, last, median
    metric = "max"

    main()
