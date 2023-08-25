import os
import csv


def extract_data():
    # Iterate over training datasets
    for training_dataset in training_datasets:
        # Iterate over evaluation ranges
        for eval_range in evaluation_ranges:
            # Iterate over evaluation datasets
            for eval_dataset in evaluation_datasets:
                # Construct the evaluation range folder name
                eval_range_folder = f"{eval_dataset}_{eval_range}"

                # Define the path to the evaluation folder for the current evaluation dataset
                eval_folder = os.path.join(root_dir, network, training_dataset, "eval", "eval_all_default", eval_range_folder)

                # Check if it's a directory
                if os.path.isdir(eval_folder):
                    # Find the results.csv file
                    results_csv_path = os.path.join(eval_folder, "results.csv")

                    # Check if results.csv exists
                    if os.path.exists(results_csv_path):
                        # Read the CSV file and extract the desired value
                        with open(results_csv_path, "r") as csv_file:
                            csv_reader = csv.reader(csv_file)

                            # Skip the header line
                            next(csv_reader)

                            values_list = []
                            for row in csv_reader:
                                epoch, values, _ = row
                                values_list.append(float(values.split(",")[3]))  # Extract the desired value

                                # Maintain a list with the last 'n' values
                                if len(values_list) > last_n_epochs:
                                    values_list.pop(0)

                            # Calculate the average of the last 'n' values
                            average_value = sum(values_list) / len(values_list)
                            max_value = max(values_list)

                            # Store the extracted value along with information about the dataset, eval range folder, and network
                            key = f"{network}_{training_dataset}_{eval_range_folder}"
                            extracted_data[key] = average_value

    # Create a directory to store CSV files
    output_dir = "output_csv"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Create a CSV file for the current evaluation range
        output_csv_path = os.path.join(output_dir, f"extracted_data_{eval_range}.csv")

        # Open the CSV file for writing
        with open(output_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the data rows without the first column (training dataset)
            for training_dataset in training_datasets:
                data_row = []
                for eval_dataset in evaluation_datasets:
                    key = f"{network}_{training_dataset}_{eval_dataset}_{eval_range}"
                    data_row.append(extracted_data.get(key, ""))
                csv_writer.writerow(data_row)

    print(f"Extracted data saved to {output_dir}")


def plot_results():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Define the specified bar colors
    bar_colors = {
        0: tuple(np.asarray((190, 0, 0)) / 255),
        1: tuple(np.asarray((190, 90, 90)) / 255),
        2: tuple(np.asarray((222, 235, 247)) / 255),
        3: tuple(np.asarray((200, 235, 247)) / 255),
        4: tuple(np.asarray((178, 202, 225)) / 255),
        5: tuple(np.asarray((158, 202, 225)) / 255),
        6: tuple(np.asarray((138, 202, 225)) / 255),
        7: tuple(np.asarray((49, 130, 189)) / 255),
    }

    for eval_range in evaluation_ranges:
        # Load the data from the CSV file without headers (adjust the file path as needed)
        data = pd.read_csv(f"output_csv/extracted_data_{eval_range}.csv", header=None)

        # Set column labels
        col_labels = evaluation_datasets

        # Set the extracted values as a matrix
        data_matrix = data.values

        # Create a DataFrame with column labels
        df = pd.DataFrame(data_matrix, columns=col_labels)

        # Transpose the DataFrame to have training datasets as columns and evaluation datasets as rows
        df_transposed = df.transpose()

        # Set the column names of df_transposed to the training dataset names
        df_transposed.columns = training_datasets

        # Create a grouped bar plot with adjusted bar width and spacing
        plt.figure(figsize=(12, 6))
        bar_width = 0.1
        index = np.arange(len(evaluation_datasets))

        for i, training_dataset in enumerate(df_transposed.columns):
            plt.bar(
                index + i * bar_width,
                df_transposed[training_dataset],
                bar_width,
                label=f'{training_dataset} (Training)',
                color=bar_colors[i],  # Set the specified bar color
                zorder = 2
            )

        plt.xlabel("Evaluation Dataset")
        plt.ylabel("3D AP (0.7) in %")
        plt.title(f"3D AP (0.7) for Evaluation Range {eval_range}")
        plt.xticks(index + bar_width * (len(df_transposed.columns) - 1) / 2, evaluation_datasets, rotation=45)

        # Add a legend that includes both evaluation and training dataset names
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), labels=[f'{training_dataset}' for training_dataset in df_transposed.columns], title="Training Dataset")

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
        plt.savefig(f"plot_{eval_range}.png")

    # Show the plot (optional)
    plt.show()
    print()


if __name__ == "__main__":
    # Define the list of training dataset names
    training_datasets = ["real", "real2sim8", "sim_noise", "sim_noise_obj", "retrieved_512", "sim2real6", "sim2real6_noise", "sim"]

    # Define the list of evaluation ranges
    evaluation_ranges = ["0_100", "0_33", "33_66", "66_100"]

    # Define the list of evaluation datasets
    evaluation_datasets = ["real", "sim"]

    # Select the network (pointpillar or pointrcnn)
    network = "pointrcnn"

    # Create a dictionary to store the extracted values
    extracted_data = {}

    # Define the root directory
    root_dir = "/mnt/ge75huw/01_Trainings/01_Sim2Real_OpenPCDet/indy/indy_models"

    # Number of last epochs to consider for averaging
    last_n_epochs = 5  # Change this as needed

    extract_data()

    plot_results()
