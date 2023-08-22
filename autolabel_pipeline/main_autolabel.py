
# Import libraries
from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import pathlib
import shutil
import yaml
import os


from base_functions import load_config, PathManager, autolabel_path_manager
from visualize_pcds import get_file_names, visualize_single_pcd
from voting_schemes import nms_voting, majority_voting


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


"""

FILE DESCRIPTION: 

This file is the main file of the autolabel pipeline. It serves as a central script, triggering respective functions of
the pipeline. All parameters are configurable in autolabel.yaml. 
If not triggered in pipeline, to trigger: main_pseudo_label(cfg, BATCH_SIZE_VOTING, START_AT_CHECKPOINT, START_FRAME)
BATCH_SIZE_VOTING: Number of frames processed before user input to continue. 
START_AT_CHECKPOINT: If All frames should be processed, set to False. To process form START_FRAME set to True. 
START_FRAME:  If not all frames should be processed, start at frame N.


Important: 
dataframe structure: ['ID', 'label', 'loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z', 'score']

voting schemes: 
    - non maximum suppression + confidence
    - majority voting

"""



# Function loads csv data of a specific frame to a dataframe containing all bboxes
def csv_to_dataframe(path_data, file_name):

    columns = ['ID', 'label', 'loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z', 'score']

    # Load data from csv to a dataframe.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="genfromtxt: Empty input file")
        my_file = np.genfromtxt(path_data, delimiter=',', dtype=str)

    if np.size(my_file) == 0:
        return pd.DataFrame(columns=columns)

    # Reshape data to a two-dimensional array if it is a single-dimensional array
    if my_file.ndim == 1:
        my_file = my_file.reshape(1, -1)

    df = pd.DataFrame(columns=columns)
    for row in my_file:
        row_data = [file_name] + list(row)
        df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

    df[df.columns[2:]] = df[df.columns[2:]].astype(float)

    return df

# Function that loads the bboxes predicted for a frame to dataframe for every model.
def load_frame(frame_ID, path_frames):

    file_path = os.path.join(path_frames, frame_ID + ".csv")
    df_frame_ID = csv_to_dataframe(file_path, frame_ID)

    return df_frame_ID

# Function that outputs the set of common and unique frames for n folders.
def common_set_between_datasets(cfg, folders, FLAG_GENERATE_EMPTY_MISSING_FILES):
    common_set = set()
    unique_files = set()

    for folder in folders:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_set = set([os.path.splitext(f)[0] for f in files])
        if common_set:
            common_set = common_set.intersection(file_set)
        else:
            common_set = file_set
        unique_files.update(file_set)
        if cfg.PIPELINE.PRINT_INFORMATION:
            print("Loaded: ", folder, "; Frames: ", len(files))

    unique_files -= common_set
    common_array = sorted(map(int, common_set))
    uniques_array = sorted(map(int, unique_files))

    if cfg.PIPELINE.PRINT_INFORMATION:
        print("Common set origin: ", len(common_array))

    if FLAG_GENERATE_EMPTY_MISSING_FILES:
        if len(uniques_array) > 0:
            # add empty csv files in the folder where no detections where predicted.
            add_empty_prediction_files(uniques_array, folders)
            if cfg.PIPELINE.PRINT_INFORMATION:
                print("Empty frames added in corresponding folders for IDs: ", uniques_array)

        combined_array = sorted(common_array + uniques_array)
        if len(combined_array) == 0:
            raise ValueError("No frames in the set.")

        if cfg.PIPELINE.PRINT_INFORMATION:
            print("Combined set: ", len(combined_array))

        return combined_array

    return common_array

# Function that generated empty csv files to fill null predictions.
def add_empty_prediction_files(unique_frames, folders):

    for frame in unique_frames:
        file_name = str(frame).zfill(6) + '.csv'
        for folder in folders:
            file_path = os.path.join(folder, file_name)
            if not os.path.exists(file_path):
                empty_array = np.empty((0,))
                np.savetxt(file_path, empty_array, delimiter=',', fmt='%s')

# Function that manages NMS or MAJORITY voting and feeds data to the voting schemes
def vote_pseudo_labels(cfg, path_manager, frame_set, batch_size_voting):

    show_progress_bar = not cfg.PIPELINE.PRINT_INFORMATION
    counter = 0
    counter_total_frames_processed = 0
    description = "Voting pseudo-labels (" + str(cfg.PIPELINE.VOTING_SCHEME) + ")"

    # Define the paths to save the metrics evaluating the voting schemes.
    if cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
        if not os.path.exists(path_manager.get_path("path_pseudo_labels_majority")):
            os.makedirs(path_manager.get_path("path_pseudo_labels_majority"))
        if cfg.PIPELINE.COMPUTE_EVALUATION_METRICS:
            path_metrics_pseudo_label = os.path.join(path_manager.get_path("path_pseudo_labels_majority"),"metrics_majority")

    elif cfg.PIPELINE.VOTING_SCHEME == 'NMS':
        if not os.path.exists(path_manager.get_path("path_pseudo_labels_nms")):
            os.makedirs(path_manager.get_path("path_pseudo_labels_nms"))
        if cfg.PIPELINE.COMPUTE_EVALUATION_METRICS:
            path_metrics_pseudo_label = os.path.join(path_manager.get_path("path_pseudo_labels_nms"),"metrics_nms")

    else:
        raise ValueError("No valid VOTING_SCHEME selected.")

        # Remove old voting evaluation files.
        reset_voting_metrics(path_metrics_pseudo_label)

    for frame in tqdm(frame_set, desc=description, unit="frame", disable=not show_progress_bar):

        frame_ID = str(frame).zfill(6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load all predicted frame IDs (ptrcnn & ptpillar & second) keep cfg.DATA.PROJECT.MODELS sequence.
            df_ptrcnn = load_frame(frame_ID, path_manager.get_path("path_ptrcnn_predictions"))
            df_ptpillar = load_frame(frame_ID, path_manager.get_path("path_ptpillar_predictions"))
            df_second = load_frame(frame_ID, path_manager.get_path("path_second_predictions"))

        if cfg.PIPELINE.PRINT_INFORMATION:
            print("df_ptrcnn: ", "\n", df_ptrcnn)
            print("df_ptpillar: ", "\n", df_ptpillar)
            print("df_second: ", "\n", df_second)

        if cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
            # majority_voting.
            majority_voting.majority_voting(cfg, path_manager, df_ptrcnn, df_ptpillar, df_second, frame_ID)
            counter_total_frames_processed += 1

        elif cfg.PIPELINE.VOTING_SCHEME == 'NMS':
            # NMS voting
            nms_voting.non_maximum_suppression_voting(cfg, path_manager, df_ptrcnn, df_ptpillar, df_second, frame_ID)
            counter_total_frames_processed += 1

        else:
            raise ValueError("No valid VOTING_SCHEME selected.")

        if cfg.PIPELINE.SHOW_POINT_CLOUDS:
            print("\n", "Visualization triggered:")
            visualize_single_pcd(frame_ID, cfg.VISUALISATION.BBOXES_TO_LOAD, cfg, path_manager)

        counter +=1
        if counter >= batch_size_voting:
            manual_continue = input("Continue y/ n ? ")
            if manual_continue == "y":
                counter = 0
                continue
            else:
                if cfg.PIPELINE.COMPUTE_EVALUATION_METRICS:
                    evaluate_voting_metrics(cfg, path_metrics_pseudo_label, counter_total_frames_processed)
                exit()

    if cfg.PIPELINE.COMPUTE_EVALUATION_METRICS:
        evaluate_voting_metrics(cfg, path_metrics_pseudo_label, counter_total_frames_processed)

# Function that removes frames to be processed to start at set checkpoint, if set.
def remove_smaller_numbers(numbers, value):
    return [num for num in numbers if num >= value]



# Function that resets the evaluation metrics folder. ONLY IF cfg.PIPELINE.COMPUTE_EVALUATION_METRICS
def reset_voting_metrics(path_to_reset):

    if os.path.exists(path_to_reset) and os.listdir(path_to_reset):
        shutil.rmtree(path_to_reset, ignore_errors=True)
    return

# Function that evaluates the majority_voting metrics. ONLY IF cfg.PIPELINE.COMPUTE_EVALUATION_METRICS
def evaluate_voting_metrics(cfg, path_to_voting_metrics, counter_total_processed_frames):
    print("___________________________________________________________________________")

    if not os.path.exists(path_to_voting_metrics) or not os.listdir(path_to_voting_metrics):
        print("No pseudo-labels to evaluate metrics on. Recheck voting parameters and MAJORITY_VOTING_METRICS flag in majority_voting.")
        print("___________________________________________________________________________")
        return

    else:
        metrics_data = {"frame_ID": [], "label": [], "metrics_model": [], "metrics_weight": []}
        counter_non_empty_pseudo_labels = 0

        for file_name in os.listdir(path_to_voting_metrics):

            if file_name.endswith(".txt"):
                counter_non_empty_pseudo_labels += 1
                frame_id = file_name.split(".")[0]

                with open(os.path.join(path_to_voting_metrics, file_name), "r") as file:
                    next(file)
                    for line in file:
                        label, metrics_model, metrics_weight = line.strip().split(",")
                        metrics_data["frame_ID"].append(frame_id)
                        metrics_data["label"].append(label)
                        metrics_data["metrics_model"].append(int(metrics_model))
                        metrics_data["metrics_weight"].append(int(metrics_weight))

        metrics_df = pd.DataFrame(metrics_data)
        total_rows = metrics_df.shape[0]

        for label in cfg.PIPELINE.CLASSES:
            label_df = metrics_df[metrics_df['label'] == label]
            model_counts = label_df['metrics_model'].value_counts()
            weight_counts = label_df['metrics_weight'].value_counts()
            total_rows_per_label = label_df.shape[0]

            model_percentages = (model_counts / total_rows_per_label) * 100
            weight_percentages = (weight_counts / total_rows_per_label) * 100
            print(f"Label: {label} (Number of detected objects: {total_rows_per_label})")

            for i, percentage in model_percentages.items():
                model = cfg.DATA.PROJECT.MODELS[i - 1]
                print(f"Ratio of pseudo-labels selected from model {model}: {percentage:.2f}%")
            for weight, percentage in weight_percentages.items():
                print(f"Ratio of objects selected from a set of {weight} bounding boxes: {percentage:.2f}%")
            print("\n_________")

        print(f"Total number of frames with generated pseudo-labels/ processed: "
              f"{counter_non_empty_pseudo_labels}/{counter_total_processed_frames}  "
              f"(Total detected objects: {total_rows})")
        print("___________________________________________________________________________")




def main_pseudo_label(cfg, BATCH_SIZE_VOTING, START_AT_CHECKPOINT, START_FRAME):

    # Add relative paths to PathManager for easy access
    path_manager = autolabel_path_manager(cfg)

    frame_set = common_set_between_datasets(cfg,
                                            [path_manager.get_path("path_ptrcnn_predictions"),
                                             path_manager.get_path("path_ptpillar_predictions"),
                                             path_manager.get_path("path_second_predictions")],
                                            True)

    if START_AT_CHECKPOINT:
        frame_set = remove_smaller_numbers(frame_set, START_FRAME)

    vote_pseudo_labels(cfg, path_manager, frame_set, BATCH_SIZE_VOTING)



if __name__ == "__main__":

    BATCH_SIZE_VOTING = 1
    # If not all frames should be voted on: Set START_AT_CHECKPOINT = True + set START_FRAME as first frame to be voted on.
    START_AT_CHECKPOINT = True
    START_FRAME = 5994

    # Load EasyDict to access parameters.
    cfg = load_config()
    main_pseudo_label(cfg, BATCH_SIZE_VOTING, START_AT_CHECKPOINT, START_FRAME)
