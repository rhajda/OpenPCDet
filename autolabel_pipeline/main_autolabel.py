
# Import libraries
from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import pathlib
import yaml
import os
# Import visualization functions
from visualize_pcds import get_file_names, visualize_single_pcd
# Import voting schemes
from voting_schemes import nms_voting, majority_voting


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


"""

FILE DESCRIPTION: 

This file is the main file of the autolabel pipeline. It serves as a central script, triggering respective functions of
the pipeline. All parameters are configurable in autolabel.yaml. 

Important: 
dataframe structure: ['ID', 'label', 'loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z', 'score']

voting schemes: 
    - non maximum suppression + confidence
    - majority voting

"""

BATCH_SIZE_VOTING = 10000

# If not all frames should be voted on: Set START_AT_CHECKPOINT = True + set START_FRAME as first frame to be voted on.
START_AT_CHECKPOINT = False
START_FRAME = 100




# Function that loads the YAML file to access parameters.
def load_config():
    cfg_file = os.path.join(working_path, 'autolabel_pipeline/autolabel.yaml')
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError

    with open(cfg_file, 'r') as f:
        try:
            cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print("Error parsing YAML file:", e)
            return EasyDict()

    cfg = EasyDict(cfg_dict)
    return cfg

# Function loads csv data of a specific frame to a dataframe containing all bboxes
def csv_to_dataframe(path_data, file_name):

    columns = ['ID', 'label', 'loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z', 'score']

    # Load data from csv to a dataframe.
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
def vote_pseudo_labels(frame_set, batch_size_voting):

    show_progress_bar = not cfg.PIPELINE.PRINT_INFORMATION

    counter = 0
    description = "Voting pseudo-labels (" + str(cfg.PIPELINE.VOTING_SCHEME) + ")"

    for frame in tqdm(frame_set, desc=description, unit="frame", disable=not show_progress_bar):

        frame_ID = str(frame).zfill(6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load all predicted frame IDs (ptrcnn & ptpillar & second)
            df_ptrcnn = load_frame(frame_ID, cfg.DATA.PATH_PTRCNN_PREDICTIONS)
            df_ptpillar = load_frame(frame_ID, cfg.DATA.PATH_PTPILLAR_PREDICTIONS)
            df_second = load_frame(frame_ID, cfg.DATA.PATH_SECOND_PREDICTIONS)

        if cfg.PIPELINE.PRINT_INFORMATION:
            print("df_ptrcnn: ", "\n", df_ptrcnn)
            print("df_ptpillar: ", "\n", df_ptpillar)
            print("df_second: ", "\n", df_second)

        if cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
            # majority_voting.
            majority_voting.majority_voting(cfg, df_ptrcnn, df_ptpillar, df_second, frame_ID)

        elif cfg.PIPELINE.VOTING_SCHEME == 'NMS':
            # NMS voting
            nms_voting.non_maximum_suppression_voting(cfg, df_ptrcnn, df_ptpillar, df_second, frame_ID)

        else:
            raise ValueError("No valid VOTING_SCHEME selected.")

        if cfg.PIPELINE.SHOW_POINT_CLOUDS:
            print("\n", "Visualization triggered:")
            visualize_single_pcd(frame_ID, cfg.VISUALISATION.BBOXES_TO_LOAD, cfg)

        counter +=1
        if counter >= batch_size_voting:
            manual_continue = input("Continue y/ n ? ")
            if manual_continue == "y":
                counter = 0
                continue
            else:
                exit()


def remove_smaller_numbers(numbers, value):
    return [num for num in numbers if num >= value]



if __name__ == "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()
    frame_set = common_set_between_datasets(cfg,
                                            [cfg.DATA.PATH_PTRCNN_PREDICTIONS,
                                             cfg.DATA.PATH_PTPILLAR_PREDICTIONS,
                                             cfg.DATA.PATH_SECOND_PREDICTIONS],
                                            True)

    if START_AT_CHECKPOINT:
        frame_set = remove_smaller_numbers(frame_set, START_FRAME)

    vote_pseudo_labels(frame_set, BATCH_SIZE_VOTING)
