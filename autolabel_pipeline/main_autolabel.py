
# Import libraries
from easydict import EasyDict
import pandas as pd
import numpy as np
import pathlib
import yaml
import os
# Import visualization functions
from visualize_pcds import get_file_names, visualize_single_pcd
# Import voting schemes
from voting_schemes import nms_voting, majority_voting, test_voting


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
# Function that loads the bboxes predicted for a frame ID to dataframe for every model.
def load_frame_predictions(frame_ID, path_frames):

    # sort all frames by ascending order for all models
    all_frame_IDs = sorted(get_file_names(path_frames, '.csv'), key=lambda x: int(x))
    file_path = os.path.join(path_frames, frame_ID + ".csv")

    # Load predictions for specific frame_ID  to dataframe
    df_frame_ID = csv_to_dataframe(file_path, frame_ID)

    return all_frame_IDs, df_frame_ID


if __name__ == "__main__":

    # Choose element to display '000618',  '003658', '003676', '004769'
    frame_ID = '003676'

    # Load EasyDict to access parameters.
    cfg = load_config()

    # Load all predicted frame IDs (ptrcnn & ptpillar & second)
    ptrcnn_all_frame_IDs, df_ptrcnn= load_frame_predictions(frame_ID, cfg.DATA.PATH_PTRCNN_PREDICTIONS)
    ptpillar_all_frame_IDs, df_ptpillar = load_frame_predictions(frame_ID, cfg.DATA.PATH_PTPILLAR_PREDICTIONS)
    second_all_frame_IDs, df_second = load_frame_predictions(frame_ID, cfg.DATA.PATH_SECOND_PREDICTIONS)

    if cfg.PIPELINE.PRINT_INFORMATION:
        print("df_ptrcnn: ", "\n", df_ptrcnn)
        print("df_ptpillar: ", "\n", df_ptpillar)
        print("df_second: ", "\n", df_second)

    # nms_voting.
    nms_voting.non_maximum_suppression_voting(cfg, df_ptrcnn, df_ptpillar, df_second, frame_ID)

    # majority_voting.
    #majority_voting.majority_voting(cfg, df_ptrcnn, df_ptpillar, df_second, frame_ID)


    if cfg.PIPELINE.SHOW_POINT_CLOUDS:
        print("\n", "Visualization triggered:")
        visualize_single_pcd(frame_ID, cfg.VISUALISATION.BBOXES_TO_LOAD, cfg)

