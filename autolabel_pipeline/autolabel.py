from easydict import EasyDict
import pandas as pd
import numpy as np
import pathlib
import yaml
import os
from visualize_pcds import get_file_names

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


"""

FILE DESCRIPTION: 

dataframe structure: ['ID', 'label', 'loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z', 'score']

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


# Function that loads the object list of a specific frame into a dataframe.
def load_file_to_dataframe(path_data, file_name):

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




if __name__=="__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()

    # Load all frame IDs with object lists for ptrcnn
    ptrcnn_file_names_all = get_file_names(cfg.DATA.PATH_PTRCNN_PREDICTIONS, '.csv')
    ptrcnn_file_names_all = sorted(ptrcnn_file_names_all, key=lambda x: int(x))
    print(ptrcnn_file_names_all[:10])

    # Load all frame IDs with object lists for ptpillar
    ptpillar_file_names_all = get_file_names(cfg.DATA.PATH_PTPILLAR_PREDICTIONS, '.csv')
    ptpillar_file_names_all = sorted(ptpillar_file_names_all, key=lambda x: int(x))
    print(ptpillar_file_names_all[:10])

    count = 0
    for element in range (10):
        print("\n")
        count += 1

        # Load a single frame object list to df  --> PTRCNN
        ptrcnn_file_path = os.path.join(cfg.DATA.PATH_PTRCNN_PREDICTIONS, ptrcnn_file_names_all[element] + ".csv")
        ptrcnn_file_name = ptrcnn_file_names_all[element]
        df_ptrcnn = load_file_to_dataframe(ptrcnn_file_path, ptrcnn_file_name)
        print("df_ptrcnn: ", "\n", df_ptrcnn)

        # Load a single frame object list to df  --> PTPILLAR
        ptpillar_file_path = os.path.join(cfg.DATA.PATH_PTPILLAR_PREDICTIONS, ptpillar_file_names_all[element] + ".csv")
        ptpillar_file_name = ptpillar_file_names_all[element]
        df_ptpillar = load_file_to_dataframe(ptpillar_file_path, ptpillar_file_name)
        print("df_ptpillar: ", "\n", df_ptpillar)


        if count >= 3:
            break
