
# Import libraries
import os
import yaml
import pickle
import pathlib
from easydict import EasyDict


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]

# Function that loads the YAML file to access parameters.
def load_config():
    cfg_file = os.path.join(working_path, 'autolabel_pipeline/autolabel.yaml')
    print(cfg_file)
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


if __name__ == "__main__":

    cfg = load_config()

    # Load the pickle file
    with open(cfg.PARAMETERS_LOADING.PATH_TO_RESULTS, 'rb') as file:
        data = pickle.load(file)

        # Access the values of 'name' key for the first 10 dictionaries
    for dictionary in data[:1]:
        print(dictionary)

    exit()