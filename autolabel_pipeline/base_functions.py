
# Import libraries
from easydict import EasyDict
import pathlib
import yaml
import os

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]



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

# Class to load different directories and access them easily.
class PathManager:
    def __init__(self):
        self.paths = {}

    def add_path(self, name, path):
        self.paths[name] = path

    def get_path(self, name):
        return self.paths.get(name)

    def list_paths(self):
        return list(self.paths.keys())

# Function that adds relative paths to the PathManager for easy access.
def autolabel_path_manager(cfg):

    # Instance of a PathManager object to keep track of paths.
    path_manager = PathManager()

    # Gather paths where predictions as saved for all models.
    models_specified = [model for model in ['pointrcnn', 'pointpillar', 'second'] if model in cfg.DATA.PROJECT.MODELS]

    if not  len(models_specified) == len(cfg.DATA.PROJECT.MODELS):
        raise ValueError("Pseudo-labels can not be voted. Check selected models in cfg.DATA.PROJECT.MODELS. "
                         "Voting configured for ['pointrcnn', 'pointpillar', 'second'].")


    # Paths to model predictions
    path_ptrcnn_predictions = os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA, "predictions/pointrcnn")
    path_ptpillar_predictions = os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA, "predictions/pointpillar")
    path_second_predictions = os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA, "predictions/second")

    # Paths to pseudo-labels
    path_pseudo_labels_majority = os.path.join(working_path,
                                               cfg.DATA.PROJECT.AUTOLABEL_DATA,
                                               "predictions/pseudo_labels/majority_voting")
    path_pseudo_labels_nms = os.path.join(working_path,
                                          cfg.DATA.PROJECT.AUTOLABEL_DATA,
                                          "predictions/pseudo_labels/nms_voting")

    # Paths main_autolabel
    path_manager.add_path("path_ptrcnn_predictions", path_ptrcnn_predictions)
    path_manager.add_path("path_ptpillar_predictions", path_ptpillar_predictions)
    path_manager.add_path("path_second_predictions", path_second_predictions)
    path_manager.add_path("path_pseudo_labels_majority", path_pseudo_labels_majority)
    path_manager.add_path("path_pseudo_labels_nms", path_pseudo_labels_nms)

    # Paths main_pipeline
    path_manager.add_path("path_project_dataset", os.path.join(working_path, cfg.DATA.PROJECT.DATASET))
    path_manager.add_path("path_project_data", os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA))
    path_manager.add_path("path_cfg_dataset", os.path.join(working_path, cfg.DATA.PROJECT.CFG_DATASET))
    path_manager.add_path("path_cfg_models", os.path.join(working_path, cfg.DATA.PROJECT.CFG_MODELS))
    path_manager.add_path("path_model_ckpt_dir", os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA, "models"))
    path_manager.add_path("path_output_labels",os.path.join(working_path, cfg.DATA.PROJECT.AUTOLABEL_DATA, "output_labels"))

    # Paths PCDET outputs
    path_manager.add_path("path_pcdet_pointrcnn",
                          os.path.join(working_path, "output/home/tools/cfgs/autolabel_models/pointrcnn/default/ckpt"))
    path_manager.add_path("path_pcdet_pointpillar",
                          os.path.join(working_path, "output/home/tools/cfgs/autolabel_models/pointpillar/default/ckpt"))
    path_manager.add_path("path_pcdet_second",
                          os.path.join(working_path, "output/home/tools/cfgs/autolabel_models/second/default/ckpt"))

    return path_manager

