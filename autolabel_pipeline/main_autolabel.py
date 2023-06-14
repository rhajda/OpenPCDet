from easydict import EasyDict
import pandas as pd
import numpy as np
import pathlib
import yaml
import os
import torch
from visualize_pcds import get_file_names, visualize_single_pcd

from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

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


# Function that generates a ndarray encoding the df rows of overlapping bboxes.
def identify_overlapping_bboxes(df1, df2):
    # Uses euclidian centerpoint distance and then computes IoU.

    # Compute the euclidian distance between all elements of the dataframes to filter out non-overlapping ones.
    distances = []
    for index_df1 in range(len(df1)):
        centerpoint_1 = np.array(df1.iloc[index_df1][2:5].values.astype(float))
        distances_element = []
        for index_df2 in range(len(df2)):
            centerpoint_2 = np.array(df2.iloc[index_df2][2:5].values.astype(float))
            distances_element.append(compute_euclidian_distance(centerpoint_1, centerpoint_2))
        distances.append(distances_element)

    # Compute a threshold that guarantees that there is no intersection between bboxes.
    largest_dim_len = max(df1['dim_len'].max(), df2['dim_len'].max())
    largest_dim_wi = max(df1['dim_wi'].max(), df2['dim_wi'].max())
    largest_bbox_radius = (((largest_dim_wi / 2) ** 2) + ((largest_dim_len / 2) ** 2)) ** 0.5
    centerpoint_distance_threshold = (2 * (largest_bbox_radius * 1.1))

    detected_overlaps = np.array(
        [[1 if val < centerpoint_distance_threshold else 0 for val in inner] for inner in distances])
    detected_overlaps = detected_overlaps.astype(float)

    # for the bounding boxes below the non-overlap threshold, get IoU.
    combinations_to_check = np.argwhere(detected_overlaps == 1)

    #print('possible overlaps: ', "\n", detected_overlaps)

    for i in range(len(combinations_to_check)):
        bbox1 = df1.iloc[combinations_to_check[i][0]]
        bbox2 = df2.iloc[combinations_to_check[i][1]]
        iou = iou_2_df_objects(bbox1, bbox2)
        detected_overlaps[combinations_to_check[i][0]][combinations_to_check[i][1]] = iou

    if cfg.PIPELINE.PRINT_INFORMATION:
        # Set the printing options
        np.set_printoptions(precision=2, suppress=True)
        print("centerpoint_distance_threshold: ", centerpoint_distance_threshold)
        print("Detected overlaps with IoU: ", "\n", detected_overlaps)

    return detected_overlaps


def identify_representative_bboxes(detected_overlaps, df1, df2):

    # get the indeces of overlapping bboxes.
    overlaps_to_check = np.argwhere(detected_overlaps != 0)

    for i in range(len(overlaps_to_check)):

        bbox1 = torch.tensor(np.array(df1.iloc[overlaps_to_check[i][0]][2:9].values.astype(float)))
        bbox2 = torch.tensor(np.array(df2.iloc[overlaps_to_check[i][1]][2:9].values.astype(float)))
        boxes = torch.stack([bbox1, bbox2], dim=0).float().cuda()

        score1 = torch.tensor(np.array(df1.iloc[overlaps_to_check[i][0]][9].astype(float)))
        score2 = torch.tensor(np.array(df2.iloc[overlaps_to_check[i][1]][9].astype(float)))
        scores = torch.stack([score1, score2], dim=0).float()

        representative = iou3d_nms_utils.nms_gpu(boxes, scores, 0.1)

        if cfg.PIPELINE.PRINT_INFORMATION:
            #print("position: ", overlaps_to_check[i])
            # print("boxes: ", boxes)
            # print("scores: ", scores)
            #print("representative: ", representative)
            pass






# Function that computes the euclidian distance between two points.
def compute_euclidian_distance(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2) ** 0.5


# Function that computes  the IoU of two bboxes (pcdet)
def iou_2_df_objects(bbox1, bbox2):

    # write elements ['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z'] from df.iloc[] to torch.tensor
    box1 = torch.tensor(np.array([bbox1[2:9].values.astype(float)]))
    box2 = torch.tensor(np.array([bbox2[2:9].values.astype(float)]))
    iou = box_utils.boxes3d_nearest_bev_iou(box1, box2)
    iou = iou.item()

    return iou




if __name__ == "__main__":

    # Choose element to display '000618',  '003658', '003676'
    TEMP_choose_element = '000618'



    # Load EasyDict to access parameters.
    cfg = load_config()

    # Load all predicted frame IDs --> ptrcnn / ptpillar
    ptrcnn_file_names_all = sorted(get_file_names(cfg.DATA.PATH_PTRCNN_PREDICTIONS, '.csv'), key=lambda x: int(x))
    ptpillar_file_names_all = sorted(get_file_names(cfg.DATA.PATH_PTPILLAR_PREDICTIONS, '.csv'), key=lambda x: int(x))

    ptrcnn_file_path = os.path.join(cfg.DATA.PATH_PTRCNN_PREDICTIONS, TEMP_choose_element + ".csv")
    ptpillar_file_path = os.path.join(cfg.DATA.PATH_PTPILLAR_PREDICTIONS, TEMP_choose_element + ".csv")
    df_ptrcnn = load_file_to_dataframe(ptrcnn_file_path, TEMP_choose_element)
    df_ptpillar = load_file_to_dataframe(ptpillar_file_path, TEMP_choose_element)

    if cfg.PIPELINE.PRINT_INFORMATION:
        print("df_ptrcnn: ", "\n", df_ptrcnn)
        print("df_ptpillar: ", "\n", df_ptpillar)




    detected_overlaps = identify_overlapping_bboxes(df_ptrcnn, df_ptpillar)


    #identify_representative_bboxes(detected_overlaps, df_ptrcnn, df_ptpillar)

    # Compute intersection over union of two boxes:
    #iou_2_objects = iou_2_df_objects(df_ptrcnn.iloc[0],  df_ptpillar.iloc[0])
    #print("iou_2_objects: ", iou_2_objects)


    if cfg.PIPELINE.SHOW_POINT_CLOUDS:
        print("\n", "Visualization triggered:")
        visualize_single_pcd(TEMP_choose_element, cfg.VISUALISATION.BBOXES_TO_LOAD, cfg)
