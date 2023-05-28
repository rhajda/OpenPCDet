
# Import libraries.
import os
import open3d as o3d
import numpy as np
import random
import pathlib
from easydict import EasyDict
import yaml

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


"""

FILE DESCRIPTION: 

This file displays point clouds using open3d. The user can either specify a specific point cloud to be displayed or 
randomly display 5 point clouds. The script loads the point clouds as well as the ground truth and point rcnn bboxs.
Ground truths are displayed in red. Predicted boxes (point rcnn) in blue.  
The script can be triggered directly. It loads arguments form autolabel.yaml (VISUALIZATION)


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


# Function that loads the names of all files in a folder to an array
def get_file_names(path_to_files, file_type):

    file_names = []
    for f in os.listdir(path_to_files):
        if f.endswith(file_type):
            file_names.append(os.path.splitext(f)[0])

    return file_names


# Function that transforms predicted boxes to open3d boxes
def translate_boxes_to_open3d_instance(gt_boxes, gt=False):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6] * 1.0
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    if gt:
        box3d.color = np.asarray([1, 0, 0])
    else:
        box3d.color = np.asarray([0, 1, 1])

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if gt:
        line_set.paint_uniform_color(np.asarray([1, 0, 0]))
    else:
        line_set.paint_uniform_color(np.asarray([0, 1, 1]))

    return box3d, line_set


# This function loads the pcd, predicted boxes and ground-truths of a specified frame ID and visualizes it in open3d.
def visualize(single_pcd):

    print("Visualizing frame ID: ", single_pcd)
    prcnn_file = os.path.join(cfg.VISUALISATION.PATH_PTRCNN_PREDICTIONS, single_pcd + ".csv")
    gt_file = os.path.join(cfg.VISUALISATION.PATH_GROUND_TRUTHS, single_pcd + ".txt")
    pcd_file = os.path.join(cfg.VISUALISATION.PATH_POINT_CLOUDS, single_pcd + ".pcd")
    #print("Point-rcnn ID: ", prcnn_file)
    #print("Ground truth ID: ", gt_file)
    #print("Point cloud ID: ", pcd_file)

    # initialize
    boxes_3d = []
    line_sets = []

    # The x, y, z axis will be rendered as red, green, and blue arrows respectively.
    mesh_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_frame)

    # point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Predicted bounding boxes POINT-RCNN
    prcnn_pred_array = np.genfromtxt(prcnn_file, delimiter=',', dtype=str)

    if prcnn_pred_array.ndim == 1:
        prcnn_labels = prcnn_pred_array[0]
        prcnn_boxes = prcnn_pred_array[1:].astype(float)
        ret = translate_boxes_to_open3d_instance(prcnn_boxes, gt=False)
        boxes_3d.append(ret[0])
        line_sets.append(ret[1])

    else:
        prcnn_labels = prcnn_pred_array[:, 0]
        prcnn_boxes = prcnn_pred_array[:, 1:].astype(float)

        for box_idx, box in enumerate(prcnn_boxes[prcnn_boxes[:, 0].argsort()]):
            # print(f"X,Y,Z,L,W,H,RotY,Score: {box}")
            ret = translate_boxes_to_open3d_instance(box, gt=False)
            boxes_3d.append(ret[0])
            line_sets.append(ret[1])

    # Ground truth bounding boxes
    gt_array = np.genfromtxt(gt_file, delimiter=' ', dtype=str)

    if gt_array.ndim == 1:
        gt_labels = gt_array[0]
        gt_boxes = gt_array[1:].astype(float)

        if gt_labels[0] == 'DontCare':
            pass

        # Convert groundtruth boxes to format: X,Y,Z,L,W,H,RotY
        gt_boxes_reformatted = np.zeros(7)
        gt_boxes_reformatted[0] = gt_boxes[10]  # loc_x
        gt_boxes_reformatted[1] = gt_boxes[11]  # loc_y
        gt_boxes_reformatted[2] = gt_boxes[12]  # loc_z
        gt_boxes_reformatted[3] = gt_boxes[9]  # dim_length
        gt_boxes_reformatted[4] = gt_boxes[8]  # dim_width
        gt_boxes_reformatted[5] = gt_boxes[7]  # dim_height
        gt_boxes_reformatted[6] = gt_boxes[13]  # rot_y
        # box: X, Y, Z, L, W, H, Rot_Y
        ret = translate_boxes_to_open3d_instance(gt_boxes_reformatted, gt=True)
        boxes_3d.append(ret[0])
        line_sets.append(ret[1])

    else:
        gt_labels = gt_array[:, 0]
        gt_boxes = gt_array[:, 1:].astype(float)
        gt_boxes = gt_boxes[gt_labels != 'DontCare']
        # Convert groundtruth boxes to format: X,Y,Z,L,W,H,RotY
        gt_boxes_reformatted = np.zeros((gt_boxes.shape[0], 7))
        gt_boxes_reformatted[:, 0] = gt_boxes[:, 10]  # loc_x
        gt_boxes_reformatted[:, 1] = gt_boxes[:, 11]  # loc_y
        gt_boxes_reformatted[:, 2] = gt_boxes[:, 12]  # loc_z
        gt_boxes_reformatted[:, 3] = gt_boxes[:, 9]  # dim_length
        gt_boxes_reformatted[:, 4] = gt_boxes[:, 8]  # dim_width
        gt_boxes_reformatted[:, 5] = gt_boxes[:, 7]  # dim_height
        gt_boxes_reformatted[:, 6] = gt_boxes[:, 13]  # rot_y

        for box_idx, box in enumerate(gt_boxes_reformatted):
            # box: X, Y, Z, L, W, H, Rot_Y
            ret = translate_boxes_to_open3d_instance(box, gt=True)
            boxes_3d.append(ret[0])
            line_sets.append(ret[1])

    print("Predicted pt-rcnn, ID ", my_pcd, ": ",  prcnn_labels)
    print("Ground-truths, ID ", my_pcd, ": ", gt_labels)

    vis.add_geometry(pcd)
    for box in boxes_3d:
        vis.add_geometry(box)
    for line_set in line_sets:
        vis.add_geometry(line_set)

    viewctrl = vis.get_view_control()
    viewctrl.set_lookat(np.array([0.0, 0.0, 0.0]))
    viewctrl.set_zoom(30)
    rend_opt = vis.get_render_option()
    rend_opt.point_size = 2
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


if __name__== "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()

   # Load point clouds / Groundtruth data / Point-rcnn data:
    point_cloud_files = get_file_names(cfg.VISUALISATION.PATH_POINT_CLOUDS, '.pcd')
    groundtruth_file_names = get_file_names(cfg.VISUALISATION.PATH_GROUND_TRUTHS, '.txt')
    pointrcnn_file_names = get_file_names(cfg.VISUALISATION.PATH_PTRCNN_PREDICTIONS, '.csv')
    my_pcd = cfg.VISUALISATION.POINT_CLOUD_ID_TO_VISUALIZE

    # Check if a single file or multiple random files should be loaded.
    if my_pcd =='':
        # Load multiple random files.
        pcds_common = []

        for element in pointrcnn_file_names:
            if element in point_cloud_files and element in groundtruth_file_names:
                pcds_common.append(element)

        random.shuffle(pcds_common)
        pcds_selected = pcds_common[:5]

        for i in range (0, (len(pcds_selected))):
            visualize(pcds_selected[i])
        print("done.")

    else:
        # Load single specified file.
        if all(my_pcd in arr for arr in [point_cloud_files, groundtruth_file_names, pointrcnn_file_names]):
            visualize(my_pcd)
        else:
            raise ValueError('This point cloud ID is not available. Choose a different one.')
