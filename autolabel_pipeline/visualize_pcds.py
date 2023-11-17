
# Import libraries.
import os
import open3d as o3d
import numpy as np
import random
import pathlib
from easydict import EasyDict
import yaml

from base_functions import load_config, autolabel_path_manager

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


"""

FILE DESCRIPTION: 

This file displays point clouds using open3d. The user can either specify a specific point cloud to be displayed or 
randomly display 5 point clouds. 
The script can load ['groundtruths', 'pointrcnn', 'pointpillar', 'second, 'pseudo_labels']. 
The point cloud will always be displayed, bbox sources can be defined in the .yaml file. 
Ex.: If one wants to load only groundtruths --> BBOXES_TO_LOAD:['groundtruths'].

Ground truths are displayed in red. 
Predicted boxes (pointrcnn) in dark blue.  
Predicted boxes (pointpillar) in medium blue. 
Predicted boxes (second) in light blue. 
Pseudo_labels are displayed in green.

The script can be triggered directly. It loads arguments form autolabel.yaml (VISUALIZATION)

"""




# Function that generates an array containing only common elements present in every bbox source.
def load_common_pcds(path_manager, bbox_source):

    bbox_source_elements = []

    if not bbox_source:
        print("Only pointcloud will be loaded, no bbox sources are selected.")

    else:
        for element in bbox_source:

            if element == 'groundtruths':
                groundtruth_file_names = get_file_names(cfg.DATA.PATH_GROUND_TRUTHS, '.txt')
                bbox_source_elements.append(groundtruth_file_names)

            if element == 'pointrcnn':
                pointrcnn_file_names = get_file_names(path_manager.get_path("path_ptrcnn_predictions"), '.csv')
                bbox_source_elements.append(pointrcnn_file_names)

            if element == 'pointpillar':
                pointpillar_file_names = get_file_names(path_manager.get_path("path_ptpillar_predictions"), '.csv')
                bbox_source_elements.append(pointpillar_file_names)

            if element == 'second':
                second_file_names = get_file_names(path_manager.get_path("path_second_predictions"), '.csv')
                bbox_source_elements.append(second_file_names)

            if element == 'pseudo_labels':

                if cfg.PIPELINE.VOTING_SCHEME == 'NMS':
                    pseudo_labels_file_names = get_file_names(path_manager.get_path("path_pseudo_labels_nms"), '.csv')
                    if os.path.exists(os.path.join(path_manager.get_path("path_pseudo_labels_nms"), 'control')):
                        pseudo_labels_file_names_control = get_file_names(os.path.join(path_manager.get_path("path_pseudo_labels_nms"),
                                                                                       'control'), '.csv')
                        pseudo_labels_file_names += pseudo_labels_file_names_control

                elif cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
                    pseudo_labels_file_names = get_file_names(path_manager.get_path("path_pseudo_labels_majority"), '.csv')
                    if os.path.exists(os.path.join(path_manager.get_path("path_pseudo_labels_majority"), 'control')):
                        pseudo_labels_file_names_control = get_file_names(os.path.join(path_manager.get_path("path_pseudo_labels_majority"),
                                                                                       'control'), '.csv')
                        pseudo_labels_file_names += pseudo_labels_file_names_control

                bbox_source_elements.append(pseudo_labels_file_names)

    # Load point clouds / Groundtruth data / Point-rcnn data:
    point_cloud_files = get_file_names(cfg.DATA.PATH_POINT_CLOUDS, '.pcd')

    # write all files that have all necessary information into common_elements_array.
    if len(bbox_source_elements) > 0:
        common_elements = set(bbox_source_elements[0])
        for subarray in bbox_source_elements[1:]:
            common_elements = common_elements.intersection(subarray)
        common_elements_array = np.array(list(common_elements))

    else:
        common_elements_array = point_cloud_files

    return common_elements_array

# Function that loads the names of all files in a folder to an array
def get_file_names(path_to_files, file_type):

    file_names = []
    for f in os.listdir(path_to_files):
        if f.endswith(file_type):
            file_names.append(os.path.splitext(f)[0])

    return file_names

# Function that transforms predicted boxes to open3d boxes
def translate_boxes_to_open3d_instance(gt_boxes, gt):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    if gt == 'groundtruths':
        color_groundtruths = [1, 0, 0]      # red

    if gt == 'pointrcnn':
        color_pointrcnn = [0, 0, 1]         # dark blue

    if gt == 'pointpillar':
        color_pointpillar = [0, 0.7, 1]     # medium blue

    if gt == 'second':
        color_second = [0, 1, 1]            # light blue

    if gt == 'pseudo_labels':
        color_pseudo_labels = [0, 0.7, 0]     # green

    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6] * 1.0
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    if gt == 'groundtruths':
        box3d.color = np.asarray(color_groundtruths)

    if gt == 'pointrcnn':
        box3d.color = np.asarray(color_pointrcnn)

    if gt == 'pointpillar':
        box3d.color = np.asarray(color_pointpillar)

    if gt == 'second':
        box3d.color = np.asarray(color_second)

    if gt == 'pseudo_labels':
        box3d.color = np.asarray(color_pseudo_labels)


    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if gt == 'groundtruths':
        line_set.paint_uniform_color(np.asarray(color_groundtruths))

    if gt == 'pointrcnn':
        line_set.paint_uniform_color(np.asarray(color_pointrcnn))

    if gt == 'pointpillar':
        line_set.paint_uniform_color(np.asarray(color_pointpillar))

    if gt == 'second':
        line_set.paint_uniform_color(np.asarray(color_second))

    if gt == 'pseudo_labels':
        line_set.paint_uniform_color(np.asarray(color_pseudo_labels))

    return box3d, line_set

# This function loads the pcd, predicted boxes and ground-truths of a specified frame ID and visualizes it in open3d.
def visualize_single_pcd(single_pcd, bbox_source, cfg, path_manager):

    print("Visualizing frame ID: ", single_pcd)
    print("Sources: ", bbox_source)

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
    pcd_file = os.path.join(cfg.DATA.PATH_POINT_CLOUDS, single_pcd + ".pcd")
    # print("Point cloud ID: ", pcd_file)
    pcd = o3d.io.read_point_cloud(pcd_file)

    for element in bbox_source:

        if element == 'groundtruths':
            gt_file = os.path.join(cfg.DATA.PATH_GROUND_TRUTHS, single_pcd + ".txt")

            # Ground truth bounding boxes
            gt_array = np.genfromtxt(gt_file, delimiter=' ', dtype=str)

            if np.size(gt_array) == 0:
                print("Ground-truths, ID ", single_pcd, ": ", "EMPTY ARRAY.")
                continue

            if gt_array.ndim == 1:
                gt_array = gt_array.reshape(1, -1)

            gt_labels = gt_array[:, 0]
            gt_boxes = gt_array[:, 1:].astype(float)
            gt_boxes = gt_boxes[gt_labels != 'DontCare']
            #gt_boxes = gt_boxes[gt_labels == 'Car']
            # Convert groundtruth boxes to format: X,Y,Z,L,W,H,RotY
            gt_boxes_reformatted = np.zeros((gt_boxes.shape[0], 7))
            gt_boxes_reformatted[:, 0] = gt_boxes[:, 10]  # loc_x
            gt_boxes_reformatted[:, 1] = gt_boxes[:, 11]  # loc_y
            gt_boxes_reformatted[:, 2] = gt_boxes[:, 12]  # loc_z
            gt_boxes_reformatted[:, 3] = gt_boxes[:, 9]  # dim_length
            gt_boxes_reformatted[:, 4] = gt_boxes[:, 8]  # dim_width
            gt_boxes_reformatted[:, 5] = gt_boxes[:, 7]  # dim_height
            gt_boxes_reformatted[:, 6] = gt_boxes[:, 13]  # rot_y

            # 10 --> x kitti
            # 11 --> y kitti
            # 12 --> z kitti
            #gt_boxes_reformatted[:, 0] = gt_boxes[:, 10]  # loc_x
            #gt_boxes_reformatted[:, 1] = gt_boxes[:, 12]  # loc_y
            #gt_boxes_reformatted[:, 2] = gt_boxes[:, 11]  # loc_z
            #gt_boxes_reformatted[:, 3] = gt_boxes[:, 9]  # dim_length
            #gt_boxes_reformatted[:, 4] = gt_boxes[:, 8]  # dim_width
            #gt_boxes_reformatted[:, 5] = gt_boxes[:, 7]  # dim_height
            #gt_boxes_reformatted[:, 6] = gt_boxes[:, 13]  # rot_y


            for box_idx, box in enumerate(gt_boxes_reformatted):
                # box: X, Y, Z, L, W, H, Rot_Y
                ret = translate_boxes_to_open3d_instance(box, gt='groundtruths')
                boxes_3d.append(ret[0])
                line_sets.append(ret[1])

            print("Ground-truths (red), ID ", single_pcd, ": ", gt_labels)
            # print("Ground truth ID: ", gt_file)

        if element == 'pointrcnn':
            prcnn_file = os.path.join(path_manager.get_path("path_ptrcnn_predictions"), single_pcd + ".csv")

            # Predicted bounding boxes POINT-RCNN
            prcnn_pred_array = np.genfromtxt(prcnn_file, delimiter=',', dtype=str)

            if np.size(prcnn_pred_array) == 0:
                print("Predicted pt-rcnn, ID ", single_pcd, ": ", "EMPTY ARRAY.")
                continue

            if prcnn_pred_array.ndim == 1:
                prcnn_pred_array = prcnn_pred_array.reshape(1, -1)

            prcnn_labels = prcnn_pred_array[:, 0]
            prcnn_boxes = prcnn_pred_array[:, 1:].astype(float)

            for box_idx, box in enumerate(prcnn_boxes[prcnn_boxes[:, 0].argsort()]):
                # print(f"X,Y,Z,L,W,H,RotY,Score: {box}")
                ret = translate_boxes_to_open3d_instance(box, gt='pointrcnn')
                boxes_3d.append(ret[0])
                line_sets.append(ret[1])

            print("Predicted pt-rcnn (dark blue), ID ", single_pcd, ": ", prcnn_labels)
            # print("Pt-rcnn ID: ", prcnn_file)

        if element == 'pointpillar':
            ppillar_file = os.path.join(path_manager.get_path("path_ptpillar_predictions"), single_pcd + ".csv")

            # Predicted bounding boxes POINT-PILLAR
            ppillar_pred_array = np.genfromtxt(ppillar_file, delimiter=',', dtype=str)

            if np.size(ppillar_pred_array) == 0:
                print("Predicted pt-pillar, ID ", single_pcd, ": ", "EMPTY ARRAY.")
                continue

            if ppillar_pred_array.ndim == 1:
                ppillar_pred_array = ppillar_pred_array.reshape(1, -1)

            ppillar_labels = ppillar_pred_array[:, 0]
            ppillar_boxes = ppillar_pred_array[:, 1:].astype(float)

            for box_idx, box in enumerate(ppillar_boxes[ppillar_boxes[:, 0].argsort()]):
                # print(f"X,Y,Z,L,W,H,RotY,Score: {box}")
                ret = translate_boxes_to_open3d_instance(box, gt='pointpillar')
                boxes_3d.append(ret[0])
                line_sets.append(ret[1])

            print("Predicted pt-pillar (medium blue), ID ", single_pcd, ": ", ppillar_labels)
            # print("Pt-pillar ID: ", ppillar_file)

        if element == 'second':
            second_file = os.path.join(path_manager.get_path("path_second_predictions"), single_pcd + ".csv")

            # Predicted bounding boxes SECOND
            second_pred_array = np.genfromtxt(second_file, delimiter=',', dtype=str)

            if np.size(second_pred_array) == 0:
                print("Predicted second, ID ", single_pcd, ": ", "EMPTY ARRAY.")
                continue

            if second_pred_array.ndim == 1:
                second_pred_array = second_pred_array.reshape(1, -1)

            second_labels = second_pred_array[:, 0]
            second_boxes = second_pred_array[:, 1:].astype(float)

            for box_idx, box in enumerate(second_boxes[second_boxes[:, 0].argsort()]):
                # print(f"X,Y,Z,L,W,H,RotY,Score: {box}")
                ret = translate_boxes_to_open3d_instance(box, gt='second')
                boxes_3d.append(ret[0])
                line_sets.append(ret[1])

            print("Predicted second (light blue), ID ", single_pcd, ": ", second_labels)
            # print("Second ID: ", ppillar_file)

        if element == 'pseudo_labels':

            if cfg.PIPELINE.VOTING_SCHEME == 'NMS':
                pseudo_label_file = os.path.join(path_manager.get_path("path_pseudo_labels_nms"), single_pcd + ".csv")

            elif cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
                pseudo_label_file = os.path.join(path_manager.get_path("path_pseudo_labels_majority"), single_pcd + ".csv")
                if not os.path.exists(pseudo_label_file):
                    pseudo_label_file = os.path.join(path_manager.get_path("path_pseudo_labels_majority"), 'control', single_pcd + ".csv")
            else:
                raise ValueError("Selected VOTING_SCHEME not valid. ")


            # Predicted bounding boxes PSEUDO_LABELS
            pseudo_label_pred_array = np.genfromtxt(pseudo_label_file, delimiter=',', dtype=str)

            if np.size(pseudo_label_pred_array) == 0:
                print("Pseudo_labels, ID ", single_pcd, ": ", "EMPTY ARRAY.")
                continue

            if pseudo_label_pred_array.ndim == 1:
                pseudo_label_pred_array = pseudo_label_pred_array.reshape(1, -1)

            pseudo_label_labels = pseudo_label_pred_array[:, 0]
            pseudo_label_boxes = pseudo_label_pred_array[:, 1:].astype(float)

            for box_idx, box in enumerate(pseudo_label_boxes[pseudo_label_boxes[:, 0].argsort()]):
                # print(f"X,Y,Z,L,W,H,RotY,Score: {box}")
                ret = translate_boxes_to_open3d_instance(box, gt='pseudo_labels')
                boxes_3d.append(ret[0])
                line_sets.append(ret[1])

            print("Pseudo_labels (green), ID ", single_pcd, ": ", pseudo_label_labels)
            # print("Pseudo_label ID: ", pseudo_label_file)

    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd)
    for box in boxes_3d:
        vis.add_geometry(box)

    for line_set in line_sets:
        vis.add_geometry(line_set)

    viewctrl = vis.get_view_control()
    viewctrl.set_lookat(np.array([0.0, 0.0, 0.0]))
    viewctrl.set_zoom(30)
    rend_opt = vis.get_render_option()
    rend_opt.point_size = 3
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    print("\n")

# This function triggers the open3D visualization depending on the input given: my_pcd = 'xxxxx' , ''
def trigger_visualization(cfg, my_pcd, bbox_source):

    # Add PathManager for easy access to paths.
    path_manager = autolabel_path_manager(cfg)

    pcds_common = load_common_pcds(path_manager, bbox_source)

    if my_pcd == '':
        # Load multiple random files.
        random.shuffle(pcds_common)
        pcds_selected = pcds_common[:5]

        for i in range(0, (len(pcds_selected))):
            visualize_single_pcd(pcds_selected[i], bbox_source, cfg, path_manager)
        print("done.")

    else:
        # Load single specified file.
        if my_pcd in pcds_common:
            visualize_single_pcd(my_pcd, bbox_source, cfg, path_manager)
        else:
            raise ValueError('This point cloud ID is not available. Choose a different one.')


if __name__== "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()

    # Check if a single file or multiple random files should be loaded.
    my_pcd_to_visualize = cfg.VISUALISATION.POINT_CLOUD_ID_TO_VISUALIZE

    # define which bboxes to load for visualization.
    bbox_source = cfg.VISUALISATION.BBOXES_TO_LOAD

    trigger_visualization(cfg, my_pcd_to_visualize, bbox_source)
