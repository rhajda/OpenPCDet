
# Import libraries
import os
import numpy as np
import pandas as pd
import torch


# Functions from openpcdet_autolabel.pcdet.utils:
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


def load_boxes_to_df(filepath):
    column_names = ['label', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                    'dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'waymo_info']

    df = pd.read_csv(filepath, delimiter=' ', header=None, names=column_names)
    df = df.drop(columns=['waymo_info'])

    return df


def find_next_txt_file(folder_path):

    if not os.path.exists(folder_path):
        return None

    files = os.listdir(folder_path)
    txt_files = [file for file in files if file.endswith('.txt')]
    if len(txt_files) > 0:
        numbers = [int(file.split('.')[0]) for file in txt_files]
        highest_number = max(numbers)
        next_number = highest_number + 1
    else:
        next_number = 100000

    next_file_name = f'{next_number:06d}.txt'
    next_file_name_pcd =  f'{next_number:06d}.pcd'
    next_file_path = os.path.join(folder_path, next_file_name)
    return next_file_name_pcd, next_file_path

def main_rename_kitti_carla_dataset(POINT_CLOUD_RANGE, label_path, velodyne_path, save_path):

    # Create save folder:
    os.makedirs(save_path, exist_ok=True)

    # List the filenames in the label folder
    label_filenames = []
    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            label_filenames.append(filename)
    label_filenames = sorted(label_filenames)

    # Filter ground truth labels to kitti POINT_CLOUD_RANGE
    for filename in label_filenames:

        df_frame = load_boxes_to_df(os.path.join(label_path, filename))
        # Filter to front of the car only.
        df_frame = df_frame[df_frame['loc_x'] > 0].reset_index(drop=True)
        # filter boxes outside range:
        df_boxes = df_frame[['loc_x', 'loc_y', 'loc_z', 'dim_length', 'dim_width', 'dim_height', 'rotation_y']]
        # Rename the columns to match the desired format
        df_boxes.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'heading']
        data_array = df_boxes.to_numpy()
        # True if at least one corner falls within the area. False else.
        mask = mask_boxes_outside_range_numpy(data_array, POINT_CLOUD_RANGE, min_num_corners=1)
        df_frame = df_frame[mask]

        # Find the matching .pcd file.
        name, extension = os.path.splitext(filename)
        pcd_filename = name + ".pcd"
        pcd_path = os.path.join(velodyne_path, pcd_filename)

        if os.path.exists(pcd_path):
            print(f"Processing: {pcd_filename}")
            next_file_name_pcd, next_file_path = find_next_txt_file(save_path)
            df_frame.to_csv(next_file_path, sep=' ', header=False, index=False, float_format='%g')
            os.rename(pcd_path, os.path.join(velodyne_path, next_file_name_pcd))

        else:
            continue


if __name__ == "__main__":

    POINT_CLOUD_RANGE = [0, -40, -3, 70.4, 40, 1]
    root_path = os.path.join("/home/data/converted_to_kitti_format/waymo_in_kitti_format_all/training/")

    label_path = os.path.join(root_path, "label_kitti_carla")
    velodyne_path = os.path.join(root_path, "velodyne")
    save_path = os.path.join(root_path, "label_autolabel")

    main_rename_kitti_carla_dataset(POINT_CLOUD_RANGE, label_path, velodyne_path, save_path)

