

# Import Libraries
import pandas as pd
import os
import numpy as np

from autolabel_pipeline.base_functions import load_config, autolabel_path_manager
import sys
sys.path.append('/home/autolabel_pipeline/')
from autolabel_pipeline.visualize_pcds import visualize_single_pcd




# Function to create a rotation matrix around the x-axis
def rotation_matrix_x(degrees):
    radians = np.radians(degrees)
    return np.array([[1, 0, 0],
                     [0, np.cos(radians), -np.sin(radians)],
                     [0, np.sin(radians), np.cos(radians)]])

# Function to create a rotation matrix around the y-axis
def rotation_matrix_y(degrees):
    radians = np.radians(degrees)
    return np.array([[np.cos(radians), 0, np.sin(radians)],
                     [0, 1, 0],
                     [-np.sin(radians), 0, np.cos(radians)]])

# Function to create a rotation matrix around the z-axis
def rotation_matrix_z(degrees):
    radians = np.radians(degrees)
    return np.array([[np.cos(radians), -np.sin(radians), 0],
                     [np.sin(radians), np.cos(radians), 0],
                     [0, 0, 1]])


if __name__ == "__main__":


    frame = '0001000'
    base_folder = '/home/data/converted/waymo_mmdet_convert/kitti_format/training/label_debug/'

    # Load EasyDict to access parameters.
    cfg = load_config()
    # Load path manager to access paths easily.
    path_manager = autolabel_path_manager(cfg)


    load_path = os.path.join(base_folder, "0001000_og.txt")
    save_path = os.path.join(base_folder, str(frame +'.txt'))
    column_names = ['label', 'trunc', 'occl', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length',
                    'x', 'y', 'z', 'rot', 'unn']

    df = pd.read_csv(load_path, delimiter=' ', header=None, names=column_names)
    df = df.drop(columns=['unn'])


    if False:
        # Specify rotation angles in degrees for each axis
        rot_x = 90  # Rotation around the x-axis in degrees
        rot_y = 180  # Rotation around the y-axis in degrees
        rot_z = 90  # Rotation around the z-axis in degrees

        tx = 1
        ty = 0
        tz = 4

        # Create individual rotation matrices
        Rx = rotation_matrix_x(rot_x)
        Ry = rotation_matrix_y(rot_y)
        Rz = rotation_matrix_z(rot_z)

    else:
        # Specify rotation angles in degrees for each axis
        rot_x = 0  # Rotation around the x-axis in degrees
        rot_y = 0  # Rotation around the y-axis in degrees
        rot_z = 0  # Rotation around the z-axis in degrees

        tx = 0
        ty = 0
        tz = 0

        # Create individual rotation matrices
        Rx = rotation_matrix_x(rot_x)
        Ry = rotation_matrix_y(rot_y)
        Rz = rotation_matrix_z(rot_z)



    # Specify translation vector
    translation = np.array([tx, ty, tz])  # Replace tx, ty, tz with your desired translation values

    # Create the transformation matrix T
    T_camera_to_velodyne = np.eye(4)  # Identity matrix
    T_camera_to_velodyne[:3, :3] = np.dot(Rz, np.dot(Ry, Rx))
    T_camera_to_velodyne[:3, 3] = translation

    # Now, you can apply this transformation to your data
    xyz_columns = df[['x', 'y', 'z']].values
    xyz_columns = np.hstack((xyz_columns, np.ones((xyz_columns.shape[0], 1))))
    transformed_xyz = xyz_columns.dot(T_camera_to_velodyne.T)[:, :3]
    df[['x', 'y', 'z']] = transformed_xyz


    #T_camera_to_velodyne = np.dot(Rz, np.dot(Ry, Rx))
    #xyz_columns = df[['x', 'y', 'z']]
    #transformed_xyz = xyz_columns.dot(T_camera_to_velodyne[:3, :3])
    #df[['x', 'y', 'z']] = transformed_xyz
    print(df)

    # Remove the header
    df = df.iloc[1:]
    df.to_csv(save_path, sep=' ', header=False, index=False, float_format='%g')

    visualize_single_pcd('0001000', cfg.VISUALISATION.BBOXES_TO_LOAD, cfg, path_manager)
