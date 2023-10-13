
# Import libraries
from os import path as osp
import os
import argparse
import tqdm
import subprocess

from convert_waymo_to_kitti import waymo_data_prep, Waymo2KITTI
from convert_kitti_cam_to_kitti_carla import main_convert_kitti_cam_to_kitti_carla




if __name__ == "__main__":

    root_path = os.path.join("/home/data/waymo_open_dataset/")
    out_dir = os.path.join("/home/data/converted_to_kitti_format/")

    waymo_in_kitti_camera_coordinate_folder = os.path.join(out_dir, "waymo_in_kitti_format_all/training")
    label_source_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "label_all")
    label_target_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "label_kitti_carla")
    calib_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "calib")

    if True:
        print("Triggered convert_waymo_to_kitti. ")
        command = f"python convert_waymo_to_kitti.py --root-path {root_path} --out-dir {out_dir} waymo"

        try:
            subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running the training process: {e}")

    main_convert_kitti_cam_to_kitti_carla(waymo_in_kitti_camera_coordinate_folder,
                                          label_source_path,
                                          label_target_path,
                                          calib_path)
