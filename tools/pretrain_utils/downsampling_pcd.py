import os
import numpy as np
import open3d as o3d
import math


# Directions to the point clouds
path_to_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/new/sim_no_noise/training/velodyne"
path_to_downsampled_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/new/sim_no_noise_downsampled/training/velodyne"
os.makedirs(path_to_downsampled_point_clouds, exist_ok=True)
downsample_factor = 0.8  # keeps downsample_factor*100 % of the points

# Counting all point clouds
initial_count = len(os.listdir(path_to_point_clouds))

# Load saved point cloud
data_type = ".pcd"
for i in range(0, initial_count):
    print(f"{i}/{initial_count}")
    pcd_file_name = str(i).zfill(6) + data_type
    pcd = np.asarray(o3d.io.read_point_cloud(os.path.join(path_to_point_clouds, pcd_file_name)).points)

    # Downsampling the array
    mask = np.random.choice(range(pcd.shape[0]), math.floor(pcd.shape[0]*downsample_factor), replace=False)

    # Convert the downsampled array into a point cloud and save them
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(pcd[mask])
    o3d.io.write_point_cloud(os.path.join(path_to_downsampled_point_clouds, pcd_file_name), downsampled_pcd)
