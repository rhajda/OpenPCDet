import os
import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directions to the point clouds
path_to_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_no_noise/training/velodyne"
path_to_downsampled_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_no_noise_downsampled_range/training/velodyne"
os.makedirs(path_to_downsampled_point_clouds, exist_ok=True)
downsample_factor = 0.8

# Counting all point clouds
initial_count = len(os.listdir(path_to_point_clouds))

# Load saved point cloud
data_type = ".pcd"

bins = np.arange(0, 150, 1)
prob_diff = np.maximum(np.load("prob_diff.npy"), 0.0)
prob_keep = 1 - prob_diff

for i in tqdm(range(0, initial_count)):
    pcd_file_name = str(i).zfill(6) + data_type
    pcd_o3d = o3d.io.read_point_cloud(os.path.join(path_to_point_clouds, pcd_file_name), format="pcd")
    pcd = np.asarray(pcd_o3d.points)

    ranges = np.linalg.norm(pcd, axis=1)

    point_keep_prob = prob_keep[np.minimum(np.floor(ranges).astype(int), 148)]
    point_keep_prob /= point_keep_prob.sum()

    # Downsampling
    mask = np.random.choice(range(pcd.shape[0]), math.floor(pcd.shape[0]*downsample_factor), replace=False, p=point_keep_prob)

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(pcd[mask])

    o3d.io.write_point_cloud(os.path.join(path_to_downsampled_point_clouds, pcd_file_name), downsampled_pcd)

