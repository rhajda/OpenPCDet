import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

mode = "s"  # "s":sim, "r":real, "sd":sim_downsampled

# Directions to the point clouds
if mode == "s":
        path_to_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_no_noise/training/velodyne"
elif mode == "r":
        path_to_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/real/training/velodyne"
elif mode == "sd":
        path_to_point_clouds = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_no_noise_downsampled_range/training/velodyne"

# Counting all point clouds
initial_count = len(os.listdir(path_to_point_clouds))

# Load saved point cloud
data_type = ".pcd"

bins = np.arange(0, 150, 1)
sum_counts = np.zeros(len(bins) - 1)

for i in tqdm(range(0, initial_count)):
        pcd_file_name = str(i).zfill(6) + data_type
        pcd = np.asarray(o3d.io.read_point_cloud(os.path.join(path_to_point_clouds, pcd_file_name), format="pcd").points)
        ranges = np.linalg.norm(pcd, axis=1)
        counts, _ = np.histogram(ranges, bins=bins, density=True)
        sum_counts += counts

np.save(f"hist_{mode}.npy", sum_counts / initial_count)
