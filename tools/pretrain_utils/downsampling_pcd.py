import os
import copy
import numpy as np
import open3d as o3d
import math


# Directions to the point clouds
path_to_point_clouds = r"C:\Users\gima-\Desktop\pythonProject1\velodyne"
path_to_downsampled_point_clouds = r"C:\Users\gima-\Desktop\pythonProject1\sim_no_noise_downsampled"

# Counting all pointclouds
initial_count = 0
dir = path_to_point_clouds
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        initial_count += 1
print('initial_count:', initial_count)

# Load saved point cloud
for i in range(0, initial_count):
    pointcloud = str(i).zfill(6)
    datatyp = ".pcd"
    pointcloud = pointcloud + datatyp
    pcd_load = o3d.io.read_point_cloud(os.path.join(path_to_point_clouds, pointcloud))
###    o3d.visualization.draw_geometries([pcd_load])

    # Convert PointCloud to numpy array
    arr = np.asarray(pcd_load.points)
    rows, cols = arr.shape


    # Downsampling the array
    downsampled_arr = arr
    rows_to_delete = np.array
    rows_to_delete = np.random.choice(range(rows), math.floor(rows*0.2), replace=False)
    downsampled_arr = np.delete(downsampled_arr, rows_to_delete, 0)
    downsampled_rows, downsampled_cols = downsampled_arr.shape


    # Convert the downsampled array into a point cloud an save them
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(downsampled_arr)
    o3d.io.write_point_cloud(os.path.join(path_to_downsampled_point_clouds, pointcloud), pcd)

    # print('Rows before downsampling: ', rows)
    # print('Rows to delete: ', rows_to_delete)
    # print('Rows after downsampling: ', downsampled_rows)