import os
import struct
import numpy as np
import open3d as o3d


def process_bin_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bin_files = [f for f in os.listdir(input_folder) if f.endswith(".bin")]

    for bin_file in bin_files:
        bin_path = os.path.join(input_folder, bin_file)
        pcd_file = os.path.join(output_folder, bin_file.replace(".bin", ".pcd"))

        size_float = 4
        list_pcd = []
        with open(bin_path, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)

        np_pcd = np.asarray(list_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        o3d.io.write_point_cloud(pcd_file, pcd)

        print(f"Processed {bin_file} and saved as {pcd_file}")

if __name__ == "__main__":

    input_folder = "/home/data/converted/waymo_mmdet_convert/kitti_format/training/velodyne"
    output_folder = "/home/data/converted/waymo_mmdet_convert/kitti_format/training/velodyne"
    process_bin_files(input_folder, output_folder)
