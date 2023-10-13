
import os

folder1 = '/home/data/converted_to_kitti_format/waymo_in_kitti_format/training/label_kitti_carla_all'
folder2 = '/home/data/converted_to_kitti_format/waymo_in_kitti_format/training/label_kitti_carla'




# Get the list of .txt files in both directories
txt_files1 = [f for f in os.listdir(folder1) if f.endswith('.txt')]
txt_files2 = [f for f in os.listdir(folder2) if f.endswith('.txt')]

print(f"{folder1} length: {len(txt_files1)}")
print(f"{folder2} length: {len(txt_files2)}")

