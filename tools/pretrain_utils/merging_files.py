import os
import sys
import shutil


####
pcd = False  # copy point clouds or labels
src_dataset = ""  # "" for real or "simulation/01_no_noise" / "simulation/02_noise_002" / "simulation/03_noise_005" for sim
dst_dataset = "real"
extracted_data_path = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data"
####

if pcd:
    srcpath = os.path.join(src_dataset, "pcd_valid")
    savepath = os.path.join("for_training", dst_dataset, "velodyne")
    data_type = '.pcd'
else:
    srcpath = os.path.join(src_dataset, "label_valid")
    savepath = os.path.join("for_training", dst_dataset, "label_2")
    data_type = '.txt'

# Source data path for each day
src_day1 = os.path.join(extracted_data_path, "20220103", srcpath)
src_day2 = os.path.join(extracted_data_path, "20220106", srcpath)
src_day3 = os.path.join(extracted_data_path, "20220107", srcpath)

# List of all files for each day
filesday1 = os.listdir(src_day1)
filesday2 = os.listdir(src_day2)
filesday3 = os.listdir(src_day3)
files_num_total = len(filesday1) + len(filesday2) + len(filesday3)

# Move files from each day to "training" folder
src_num = 0
dst_num = 0
for file in filesday1:
    print(f"{dst_num} / {files_num_total}")
    shutil.copyfile(src=os.path.join(src_day1, str(src_num).zfill(6) + data_type), dst=os.path.join(extracted_data_path, savepath, str(dst_num).zfill(6) + data_type))
    src_num += 1
    dst_num += 1

src_num = 0
for file in filesday2:
    print(f"{dst_num} / {files_num_total}")
    shutil.copyfile(src=os.path.join(src_day2, str(src_num).zfill(6) + data_type), dst=os.path.join(extracted_data_path, savepath, str(dst_num).zfill(6) + data_type))
    src_num += 1
    dst_num += 1

src_num = 0
for file in filesday3:
    print(f"{dst_num} / {files_num_total}")
    shutil.copyfile(src=os.path.join(src_day3, str(src_num).zfill(6) + data_type), dst=os.path.join(extracted_data_path, savepath, str(dst_num).zfill(6) + data_type))
    src_num += 1
    dst_num += 1
