import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from chamferdist import ChamferDistance
import geomloss
import torch


mode = "calc"  # "load": from disk, "calc": calculate from scratch

if torch.cuda.is_available():
    print("Using GPU")
    dev = "cuda:0"
else:
    print("Using CPU")
    dev = "cpu"

# Directions to the point clouds
root_path = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training"
imageset_subdir = "ImageSets"
pcl_subdir = "training/velodyne"

datasets = {0: "real",
            1: "sim_no_noise",
            2: "sim_noise_002",
            3: "sim_no_noise_downsampled",
            4: "sim_no_noise_downsampled_range"}

dataset1 = 0
dataset2 = 1

if mode == "calc":
    sample_id_list = [x.strip() for x in open(os.path.join(root_path, datasets[dataset1], imageset_subdir, "train.txt")).readlines()]

    # Load saved point cloud
    data_type = ".pcd"

    # metrics
    cd = ChamferDistance()
    emd = geomloss.SamplesLoss()

    cd_losses = []
    emd_losses = []
    min_bound = np.asarray([-100, -100, -50])
    max_bound = np.asarray([100, 100, 50])
    aa_bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    for i in tqdm(range(len(sample_id_list))):
            pcd_file_name = str(sample_id_list[i]).zfill(6) + data_type

            pcd1_path = os.path.join(root_path, datasets[dataset1], pcl_subdir, pcd_file_name)
            pcd2_path = os.path.join(root_path, datasets[dataset2], pcl_subdir, pcd_file_name)

            pcd1_o3d = o3d.io.read_point_cloud(pcd1_path, format="pcd").crop(aa_bb)
            pcd2_o3d = o3d.io.read_point_cloud(pcd2_path, format="pcd").crop(aa_bb)

            pcd1 = np.asarray(pcd1_o3d.points)
            pcd2 = np.asarray(pcd2_o3d.points)

            pcd1_tensor = torch.Tensor(pcd1).view(1, -1, 3).to(dev)
            pcd2_tensor = torch.Tensor(pcd2).view(1, -1, 3).to(dev)

            cd_loss = cd(pcd1_tensor, pcd2_tensor, bidirectional=True).item()
            emd_loss = torch.mean(emd(pcd1_tensor, pcd2_tensor)).item()

            cd_losses.append(cd_loss)
            emd_losses.append(emd_loss)

    cd_losses = np.asarray(cd_losses)
    emd_losses = np.asarray(emd_losses)

    np.save(os.path.join(root_path, f"cd_losses_{datasets[dataset1]}_{datasets[dataset2]}.npy"), cd_losses)
    np.save(os.path.join(root_path, f"emd_losses_{datasets[dataset1]}_{datasets[dataset2]}.npy"), emd_losses)

elif mode == "load":
    cd_losses = np.load(f"/ssd/tmp/CD_EMD/cd_losses_{datasets[dataset1]}_{datasets[dataset2]}.npy")
    emd_losses = np.load(f"/ssd/tmp/CD_EMD/emd_losses_{datasets[dataset1]}_{datasets[dataset2]}.npy")

print("CD: Mean/Min/Max")
print(int(cd_losses.mean()), int(cd_losses.min()), int(cd_losses.max()))

print("EMD: Mean/Min/Max")
print(round(emd_losses.mean(), 3), round(emd_losses.min(), 3), round(emd_losses.max(), 3))
