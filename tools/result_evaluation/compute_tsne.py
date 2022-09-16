import numpy as np
from sklearn.manifold import TSNE
import open3d as o3d
import matplotlib.pyplot as plt


NUM_POINTS = 100000
filename1 = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/real/training/velodyne/000005.pcd"
filename2 = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_no_noise/training/velodyne/000005.pcd"

pcl1 = o3d.io.read_point_cloud(filename1, format="pcd")
pcl2 = o3d.io.read_point_cloud(filename2, format="pcd")
pcl1 = np.asarray(pcl1.points)
pcl2 = np.asarray(pcl2.points)

mask1 = np.random.choice(len(pcl1), min(NUM_POINTS, len(pcl1), len(pcl2)))
mask2 = np.random.choice(len(pcl2), min(NUM_POINTS, len(pcl1), len(pcl2)))
pcl1_masked = pcl1[mask1, :]
pcl2_masked = pcl2[mask2, :]

pcl_concat = np.concatenate((pcl1_masked, pcl2_masked))

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(pcl_concat)

pcl1_o3d = o3d.geometry.PointCloud()
pcl2_o3d = o3d.geometry.PointCloud()
pcl1_o3d.points = o3d.utility.Vector3dVector(pcl1_masked)
pcl2_o3d.points = o3d.utility.Vector3dVector(pcl2_masked)
pcl1_o3d.colors = o3d.utility.Vector3dVector(np.full_like(pcl1_masked, 0.1))
pcl2_o3d.colors = o3d.utility.Vector3dVector(np.full_like(pcl2_masked, 0.6))
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcl1_o3d)
vis.add_geometry(pcl2_o3d)
vis.run()

color = np.full_like(pcl_concat[:,0], 0)
color[len(pcl1_masked[:,0]):] = 1
fig, ax = plt.subplots()
ax.scatter(X_embedded[:,0], X_embedded[:,1], c=color)
plt.show()


print()