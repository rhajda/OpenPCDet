
import os
import open3d as o3d
import numpy as np


# Path to the point cloud and label folders
point_cloud_folder = "/home/Kitti/training/velodyne"
label_folder = "/home/Kitti/training/label_2"

file = "002645"

print("loading file: ", file)
pcd_file = os.path.join(point_cloud_folder, file + ".pcd")
label_file = os.path.join(label_folder, file + ".txt")


# Load point cloud data from file
pcd = o3d.io.read_point_cloud(pcd_file)

# Load label data from file
with open(label_file, 'r') as f:
    lines = f.readlines()


combined_list_all = []
# Loop through each line
for line in lines:
    elements = line.split()  # split the string by whitespace to get separate elements
    elements[1:] = [float(e) for e in elements[1:]]  # convert all elements except the first one to float

    # combine the elements into a single list
    combined_list = [elements[0]] + elements[1:]
    combined_list_all.append(combined_list)


# Create a list of bounding box LineSets
box_aabbs = []
for label in combined_list_all:

    h, w, l, x, y, z, = label[8:14]
    aabb = o3d.geometry.AxisAlignedBoundingBox([x - l / 2, y - w / 2, z], [x + l / 2, y + w / 2, z + h])
    box_aabbs.append(aabb)

color_map = {
    'Pedestrian': [1, 0, 0],    # white
    'Car': [0, 1, 0],           # green
    'Bicycle': [0, 0, 1]        # blue
    }

# Create a visualizer and add the point cloud and boxes
visualizer = o3d.visualize.Visualizer()
visualizer.create_window()
visualizer.get_render_option().point_size = 1.0
visualizer.get_render_option().background_color = (0.5, 0.5, 0.5)
visualizer.add_geometry(pcd)

# Loop over the bounding boxes and add them to the visualizer
for i, box in enumerate(box_aabbs):
    label = combined_list_all[i][0]
    color = color_map.get(label, [1, 1, 1])
    visualizer.add_geometry(box)

# Run the visualizer
visualizer.run()