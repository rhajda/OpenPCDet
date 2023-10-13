
# Import libraries
import os
import struct
import numpy as np
import open3d as o3d
import tqdm



# Function for step 1.
def process_bin_files(input_folder):


    bin_files = [f for f in os.listdir(input_folder) if f.endswith(".bin")]

    if bin_files:
        for bin_file in bin_files:
            bin_path = os.path.join(input_folder, bin_file)
            pcd_file = os.path.join(input_folder, bin_file.replace(".bin", ".pcd"))

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

            # Delete the original .bin file
            os.remove(bin_path)

            print(f"Processed {bin_file} and saved.")
    else:
        print("No .bin files to process.")

    return



# Function for step 2.
'''
From https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/calibration_kitti.py
'''
def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

'''

    KITTI to CARLA
    X (right) -> -Y
    Y (down) -> -Z
    Z (front) -> X

    CARLA to KITTI
    X (front) -> Z
    Y (left) -> -X
    Z (up) -> -Y

'''

def switch_dims(data_source_path, data_target_path):
    """
        BOX:
            0: type
            1: truncated
            2: occluded
            3: alpha
            4: camera_bbox_left
            5: camera_bbox_top
            6: camera_bbox_right
            7: camera_bbox_bottom
            8: height
            9: length(CARLA)/width(KITTI)
            10: width(CARLA)/length(KITTI)
            11: x
            12: y
            13: z
            14: rotation_y
    """

    for file in sorted(os.listdir(data_source_path)):
        filename_source = os.path.join(data_source_path, file)
        filename_target = os.path.join(data_target_path, file)

        with open(filename_source, "r") as f:
            boxes = f.read()
            boxes = np.asarray([[el for el in row if el] for row in [box.split(" ") for box in boxes.split("\n")]])
            types = boxes[:, 0]
            boxes = boxes[:, 1:].astype(float)

            # switch width with length
            boxes[:, [9, 8]] = boxes[:, [8, 9]]

            # limit values to two decimal places
            boxes = np.around(boxes, 2)

            new_label = np.concatenate([np.expand_dims(types, axis=1), boxes], axis=1)
            np.savetxt(filename_target, new_label, fmt="%s")

def convert_to_lidar(data_source_path, data_target_path, calib_path):
    pbar = tqdm.tqdm(total=len(os.listdir(data_source_path)))
    for file in sorted(os.listdir(data_source_path)):
        filename_source = os.path.join(data_source_path, file)
        filename_target = os.path.join(data_target_path, file)

        calib_file = os.path.join(calib_path, file)
        calib = Calibration(calib_file)


        with open(filename_source, "r") as f:
            labels = f.read().splitlines()
            if not len(labels) == 0:
                labels = [box.split(" ") for box in labels]
                types = np.asarray([[str(item) for idx, item in enumerate(box) if idx == 0] for box in labels])
                boxes = np.asarray([[float(item) for idx, item in enumerate(box) if idx > 0] for box in labels])

                # Convert XYZ_camera to XYZ_lidar
                boxes[:, 10:13] = calib.rect_to_lidar(boxes[:, 10:13])  # xyz

                # Z is bounding box center, not ground
                boxes[:, 12] += boxes[:, 7] / 2

                # convert rot_y_camera to rot_y_lidar
                boxes[:, 13] = -boxes[:, 13] - np.pi / 2

                # shift rot_y from range [-3/2*pi, pi/2] to range [-pi, pi]
                for idx, rot_y in enumerate(boxes[:, 13]):
                    if rot_y > np.pi:
                        boxes[idx, 13] = (rot_y % np.pi) - np.pi
                    if rot_y < -np.pi:
                        boxes[idx, 13] = (rot_y + np.pi) % np.pi

                # limit values to two decimal places
                boxes = np.around(boxes, 2)

                new_label = np.concatenate([types, boxes], axis=1)
            else:
                new_label = np.array([''])
            np.savetxt(filename_target, new_label, fmt="%s")
        pbar.update(1)

def main_convert_kitti_cam_to_kitti_carla(waymo_in_kitti_camera_coordinate_folder,
                                          label_source_path,
                                          label_target_path,
                                          calib_path):

    # Step 1 --> convert .bin files into .pcd files
    print("Converting velodyne .bin files to .pcd files...")
    process_bin_files(os.path.join(waymo_in_kitti_camera_coordinate_folder, "velodyne"))

    # Step 2 --> convert kitti reference camera coordinate system to kitti carla coordinate system.
    print("Converting ground truths from kitti reference camera coordinate to carla coordinate system..")
    os.makedirs(label_target_path, exist_ok=True)
    convert_to_lidar(label_source_path, label_target_path, calib_path)


if __name__ == "__main__":

    # Define paths to data
    waymo_in_kitti_camera_coordinate_folder = "/home/data/converted_to_kitti_format/waymo_in_kitti_format/training"
    label_source_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "label_all")
    label_target_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "label_kitti_carla")
    calib_path = os.path.join(waymo_in_kitti_camera_coordinate_folder, "calib")

    main_convert_kitti_cam_to_kitti_carla(waymo_in_kitti_camera_coordinate_folder,
                                          label_source_path,
                                          label_target_path,
                                          calib_path)