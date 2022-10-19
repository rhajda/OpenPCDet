from tqdm import tqdm
import numpy as np
import math
from typing import List, Dict
from pcdet.utils import box_utils
from .point_cloud_math import rotate_pointcloud_y

class AnalyzableDataset:
    def __init__(self, dataset: Dict):
        assert 'train_set' in dataset.keys()
        self.dataset = dataset
        self.infos = None

    def set_indices_train(self, indices: List[int]):
        frame_ids = []
        for i in tqdm(sorted(indices), 'Load frame ids'):
            frame_ids.append(self.dataset.train_set[i]['frame_id'])
        self.infos = self.dataset.train_set.get_infos(sample_id_list=frame_ids)

    def get_box_locations(self, get_center_coords: bool = False):
        """ Give the location of the annotated target 3d boxes ([0, 0, 0] in relative coords).
        """
        box_locations = []
        for info in tqdm(self.infos, 'Load box locations'):
            box_location = info['annos']['location'][0]
            # if get_center_coords:
            #     box_location += info['annos']['dimensions'][0]/2 # It should already be the center
            box_locations.append(box_location)
        return box_locations


    def get_box_rotation(self):
        """Give the rotation of the annotated 3d boxes (y_axis angle).
        """
        box_rotation = []
        for info in tqdm(self.infos, 'Load box orientations'):
            box_rotation.append(info['annos']['location'][0])
        return box_rotation

    def get_box_size(self, assume_hardcoded: bool = False):
        """Get size of the annotated 3d boxes.

        :param assume_hardcoded: Assumes all sizes are the same, defaults to False
        """
        if assume_hardcoded:
            return [self.infos[0]['annos']['dimensions'][0]]
        box_size = []
        for info in tqdm(self.infos, 'Load box size'):
            box_size.append(info['annos']['dimensions'][0])
        return box_size

    def get_normalized_target(self):
        result = {'target_point_num': [],
                  'normalized_target': []}
        for info in tqdm(self.infos, 'Load point_clouds'):
            point_cloud_loaded = self.dataset.train_set.get_lidar(info['point_cloud']['lidar_idx'])
            # point_cloud_original = self.dataset.train_set.get_lidar(current_idx)['points']
            target_point_flags = box_utils.in_hull(point_cloud_loaded,
                                                   box_utils.boxes_to_corners_3d(info['annos']['gt_boxes_lidar'])[0])
            result['target_point_num'].append(target_point_flags.sum())
            # assert info['annos']['num_points_in_gt'][0] == target_point_flags.sum()
            # num_points_original.append(info['annos']['num_points_in_gt'][0])

            target_point_cloud_loaded = point_cloud_loaded[target_point_flags]
            location_normalized_point_cloud = target_point_cloud_loaded - info['annos']['gt_boxes_lidar'][0][:3]
            rotation_normalized_point_cloud = rotate_pointcloud_y(location_normalized_point_cloud, info['annos']['rotation_y'])
            result['normalized_target'].append(rotation_normalized_point_cloud)
        return result

    def get_point_clouds(self):
        point_clouds = []
        for i in tqdm(self.indices, 'Load point_clouds'):
            point_clouds.append(self.dataset.train_set[i]['points'])
        return point_clouds

    def get_point_cloud_at_index(self, pcd_index: str):
        return self.dataset.train_set.get_lidar(pcd_index)


