'''
Pipeline for comparing real vs simulated data.
Currently written to be run on single GPU.
'''
import argparse
import datetime
import trimesh
from typing import List, Dict
import os
import easydict
from pathlib import Path
from typing import Tuple, List
import numpy as np
import random
from tabulate import tabulate
import subprocess
from tqdm import tqdm
from logging import Logger
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pyvista as pv  # Install with "vtk==8.1.2"!!
import pandas as pd

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from tools.comp_pipe.analyzable_dataset import AnalyzableDataset
import random
random.seed(1234)

ZERO_ARRAY = np.zeros((1, 3))
REAL_COLOR_R = "#FF7F7F" # (255, 127, 127)
SIM_COLOR_B = "#7F7FFF"  # (127, 127, 255)  # BLUE


def calc_pcd_pairwise_volume_iou(point_cloud_1: List[np.array],
                                 point_cloud_2: List[np.array]) -> np.array:
    result = []
    for pcd1, pcd2 in tqdm(zip(point_cloud_1, point_cloud_2),
                           "Calculating Volume IOU"):
        result.append(volume_iou_based_on_mesh_from_pcd(pcd1, pcd2))
    return np.array(result)


def volume_iou_based_on_mesh_from_pcd(point_cloud_1: np.array,
                                      point_cloud_2: np.array) -> float:
    if point_cloud_1.size == 0 or point_cloud_2.size == 0:
        return 0
    try:
        pcd1 = trimesh.PointCloud(point_cloud_1)
        pcd2 = trimesh.PointCloud(point_cloud_2)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=np.inf)
        mesh_pv_1 = pv.wrap(pcd1.convex_hull)
        mesh_pv_2 = pv.wrap(pcd2.convex_hull)
        union_volume = mesh_pv_1.boolean_union(mesh_pv_2).volume
        intersect_volume = mesh_pv_1.boolean_intersection(mesh_pv_2).volume
        result = 0 if union_volume == 0 else min(
            1, intersect_volume / union_volume)
    except Exception:  # no Volume IOU could be calculated
        return 0
    return result


def rotation_matrix(axis, theta):
    """
    Adapted from: https://stackoverflow.com/a/6802723/9621080
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_pointcloud_y(point_cloud, y_axis_angle):
    rotation_correction = rotation_matrix(axis=[0, 1, 0], theta=y_axis_angle)
    return point_cloud @ rotation_correction.T


def log_git_data(logger: Logger) -> None:
    commit_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    git_branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    logger.info(f'Git commit hash: {commit_hash}')
    logger.info(f'Git current branch: {git_branch}')


def parse_config() -> Tuple[argparse.Namespace, easydict.EasyDict]:
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config used for training e.g. cfgs/indy_models/pointrcnn.yaml' + \
                        '-> used to load data with specified augmentation')
    parser.add_argument('--extra_tag',
                        type=str,
                        default='default',
                        help='extra tag for this experiment')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--fix_random_seed',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--set',
                        dest='set_cfgs',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    # set data config
    cfg_from_yaml_file(str(Path(args.cfg_file).absolute()), cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(
        args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    assert [i for i in cfg.DATA_CONFIG.DATA_PROCESSOR if i['NAME'] == 'sample_points'] == [], \
        'sample_points not allowed, as analysis should be done on the original point_cloud'
    return args, cfg


def setup_datasets(args: argparse.Namespace,
                   cfg: easydict.EasyDict,
                   data_type: str = "real",
                   remove_missing_gt: bool = False,
                   disable_cfg_aug=False,
                   shuffle=True,
                   training=True) -> None:
    """_summary_
    :param args: _description_
    :param cfg: _description_
    :param data_type: can be "simulated" or "real" and defines which set to load.
    :param disable_cfg_aug: _description_, defaults to False
    :param shuffle: _description_, defaults to True
    :param training: Defines if the train_set will be used for training,
        this is relevant as in that case of (training=True) training sample
        with not gt boxes will be skipped, defaults to True.
    :return: _description_
    """
    if data_type == "real":
        cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.DATA_PATH_REAL
    elif data_type == "simulated":
        cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.DATA_PATH_SIM
    else:
        assert False

    if disable_cfg_aug:
        all_augmentation_names = [
            i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
        ]
        tmp_disable_aug = cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST
        if tmp_disable_aug != all_augmentation_names:
            cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = all_augmentation_names
        print("No data augmentation for the current data_set!")

    args.logger.info(
        f'**********************Setup datasets and training of {data_type} data:**********************'
    )
    dataset = easydict.EasyDict()
    # delay shuffle as we need the 1-to-1 mapping between real and simulated data for the analysis
    dataset.train_set, dataset.train_loader, dataset.train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=32,
        dist=False,
        workers=args.workers,
        logger=args.logger,
        training=training,
        merge_all_iters_to_one_epoch=False,
        total_epochs=10,
        shuffle=shuffle,
        remove_missing_gt=remove_missing_gt)
    return dataset


def make_matching_datasets(dataset_real: dict, dataset_sim: dict,
                           args: dict) -> None:
    """Make the passed datasets matching each other one-to-one.
    :param dataset_real: _description_
    :param dataset_sim: _description_
    :param args: _description_
    """

    # analyze the training samples
    args.logger.info(
        '**********************Start making datasets one-to-one matching**********************'
    )

    # FOR TRAINING SET:
    len_of_data_real = dataset_real.train_set.original_dataset_size
    len_of_data_sim = dataset_sim.train_set.original_dataset_size
    # assert len_of_data_real == len_of_data_sim

    used_real = [
        x['point_cloud']['lidar_idx']
        for x in dataset_real.train_set.kitti_infos
    ]
    used_sim = [
        x['point_cloud']['lidar_idx']
        for x in dataset_sim.train_set.kitti_infos
    ]

    diff_real_to_sim = set(used_real).difference(
        set(used_sim))  # <=> used_real - used_sim
    diff_sim_to_real = set(used_sim).difference(
        set(used_real))  # <=> used_sim - used_real

    args.logger.info(
        f"Difference of samples: these are used in training with real but not with sim \
(idx multiplied by 5 := 'frame_id'): \n{diff_real_to_sim}"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              )
    args.logger.info(
        f"Difference of samples: these are used in training with sim but not with real \
(idx multiplied by 5 := 'frame_id'): \n{diff_sim_to_real}"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              )
    args.logger.info(
        f"If both sets above are not empty we train with different training-set (correlation affected)."
    )

    counter_real_gt = len(used_real)
    counter_sim_gt = len(used_sim)

    table = [[
        'usable with gt:', f'{counter_real_gt} of {len_of_data_real}',
        f'{counter_sim_gt} of {len_of_data_sim}'
    ],
             [
                 'ratio with gt:',
                 f'{counter_real_gt/ len_of_data_real* 100:.1f}%',
                 f'{counter_sim_gt/len_of_data_sim * 100:.1f}%'
             ]]

    args.logger.info(
        f"As \"make_datasets_one_to_one\" option is used the samples not both sets are removed."
    )
    to_del_s = [
        x for x in dataset_sim.train_set.kitti_infos
        if x['point_cloud']['lidar_idx'] in diff_sim_to_real
    ]
    to_del_r = [
        x for x in dataset_real.train_set.kitti_infos
        if x['point_cloud']['lidar_idx'] in diff_real_to_sim
    ]

    for i in tqdm(to_del_s,
                  desc="Remove samples not in intersection from sim:"):
        dataset_sim.train_set.kitti_infos.remove(i)

    for i in tqdm(to_del_r,
                  desc="Remove samples not in intersection from real:"):
        dataset_real.train_set.kitti_infos.remove(i)

    assert len(dataset_sim.train_set) == len(dataset_real.train_set)
    args.logger.info(
        f"\nDataset size after removing samples not in intersection and unused in training: \
        {len(dataset_sim.train_set)} of originally {len_of_data_real} (both real and sim) \n"
    )
    counter_real = len(dataset_real.train_set)
    counter_sim = len(dataset_sim.train_set)

    table += [[
        'usable one_to_one:', f'{counter_real} of {len_of_data_real}',
        f'{counter_sim} of {len_of_data_sim}'
    ],
              [
                  'ratio one_to_one:',
                  f'{counter_real/ len_of_data_real* 100:.1f}%',
                  f'{counter_sim/len_of_data_sim * 100:.1f}%'
              ]]

    args.logger.info(
        "Check how many of the training samples from the given datasets contain gt_obj_boxes and \
thus will be used for training with the current configuration.\n\n"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  + tabulate(
            table,
            headers=['Attribute', 'dataset_real_train', 'dataset_sim_train'],
            tablefmt='orgtbl') + "\n")

    # CAN BE REMOVED: IT KEPT AS WHOLE
    # FOR VALIDATION SET:
    args.logger.info(f"In the validation set samples are not checked.")

    args.logger.info(
        '**********************End making datasets one-to-one matching**********************'
    )


def log_min_max_mean(args, title, header, train_real, train_sim, to_csv=None):
    headers = ['Attribute', 'real_aug', 'simulated_aug', "num_of_samples"]
    stacked_values = [np.vstack(x) for x in [train_real, train_sim]]
    mean_values = ['mean'] + [
        list(np.round(np.mean(x, axis=(0)), 4)) for x in stacked_values
    ] + [len(train_real)]
    min_values = ['min'] + [
        list(np.round(np.min(x, axis=(0)), 4)) for x in stacked_values
    ] + [len(train_real)]
    max_values = ['max'] + [
        list(np.round(np.max(x, axis=(0)), 4)) for x in stacked_values
    ] + [len(train_real)]
    args.logger.info(title + "\n" +
                     tabulate([mean_values, min_values, max_values],
                              headers=headers,
                              tablefmt='orgtbl'))
    if to_csv is not None:
        target_csv = args.output_dir / to_csv
        with target_csv.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(mean_values)
            writer.writerow(min_values)
            writer.writerow(max_values)


def analyze_data_pairs(dataset_real, dataset_sim, args=None, cfg=None, num=20):
    """
    Compare the given datasets and log results.
    :param dataset_real: _description_
    :param dataset_sim: _description_
    :param dataset_real_no_aug: _description_
    :param dataset_sim_no_aug: _description_
    :param args: _description_
    :param num: _description_, defaults to 20
    Note: supports additional comparisons with augmented data, but this is not used for the indy dataset.
    """
    args.logger.info(
        '**********************Start comparison of data pairs**********************'
    )
    sns.set_theme(style="ticks")
    # check for matching lengths
    train_len = len(dataset_real.train_set)
    assert train_len == len(dataset_sim.train_set), \
        'The number of train data pairs in real and simulated dataset should be the same'

    if num == -1:  # use all samples
        idxs_train = range(0, train_len - 1)
    else:
        idxs_train = random.sample(range(0, train_len), k=num)

    dataset_real_analyze = AnalyzableDataset(dataset_real)
    dataset_real_analyze.set_indices_train(idxs_train)

    dataset_sim_analyze = AnalyzableDataset(dataset_sim)
    dataset_sim_analyze.set_indices_train(idxs_train)

    real_box_locations = dataset_real_analyze.get_box_locations()
    sim_box_locations = dataset_sim_analyze.get_box_locations()
    real_target_pointclouds = dataset_real_analyze.get_normalized_target()
    sim_target_pointclouds = dataset_sim_analyze.get_normalized_target()

    create_tables = False
    if create_tables:
        log_min_max_mean(
            args,
            title=
            "Some metrics on the selected corresponding point cloud pairs. Due to the sampling process to guarantee same size => points wont be the same each time!",
            header=[
                'Attribute', 'real_aug', 'simulated_aug', 'real_no_aug',
                'simulated_no_aug'
            ],
            train_real=dataset_real_analyze.get_point_clouds(),
            train_sim=dataset_sim_analyze.get_point_clouds(),
            to_csv="obj_and_background_point_ranges.csv")
        # Locations of Bboxes:
        log_min_max_mean(args,
                         title="Bounding Box locations",
                         header=['Attribute', 'real', 'simulated'],
                         train_real=real_box_locations,
                         train_sim=sim_box_locations,
                         to_csv="obj_location_ranges.csv")
        # Distances between Bboxes:
        distances_loc = np.linalg.norm(np.array(real_box_locations) -
                                       np.array(sim_box_locations),
                                       axis=1)
        log_min_max_mean(args,
                         title="L-2 distance between bounding box locations",
                         header=[
                             'Attribute', 'real_aug', 'simulated_aug',
                             'real_no_aug', 'simulated_no_aug'
                         ],
                         train_real=distances_loc,
                         train_sim=distances_loc,
                         to_csv="obj_box_location_l2_distance.csv")
        # Rotation of Bboxes:
        log_min_max_mean(
            args,
            title="Absolute difference in rotation(heading angle)",
            header=[
                'Attribute', 'real_aug', 'simulated_aug', 'real_no_aug',
                'simulated_no_aug'
            ],
            train_real=dataset_real_analyze.get_box_rotation(),
            train_sim=dataset_sim_analyze.get_box_rotation(),
            to_csv="obj_orientation_angle_ranges.csv")
        # Diff. between rotations:
        distances_rot = np.abs(
            np.array(dataset_real_analyze.get_box_rotation()) -
            np.array(dataset_sim_analyze.get_box_rotation()))
        log_min_max_mean(
            args,
            title="Absolute difference in rotation(heading angle)",
            header=[
                'Attribute', 'real_aug', 'simulated_aug', 'real_no_aug',
                'simulated_no_aug'
            ],
            train_real=distances_rot,
            train_sim=distances_rot,
            to_csv="obj_orientation_angle_ranges.csv")
        log_min_max_mean(
            args,
            title="Number of target points",
            header=['Attribute', 'real', 'simulated'],
            train_real=real_target_pointclouds['target_point_num'],
            train_sim=sim_target_pointclouds['target_point_num'],
            to_csv="obj_number_of_points_ranges.csv")

    # CREATE BUCKETS:
    real_box_dist = [((1, -1)[x[0] < 0]) * np.linalg.norm(x)
                     for x in real_box_locations]
    real_sorted = sorted(zip(real_box_dist,
                             real_target_pointclouds['normalized_target']),
                         key=lambda x: np.abs(x[0]))
    sim_box_dist = [((1, -1)[x[0] < 0]) * np.linalg.norm(x)
                    for x in sim_box_locations]
    sim_sorted = sorted(zip(sim_box_dist,
                            sim_target_pointclouds['normalized_target']),
                        key=lambda x: np.abs(x[0]))
    thresholds = list(np.arange(0, 99, 33.33)) + [100]
    bins_real = binning_based_on_threshold(thresholds, real_sorted)
    bins_sim = binning_based_on_threshold(thresholds, sim_sorted)

    # 2D HISTOGRAM PLOTS CAR:
    scatter_sideplot = True
    if scatter_sideplot:
        for i in [0, 0.1, 1, 2]:
            point_clouds = np.concatenate(bins_real[0]['pcds'])[:25000]
            side_plot(args, point_clouds, dim_to_skip=i, reverse=(i == 0.1))

    # BIRD EYE VIEW PLOTS OF ENV:
    scatter_plot_bev = True
    if scatter_plot_bev:
        point_clouds_s = dataset_sim_analyze.get_point_clouds()
        point_clouds_r = dataset_real_analyze.get_point_clouds()
        plot_full_pcd(args,
                      point_clouds_s[-1],
                      name="full_pcd_s.pdf",
                      s=7,
                      color=SIM_COLOR_B)
        plot_full_pcd(args,
                      point_clouds_r[-1],
                      name="full_pcd_r.pdf",
                      s=7,
                      color=REAL_COLOR_R)

    # IOU:
    make_iou_tables = False
    if make_iou_tables:
        vicinity_slack = 5  # how many successing an preciding point clouds to compare to select the best match.
        iou_distances = []
        # TODO: Denoise!
        for i, br in tqdm(enumerate(dataset_real_analyze.infos)):
            br_idx = br['point_cloud']['lidar_idx']
            br_loc = br['annos']['location'][0]
            if np.linalg.norm(br_loc) < 33:
                closest_sample_idx = find_closest_sample_index(
                    br_loc, dataset_sim_analyze, i, vicinity_slack)
                pcd_1 = dataset_real_analyze.get_point_cloud_at_index(br_idx)
                pcd_2 = dataset_sim_analyze.get_point_cloud_at_index(
                    closest_sample_idx)
                iou_distances.append(
                    volume_iou_based_on_mesh_from_pcd(pcd_1, pcd_2))

        iou_distances = np.array(iou_distances)
        print(np.min(iou_distances), np.max(iou_distances),
              np.mean(iou_distances))
        if iou_distances.size > 0:
            log_min_max_mean(
                args,
                title="IOU distance",
                header=['Attribute - volume_iou', 'real', 'simulated'],
                train_real=iou_distances,
                train_sim=iou_distances,
                to_csv=f"volume_iou_0to100_ranges.csv")

    # Plot 2: HOW CARS LOOK TO THE NETWORK:
    plot_car_3d = True
    if plot_car_3d:
        num_points = [9731, 20000]  # This looks the best
        alphas = [0.85]  # This looks the best
        # REAL:
        try:
            side_plot_car_3d_for_range(args,
                                       bins=bins_real,
                                       alphas=alphas,
                                       num_points=num_points,
                                       data_type="real")
        except ValueError:
            print("WARNING: not enough points to available.")
        # SIMULATED:
        try:
            side_plot_car_3d_for_range(args,
                                       bins=bins_sim,
                                       alphas=alphas,
                                       num_points=num_points,
                                       data_type="sim")
        except ValueError:
            print("WARNING: not enough points to available.")

    args.logger.info(
        '********************** End comparison of data pairs **********************'
    )


def find_closest_sample_index(target_loc, data_set_analyze, i, vicinity_slack):
    idx_from = max(i - vicinity_slack, 0)
    idx_to = min(i + vicinity_slack, len(data_set_analyze.infos) - 1)
    closest_cloud = np.inf, -1
    for sample in data_set_analyze.infos[idx_from:idx_to]:
        dist = np.linalg.norm(target_loc - sample['annos']['location'][0])
        if dist < closest_cloud[0]:
            closest_cloud = dist, sample['point_cloud']['lidar_idx']
    return closest_cloud[1]  # return corresponding index


def side_plot_car_3d_for_range(
    args,
    bins: List[Dict],
    alphas: List[int],
    num_points: List[int],
    data_type: str,
    use_color_gradient: bool = True,
    car_size=np.array([4.88, 1.9, 1.18])) -> None:
    for pcd_bin in bins:
        for alpha in alphas:
            for num_point in num_points:
                side_plot_car_3d(args, pcd_bin, num_point, alpha, data_type,
                                 use_color_gradient, car_size)


def side_plot_car_3d(
    args: Dict,
    pcd_bin: List[Dict],
    num_point: int,
    alpha: float,
    data_type: str,
    use_color_gradient: bool = True,
    car_size=np.array([4.88, 1.9, 1.18])) -> None:

    name = f'car_plot_3d_{data_type}_NORM_range{str(int(pcd_bin["th"][0]))}_{str(int(pcd_bin["th"][1]))}' + \
        f'points{num_point}_alpha{str(alpha).replace(".", "")[1:]}.pdf'

    points = np.concatenate(pcd_bin['pcds'], axis=0)
    if num_point != -1 and num_point <= points.shape[
            0]:  # randomly sample num_points
        index = np.random.choice(points.shape[0], num_point, replace=False)
        points = points[index]
        # add two negative and two positive scans with the most points.
        best_pos = (0, None)
        best_neg = (0, None)
        for i, loc in enumerate(pcd_bin['loc']):
            if loc > 0 and pcd_bin['count'][i] > best_pos[0]:
                best_pos = (pcd_bin['count'][i], pcd_bin['pcds'][i])
            elif loc <= 0 and pcd_bin['count'][i] > best_neg[0]:
                best_neg = (pcd_bin['count'][i], pcd_bin['pcds'][i])
        if best_pos[1] is not None:
            points = np.concatenate([points, best_pos[1]], axis=0)
        if best_neg[1] is not None:
            points = np.concatenate([points, best_neg[1]], axis=0)
    else:
        raise ValueError(
            f"'num_point': {num_point} more than available points: {points.shape[0]}"
        )

    norm_c = None
    if use_color_gradient:
        norm_c = (points[:, 1] - points[:, 1].min())
        norm_c = norm_c / norm_c.max()
        norm_c[norm_c >= 0.5] = -1 * norm_c[norm_c >= 0.5] + 1
        norm_c[np.argmax(norm_c)] = 1
        alpha = np.ones_like(norm_c) * alpha
        alpha[np.argmax(norm_c)] = 0.0
    if data_type == "real":
        cmap = 'viridis'
    elif data_type == "sim":
        cmap = None
    elif data_type == "sim_noise":
        cmap = 'inferno'
    else:
        raise ValueError(
            "'data_type' needs to be in ['real', 'sim', 'sim_noise']!")

    points_axis_split = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_xlim([
        -car_size[0] / 2 - car_size[0] * 0.15,
        car_size[0] / 2 + car_size[0] * 0.15
    ])
    ax.set_ylim([
        -car_size[1] / 2 - car_size[1] * 0.15,
        car_size[1] / 2 + car_size[1] * 0.15
    ])
    ax.set_zlim([-car_size[2] / 2, car_size[2] / 2 + car_size[2] * 0.2])
    ax.scatter(*points_axis_split, c=norm_c, alpha=alpha, cmap=cmap)
    plt.savefig(args.output_dir / name, bbox_inches='tight', dpi=300)


def plot_full_pcd(args, points, name="full_pcd.pdf", s=5, color=None):
    points_axis_split = points[:, 0], points[:, 1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.set_xlabel('meter')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-50, 50])
    ax.scatter(*points_axis_split, s=s, color=color)
    plt.savefig(args.output_dir / ("top_" + name),
                bbox_inches='tight',
                dpi=300)


def side_plot(args,
              point_clouds,
              dim_to_skip,
              car_size=np.array([4.88, 1.9, 1.18]),
              reverse=True):
    """
    :param args:
    :param point_clouds:
    :param dim_to_skip:
    :param car_size: (length, width, height), defaults to np.array([4.88, 1.9, 1.18])

    Color closer points differently based distance => use varaible c for the removed dim
    See: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """
    dims_to_take = [0, 1, 2]
    dims_to_take.remove((0 if dim_to_skip == 0.1 else dim_to_skip))
    dim_to_skip = int(dim_to_skip)
    point_clouds = point_clouds[point_clouds[:, dim_to_skip].argsort(
    )[::-1]] if reverse else point_clouds[point_clouds[:,
                                                       dim_to_skip].argsort()]
    points = point_clouds[:, dims_to_take[0]], point_clouds[:, dims_to_take[1]]
    car_size = car_size.astype(np.int32)
    fig_size = 9 * 0.8, int(
        12 * (car_size[dims_to_take[1]] / car_size[dims_to_take[0]])) * 0.8
    if dim_to_skip in [0.1, 0]:
        fig_size = 5.8 * 0.8, 3.2 * 0.8
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()
    axis_labels = ['x [m]', 'y [m]', 'z [m]']
    axis_labels = axis_labels[:dim_to_skip] + axis_labels[dim_to_skip+1:]
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    norm_c = (point_clouds[:, dim_to_skip] -
              point_clouds[:, dim_to_skip].min())
    norm_c = norm_c / norm_c.max()
    norm_c = norm_c[::-1] if not reverse else norm_c
    alpha = (norm_c[::-1] + 0.1) * 0.3 if dim_to_skip == 0 else 0.3
    s = alpha * 2 * 24 if dim_to_skip == 0 else 6
    ax.scatter(*points, c=norm_c, alpha=alpha, s=s,
               cmap='viridis')  # c=1 : black=Front , c=0 : white
    plt.savefig(
        args.output_dir /
        f"car_angle_plot_real_skipdim{dim_to_skip}_{point_clouds.shape[0]}points_{'R'*reverse}.pdf",
        dpi=300,
        bbox_inches='tight')


def binning_based_on_threshold(
        thresholds: List[int],
        sorted_values: List[Tuple[np.array, np.array]]) -> List[Dict]:
    bins = [{
        "pcds": [],
        "th": [],
        "loc": [],
        "count": []
    } for _ in range(len(thresholds) - 1)]
    for i in range(len(thresholds) - 1):
        bins[i]["th"] = (thresholds[i], thresholds[i + 1])
        for loc, pc in sorted_values:
            if thresholds[i] <= np.abs(loc) <= thresholds[i + 1]:
                bins[i]["pcds"].append(pc)
                bins[i]["count"].append(pc.shape[0])
                bins[i]["loc"].append(loc)
    return bins


def main():
    root = (Path(__file__).parent / '../..').resolve()
    os.chdir(
        str(root /
            'tools'))  # directory change makes path handling easier later on
    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # Set up logging and output directories
    args.output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.output_dir / "logs" / (
        'log_pipeline_%s.txt' %
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    args.logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    args.output_dir = args.output_dir / datetime.datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # start logging
    args.logger.info(
        '**********************Start logging**********************')
    log_git_data(args.logger)

    # ######### LOAD DATA SETS: ##########
    dataset_real = setup_datasets(args,
                                  cfg,
                                  data_type="real",
                                  shuffle=False,
                                  remove_missing_gt=True)
    dataset_sim = setup_datasets(args,
                                 cfg,
                                 data_type="simulated",
                                 shuffle=False,
                                 remove_missing_gt=True)

    make_matching_datasets(dataset_real, dataset_sim, args)

    # ######### DATASET ANALYSIS: ########## TODO:fix aug vs no aug setting!- CAN PROBABLY BE REMOVED
    dataset_real_original = None
    dataset_sim_original = None
    # no augmentation:
    all_augmentation_names = [
        i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
    ]
    if cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST != all_augmentation_names:
        dataset_real_original = setup_datasets(args,
                                               cfg,
                                               data_type="real",
                                               disable_cfg_aug=False,
                                               shuffle=False)
        dataset_sim_original = setup_datasets(args,
                                              cfg,
                                              data_type="simulated",
                                              disable_cfg_aug=False,
                                              shuffle=False)
        make_matching_datasets(dataset_real_original, dataset_sim_original,
                               args)

    # analyze, compare and visualize *num* samples of 1-to-1 correspondences:
    analyze_data_pairs(dataset_real, dataset_sim, args, cfg, num=-1)


if __name__ == "__main__":
    main()
