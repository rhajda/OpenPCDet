'''
Pipeline for comparing real vs simulated data.
Currently written to be run on single GPU.
'''
import time
import argparse
import datetime
import glob
import os
import easydict
from pathlib import Path
from typing import Tuple, List
import numpy as np
import time
import random
from tabulate import tabulate
import subprocess
from tqdm import tqdm
from logging import Logger
import csv
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from tools.comp_pipe.analyzable_dataset import AnalyzableDataset

ZERO_ARRAY = np.zeros((1, 3))

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
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    git_branch =  subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    logger.info(f'Git commit hash: {commit_hash}')
    logger.info(f'Git current branch: {git_branch}')

def parse_config() -> Tuple[argparse.Namespace, easydict.EasyDict]:
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config used for training e.g. cfgs/indy_models/pointrcnn.yaml' + \
                        '-> used to load data with specified augmentation')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    # set data config
    cfg_from_yaml_file(str(Path(args.cfg_file).absolute()), cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    assert [i for i in cfg.DATA_CONFIG.DATA_PROCESSOR if i['NAME'] == 'sample_points'] == [], \
        'sample_points not allowed, as analisys should be done on the original point_cloud' # TODO this might be a contradiction w.r.t to idea of this script!!
    return args, cfg

def setup_datasets(args: argparse.Namespace, cfg: easydict.EasyDict, data_type:str = "real",
                   remove_missing_gt: bool = False, disable_cfg_aug=False, shuffle=True, training=True) -> None:
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
        all_augmentation_names = [i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
        tmp_disable_aug = cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST
        if tmp_disable_aug != all_augmentation_names:
            cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = all_augmentation_names
        print("No data augmentation for the current data_set!")

    args.logger.info(f'**********************Setup datasets and training of {data_type} data:**********************')
    dataset = easydict.EasyDict()
    # delay shuffle as we need the 1-to-1 mapping between real and simulated data for the analysis
    dataset.train_set, dataset.train_loader, dataset.train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=32,
        dist=False, workers=args.workers,
        logger=args.logger,
        training=training,
        merge_all_iters_to_one_epoch=False,
        total_epochs=10,
        shuffle=shuffle,
        remove_missing_gt = remove_missing_gt
    )
    return dataset

def make_matching_datasets(dataset_real: dict, dataset_sim: dict, args: dict) -> None:
    """Make the passed datasets matching each other one-to-one.
    :param dataset_real: _description_
    :param dataset_sim: _description_
    :param args: _description_
    """

    # analyze the training samples
    args.logger.info('**********************Start making datasets one-to-one matching**********************')

    # FOR TRAINING SET:
    len_of_data_real = dataset_real.train_set.original_dataset_size
    len_of_data_sim = dataset_sim.train_set.original_dataset_size
    # assert len_of_data_real == len_of_data_sim

    used_real = [x['point_cloud']['lidar_idx'] for x in dataset_real.train_set.kitti_infos]
    used_sim =  [x['point_cloud']['lidar_idx'] for x in dataset_sim.train_set.kitti_infos]

    diff_real_to_sim = set(used_real).difference(set(used_sim)) # <=> used_real - used_sim
    diff_sim_to_real= set(used_sim).difference(set(used_real)) # <=> used_sim - used_real

    args.logger.info(f"Difference of samples: these are used in training with real but not with sim \
(idx multiplied by 5 := 'frame_id'): \n{diff_real_to_sim}")
    args.logger.info(f"Difference of samples: these are used in training with sim but not with real \
(idx multiplied by 5 := 'frame_id'): \n{diff_sim_to_real}")
    args.logger.info(f"If both sets above are not empty we train with different training-set (correlation affected).")

    counter_real_gt = len(used_real)
    counter_sim_gt = len(used_sim)

    table = [['usable with gt:', f'{counter_real_gt} of {len_of_data_real}', f'{counter_sim_gt} of {len_of_data_sim}'],
            ['ratio with gt:', f'{counter_real_gt/ len_of_data_real* 100:.1f}%', f'{counter_sim_gt/len_of_data_sim * 100:.1f}%']]

    args.logger.info(f"As \"make_datasets_one_to_one\" option is used the samples not both sets are removed.")
    to_del_s = [x for x in dataset_sim.train_set.kitti_infos if x['point_cloud']['lidar_idx'] in diff_sim_to_real]
    to_del_r = [x for x in dataset_real.train_set.kitti_infos if x['point_cloud']['lidar_idx'] in diff_real_to_sim]

    for i in tqdm(to_del_s, desc="Remove samples not in intersection from sim:"):
        dataset_sim.train_set.kitti_infos.remove(i)

    for i in tqdm(to_del_r, desc="Remove samples not in intersection from real:"):
        dataset_real.train_set.kitti_infos.remove(i)

    assert len(dataset_sim.train_set) == len(dataset_real.train_set)
    args.logger.info(f"\nDataset size after removing samples not in intersection and unused in training: \
        {len(dataset_sim.train_set)} of originally {len_of_data_real} (both real and sim) \n")
    counter_real = len(dataset_real.train_set)
    counter_sim = len(dataset_sim.train_set)

    table += [['usable one_to_one:', f'{counter_real} of {len_of_data_real}', f'{counter_sim} of {len_of_data_sim}'],
        ['ratio one_to_one:', f'{counter_real/ len_of_data_real* 100:.1f}%', f'{counter_sim/len_of_data_sim * 100:.1f}%']]

    args.logger.info("Check how many of the training samples from the given datasets contain gt_obj_boxes and \
thus will be used for training with the current configuration.\n\n" + tabulate(table,
        headers=['Attribute', 'dataset_real_train', 'dataset_sim_train'], tablefmt='orgtbl') + "\n")

    # CAN BE REMOVED: IT KEPT AS WHOLE
    # FOR VALIDATION SET:
    args.logger.info(f"In the validation set samples are not checked.")


    args.logger.info('**********************End making datasets one-to-one matching**********************')

def log_min_max_mean(args, title, header, train_real, train_sim, to_csv=None):
    headers = ['Attribute', 'real_aug', 'simulated_aug', "num_of_samples"]
    stacked_values = [np.vstack(x) for x in [train_real, train_sim]]
    mean_values = ['mean'] + [list(np.round(np.mean(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    min_values = ['min'] + [list(np.round(np.min(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    max_values = ['max'] + [list(np.round(np.max(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    args.logger.info(title + "\n" + tabulate(
        [ mean_values, min_values, max_values], headers=headers, tablefmt='orgtbl'))
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
    args.logger.info('**********************Start comparison of data pairs**********************')
    # check for matching lengths
    train_len = len(dataset_real.train_set)
    assert train_len == len(dataset_sim.train_set), \
        'The number of train data pairs in real and simulated dataset should be the same'

    if num == -1: # use all samples
        idxs_train = range(0, train_len-1)
    else:
        idxs_train = random.sample(range(0, train_len), k=num)

    dataset_real_analyze = AnalyzableDataset(dataset_real)
    dataset_real_analyze.set_indices_train(idxs_train)
    dataset_sim_analyze = AnalyzableDataset(dataset_sim)
    dataset_sim_analyze.set_indices_train(idxs_train)

    log_min_max_mean(args,
                     title="Some metrics on the selected corresponding point cloud pairs. Due to the sampling process to guarantee same size => points wont be the same each time!",
                     header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                     train_real=dataset_real_analyze.get_point_clouds(),
                     train_sim=dataset_sim_analyze.get_point_clouds(), to_csv="obj_and_background_point_ranges.csv")

    # Location coordinates of all points in bbox: # TODO

    real_box_locations = dataset_real_analyze.get_box_locations()
    sim_box_locations = dataset_sim_analyze.get_box_locations()
    real_target_pointclouds = dataset_real_analyze.get_normalized_target()
    sim_target_pointclouds = dataset_sim_analyze.get_normalized_target()
    # Locations of Bboxes: ALSO: SEE PLOT FOR THIS!
    log_min_max_mean(args,
                    title="Bounding Box locations",
                    header=['Attribute', 'real', 'simulated'],
                    train_real=real_box_locations, train_sim=sim_box_locations, to_csv="obj_location_ranges.csv")
    # Distances between Bboxes:
    distances_loc = np.linalg.norm(np.array(real_box_locations) - np.array(sim_box_locations), axis=1)
    log_min_max_mean(args,
                title="L-2 distance between bounding box locations",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=distances_loc, train_sim=distances_loc,
                to_csv="obj_box_location_l2_distance.csv")
    # Rotation of Bboxes:
    log_min_max_mean(args,
                title="Absolute difference in rotation(heading angle)",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=dataset_real_analyze.get_box_rotation(),
                train_sim=dataset_sim_analyze.get_box_rotation(), to_csv="obj_orientation_angle_ranges.csv")
    # Diff. between rotations:
    distances_rot = np.abs(np.array(
        dataset_real_analyze.get_box_rotation()) - np.array(dataset_sim_analyze.get_box_rotation()))
    log_min_max_mean(args,
                title="Absolute difference in rotation(heading angle)",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=distances_rot, train_sim=distances_rot,
                to_csv="obj_orientation_angle_ranges.csv")
    log_min_max_mean(args,
                    title="Number of target points",
                    header=['Attribute', 'real', 'simulated'],
                    train_real=real_target_pointclouds['target_point_num'],
                    train_sim=sim_target_pointclouds['target_point_num'], to_csv="target_num_points_ranges.csv")

    # PLOTS:
    sns.set_theme(style="ticks")
    # WITH BUCKETS:
    # sort all point clouds into  buckets:
    import time
    t0 = time.time()
    real_box_dist = [np.linalg.norm(x) for x in real_box_locations]
    real_sorted = sorted(zip(real_box_dist, real_target_pointclouds['normalized_target']), key=lambda x: x[0])
    sim_box_dist = [np.linalg.norm(x) for x in sim_box_locations]
    sim_sorted = sorted(zip(sim_box_dist, sim_target_pointclouds['normalized_target']), key=lambda x: x[0])
    thresholds = list(np.arange(0, 100, 33.33)) + [100]
    bins_real = sort_based_on_thres(thresholds, real_sorted)
    bins_sim = sort_based_on_thres(thresholds, sim_sorted)
    print(time.time() - t0)

    # 2D HISTOGRAM PLOTS GENERAL DATA:
    # import pandas as pd
    # df_normal_a = pd.DataFrame(data = [sum([cur_bin.shape[0] for cur_bin in bins]) for bins in bins_sim],
    #                            columns=['average number of points'])
    # df_normal_b = pd.DataFrame(data = [xx + 5 for xx in thresholds[:-1]],
    #                            columns=['meter']).assign(group = 'Group B')
    # score_data = pd.concat([df_normal_a, df_normal_b])
    # point_sum_per_bin = [(i*5, sum([cur_bin.shape[0] for cur_bin in bins])) for i, bins in enumerate(bins_sim)]
    # point_avg_per_bin = [(i*5, np.mean([cur_bin.shape[0] for cur_bin in bins])) for i, bins in enumerate(bins_sim)]
    # bins_points = []
    # data = pd.DataFrame(data = point_sum_per_bin)
    # fig = sns.histplot(data=data).get_figure()
    # fig.axes[0].set_xlabel('total num of points w.r.t. range')
    # fig.savefig("hist_of_total_point_num.png", bbox_inches='tight')
    # fig = sns.histplot(data=point_avg_per_bin).get_figure()
    # fig.axes[0].set_xlabel('average number of points w.r.t. range')
    # fig.savefig("hist_of_avg_point_num.png", bbox_inches='tight')

    # 2D HISTOGRAM PLOTS CAR:
    # color closer points differently based distance => use varaible c for the removed dim
    # See: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    scatter_sideplot = False
    if scatter_sideplot:
        for i in [0, 0.1, 1, 2]:
            point_clouds = np.concatenate(bins_real[0])
            side_plot(args, point_clouds, dim_to_skip=i, reverse=(i==0.1))
    print()

    # point_cloud_range = [int(x/4) for x in cfg.DATA_CONFIG["POINT_CLOUD_RANGE"]]
    # # Plot 1: DATA DISTRIBUTIONS!
    # # REAL:
    # x = np.array(real_box_locations)[:, 0]
    # y = np.array(real_box_locations)[:, 1]
    # plot = sns.jointplot(x=x, y=y, color="#4CB391")
    # plot.fig.savefig(args.output_dir / 'hexbin_plot_real_bbox_locations_sns.png', bbox_inches='tight')
    # plt.hexbin(x, y, gridsize=(point_cloud_range[3]-point_cloud_range[0], point_cloud_range[4]-point_cloud_range[1]))
    # plt.savefig(args.output_dir / 'hexbin_plot_real_bbox_locations.png', bbox_inches='tight')
    # # SIMULATED:
    # x = np.array(sim_box_locations)[:, 0]
    # y = np.array(sim_box_locations)[:, 1]
    # plot = sns.jointplot(x=x, y=y, color="#4CB391")
    # plot.fig.savefig(args.output_dir / 'hexbin_plot_sim_bbox_locations_sns.png', bbox_inches='tight')
    # plt.hexbin(x, y, gridsize=(point_cloud_range[3]-point_cloud_range[0], point_cloud_range[4]-point_cloud_range[1]))
    # plt.savefig(args.output_dir / 'hexbin_plot_sim_bbox_locations.png', bbox_inches='tight', dpi=300)


    # Plot 2: HOW CARS LOOK TO THE NETWORK:
    # for alpha in [0.001, 0.002, 0.005, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     # REAL:
    #     for b, th in zip(bins_real, thresholds[1:]):
    #         name = f'car_plot_real_normalized_to_{int(th)}_alpha{str(alpha).replace(".", "")}.png'
    #         make_plot(args, b, name, alpha, num_points=-1)
    #     # SIMULATED:
    #     for b, th in zip(bins_sim, thresholds[1:]):
    #         name = f'car_plot_sim_normalized_to_{int(th)}_alpha{str(alpha).replace(".", "")}.png'
    #         make_plot(args, b, name, alpha, num_points=-1)
    # for num_points, alpha in zip([40, 80, 100, 200, 300],[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    #     # REAL:
    #     for i, (b, th) in enumerate(zip(bins_real, thresholds[1:])):
    #         name = help_i(th, num_points, alpha, typ="real")
    #         make_plot(args, b, name, alpha, num_points)
    #     # SIMULATED:
    #     for i, (b, th) in enumerate(zip(bins_sim, thresholds[1:])):
    #         name = help_i(th, num_points, alpha, typ="sim")
    #         make_plot(args, b, name, alpha, num_points)

    args.logger.info('********************** End comparison of data pairs **********************')

def side_plot(args, point_clouds, dim_to_skip, car_size = np.array([4.88, 1.9, 1.18]), reverse=True):
    """
    :param args:
    :param point_clouds:
    :param dim_to_skip:
    :param car_size: (length, width, height), defaults to np.array([4.88, 1.9, 1.18])
    """
    dims_to_take = [0, 1, 2]
    dims_to_take.remove((0 if dim_to_skip == 0.1 else dim_to_skip))
    dim_to_skip = int(dim_to_skip)
    point_clouds = point_clouds[point_clouds[:, dim_to_skip].argsort()[::-1]] if reverse else point_clouds[point_clouds[:, dim_to_skip].argsort()]
    points = point_clouds[:, dims_to_take[0]], point_clouds[:, dims_to_take[1]]
    car_size = car_size.astype(np.int32)
    fig_size = 9, int(12 * (car_size[dims_to_take[1]] / car_size[dims_to_take[0]]))
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()
    ax.set_xlabel('meter')
    norm_c = (point_clouds[:, dim_to_skip] - point_clouds[:, dim_to_skip].min())
    norm_c = norm_c / norm_c.max()
    norm_c = norm_c[::-1] if not reverse else norm_c
    s = 12 if dim_to_skip == 0 else 6
    alpha = (norm_c[::-1] + 0.1) * 0.3 if dim_to_skip == 0 else 0.3
    ax.scatter(*points, c=norm_c, alpha=alpha, s=s) # c=1 : black=Front , c=0 : white
    plt.savefig(args.output_dir / f"car_angle_plot_real_skipdim{dim_to_skip}_{point_clouds.shape[0]}points_{'R'*reverse}.png", dpi=300)

def help_i(th, num_points, alpha, typ):
    # to 33 use 40 pointclouds -> to 66 use 80 -> to 100 use 160
    name = f'car_plot_{typ}_normalized_{int(th-33)}_to_{int(th)}_alpha{str(alpha).replace(".", "")}_points_{num_points}.png'
    return name

def make_plot(args, b, name, alpha, num_points):
    points_all = np.concatenate(b[:num_points], axis=0) if num_points != -1 else np.concatenate(b, axis=0)
    plot_pointcloud_stacked(points_all, alpha, name, args)

def plot_pointcloud_stacked(point_clouds, alpha, name, args):
    points = point_clouds[:,0], point_clouds[:,1], point_clouds[:,2]
    color = "blue" if "_real_" in name else "red"
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('meter')
    ax.scatter(*points, c=color, alpha=alpha)
    plt.savefig(args.output_dir / name, bbox_inches='tight', dpi=300)

def sort_based_on_thres(thresholds, sorted_values):
    bins = [[] for _ in range(len(thresholds)-1)]
    for i in range(len(thresholds)-1):
        for loc, pc in sorted_values:
            if thresholds[i] <= loc <= thresholds[i+1]:
                bins[i].append(pc)
    return bins

def main():
    root = (Path(__file__).parent / '../..').resolve()
    os.chdir(str(root / 'tools')) # directory change makes path handling easier later on
    args, cfg = parse_config()

    # check the passed parameters:
    # args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    # args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # Set up logging and output directories
    args.output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    # args.ckpt_dir = args.output_dir / 'ckpt'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    # args.eval_output_dir = args.output_dir / 'eval' / 'eval_with_train'
    # args.eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.output_dir / "logs" / ('log_pipeline_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    args.logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # start logging
    args.logger.info('**********************Start logging**********************')
    log_git_data(args.logger)

    # log available gpus
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    # args.logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    # for key, val in vars(args).items():
    #     args.logger.info('{:16} {}'.format(key, val))
    # log_config_to_file(cfg, logger=args.logger)
    # os.system('cp %s %s' % (args.cfg_file, args.output_dir))

    # ######### LOAD DATA SETS: ##########
    dataset_real = setup_datasets(args, cfg, data_type="real", shuffle=False, remove_missing_gt=True)
    dataset_sim = setup_datasets(args, cfg, data_type="simulated", shuffle=False, remove_missing_gt=True)

    make_matching_datasets(dataset_real, dataset_sim, args)

    # ######### DATASET ANALYSIS: ########## TODO:fix aug vs no aug setting!- CAN PROBABLY BE REMOVED
    dataset_real_original = None
    dataset_sim_original = None
    # no augmentation:
    all_augmentation_names = [i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    if cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST != all_augmentation_names:
        dataset_real_original = setup_datasets(args, cfg, data_type="real", disable_cfg_aug=False, shuffle=False)
        dataset_sim_original = setup_datasets(args, cfg, data_type="simulated", disable_cfg_aug=False, shuffle=False)
        make_matching_datasets(dataset_real_original, dataset_sim_original, args)

    # analyze, compare and visualize *num* samples of 1-to-1 correspondences:
    analyze_data_pairs(dataset_real, dataset_sim, args, cfg, num=100)
    # analyze_data_pairs(dataset_real_no_aug, dataset_sim_no_aug, args, cfg, num=50) TODO: this depends on the setting of the current dataset config file!

if __name__ == "__main__":
    main()