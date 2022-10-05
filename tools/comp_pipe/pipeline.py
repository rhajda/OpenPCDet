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

from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils, box_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model
from tools.test_utils.test_utils import repeat_eval_ckpt
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
    dataset.test_set, dataset.test_loader, dataset.sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=32,
        dist=False, workers=args.workers, logger=args.logger, training=False,
        shuffle=shuffle,
        remove_missing_gt = False # not necessary
    )
    
    return dataset

def execute_training(args: argparse.Namespace, cfg:easydict.EasyDict, dataset:easydict.EasyDict, evaluate=True) -> None:
    """_summary_

    :param args: _description_
    :param cfg: _description_
    """
    # setup model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset.train_set, epoch_eval=True)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=args.logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=False, optimizer=optimizer, logger=args.logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(args.ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=False, optimizer=optimizer, logger=args.logger
            )
            last_epoch = start_epoch + 1


    args.logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(dataset.train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    args.logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # tensorboard logging:
    tb_log = SummaryWriter(log_dir=str(args.output_dir / 'tensorboard'))

    model.train()
    train_model(
        model,
        optimizer,
        dataset.train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=args.ckpt_dir,
        train_sampler=dataset.train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        cfg=cfg,
        test_loader=dataset.test_loader,
        logger=args.logger,
        eval_output_dir=args.eval_output_dir
    )
    if hasattr(dataset.train_set, 'use_shared_memory') and dataset.train_set.use_shared_memory:
        dataset.train_set.clean_shared_memory()

    args.logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if evaluate:
        args.logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

        args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

        repeat_eval_ckpt(model, dataset.test_loader, args, args.eval_output_dir,
                        args.logger, args.ckpt_dir, cfg=cfg, dist_test=False)
        args.logger.info('**********************End evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

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
    test_len = len(dataset_real.test_set)
    assert train_len == len(dataset_sim.train_set), \
        'The number of train data pairs in real and simulated dataset should be the same'
    assert test_len == len(dataset_sim.test_set), \
        'The number of test data pairs in real and simulated dataset should be the same'


    # get some corresponding data pairs
    test_train_ratio = test_len / train_len
    if 0.5 <= test_train_ratio: print('The train test ration seems off.')
    test_samples = 1 if int(num * test_train_ratio) == 0 else int(num * test_train_ratio)

    if num == -1: # use all samples # TODO: also check test_set?!
        idxs_test = range(0, test_len-1)
        idxs_train = range(0, train_len-1)
    else:
        idxs_train = random.sample(range(0, train_len), k=num-test_samples)
        idxs_test = random.sample(range(0, test_len), k=test_samples)

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

    real_target_pointclouds = dataset_real_analyze.get_normalized_target()
    sim_target_pointclouds = dataset_sim_analyze.get_normalized_target()
    
    log_min_max_mean(args, 
                    title="Number of target points",
                    header=['Attribute', 'real', 'simulated'],
                    train_real=real_target_pointclouds['target_point_num'], 
                    train_sim=sim_target_pointclouds['target_point_num'], to_csv="target_num_points_ranges.csv")
    

    point_cloud_range = [int(x/4) for x in cfg.DATA_CONFIG["POINT_CLOUD_RANGE"]]
    sns.set_theme(style="ticks")
    # Plot 1: Data distributions!
    # REAL:
    x = np.array(real_box_locations)[:, 0]
    y = np.array(real_box_locations)[:, 1]
    plot = sns.jointplot(x=x, y=y, color="#4CB391")
    plot.fig.savefig(args.output_dir / 'hexbin_plot_real_bbox_locations_sns.png', bbox_inches='tight')
    plt.hexbin(x, y, gridsize=(point_cloud_range[3]-point_cloud_range[0], point_cloud_range[4]-point_cloud_range[1]))
    plt.savefig(args.output_dir / 'hexbin_plot_real_bbox_locations.png', bbox_inches='tight')
    # SIMULATED:
    x = np.array(sim_box_locations)[:, 0]
    y = np.array(sim_box_locations)[:, 1]
    plot = sns.jointplot(x=x, y=y, color="#4CB391")
    plot.fig.savefig(args.output_dir / 'hexbin_plot_sim_bbox_locations_sns.png', bbox_inches='tight')
    plt.hexbin(x, y, gridsize=(point_cloud_range[3]-point_cloud_range[0], point_cloud_range[4]-point_cloud_range[1]))
    plt.savefig(args.output_dir / 'hexbin_plot_sim_bbox_locations.png', bbox_inches='tight', dpi=300)


    # Plot 2:   
    # REAL:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('meter')
    points_blue = np.concatenate(real_target_pointclouds['normalized_target'], axis=0)
    ax.scatter(points_blue[:,0], points_blue[:,1], points_blue[:,2])
    plt.savefig(args.output_dir / 'car_plot_real_normalized.png', bbox_inches='tight', dpi=300)
    # SIMULATED:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('meter')
    points_red_not_rotated = np.concatenate(sim_target_pointclouds['normalized_target'], axis=0)
    ax.scatter(points_red_not_rotated[:,0], points_red_not_rotated[:,1], points_red_not_rotated[:,2], c="red")
    plt.savefig(args.output_dir / 'car_plot_sim_normalized.png', bbox_inches='tight', dpi=300)

    args.logger.info('********************** End comparison of data pairs **********************')

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

    log_file = args.output_dir / ('log_pipeline_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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
    analyze_data_pairs(dataset_real, dataset_sim, args, cfg, num=50)
    # analyze_data_pairs(dataset_real_no_aug, dataset_sim_no_aug, args, cfg, num=50) TODO: this depends on the setting of the current dataset config file!

if __name__ == "__main__":
    main()