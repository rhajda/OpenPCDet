'''
Pipeline for comparing real vs simulated data.

Currently written to be run on single GPU.
'''
from pickle import TRUE
import time
from path_handle import paths
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
import seaborn as sns

from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils, box_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model
from tools.test_utils.test_utils import repeat_eval_ckpt

ZERO_ARRAY = np.zeros((1, 3))

def log_git_data(logger: Logger) -> None:
    """_summary_

    :param args: _description_
    """
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    git_branch =  subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    logger.info(f'Git commit hash: {commit_hash}')
    logger.info(f'Git current branch: {git_branch}')

def parse_config() -> Tuple[argparse.Namespace, easydict.EasyDict]:
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training e.g. cfgs/indy_models/pointrcnn.yaml')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')

    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')


    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    # set data config
    cfg_from_yaml_file(str(Path(args.cfg_file).absolute()), cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    assert [i for i in cfg.DATA_CONFIG.DATA_PROCESSOR if i['NAME'] == 'sample_points'] == [], \
        'sample_points not allowed, as analisys should be done on the original pointcloud'
    return args, cfg

def setup_datasets(args: argparse.Namespace, cfg:easydict.EasyDict, data_type:str = "real",
                   remove_missing_gt: bool = False,
                   disable_cfg_aug=False, shuffle=True, training=True) -> None:
    """_summary_
    :param args: _description_
    :param cfg: _description_
    :param data_type: _description_, defaults to "real"
    :param disable_cfg_aug: _description_, defaults to False
    :param shuffle: _description_, defaults to True
    :param training: Defines if the train_set will be used for training,
        this is relevant as in that case of (training=True) training sample
        with not gt boxes will be skipped, defaults to True.
    :return: _description_
    """
    assert data_type == "real" or data_type == "simulated"
    if data_type == "real":
        tmp_data_path = cfg.DATA_CONFIG.DATA_PATH
        cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.DATA_PATH_REAL 
    elif data_type == "simulated":
        tmp_data_path = cfg.DATA_CONFIG.DATA_PATH
        cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.DATA_PATH_SIM
    else:
        assert False
        
    if disable_cfg_aug:
        all_augmentation_names = [i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
        tmp_disable_aug = cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST
        if tmp_disable_aug != all_augmentation_names:
            cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = all_augmentation_names
        print("WARNING: No data augmentation for the current data_set!")

    args.logger.info(f'**********************Setup datasets and training of {data_type} data:**********************')
    dataset = easydict.EasyDict()
    # delay shuffle as we need the 1-to-1 mapping between real and simulated data for the analysis
    dataset.train_set, dataset.train_loader, dataset.train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=args.logger,
        training=training,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        shuffle=shuffle,
        remove_missing_gt = remove_missing_gt
    )
    dataset.test_set, dataset.test_loader, dataset.sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=args.logger, training=False,
        shuffle=shuffle,
        remove_missing_gt = False # not necessary
    )

    # restore the settings
    # if data_type == "real":
    #     cfg.DATA_CONFIG.DATA_PATH = tmp_data_path
    # if disable_cfg_aug:
    #     cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = tmp_disable_aug

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


def log_example_pointcloud_pairs(dataset_real, dataset_sim, args): # TODO
    """_summary_

    :param dataset_real: _description_
    :param dataset_sim: _description_
    :param args: _description_
    """
    pc1 = dataset_real.train_set[0]['pointcloud']

    summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)

def log_min_max_mean(args, title, header, train_real, train_sim, train_real_no_aug, train_sim_no_aug, to_csv=None):
    headers = ['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug', "num_of_samples"]
    stacked_values = [np.vstack(x) for x in [train_real, train_sim, train_real_no_aug, train_sim_no_aug]]
    mean_values = [list(np.round(np.mean(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    min_values = [list(np.round(np.min(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    max_values = [list(np.round(np.max(x, axis=(0)), 4)) for x in stacked_values] + [len(train_real)]
    args.logger.info(title + "\n" + tabulate(
        [['mean'] + mean_values, 
         ['min'] + min_values, 
         ['max'] + max_values], headers=headers, tablefmt='orgtbl'))
    if to_csv is not None:
        target_csv = args.output_dir / to_csv
        with target_csv.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(mean_values)
            writer.writerow(min_values)
            writer.writerow(max_values)

def analyze_data_pairs(dataset_real, dataset_sim, dataset_real_no_aug=None, dataset_sim_no_aug=None, args=None, num=20):
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

    # Load augmented and non_augmented values:
    train_real, train_sim, train_real_no_aug, train_sim_no_aug = [], [], [], []
    for i in tqdm(idxs_train, desc="Load corresponding samples"):
        loaded_sample_r = dataset_real.train_set[i]
        loaded_sample_s = dataset_sim.train_set[i]
        assert loaded_sample_r['frame_id'] == loaded_sample_s['frame_id'] 
        # loaded_sample_idx = int(loaded_sample['frame_id']) // 5 # usually the same as i, but if no gt boxes, use the new returned id
        train_real.append(loaded_sample_r['points'])
        train_sim.append(loaded_sample_s['points'])

        if dataset_real_no_aug is not None:
            assert train_sim_no_aug is not None
            loaded_sample_r = dataset_real_no_aug.train_set[i]
            loaded_sample_s = dataset_sim_no_aug.train_set[i]
            assert loaded_sample_r['frame_id'] == loaded_sample_s['frame_id'] 
        train_real_no_aug.append(loaded_sample_r['points'] if dataset_real_no_aug is not None else ZERO_ARRAY)
        train_sim_no_aug.append(loaded_sample_s['points'] if dataset_sim_no_aug is not None else ZERO_ARRAY)

    log_min_max_mean(args, 
                     title="Some metrics on the selected corresponding point cloud pairs. Due to the sampling process to guarantee same size => points wont be the same each time!",
                     header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                     train_real=train_real, train_sim=train_sim, 
                     train_real_no_aug=train_real_no_aug, train_sim_no_aug=train_sim_no_aug, 
                     to_csv="obj_and_background_point_ranges.csv")

    # Get infos multithreaded:
    infos_real = dataset_real.train_set.get_infos(
        sample_id_list=[dataset_real.train_set[i]['frame_id'] for i in idxs_train])
    infos_sim = dataset_sim.train_set.get_infos(
        sample_id_list=[dataset_sim.train_set[i]['frame_id'] for i in idxs_train])

    gt_target_locations_real = []
    gt_target_locations_sim = []
    gt_target_rotation_real = []
    gt_target_rotation_sim = []
    gt_target_size_real = []
    gt_target_size_sim = []
    gt_target_num_points_original = []
    gt_target_num_points_sampled = []
    gt_target_point_cloud_sampled = []
    gt_target_point_cloud_original = []

    for i_r, i_s, i in tqdm(zip(infos_real, infos_sim, idxs_train)):
        gt_target_locations_real.append(i_r['annos']['location'][0])
        gt_target_locations_sim.append(i_s['annos']['location'][0])
        
        gt_target_rotation_real.append(i_r['annos']['rotation_y'])
        gt_target_rotation_sim.append(i_s['annos']['rotation_y'])

        gt_target_size_real.append(i_r['annos']['dimensions'])
        gt_target_size_sim.append(i_s['annos']['dimensions'])
        
        loaded_sample_r = dataset_real.train_set[i]
        loaded_sample_s = dataset_sim.train_set[i]
        assert loaded_sample_r['frame_id'] == loaded_sample_s['frame_id']
        current_idx = loaded_sample_r['frame_id']
        points_loaded_r = loaded_sample_r['points']
        points_loaded_s = loaded_sample_s['points']
        points_original_r = dataset_real.train_set.get_lidar(current_idx)
        points_original_s = dataset_sim.train_set.get_lidar(current_idx)
        # convex hull of box
        flag_r = box_utils.in_hull(points_loaded_r, box_utils.boxes_to_corners_3d(i_r['annos']['gt_boxes_lidar'])[0])
        flag_s = box_utils.in_hull(points_loaded_s, box_utils.boxes_to_corners_3d(i_s['annos']['gt_boxes_lidar'])[0])
        flag_r_orig = box_utils.in_hull(points_original_r, box_utils.boxes_to_corners_3d(i_r['annos']['gt_boxes_lidar'])[0])
        flag_s_orig = box_utils.in_hull(points_original_s, box_utils.boxes_to_corners_3d(i_s['annos']['gt_boxes_lidar'])[0])

        gt_target_num_points_sampled.append((flag_r.sum(), flag_s.sum()))
        gt_target_num_points_original.append((i_r['annos']['num_points_in_gt'][0], i_s['annos']['num_points_in_gt'][0]))
        
        gt_target_point_cloud_sampled.append((points_original_r[flag_r_orig], points_original_s[flag_s_orig]))  
        gt_target_point_cloud_original.append((points_original_r, points_original_s))  

    # TODO: All values are here: do evaluation:
    # Location coordinates of bbox (independent of sampling - currently our only aug):
    log_min_max_mean(args, 
                    title="Bounding Box locations",
                    header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                    train_real=gt_target_locations_real, train_sim=gt_target_locations_sim, 
                    train_real_no_aug=ZERO_ARRAY, train_sim_no_aug=ZERO_ARRAY, 
                    to_csv="obj_location_ranges.csv")
    # Plot:
    # create data
    x = np.random.normal(size=50000)
    y = (x * 3 + np.random.normal(size=50000)) * 5
    
    # Make the plot
    plt.hexbin(x, y, gridsize=(15,15) )
    plt.show()
    
    # We can control the size of the bins:
    plt.hexbin(x, y, gridsize=(150,150) )
    plt.show()

    # Distances between Bboxes (independent of sampling - currently our only aug):
    distances_loc = np.linalg.norm(np.array(gt_target_locations_real) - np.array(gt_target_locations_sim), axis=1)
    log_min_max_mean(args, 
                title="L-2 distance between bounding box locations",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=distances_loc, train_sim=distances_loc, 
                train_real_no_aug=ZERO_ARRAY, train_sim_no_aug=ZERO_ARRAY, 
                to_csv="obj_box_location_l2_distance.csv")

    # Diff between Rotations (independent of sampling - currently our only aug):
    distances_rot = np.abs(np.array(gt_target_rotation_real) - np.array(gt_target_rotation_sim))
    log_min_max_mean(args, 
                title="Absolute difference in rotation(heading angle)",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=distances_rot, train_sim=distances_rot, 
                train_real_no_aug=ZERO_ARRAY, train_sim_no_aug=ZERO_ARRAY, 
                to_csv="obj_orientation_angle_ranges.csv")

    # Target sizes # HARDCODED ANYWAY!:
    # log_min_max_mean(args, 
    #             title="Bounding Box sizes",
    #             header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
    #             train_real=gt_target_size_real, train_sim=gt_target_size_sim, 
    #             train_real_no_aug=ZERO_ARRAY, train_sim_no_aug=ZERO_ARRAY, 
    #             to_csv="obj_size.csv")
    
    # Number of points in the gt target bbox:
    log_min_max_mean(args, 
                title="Absolute difference in rotation(heading angle)",
                header=['Attribute', 'real_aug', 'simulated_aug', 'real_no_aug', 'simulated_no_aug'],
                train_real=list(zip(*gt_target_num_points_sampled))[0], 
                train_sim=list(zip(*gt_target_num_points_sampled))[1], 
                train_real_no_aug=list(zip(*gt_target_num_points_original))[0],
                train_sim_no_aug=list(zip(*gt_target_num_points_original))[1], 
                to_csv="obj_number_of_points_ranges.csv")

    # IOU Convex hull: 
    # from Geometry3D import ConvexPolyhedron, Point, ConvexPolygon
    # import Geometry3D
    # import open3d as o3d
    # device = o3d.core.Device("CPU:0")

    # for a, b in gt_target_point_cloud_sampled:#
    #     a_ch = ConvexHull(a).volume
    #     b_ch = ConvexHull(b).volume
    #     c_ch = ConvexHull(np.concatenate((a, b), axis=0)).volume
    #     print((c_ch - a_ch - b_ch) / c_ch)
        # THROWS ERROR:
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(a)
        # hull1, _ = pcd1.compute_convex_hull()
        
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(b)
        # hull2, _  = pcd2.compute_convex_hull()

        # conv_polys1, conv_polys2 = [], []
        # for tri in list(hull1.triangles):
        #     conv_polys1.append(ConvexPolygon([Point(xyz) for xyz in a[tri]]))
        # for tri in list(hull2.triangles):
        #     conv_polys2.append(ConvexPolygon([Point(xyz) for xyz in b[tri]]))
        # intersection_polyh = Geometry3D.intersection(ConvexPolyhedron(conv_polys1), ConvexPolyhedron(conv_polys2))
        # intersection_vol = 0
        # if intersection_polyh is not None:
        #     intersection_hull_points = [(p.x, p.y, p.z) for p in list(intersection_polyh.point_set)]
        #     intersection_vol = ConvexHull(intersection_hull_points).volume
        # print(intersection_vol)

    args.logger.info('********************** End comparison of data pairs **********************')

def main():
    os.chdir(paths.tools) # directory change makes life easier with path handling later on
    args, cfg = parse_config()

    # check the passed parameters:
    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # Set up logging and output directories
    args.output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    args.ckpt_dir = args.output_dir / 'ckpt'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.eval_output_dir = args.output_dir / 'eval' / 'eval_with_train'
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.output_dir / ('log_pipeline_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    args.logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # start logging
    args.logger.info('**********************Start logging**********************')
    log_git_data(args.logger)

    # log available gpus
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    args.logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        args.logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=args.logger)
    os.system('cp %s %s' % (args.cfg_file, args.output_dir))

    # ######### LOAD DATA FOR TRAINING: ##########
    dataset_real = setup_datasets(args, cfg, data_type="real", shuffle=False, remove_missing_gt=True)
    dataset_sim = setup_datasets(args, cfg, data_type="simulated", shuffle=False, remove_missing_gt=True)

    make_matching_datasets(dataset_real, dataset_sim, args)
    
    # ######### DATASET ANALYSIS: ########## TODO:fix aug vs no aug setting!- CAN PROBABLY BE REMOVED
    dataset_sim_no_aug = None
    dataset_real_no_aug = None
    # no augmentation:
    all_augmentation_names = [i.NAME for i in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    if cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST != all_augmentation_names:
        dataset_sim_no_aug = setup_datasets(args, cfg, data_type="simulated", disable_cfg_aug=False, shuffle=False)
        dataset_real_no_aug = setup_datasets(args, cfg, data_type="real", disable_cfg_aug=False, shuffle=False)
        make_matching_datasets(dataset_real_no_aug, dataset_sim_no_aug, args)
    
    # analyze, compare and visualize *num* samples of 1-to-1 correspondences:
    analyze_data_pairs(dataset_real, dataset_sim, dataset_real_no_aug, dataset_sim_no_aug, args, num=40)

if __name__ == "__main__":
    main()