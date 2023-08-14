import _init_path
import argparse
import datetime
import glob
import os
import re
import time
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter


def set_random_seed(seed):
    # set fixed determinism seed
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


set_random_seed(777)

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--dataset', type=str, default='default', help='dataset used for testing')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--vis', action='store_true', default=False, help='visualize every point cloud, GT, and predictions')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    cfg["DATA_CONFIG"]["DATA_PATH"] = os.path.join(cfg["DATA_CONFIG"]["DATA_PATH"], args.dataset)
    print(f"Using dataset {cfg['DATA_CONFIG']['DATA_PATH']} for testing!")

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False, tb_log=None):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # Weights and biases histogram to tensorboard
    for tag, parm in model.named_parameters():
        tb_log.add_histogram(tag, parm.data.cpu().numpy(), epoch_id)

    # start evaluation
    tb_dict, _ = eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file, vis=args.vis
    )

    result_str = ""
    header_str = ""
    for idx, (key, val) in enumerate(sorted(tb_dict.items())):
        tb_log.add_scalar(f"eval_offline/{key}", val, epoch_id)
        result_str += str(val) + ","
        header_str += str(key) + ","
    result_str = result_str[:-1]
    header_str = header_str[:-1]

    # Save results to csv
    if os.path.getsize(eval_output_dir / 'result.pkl') > 0:
        with open(eval_output_dir / 'result.pkl', 'rb') as f:
            det_annos = pickle.load(f)
            total_pred_objects = 0
            for anno in det_annos:
                total_pred_objects += anno['name'].__len__()
            avg_pred_obj = total_pred_objects / max(1, len(det_annos))
    else:
        # results.pkl empty, no predictions
        avg_pred_obj = 0.0
    with open(os.path.join(eval_output_dir, "results.csv"), "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch_id", header_str, "avg_pred_obj"])
        csv_writer.writerow([epoch_id, result_str, avg_pred_obj])


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False, tb_log=None):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    total_time = 0
    first_eval = True

    with open(os.path.join(eval_output_dir, "results.csv"), "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch_id",
                             "AP_Car_3d/0.5_R11",
                             "AP_Car_3d/0.5_R40",
                             "AP_Car_3d/0.7_R11",
                             "AP_Car_3d/0.7_R40",
                             "recall/rcnn_0.3",
                             "recall/rcnn_0.5",
                             "recall/rcnn_0.7",
                             "recall/roi_0.3",
                             "recall/roi_0.5",
                             "recall/roi_0.7",
                             "avg_pred_obj"])

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 1
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60:
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # Weights and biases histogram to tensorboard
        for tag, parm in model.named_parameters():
            tb_log.add_histogram(tag, parm.data.cpu().numpy(), cur_epoch_id)

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict, _ = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        result_str = ""
        if cfg.LOCAL_RANK == 0:
            for key, val in sorted(tb_dict.items()):
                tb_log.add_scalar(f"eval_offline/{key}", val, cur_epoch_id)
                result_str += str(val) + ","
            result_str = result_str[:-1]

        # Save results to csv
        if os.path.getsize(cur_result_dir / 'result.pkl') > 0:
            with open(cur_result_dir / 'result.pkl', 'rb') as f:
                det_annos = pickle.load(f)
                total_pred_objects = 0
                for anno in det_annos:
                    total_pred_objects += anno['name'].__len__()
                avg_pred_obj = total_pred_objects / max(1, len(det_annos))
        else:
            # results.pkl empty, no predictions
            avg_pred_obj = 0.0
        with open(os.path.join(eval_output_dir, "results.csv"), "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([cur_epoch_id, result_str, avg_pred_obj])

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.vis:
        assert args.batch_size == 1, "Set batch size to 1 if visualizing point clouds and boxes"

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.dataset is not None:
        eval_output_dir = eval_output_dir / f"{args.dataset}_{int(cfg['DATA_CONFIG']['EVAL_RANGE'][0])}_{int(cfg['DATA_CONFIG']['EVAL_RANGE'][1])}"

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger,
        training=False,
        eval_mode=True,
        test=True
    )

    # Tensorboard
    tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    print(f"Using dataset {cfg['DATA_CONFIG']['DATA_PATH']} for testing!")

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test, tb_log=tb_log)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test, tb_log=tb_log)


if __name__ == '__main__':
    main()
