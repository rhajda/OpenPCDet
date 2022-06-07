'''
Pipeline for comparing real vs simulated data. 

Currently writte to be run on single GPU.
'''
import time
from path_handle import paths
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training e.g. cfgs/indy_models/pointrcnn.yaml')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
        
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')


    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none') only single GPU as of now
    # parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    # parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    # parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def training_setup(args, cfg, logger):
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )


    
def main():
    args, cfg = parse_config()
    
    # check the passed parameters:
    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.fix_random_seed:
        common_utils.set_random_seed(666)
        
    # Set up logging and output directories
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    # start logging
    logger.info('**********************Start logging**********************')
    # log available gpus
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    os.system('cp %s %s' % (args.cfg_file, output_dir))
    
    # tensorboard logging:
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # load the two datsets:
    # load real:
    training_setup(args, cfg, logger)
    # load simulated:
    training_setup(args, cfg, logger)
    
    # setup model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, epoch_eval=True)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    
    # load example pointclouds
    
    
    return



if __name__ == "__main__":
    t0 = time.time()
    print("Running...")
    main()
    print("Time: {:.2f}s".format(time.time() - t0))