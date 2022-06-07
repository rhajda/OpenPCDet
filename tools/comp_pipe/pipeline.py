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
import easydict
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model
from tools.test_utils.test_utils import repeat_eval_ckpt

def parse_config() -> Tuple[argparse.Namespace, easydict.EasyDict]:
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

    # set data config
    os.chdir(paths.tools)
    cfg_from_yaml_file(paths.cfg_indy_pointrcnn, cfg)
    os.chdir(paths.root)
    
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def setup_datasets(args: argparse.Namespace, cfg:easydict.EasyDict, data_type:str = "real") -> None:
    """_summary_

    :param args: _description_
    :param cfg: _description_
    :param data_type: _description_, defaults to "real"
    """
    assert data_type == "real" or data_type == "simulated"
    args.logger.info(f'**********************Setup datasets and training of {data_type} data:**********************')
    dataset = easydict.EasyDict()
    os.chdir(paths.tools)
    dataset.train_set, dataset.train_loader, dataset.train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=args.logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    dataset.test_set, dataset.test_loader, dataset.sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=args.logger, training=False
    )
    # os.chdir(paths.root)
    return dataset
    
def execute_training(args: argparse.Namespace, cfg:easydict.EasyDict, dataset:easydict.EasyDict) -> None:
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

    model.train()
    args.logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(dataset.train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    args.logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    # tensorboard logging:
    tb_log = SummaryWriter(log_dir=str(args.output_dir / 'tensorboard'))
    
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
    
    args.logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(model, dataset.test_loader, args, args.eval_output_dir, 
                     args.logger, args.ckpt_dir, dist_test=args.dist_train)
    args.logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


    
def main():
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
    # log available gpus
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    args.logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        args.logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=args.logger)
    os.system('cp %s %s' % (args.cfg_file, args.output_dir))
    
    # load the two datsets:
    # real:
    dataset_real = setup_datasets(args, cfg, data_type="real")
    # simulated:
    dataset_sim = setup_datasets(args, cfg, data_type="simulated")
    
    # train on real: TODO investigate why we still nee to have: os.chdir(paths.tools)
    execute_training(args, cfg, dataset_real)
    # train on simulated:
    execute_training(args, cfg, dataset_sim)


if __name__ == "__main__":
    t0 = time.time()
    print("Running...")
    main()
    print("Time: {:.2f}s".format(time.time() - t0))