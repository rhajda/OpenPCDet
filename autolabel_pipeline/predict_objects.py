

# Import necessary libraries
from pathlib import Path
import sys
import os
import re
import datetime
import torch
import tqdm
import numpy as np
from tensorboardX import SummaryWriter

package_path = Path(__file__).resolve().parents[1]
sys.path.append(os.path.join(package_path, 'tools'))


from tools.test import parse_config
from tools.test import set_random_seed
set_random_seed(777)

from eval_utils import eval_utils
from pcdet.utils import common_utils
from pcdet.config import log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu


"""

FILE DESCRIPTION: 

This file generates object predictions from a trained model used as input. 
The script can be triggered via predict_objects.sh.

# Arguments to the variables are specified in predict_objects.sh.
CFG_FILE="/.../..."
CKPT_DIR="/.../ckpt/"
CKPT='checkpoint_epoch_XX.pth'


"""


# Function adapted from tools.test import set_random_seed to generate prediction list.
def predict_single_ckpt(model, dataloader, args, predict_output_dir, logger, epoch_id, dist_test=False):

    model.load_params_from_file(filename=os.path.join(args.ckpt_dir, args.ckpt), logger=logger, to_cpu=dist_test)
    model.cuda()

    logger.info('MODEL BUILT SUCCESSFULLY')

    # MAKE PREDICTIONS
    dataset = dataloader.dataset
    class_names = dataset.class_names

    logger.info('*************** EPOCH %s PREDICTIONS *****************' % epoch_id)

    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    model.eval()
    model.eval_mode = True
    model.test = True

    for module in model.module_list:
        if hasattr(module, "eval_mode"):
            module.eval_mode = True
        if hasattr(module, "test"):
            module.test = True

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='predict', dynamic_ncols=True)

    for i, batch_dict in enumerate(dataloader):

        #print(batch_dict)

        frame_ids = batch_dict['frame_id']
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict, feat = model(batch_dict)



            for j in range(0, len(frame_ids)):

                # Write predictions into one array per frame:
                pred_labels_encoded = pred_dicts[j]['pred_labels'].cpu().numpy()
                pred_labels = np.array([class_names[label - 1] for label in pred_labels_encoded])
                pred_boxes = pred_dicts[j]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[j]['pred_scores'].cpu().numpy()
                frame_predictions = np.column_stack((pred_labels, pred_boxes, pred_scores))

                # Save the ndarray to the CSV file
                file_name = str(frame_ids[j]) + '.csv'
                file_path = os.path.join(predict_output_dir, file_name)
                np.savetxt(file_path, frame_predictions, delimiter=',', fmt='%s')

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    logger.info('Predictions saved for all frames.')



if __name__ == '__main__':

    LOG_FLAG = False

    # Load cfg file and args
    args, cfg = parse_config()


    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl')
        dist_test = True

    if args.vis:
        assert args.batch_size == 1, "Set batch size to 1 if visualizing point clouds and boxes"

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus


    # Write data to /autolabel_pipeline/../predictions
    output_dir = cfg.ROOT_DIR / 'autolabel_data' / 'autolabel' / 'predictions' / cfg.TAG
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        #predict_output_dir = output_dir / ('epoch_%s' % epoch_id)
        predict_output_dir = output_dir
        print("Output directory for predictions: ", predict_output_dir)

    else:
        predict_output_dir = output_dir / 'eval_all_default'

    predict_output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = predict_output_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir/ ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    if LOG_FLAG:
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))


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

    logger.info('BUILDING MODEL')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        predict_single_ckpt(model, test_loader, args, predict_output_dir, logger, epoch_id, dist_test=dist_test)
