import pickle
import time

import numpy as np
import torch
import tqdm
import open3d as o3d

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def translate_boxes_to_open3d_instance(gt_boxes, box3d, line_set, gt=False):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6] * 1.0
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d.clear()
    box3d.center = center
    box3d.extent = lwh
    box3d.R = rot
    if gt:
        box3d.color = np.asarray([1, 0, 0])
    else:
        box3d.color = np.asarray([0, 0, 1])

    new_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    line_set.points = new_line_set.points
    line_set.lines = o3d.utility.Vector2iVector(np.array([[1, 4], [7, 6]]))

    return box3d, line_set


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None,
                   get_val_loss=False, vis=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    model.eval_mode = True
    model.test = True
    for module in model.module_list:
        if hasattr(module, "eval_mode"):
            module.eval_mode = True
        if hasattr(module, "test"):
            module.test = True

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    val_loss_list = list()
    glob_feats = []

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict, feat = model(batch_dict)
            glob_feats.append(feat)
            if get_val_loss:
                loss, tb_dict, disp_dict = model.get_training_loss()
                val_loss_list.append([loss, tb_dict, disp_dict])
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        if vis:
            # initialize
            pcl = o3d.geometry.PointCloud()
            boxes_3d = []
            line_sets = []
            for idx in range(50):
                boxes_3d.append(o3d.geometry.OrientedBoundingBox())
                line_sets.append(o3d.geometry.LineSet())

            # The x, y, z axis will be rendered as red, green, and blue arrows respectively.
            mesh_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1, origin=[0, 0, 0])

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh_frame)
            vis.add_geometry(pcl)
            for box_3d in boxes_3d:
                vis.add_geometry(box_3d)
            for line_set in line_sets:
                vis.add_geometry(line_set)

            # clear boxes
            for box in boxes_3d:
                box.clear()
            for line_set in line_sets:
                line_set.clear()

            # point cloud
            points = batch_dict["points"].detach().cpu().numpy()[:, 1:]
            pcl.points = o3d.utility.Vector3dVector(points)

            # Only filter "Car" objects of GT data
            car_mask_gt = np.asarray([obj_name == "Car" for obj_name in batch_dict["gt"][0]["annos"]["name"] if obj_name != "DontCare"])
            boxes_gt = batch_dict["gt"][0]["annos"]["gt_boxes_lidar"][car_mask_gt]
            box_total = 0
            for box_idx, box in enumerate(boxes_gt):
                # box: X, Y, Z, L, W, H, Rot_Y
                translate_boxes_to_open3d_instance(box, boxes_3d[box_idx], line_sets[box_idx], gt=True)
                box_total += 1

            # predicted bounding boxes
            boxes_pred = annos[0]["boxes_lidar"]
            for box_idx, box in enumerate(boxes_pred):
                # box: X, Y, Z, L, W, H, Rot_Y
                translate_boxes_to_open3d_instance(box, boxes_3d[box_total], line_sets[box_total], gt=False)
                box_total += 1

            vis.update_geometry(pcl)
            for box in boxes_3d:
                vis.update_geometry(box)
            for line_set in line_sets:
                vis.update_geometry(line_set)

            viewctrl = vis.get_view_control()
            viewctrl.set_lookat(np.array([0.0, 0.0, 0.0]))
            viewctrl.set_zoom(30)

            rend_opt = vis.get_render_option()
            rend_opt.point_size = 2

            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.close()
            vis.destroy_window()

    with open(result_dir / f'glob_feat_{str(epoch_id).zfill(2)}.pkl', "wb") as file:
        pickle.dump(glob_feats, file)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    model.eval_mode = False
    model.test = False
    for module in model.module_list:
        if hasattr(module, "eval_mode"):
            module.eval_mode = False
        if hasattr(module, "test"):
            module.test = False

    return ret_dict, val_loss_list


if __name__ == '__main__':
    pass
