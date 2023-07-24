
# Import libraries
import matplotlib.pyplot as plt
from easydict import EasyDict
import pandas as pd
import numpy as np
import pathlib
import numba
import yaml
import os
# Import visualization functions
from visualize_pcds import visualize_single_pcd
from main_autolabel import csv_to_dataframe, common_set_between_datasets

from pcdet.datasets.autolabel.kitti_object_eval_python.kitti_common import get_image_index_str
from pcdet.datasets.autolabel.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.autolabel.kitti_object_eval_python.eval import get_thresholds, clean_data, image_box_overlap, bev_box_overlap, d3_box_overlap_kernel, d3_box_overlap, get_split_parts, _prepare_data, get_mAP, get_mAP_R40, print_str

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]




## Adapted from PCDET
# Function from pcdet, adapted for autolabel
def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    print("here")
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            gt_boxes = gt_boxes[:, [0, 2, 1, 3, 5, 4, 6]]  # x,y,z,l,w,h,rot -> x,z,y,l,h,w,rot
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            dt_boxes = dt_boxes[:, [0, 2, 1, 3, 5, 4, 6]]  # x,y,z,l,w,h,rot -> x,z,y,l,h,w,rot
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

# Function from pcdet, adapted for autolabel
@numba.jit(nopython=True)
def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas, dt_datas, dontcares, ignored_gts,
                             ignored_dets, metric, min_overlap, thresholds, compute_aos=False):
    print("-> fused_compute_stat")
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]

# Function from pcdet, adapted for autolabel
@numba.jit(nopython=True)
def compute_statistics_jit(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes, metric, min_overlap,
                           thresh=0, compute_fp=False, compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    ### AUTOLABEL ###
    tp_matches = []
    fp_overlaps = []
    fp_indeces = []
    #################

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size

    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True


    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    print(ignored_gt)


    for i in range(gt_size):

        # ignored_gt array with 0 if gt class matches current class -1 else. Ex. [ 0  0  0  0 -1 -1] 0-> Car, -1-> other
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False


        for j in range(det_size):

            # 0 if current class prediction, -1 else. Ex. [0 0 0 0 0] 0-> Car, -1 -> other
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            overlap = overlaps[j, i]
            dt_score = dt_scores[j]

            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score

            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j

                ### AUTOLABEL ###
                #print("FP: ", [j, i])
                fp_indeces.append(j)
                #################

                valid_detection = 1
                assigned_ignored_det = False

            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j

                ### AUTOLABEL ###
                #print("FP: ", [j, i])
                #################

                valid_detection = 1
                assigned_ignored_det = True


        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
            print("FN: ", [j, i])

        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
            print("here: ", [j, i])

        elif valid_detection != NO_DETECTION:
            tp += 1

            ### AUTOLABEL ###
            print("TP_det: ", [det_idx, i])
            tp_matches.append([det_idx, i])
            #################

            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True


    print("ignored_det: ", ignored_det)
    print("assigned_detection: ", assigned_detection)
    print("ignored_threshold: ", ignored_threshold)
    print("fp_indeces: ", fp_indeces)

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
                ### AUTOLABEL ###
                fp_overlaps.append(overlap)

                #################


        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1


    return tp, fp, fn, similarity, thresholds[:thresh_idx], tp_matches

# Function from pcdet, adapted for autolabel
def eval_class(gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, compute_aos=False, num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    ## AUTOLABEL ##
    eval_dict = {}
    ###############

    for m, current_class in enumerate(current_classes):

        ## AUTOLABEL ##
        #print("current_class: ", current_class)
        eval_dict[current_class] = {}
        ###############

        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):

                ## AUTOLABEL ##
                #print("min_overlap: ", min_overlap)
                tp_all = 0
                fp_all = 0
                fn_all = 0
                mre_error_all = [[],[],[],[],[],[],[],[]]
                ###############

                thresholdss = []
                for i in range(len(gt_annos)):
                    print("_______________________________")
                    print(gt_annos[i]["frame_ID"])


                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=True)
                    ### AUTOLABEL ###
                    tp, fp, fn, similarity, thresholds, tp_matches = rets

                    #if fp > 0:
                    #    fn = fn - fp
                    print("TP: ", tp)
                    print("FP: ", fp)
                    print("FN: ", fn)


                    tp_all += tp
                    fp_all += fp
                    fn_all += fn



                    if tp != 0:
                        mre_frame = compute_mean_relative_error(tp_matches, i, gt_annos, dt_annos, overlaps)
                        mre_error_all[0].extend(mre_frame[0]) # error_loc_x
                        mre_error_all[1].extend(mre_frame[1]) # error_loc_y
                        mre_error_all[2].extend(mre_frame[2]) # error_loc_z
                        mre_error_all[3].extend(mre_frame[3]) # error_dim_len
                        mre_error_all[4].extend(mre_frame[4]) # error_dim_wi
                        mre_error_all[5].extend(mre_frame[5]) # error_dim_ht
                        mre_error_all[6].extend(mre_frame[6]) # error_rot
                        mre_error_all[7].extend(mre_frame[7]) # overlap_bev
                    ###################
                    thresholdss += thresholds.tolist()

                ### AUTOLABEL ###
                print(overlaps)
                eval_dict[current_class][min_overlap] = {'confusion_mat': {'TP': tp_all, 'FP': fp_all, 'FN': fn_all}}
                eval_dict[current_class][min_overlap]['mre'] = { 'error': {'loc_x': mre_error_all[0],
                                                                           'loc_y': mre_error_all[1],
                                                                           'loc_z': mre_error_all[2],
                                                                           'dim_len': mre_error_all[3],
                                                                           'dim_wi': mre_error_all[4],
                                                                           'dim_ht': mre_error_all[5],
                                                                           'rot_z': mre_error_all[6]},
                                                                 'overlap': mre_error_all[7]}
                #################

                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1] + np.finfo(np.float64).eps)  # avoid zerodiv
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)


    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }

    return ret_dict, eval_dict

# Function from pcdet, adapted for autolabel
def do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos=False, PR_detail_dict=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0]
    '''
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']
    '''
    ret, confusion_mat = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps)



    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return None, None, mAP_3d, None, None, None, mAP_3d_R40, None, confusion_mat

# Function from pcdet, adapted for autolabel
def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.5, 0.5, 0.5, 0.7,
                             0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40, confusion_mat = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += "\n"
            result += print_str(
                (f"{class_to_name[curcls]} "
                 f"AP_R11@{min_overlaps[i, 0, j]:.2f}: {mAP3d[j, 0, i]:.4f}"))
            #result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
            #                     f"{mAPbbox[j, 1, i]:.4f}, "
            #                     f"{mAPbbox[j, 2, i]:.4f}"))
            #result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
            #                     f"{mAPbev[j, 1, i]:.4f}, "
            #                     f"{mAPbev[j, 2, i]:.4f}"))
            #result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}"))

            #if compute_aos:
            #    result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
            #                         f"{mAPaos[j, 1, i]:.2f}, "
            #                         f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(
                (f"{class_to_name[curcls]} "
                 f"AP_R40@{min_overlaps[i, 0, j]:.2f}: {mAP3d_R40[j, 0, i]:.4f}"))
            #result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
            #                     f"{mAPbbox_R40[j, 1, i]:.4f}, "
            #                     f"{mAPbbox_R40[j, 2, i]:.4f}"))
            #result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
            #                     f"{mAPbev_R40[j, 1, i]:.4f}, "
            #                     f"{mAPbev_R40[j, 2, i]:.4f}"))
            #result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}"))
            #if compute_aos:
            #    result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
            #                         f"{mAPaos_R40[j, 1, i]:.2f}, "
            #                         f"{mAPaos_R40[j, 2, i]:.2f}"))
            #    if i == 0:
            #       ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
            #       ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
            #       ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
            # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
            # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
            # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
            # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
            # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
            # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
            # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
            # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

            ret_dict[f"AP_{str(class_to_name[curcls])}_3d/{str(min_overlaps[i, 0, j])}_R11"] = mAP3d[j, 0, i]
            ret_dict[f"AP_{str(class_to_name[curcls])}_3d/{str(min_overlaps[i, 0, j])}_R40"] = mAP3d_R40[j, 0, i]
            '''
            ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
            ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
            ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
            ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
            ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
            ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
            ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
            ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]
            '''

    return result, ret_dict, confusion_mat



## Strongly adapted from PCDET or written entirely
# From pcdet.datasets.autolabel.kitti_object_eval_python.kitti_common.
def get_kitti_gt_annos(label_type, label_folder, frame_ids=None):

    if label_type == "ground_truths":
        file_type = '.txt'

    elif label_type == "pseudo_labels":
        file_type = '.csv'

    else:
        raise ValueError("Invalid label_type. Expected 'ground_truths' or 'pseudo_labels'.")

    if frame_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*' + file_type)
        prog = re.compile(r'^\d{6}' + re.escape(file_type) + '$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        frame_ids = [int(p.stem) for p in filepaths]
        frame_ids = sorted(frame_ids)
    if not isinstance(frame_ids, list):
        frame_ids = list(range(frame_ids))

    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in frame_ids:
        frame_idx = get_image_index_str(idx)
        label_filename = label_folder / (frame_idx + file_type)
        annos.append(get_label_anno(label_filename, label_type, frame_idx))

    return annos

# From pcdet.datasets.autolabel.kitti_object_eval_python.kitti_common.
def get_label_anno(label_path, label_type, frame_id):

    annotations = {}
    annotations.update({ 'frame_ID': [],
                         'name': [],
                         'truncated': [],
                         'occluded': [],
                         'alpha': [],
                         'bbox': [],
                         'dimensions': [],
                         'location': [],
                         'rotation_y': [],
                         'score': []})

    # FOR KITTI, MIGHT BE DIFFERENT FOR NUSCENES
    if label_type == 'ground_truths':
        with open(label_path, 'r') as f:
            lines = f.readlines()

        annotations['frame_ID'] = frame_id
        content = [line.strip().split(' ') for line in lines]
        annotations['name'] = np.array([x[0] for x in content])

        FLAG_KITTI = False

        if FLAG_KITTI:
            annotations['truncated'] = np.array([float(x[1]) for x in content]) # Kitti specific
            annotations['occluded'] = np.array([int(float(x[2])) for x in content]) # Kitti specific
            annotations['alpha'] = np.array([float(x[3]) for x in content]) # Kitti specific
            annotations['bbox'] = np.array(
                [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4) #Kitti specific
        else:
            annotations['truncated'] = np.zeros(len(content), dtype=float)
            annotations['occluded'] = np.zeros(len(content), dtype=int)
            annotations['alpha'] = np.zeros(len(content), dtype=float)
            annotations['bbox'] = np.zeros((len(content), 4), dtype=float)

        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations['dimensions'] = np.array(
            [[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
        annotations['location'] = np.array(
            [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
        annotations['rotation_y'] = np.array(
            [float(x[14]) for x in content]).reshape(-1)
        annotations['score'] = np.zeros([len(annotations['name'])])

        return annotations

    if label_type == 'pseudo_labels':
        df_pseudo_label_frame = csv_to_dataframe(label_path, frame_id)

        annotations['frame_ID'] = frame_id
        annotations['name'] = np.array(df_pseudo_label_frame['label'].values)
        annotations['alpha'] = np.zeros([len(annotations['name'])])
        annotations['truncated'] = np.zeros([len(annotations['name'])])
        annotations['occluded'] = np.zeros([len(annotations['name'])])
        annotations['bbox'] = np.zeros((len(annotations['name']), 4))
        annotations['dimensions'] = np.array(df_pseudo_label_frame[['dim_len', 'dim_ht', 'dim_wi']].values)
        annotations['location'] = np.array(df_pseudo_label_frame[['loc_x', 'loc_y', 'loc_z']].values)
        annotations['rotation_y'] = np.array(df_pseudo_label_frame['rot_z'].values)
        annotations['score'] = np.array(df_pseudo_label_frame['score'].values)

        return annotations

# Function that loads the YAML file to access parameters.
def load_config():
    cfg_file = os.path.join(working_path, 'autolabel_pipeline/autolabel.yaml')
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError

    with open(cfg_file, 'r') as f:
        try:
            cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print("Error parsing YAML file:", e)
            return EasyDict()

    cfg = EasyDict(cfg_dict)
    return cfg

# Computes the MRE as proposed by Peng et al. [30]
def compute_mean_relative_error(matches, current_index ,gt_annos, dt_annos, overlaps):
    """
    matches: 2D array (N,2) --> [[index_detection, index_groundtruth]] ([[det_idx, i]])
    outputs : error_loc_x, error_loc_y, error_loc_z, error_dim_len, error_dim_wi, error_dim_ht, error_rot
    """

    overlap_bev = []
    error_loc_x = []
    error_loc_y = []
    error_loc_z = []
    error_dim_len = []
    error_dim_wi = []
    error_dim_ht = []
    error_rot = []

    #print("matches: ", matches)

    for j in range (len(matches)):

        dt_frame_index = matches[j][0]
        gt_frame_index = matches[j][1]

        gt_loc =  gt_annos[current_index]['location'][gt_frame_index]
        gt_dim = gt_annos[current_index]['dimensions'][gt_frame_index]
        gt_rot = gt_annos[current_index]['rotation_y'][gt_frame_index]

        dt_loc = dt_annos[current_index]['location'][dt_frame_index]
        dt_dim = dt_annos[current_index]['dimensions'][dt_frame_index]
        dt_rot = dt_annos[current_index]['rotation_y'][dt_frame_index]

        #print("GT: ", gt_loc, gt_dim, gt_rot)
        #print("DT: ", dt_loc, dt_dim, dt_rot)

        overlap_bev.append(overlaps[current_index][dt_frame_index][gt_frame_index])
        print("metrics compute: ", current_index, dt_frame_index, gt_frame_index)

        error_loc_x.append(abs(gt_loc[0] - dt_loc[0]) / abs(gt_loc[0]))
        error_loc_y.append(abs(gt_loc[1] - dt_loc[1]) / abs(gt_loc[1]))
        error_loc_z.append(abs(gt_loc[2] - dt_loc[2]) / abs(gt_loc[2]))
        error_dim_len.append(abs(gt_dim[0] - dt_dim[0]) / abs(gt_dim[0]))
        error_dim_wi.append(abs(gt_dim[2] - dt_dim[2]) / abs(gt_dim[2]))
        error_dim_ht.append(abs(gt_dim[1] - dt_dim[1]) / abs(gt_dim[1]))
        error_rot.append(abs(gt_rot - dt_rot) / (2 * np.pi))

    #print("overlap_bev: ", overlap_bev)
    #print("error_loc_x: ", error_loc_x)
    #print("error_loc_y: ", error_loc_y)
    #print("error_loc_z: ", error_loc_z)
    #print("error_dim_len: ", error_dim_len)
    #print("error_dim_wi: ", error_dim_wi)
    #print("error_dim_ht: ", error_dim_ht)
    #print("error_rot: ", error_rot)

    return [error_loc_x, error_loc_y, error_loc_z, error_dim_len, error_dim_wi, error_dim_ht, error_rot, overlap_bev]

# Function generates plots for mean relative error.
def generate_plots(eval_dict):

    folder_path = "/home/autolabel_pipeline/debug"
    os.makedirs(folder_path, exist_ok=True)

    current_class = 0
    current_min_overlap = 0.7

    error_loc_x = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['loc_x']]
    error_loc_y = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['loc_y']]
    error_loc_z = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['loc_z']]
    error_dim_len = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['dim_len']]
    error_dim_wi = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['dim_wi']]
    error_dim_ht = [i * 100 for i in eval_dict[current_class][current_min_overlap]['mre']['error']['dim_ht']]


    data = [error_loc_x, error_loc_y, error_loc_z, error_dim_len, error_dim_wi, error_dim_ht]
    names = ['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht']

    plt.boxplot(data, vert=False)
    plt.yticks(range(1, len(names) + 1), names)
    plt.xlabel('Relative error in %')

    file_path = os.path.join(folder_path, 'boxplot.svg')
    plt.savefig(file_path, format='svg')

    print("SAVED SUCCESSFULLY")


def debug_evaluation_with_visualisation(frame_IDs):
    print("-> debug_evaluation_with_visualisation")

    for frame_ID in frame_IDs:
        counter = 0
        frame_ID_vis = str(frame_ID).zfill(6)

        gt_annos = get_kitti_gt_annos('ground_truths', folder_1, [frame_ID])
        # gt_annos = get_kitti_gt_annos('pseudo_labels', folder_1, frame_IDs)
        dt_annos = get_kitti_gt_annos('pseudo_labels', folder_2, [frame_ID])

        result, ret_dict, my_eval_dict = get_official_eval_result(gt_annos, dt_annos, cfg.PIPELINE.CLASSES)
        print("my_eval_dict: ", my_eval_dict)
        print("result: ", result)
        # generate_plots(my_eval_dict)

        # print(my_eval_dict)

        for key1, value1 in my_eval_dict.items():
            for key2, value2 in value1.items():
                confusion_mat = value2['confusion_mat']
                print(f'Combination: ({key1}, {key2})')
                print(f'Confusion Matrix: {confusion_mat}')
                print('---')

        if cfg.PIPELINE.SHOW_POINT_CLOUDS:
            print("\n", "Visualization triggered:")
            visualize_single_pcd(frame_ID_vis, cfg.VISUALISATION.BBOXES_TO_LOAD, cfg)


        counter += 1
        if counter >= 1:
            manual_continue = input("Continue y/ n ? ")
            if manual_continue == "y":
                counter = 0
                continue
            else:
                exit()


def compute_performance_metrics(frame_IDs):
    # frame_IDs = [3676]
    # frame_IDs = [1, 2, 4, 5, 6, 8, 15, 19]
    #frame_IDs = [1, 618]

    # PTPILLARS
    frames_remove = [2,61,314,394,499,878,932,1019,1344,1650,1752,2075,2136,2220, 2325,2628,2764,2863,
                                2995,3145,3292,3405,3630,3635,3707,3874,4191,4202,4305,4340,4415,4465,4502,4566,
                                4638,4683,4699,4746,4768, 4931, 5341,5359,5447,5458,5584,5826,6555,6586,6711,6712,
                                6741,6759,6816,6898,7079,7115,7212,7304,7343,7397]
    # SECOND frames_remove = [399,1019,1712,1946,2075,2628,2877,3183,4224,4305,4608,4683,4931,5584,6244,6819,7072,
    #                         7115,7227,7397]
    # MAJORITY frames_remove = [24,61,102,124,187,195,224,251,260,278,394,452,499,581,595,636,657,766,769,806,873,878,
    #                           881,889,932,948,967,1006,1019,1053,1144,1167,1173,1188,1194,1234,1235,1242,1286,1344,
    #                           1350,1353,1427,1432,1487,1527,1577,1587,1621,1635,1650,1684,1711,1712,1752,1782,1807,
    #                           1854,1881,1892,1946,1995,2036,2075,2136,2166,2179,2220,2279,2284,2325,2327,2329,2391,
    #                           2457,2504,2562,2613,2628,2686,2694,2695,2699,2810,2863,2877,2885,2889,2908,2914,2995,
    #                           3003,3031,3054,3071,3082,3136,3145,3156,3183,3202,3322,3324,3365,3419,3492,3573,3630,
    #                           3635,3689,3748,3762,3788,3841,3897,3950,4132,4191,4202,4224,4271,4278,4291,4305,4319,
    #                           4343,4367,4377,4393,4414,4415,4465,4502,4530,4566,4568,4576,4582,4588,4608,4638,4650,
    #                           4683,4697,4699,4706,4722,4726,4759,4768,4773,4914,4931,4944,4960,4996,5070,5078,5110,
    #                           5147,5230,5284,5337,5338,5341,5447,5452,5582,5584,5601,5638,5653,5662,5683,5699,5785,
    #                           5805,5812,5818,5841,5856,5939,5969,6028,6030,6057,6087,6096,6133,6165,6282,6321,6331,
    #                           6332,6349,6377,6491,6507,6555,6582,6586,6614,6650,6701,6711,6712,6741,6744,6759,6777,
    #                           6816,6819,6847,6855,6872,6994,7072,7079,7080,7115,7133,7201,7212,7227,7235,7253,7265,
    #                           7300,7369,7375,7397,7439]
    # NMS frames_remove = [5,61,102,224,499,581,636,657,769,873,932,1019,1053,1167,1173,1188,1194,1286,1344,1577,1587,
    #                      1650,1711,1712,1752,1813,1854,1881,1892,1995,2075,2160,2166,2220,2284,2325,2329,2504,2613,
    #                      2628,2810,2863,2877,2885,2995,3031,3082,3145,3183,3202,3322,3635,3679,3762,3777,3788,3950,
    #                      4132,4191,4195,4202,4278,4305,4367,4415,4465,4530,4566,4568,4588,4608,4683,4697,4699,4726,
    #                      4759,4768,4773,4914,4931,4944,5110,5147,5230,5338,5341,5447,5582,5584,5601,5638,5699,5785,
    #                      5818,5856,5969,6030,6165,6282,6331,6332,6377,6491,6517,6555,6650,6712,6741,6744,6759,6819,
    #                      6874,7072,7079,7115,7227,7235,7253,7300,7369,7397,7439]
    frame_IDs = [frame for frame in frame_IDs if frame not in frames_remove]

    # print(frame_IDs)

    gt_annos = get_kitti_gt_annos('ground_truths', folder_1, frame_IDs)
    # gt_annos = get_kitti_gt_annos('pseudo_labels', folder_1, frame_IDs)
    dt_annos = get_kitti_gt_annos('pseudo_labels', folder_2, frame_IDs)

    result, ret_dict, my_eval_dict = get_official_eval_result(gt_annos, dt_annos, cfg.PIPELINE.CLASSES)
    print("result: ", result)
    # generate_plots(my_eval_dict)

    # print(my_eval_dict)

    for key1, value1 in my_eval_dict.items():
        for key2, value2 in value1.items():
            confusion_mat = value2['confusion_mat']
            print(f'Combination: ({key1}, {key2})')
            print(f'Confusion Matrix: {confusion_mat}')
            print('---')



if __name__ == "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()

    folder_1 = cfg.DATA.PATH_GROUND_TRUTHS
    #folder_2 = cfg.DATA.PATH_PSEUDO_LABELS.PSEUDO_LABELS_MAJORITY
    folder_2 = cfg.DATA.PATH_PTPILLAR_PREDICTIONS
    #folder_1 = "/home/autolabel_pipeline/debug/debug_gt"
    #folder_2 = "/home/autolabel_pipeline/debug/debug_dt"

    #frame_IDs = common_set_between_datasets(cfg, [folder_1, folder_2], False)
    #frames_remove = [2, 61, 314, 394]
    #frame_IDs = [frame for frame in frame_IDs if frame not in frames_remove]
    #frame_IDs = [7375] #7412
    frame_IDs = [6]

    DEBUG = True

    if DEBUG:
        debug_evaluation_with_visualisation(frame_IDs)
    else:
        compute_performance_metrics(frame_IDs)


"""
    NOTE:   FP +1 -> if FP
            FN +1 -> if FN
            FP +1 & FN +1 -> if prediction TP but iou below threshold. 

"""




