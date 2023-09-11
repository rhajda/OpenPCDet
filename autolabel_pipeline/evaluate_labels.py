
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
from base_functions import load_config, autolabel_path_manager
from visualize_pcds import visualize_single_pcd
from main_autolabel import csv_to_dataframe, common_set_between_datasets

from pcdet.datasets.autolabel.kitti_object_eval_python.kitti_common import get_image_index_str
from pcdet.datasets.autolabel.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.autolabel.kitti_object_eval_python.eval import get_thresholds, clean_data, image_box_overlap, \
    calculate_iou_partly, bev_box_overlap, d3_box_overlap_kernel, d3_box_overlap, get_split_parts, _prepare_data, \
    get_mAP, get_mAP_R40, print_str


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]




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
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
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
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1

            ### AUTOLABEL ###
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
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
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

@numba.jit(nopython=True)
def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas, dt_datas, dontcares, ignored_gts,
                             ignored_dets, metric, min_overlap, thresholds, compute_aos=False):
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

def eval_class(FN_counts, gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, compute_aos=False, num_parts=100):
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
        eval_dict[current_class] = {}
        ###############

        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):

                ## AUTOLABEL ##
                tp_all = 0
                fp_all = 0
                fn_all = 0
                mre_error_all_gt = [[], [], [], [], [], [], [], []]
                mre_error_all_dt = [[], [], [], [], [], [], []]
                ###############

                thresholdss = []
                for i in range(len(gt_annos)):
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

                    ## Autolabel ##
                    tp, fp, fn, similarity, thresholds, tp_matches = rets
                    #print("TP: ", tp)
                    #print("FP: ", fp)
                    #print("FN: ", fn)
                    tp_all += tp
                    fp_all += fp
                    fn_all += fn

                    if tp != 0:
                        mre_frame_gt, mre_frame_dt = compute_relative_error(tp_matches, i, gt_annos, dt_annos, overlaps)
                        mre_error_all_gt[0].extend(mre_frame_gt[0]) # loc_x gt
                        mre_error_all_gt[1].extend(mre_frame_gt[1]) # loc_y gt
                        mre_error_all_gt[2].extend(mre_frame_gt[2]) # loc_z gt
                        mre_error_all_gt[3].extend(mre_frame_gt[3]) # dim_len gt
                        mre_error_all_gt[4].extend(mre_frame_gt[4]) # dim_wi gt
                        mre_error_all_gt[5].extend(mre_frame_gt[5]) # dim_ht gt
                        mre_error_all_gt[6].extend(mre_frame_gt[6]) # rot gt
                        mre_error_all_gt[7].extend(mre_frame_gt[7]) # overlap_bev gt

                        mre_error_all_dt[0].extend(mre_frame_dt[0])  # loc_x dt
                        mre_error_all_dt[1].extend(mre_frame_dt[1])  # loc_y dt
                        mre_error_all_dt[2].extend(mre_frame_dt[2])  # loc_z dt
                        mre_error_all_dt[3].extend(mre_frame_dt[3])  # dim_len dt
                        mre_error_all_dt[4].extend(mre_frame_dt[4])  # dim_wi dt
                        mre_error_all_dt[5].extend(mre_frame_dt[5])  # dim_ht dt
                        mre_error_all_dt[6].extend(mre_frame_dt[6])  # rot dt
                    ###############

                    thresholdss += thresholds.tolist()

                ### AUTOLABEL ###
                eval_dict[current_class][min_overlap] = {
                    'confusion_mat': {'TP': tp_all, 'FP': fp_all, 'FN': fn_all}}
                eval_dict[current_class][min_overlap]['mre'] = {'gt': {'loc_x': mre_error_all_gt[0],
                                                                       'loc_y': mre_error_all_gt[1],
                                                                       'loc_z': mre_error_all_gt[2],
                                                                       'dim_len': mre_error_all_gt[3],
                                                                       'dim_wi': mre_error_all_gt[4],
                                                                       'dim_ht': mre_error_all_gt[5],
                                                                       'rot_z': mre_error_all_gt[6],
                                                                       'overlap': mre_error_all_gt[7]},
                                                                'dt': {'loc_x': mre_error_all_dt[0],
                                                                       'loc_y': mre_error_all_dt[1],
                                                                       'loc_z': mre_error_all_dt[2],
                                                                       'dim_len': mre_error_all_dt[3],
                                                                       'dim_wi': mre_error_all_dt[4],
                                                                       'dim_ht': mre_error_all_dt[5],
                                                                       'rot_z': mre_error_all_dt[6]}
                                                                }
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
                    # recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + (pr[i, 2] + (list(FN_counts.values())[m])))
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

def do_eval(FN_counts, gt_annos, dt_annos, current_classes, min_overlaps, compute_aos=False, PR_detail_dict=None):
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
    ret, eval_dict = eval_class(FN_counts, gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return None, None, mAP_3d, None, None, None, mAP_3d_R40, None, eval_dict

def get_official_eval_result(FN_counts, gt_annos, dt_annos, current_classes, PR_detail_dict=None):
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
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
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
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40, eval_dict = do_eval(
        FN_counts, gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]

        for i in range(min_overlaps.shape[0]):

            result += "\n"
            result += print_str(
                (f"{class_to_name[curcls]} "
                 f"AP_R11@{min_overlaps[i, 0, j]:.2f}: {mAP3d[j, 0, i]:.4f}"))

            result += print_str(
                (f"{class_to_name[curcls]} "
                 f"AP_R40@{min_overlaps[i, 0, j]:.2f}: {mAP3d_R40[j, 0, i]:.4f}"))

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

    return result, ret_dict, eval_dict, mAP3d_R40



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
    annotations.update({'frame_ID': [],
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
            annotations['truncated'] = np.array([float(x[1]) for x in content])  # Kitti specific
            annotations['occluded'] = np.array([int(float(x[2])) for x in content])  # Kitti specific
            annotations['alpha'] = np.array([float(x[3]) for x in content])  # Kitti specific
            annotations['bbox'] = np.array(
                [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)  # Kitti specific
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

# Function that performs evaluation with and without empty detection files.
def eval_with_empty_detection_frames(cfg, gt_annos, dt_annos):
    classes = cfg.PIPELINE.CLASSES
    FN_counts = {cls: 0 for cls in classes}
    empty_frame_list = []

    # Check for empty input.
    if len(dt_annos) == 0:
        raise ValueError("No frames inputted.")

    # Sort empty detections from non-empty detections.
    empty_elements = []
    for i in range(len(dt_annos)):
        if len(dt_annos[i]['name']) == 0:
            empty_elements.append(i)

    # if no empty detections exist, proceed with eval.
    if len(empty_elements) == 0:
        result, ret_dict, my_eval_dict, mAP3d_R40 = get_official_eval_result(FN_counts, gt_annos, dt_annos, cfg.PIPELINE.CLASSES)
        return result, ret_dict, my_eval_dict, mAP3d_R40, FN_counts, empty_frame_list

    # else, count number of FN and then proceed with eval. Recall is computed including all FN in FN_counts.
    else:
        dt_annos_empty = [dt_annos[i] for i in empty_elements if i < len(dt_annos)]
        gt_annos_empty = [gt_annos[i] for i in empty_elements if i < len(gt_annos)]
        dt_annos = [x for i, x in enumerate(dt_annos) if i not in empty_elements]
        gt_annos = [x for i, x in enumerate(gt_annos) if i not in empty_elements]

        # get list of frames without detections
        for frame in gt_annos_empty:
            empty_frame_list.append(int(frame['frame_ID']))

        if len(gt_annos_empty) == len(dt_annos_empty) and len(gt_annos) == len(dt_annos):
            # Count FN occurences
            for frame in gt_annos_empty:
                missed_objects = frame['name']
                for obj in missed_objects:
                    if obj in classes:
                        FN_counts[obj] += 1

        else:
            raise ValueError("Error in separate_empty_detection_frames")

        result, ret_dict, my_eval_dict, mAP3d_R40 = get_official_eval_result(FN_counts, gt_annos, dt_annos, cfg.PIPELINE.CLASSES)

        for i in my_eval_dict:
           for threshold in my_eval_dict[i]:
               my_eval_dict[i][threshold]['confusion_mat']['FN'] += FN_counts[list(FN_counts.keys())[i]]

        return result, ret_dict, my_eval_dict, mAP3d_R40, FN_counts, empty_frame_list

# Function that prepares the data and calls the evaluation.
def compute_performance_metrics(cfg, frame_IDs, folder_1, folder_2):
    """
    NOTE:   FP +1 -> if FP
            FN +1 -> if FN
            FP +1 & FN +1 -> if prediction TP but iou below threshold.
    """

    gt_annos = get_kitti_gt_annos('ground_truths', folder_1, frame_IDs)
    dt_annos = get_kitti_gt_annos('pseudo_labels', folder_2, frame_IDs)
    result, ret_dict, my_eval_dict, mAP3d_R40, FN_counts, empty_frame_list = eval_with_empty_detection_frames(cfg,
                                                                                                              gt_annos,
                                                                                                              dt_annos)

    return result, ret_dict, my_eval_dict, mAP3d_R40, FN_counts, empty_frame_list

# Computes the MRE as proposed by Peng et al. [30]
def compute_relative_error(matches, current_index ,gt_annos, dt_annos, overlaps):
    """
    matches: 2D array (N,2) --> [[index_detection, index_groundtruth]] ([[det_idx, i]])
    outputs : error_loc_x, error_loc_y, error_loc_z, error_dim_len, error_dim_wi, error_dim_ht, error_rot
    """

    overlap_bev = []
    gt_loc_x, dt_loc_x = [], []
    gt_loc_y, dt_loc_y = [], []
    gt_loc_z, dt_loc_z = [], []
    gt_dim_len, dt_dim_len = [], []
    gt_dim_wi, dt_dim_wi = [], []
    gt_dim_ht, dt_dim_ht = [], []
    gt_rot, dt_rot = [], []

    for j in range (len(matches)):

        dt_frame_index = matches[j][0]
        gt_frame_index = matches[j][1]

        gt_loc_x.append(gt_annos[current_index]['location'][gt_frame_index][0])
        gt_loc_y.append(gt_annos[current_index]['location'][gt_frame_index][1])
        gt_loc_z.append(gt_annos[current_index]['location'][gt_frame_index][2])
        gt_dim_len.append(gt_annos[current_index]['dimensions'][gt_frame_index][0])
        gt_dim_wi.append(gt_annos[current_index]['dimensions'][gt_frame_index][1])
        gt_dim_ht.append(gt_annos[current_index]['dimensions'][gt_frame_index][2])
        gt_rot.append(gt_annos[current_index]['rotation_y'][gt_frame_index])

        dt_loc_x.append(dt_annos[current_index]['location'][dt_frame_index][0])
        dt_loc_y.append(dt_annos[current_index]['location'][dt_frame_index][1])
        dt_loc_z.append(dt_annos[current_index]['location'][dt_frame_index][2])
        dt_dim_len.append(dt_annos[current_index]['dimensions'][dt_frame_index][0])
        dt_dim_wi.append(dt_annos[current_index]['dimensions'][dt_frame_index][1])
        dt_dim_ht.append(dt_annos[current_index]['dimensions'][dt_frame_index][2])
        dt_rot.append(dt_annos[current_index]['rotation_y'][dt_frame_index])

        overlap_bev.append(overlaps[current_index][dt_frame_index][gt_frame_index])

        #print("GT: ",
        #      gt_annos[current_index]['location'][gt_frame_index],
        #      gt_annos[current_index]['dimensions'][gt_frame_index],
        #      gt_annos[current_index]['rotation_y'][gt_frame_index])
        #print("DT: ",
        #      dt_annos[current_index]['location'][dt_frame_index],
        #      dt_annos[current_index]['dimensions'][dt_frame_index],
        #      dt_annos[current_index]['rotation_y'][dt_frame_index])
        #print("overlaps: ", overlaps[current_index][dt_frame_index][gt_frame_index])

    return [gt_loc_x, gt_loc_y, gt_loc_z, gt_dim_len, gt_dim_wi, gt_dim_ht, gt_rot, overlap_bev], \
            [dt_loc_x, dt_loc_y, dt_loc_z, dt_dim_len, dt_dim_wi, dt_dim_ht, dt_rot]

# Function generates plots for mean relative error.
def generate_evaluation_results(cfg, path_manager, result, FN_counts, empty_frame_list, eval_dict):

    print("-> generate_evalutation_results")

    # Print info
    print("empty_frame_list: ", empty_frame_list)
    print("empty_frame_list length: ", len(empty_frame_list))
    print("FN_counts: ", FN_counts)
    print("\n", "_____________________", "\n", "result: ", result)
    for key1, value1 in eval_dict.items():
        for key2, value2 in value1.items():
            confusion_mat = value2['confusion_mat']
            print(f'Confusion Matrix: ({key1}, {key2})')
            print(confusion_mat)
            print('---')

    # Error info from eval_dict
    label_mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    overlap_mapping = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}

    for label in cfg.PIPELINE.CLASSES:
        print(f"Class: {label}")
        current_class = label_mapping[label]
        current_min_overlap = overlap_mapping[label]

        # write dictionary info to lists for current_class and current_min_overlap:
        gt_loc_x = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['loc_x'])
        gt_loc_y = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['loc_y'])
        gt_loc_z = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['loc_z'])
        gt_dim_len = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['dim_len'])
        gt_dim_wi = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['dim_wi'])
        gt_dim_ht = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['dim_ht'])
        gt_rot = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['rot_z'])
        overlap = np.array(eval_dict[current_class][current_min_overlap]['mre']['gt']['overlap'])
        dt_loc_x = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['loc_x'])
        dt_loc_y = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['loc_y'])
        dt_loc_z = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['loc_z'])
        dt_dim_len = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['dim_len'])
        dt_dim_wi = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['dim_wi'])
        dt_dim_ht = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['dim_ht'])
        dt_rot = np.array(eval_dict[current_class][current_min_overlap]['mre']['dt']['rot_z'])

        #print(f"gt_loc_x: {gt_loc_x}\ngt_loc_y: {gt_loc_y}\ngt_loc_z: {gt_loc_z}\ngt_dim_len: {gt_dim_len}\n"
        #      f"gt_dim_wi: {gt_dim_wi}\ngt_dim_ht: {gt_dim_ht}\ngt_rot: {gt_rot}\noverlap: {overlap}\n"
        #      f"dt_loc_x: {dt_loc_x}\ndt_loc_y: {dt_loc_y}\ndt_loc_z: {dt_loc_z}\ndt_dim_len: {dt_dim_len}\n"
        #      f"dt_dim_wi: {dt_dim_wi}\ndt_dim_ht: {dt_dim_ht}\ndt_rot: {dt_rot}")

        # Compute absolute error arrays
        abs_error_loc_x = np.abs(gt_loc_x - dt_loc_x)
        abs_error_loc_y = np.abs(gt_loc_y - dt_loc_y)
        abs_error_loc_z = np.abs(gt_loc_z - dt_loc_z)
        abs_error_dim_len = np.abs(gt_dim_len - dt_dim_len)
        abs_error_dim_wi = np.abs(gt_dim_wi - dt_dim_wi)
        abs_error_dim_ht = np.abs(gt_dim_ht - dt_dim_ht)
        abs_error_rot = np.abs((gt_rot - dt_rot + np.pi) % (2 * np.pi) - np.pi)

        abs_error_rot_front = abs_error_rot[(abs_error_rot >= 0) & (abs_error_rot <= np.pi / 2)]
        abs_error_rot_back = abs_error_rot[(abs_error_rot > np.pi / 2) & (abs_error_rot <= np.pi)]
        abs_error_rot_back = np.abs(abs_error_rot_back - np.pi)



        # Calculate the percentages
        if len(abs_error_rot) != 0:
            percentage_under_pi_half = (len(abs_error_rot_front) / len(abs_error_rot)) * 100
        else:
            percentage_under_pi_half = 0

        print(f"Heading: {percentage_under_pi_half}% of bboxes head in the right direction ({label}: {len(abs_error_rot)} TPs.)")

        #print(f"abs_error_loc_x (m): {abs_error_loc_x}\nabs_error_loc_y (m): {abs_error_loc_y}\n"
        #      f"abs_error_loc_z (m): {abs_error_loc_z}\nabs_error_dim_len (m): {abs_error_dim_len}\n"
        #      f"abs_error_dim_wi (m): {abs_error_dim_wi}\nabs_error_dim_ht (m): {abs_error_dim_ht}\n"
        #      f"abs_error_rotation: {abs_error_rot}")

        # Compute mean relative error
        if len(gt_loc_x) != 0:
            mean_relative_error_loc_x = np.mean(abs_error_loc_x / np.abs(gt_loc_x))
        else:mean_relative_error_loc_x = 0
        if len(gt_loc_y) != 0:
            mean_relative_error_loc_y = np.mean(abs_error_loc_y / np.abs(gt_loc_y))
        else:mean_relative_error_loc_y = 0
        if len(gt_loc_z) != 0:
            mean_relative_error_loc_z = np.mean(abs_error_loc_z / np.abs(gt_loc_z))
        else:mean_relative_error_loc_z = 0
        if len(gt_dim_len) != 0:
            mean_relative_error_dim_len = np.mean(abs_error_dim_len / np.abs(gt_dim_len))
        else:mean_relative_error_dim_len = 0
        if len(gt_dim_wi) != 0:
            mean_relative_error_dim_wi = np.mean(abs_error_dim_wi / np.abs(gt_dim_wi))
        else:mean_relative_error_dim_wi = 0
        if len(gt_dim_ht) != 0:
            mean_relative_error_dim_ht = np.mean(abs_error_dim_ht / np.abs(gt_dim_ht))
        else:mean_relative_error_dim_ht = 0

        mean_relative_error_rot = np.mean(abs_error_rot / (2 * np.pi))
        mean_overlap = np.mean(overlap)


        print(f"Mean relative error loc_x: {mean_relative_error_loc_x}\n"
              f"Mean relative error loc_y: {mean_relative_error_loc_y}\n"
              f"Mean relative error loc_z: {mean_relative_error_loc_z}\n"
              f"Mean relative error dim_len: {mean_relative_error_dim_len}\n"
              f"Mean relative error dem_wi: {mean_relative_error_dim_wi}\n"
              f"Mean relative error dim_ht: {mean_relative_error_dim_ht}\n"
              f"Mean relative error rot_y: {mean_relative_error_rot}\n"
              f"Mean overlap bev: {mean_overlap}\n")

        below_threshold_values = [value for value in overlap if value < current_min_overlap]
        print("below_threshold_values: ", below_threshold_values)



        FLAG_PLOT = True
        if FLAG_PLOT:

            path_project_evaluation = os.path.join(path_manager.get_path("path_project_data"), "evaluation")
            os.makedirs(path_project_evaluation, exist_ok=True)

            # Define data and plot details
            if len(abs_error_rot_back) != 0:
                data_list = [[abs_error_loc_x, abs_error_loc_y, abs_error_loc_z, abs_error_dim_len, abs_error_dim_wi,
                              abs_error_dim_ht],
                             [abs_error_rot_front, abs_error_rot_back],
                             [overlap]]
                title_list = [f'Bounding boxes {label} absolute error',
                              f'Bounding boxes {label} absolute rotation error',
                              f'Overlap bounding boxes {label} to ground truth']
                names_list = [['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht'],
                              [f'rot_z (support: {len(abs_error_rot_front)})',
                               f'rot_z (reversed, support: {len(abs_error_rot_back)})'],
                              ['overlap bev']]
                xlabel_list = ['Absolute error in meter', 'Absolute error in radians', 'Overlap in %']

                my_figsize = [(10, 6), (10,2.5), (10,2)]

            else:
                data_list = [[abs_error_loc_x, abs_error_loc_y, abs_error_loc_z, abs_error_dim_len, abs_error_dim_wi,
                              abs_error_dim_ht],
                             [abs_error_rot_front],
                             [overlap]]
                title_list = [f'Bounding boxes {label} absolute error',
                              f'Bounding boxes {label} absolute rotation error',
                              f'Overlap bounding boxes {label} to ground truth']
                names_list = [['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht'],
                              [f'rot_z (support: {len(abs_error_rot_front)})'],
                              ['overlap bev']]
                xlabel_list = ['Absolute error in meter', 'Absolute error in radians', 'Overlap in %']

                my_figsize = [(10, 6), (10, 2), (10, 2)]

            for i, (data, names, xlabel, title) in enumerate(zip(data_list, names_list, xlabel_list, title_list)):
                fig, ax = plt.subplots(figsize=my_figsize[i])

                ax.boxplot(data, vert=False)
                ax.set_yticks(range(1, len(names) + 1))
                ax.set_yticklabels(names)
                ax.set_xlabel(xlabel)
                ax.set_title(title)
                ax.grid(False)

                plt.tight_layout()

                plot_filename = f'boxplot_MAE_{cfg.PIPELINE.VOTING_SCHEME}_{label}_plot_{i}.svg'
                file_path = os.path.join(path_project_evaluation, plot_filename)
                plt.savefig(file_path, format='svg')
                plt.close()
                print(f"SAVED SUCCESSFULLY: {plot_filename}")

                # Print mean, median, and quantiles for the current plot
                print(f"Statistics for {title}:")
                for j, name in enumerate(names):
                    if len(data[j]) == 0:
                        print(f"No elements for {name}")
                    else:
                        mean = np.mean(data[j])
                        median = np.median(data[j])
                        quantiles = np.percentile(data[j], [25, 75])

                        print(f"{name}:")
                        print(f"  Mean: {mean:.2f}")
                        print(f"  Median: {median:.2f}")
                        print(f"  Q1: {quantiles[0]:.2f}")
                        print(f"  Q3: {quantiles[1]:.2f}")





def main_evaluate_labels(cfg, path_manager, folder_1, folder_2):

    frame_IDs = common_set_between_datasets(cfg, [folder_1, folder_2], False)  # important: False when handling GTs

    result, ret_dict, my_eval_dict, mAP3d_R40, FN_counts, empty_frame_list = compute_performance_metrics(cfg,
                                                                                                         frame_IDs,
                                                                                                         folder_1,
                                                                                                         folder_2)

    generate_evaluation_results(cfg, path_manager, result, FN_counts, empty_frame_list, my_eval_dict)

    return ret_dict, mAP3d_R40, my_eval_dict



if __name__ == "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()
    path_manager = autolabel_path_manager(cfg)

    folder_1 = cfg.DATA.PATH_GROUND_TRUTHS
    if cfg.PIPELINE.VOTING_SCHEME == "MAJORITY":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_majority")
    elif cfg.PIPELINE.VOTING_SCHEME == "NMS":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_nms")
    else:
        raise ValueError("Autolabel cfg.PIPELINE.VOTING_SCHEME is not valid. Please check.")

    list_to_evaluate = [path_pseudo_labels]
    for folder_2 in list_to_evaluate:
        print("folder_1: ", folder_1)
        print("folder_2:", folder_2)
        ret_dict, mAP3d_R40, my_eval_dict = main_evaluate_labels(cfg, path_manager, folder_1, folder_2)
