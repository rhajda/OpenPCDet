
# Import libraries
import pandas as pd
import torch
import numpy as np
import os

from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Prints additional information when True.
DEBUG_MODE = False

# Function performs non-maximum suppression voting on a frame.
def non_maximum_suppression_voting(cfg, df1, df2, df3, frame_ID):

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("\n","==> nms_voting : ")

    if df1.empty and df2.empty and df3.empty:
        save_pseudo_labels(cfg, pd.DataFrame(), frame_ID)
        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
            print("No bbox proposals. Saved empty frame", frame_ID)
        return

    # Gather all bboxes to one df
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    if DEBUG_MODE:
        print("df_all: ", "\n", df_all)

    # Split into object classes and perform NMS per class.
    df_pseudo_labels = pd.DataFrame()
    for object_class in range (len(cfg.PIPELINE.CLASSES)):
        df_class = df_all[df_all.label == cfg.PIPELINE.CLASSES[object_class]].reset_index(drop=True)
        if DEBUG_MODE:
            print("Class: ", cfg.PIPELINE.CLASSES[object_class])
            print(df_class)

        if len(df_class) != 0:

            bbox_list = []
            score_list = []
            for i in range(len(df_class)):
                bbox = torch.tensor(np.array(df_class.iloc[i][2:9].values.astype(float)))
                score = torch.tensor(np.array(df_class.iloc[i][9].astype(float)))
                bbox_list.append(bbox)
                score_list.append(score)

            boxes = torch.stack(bbox_list, dim=0).float().cuda()
            scores = torch.stack(score_list, dim=0).float().cuda()
            representatives = iou3d_nms_utils.nms_gpu(boxes, scores, cfg.PIPELINE.NMS_VOTING.THRESHOLD_NMS_OVERLAP)[0].tolist()
            df_class = df_class.loc[representatives]

            # filter out bboxes with insufficient confidence score:
            if cfg.PIPELINE.CLASSES[object_class] == 'Car':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLD_CONFIDENCE_CAR]

            if cfg.PIPELINE.CLASSES[object_class] == 'Pedestrian':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLD_CONFIDENCE_PEDESTRIAN]

            if cfg.PIPELINE.CLASSES[object_class] == 'Cyclist':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLD_CONFIDENCE_CYCLIST]

            df_pseudo_labels = pd.concat([df_pseudo_labels, df_class], ignore_index=True)

    save_pseudo_labels(cfg, df_pseudo_labels, frame_ID)

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("df_pseudo_labels: ", "\n", df_pseudo_labels)
        print("nms_voting done. ")


# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(cfg, df_pseudo_labels, frame_ID):
    # Save pseudo-labels as csv to folder.
    if not os.path.exists(cfg.PIPELINE.NMS_VOTING.PATH_SAVE_PSEUDO_LABELS):
        os.makedirs(cfg.PIPELINE.NMS_VOTING.PATH_SAVE_PSEUDO_LABELS)

    csv_filename = frame_ID + '.csv'
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(cfg.PIPELINE.NMS_VOTING.PATH_SAVE_PSEUDO_LABELS, csv_filename),
                                        index=False, header=False)
