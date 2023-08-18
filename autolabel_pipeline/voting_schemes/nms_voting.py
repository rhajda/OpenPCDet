
# Import libraries
import pandas as pd
import torch
import numpy as np
import os

from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Prints additional information when True.
DEBUG_MODE = False

# Function performs non-maximum suppression voting on a frame.
def non_maximum_suppression_voting(cfg, path_manager, df1, df2, df3, frame_ID):

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("_______________________________")
        print("\n","==> nms_voting : ")

    if df1.empty and df2.empty and df3.empty:
        save_pseudo_labels(path_manager, pd.DataFrame(), frame_ID)
        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
            print("No bbox proposals. Saved empty frame", frame_ID)
        return

    # Gather all bboxes to one df and add corresponding model for traceback.
    df1['metrics_model'] = 1
    df2['metrics_model'] = 2
    df3['metrics_model'] = 3

    df1['metrics_weight'] = 1
    df2['metrics_weight'] = 1
    df3['metrics_weight'] = 1

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
            representatives = iou3d_nms_utils.nms_gpu(boxes, scores, cfg.PIPELINE.NMS_VOTING.THRESHOLDS.THRESHOLD_NMS_OVERLAP)[0].tolist()
            df_class = df_class.loc[representatives]

            # filter out bboxes with insufficient confidence score:
            if cfg.PIPELINE.CLASSES[object_class] == 'Car':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLDS.THRESHOLD_CONFIDENCE_CAR]

            if cfg.PIPELINE.CLASSES[object_class] == 'Pedestrian':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLDS.THRESHOLD_CONFIDENCE_PEDESTRIAN]

            if cfg.PIPELINE.CLASSES[object_class] == 'Cyclist':
                df_class = df_class[df_class.score >= cfg.PIPELINE.NMS_VOTING.THRESHOLDS.THRESHOLD_CONFIDENCE_CYCLIST]

            df_pseudo_labels = pd.concat([df_pseudo_labels, df_class], ignore_index=True)

    if not df_pseudo_labels.empty and cfg.PIPELINE.COMPUTE_EVALUATION_METRICS:
        nms_voting_metrics(path_manager, df_pseudo_labels)

    df_pseudo_labels = df_pseudo_labels.drop(['metrics_model', 'metrics_weight'], axis=1).reset_index(drop=True)
    save_pseudo_labels(path_manager, df_pseudo_labels, frame_ID)

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("df_pseudo_labels: ", "\n", df_pseudo_labels)
        print("nms_voting done. ")

# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(path_manager, df_pseudo_labels, frame_ID):
    # Save pseudo-labels as csv to folder.
    if not os.path.exists(path_manager.get_path("path_pseudo_labels_nms")):
        os.makedirs(path_manager.get_path("path_pseudo_labels_nms"))

    csv_filename = frame_ID + '.csv'
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_manager.get_path("path_pseudo_labels_nms"),
                                                     csv_filename), index=False, header=False)

# Function that saves one .txt file per non-empty frame containing information on the selected bbox (model) and weight.
def nms_voting_metrics(path_manager, df_pseudo_labels_final):

    #print(df_pseudo_labels_final)

    if DEBUG_MODE:
        print("--> Generating nms voting metrics.")

    path_metrics_pseudo_label_nms = os.path.join(path_manager.get_path("path_pseudo_labels_nms"), "metrics_nms")

    if not os.path.exists(path_metrics_pseudo_label_nms):
        os.mkdir(path_metrics_pseudo_label_nms)

    unique_id = df_pseudo_labels_final["ID"].iloc[0]

    with open(os.path.join(path_metrics_pseudo_label_nms, f"{unique_id}.txt"), "w") as file:
        file.write("label, metrics_model, metrics_weight\n")
        for index, row in df_pseudo_labels_final.iterrows():
            label = row['label']
            metrics_model = row['metrics_model']
            metrics_weight = row['metrics_weight']
            file.write(f"{label}, {metrics_model}, {metrics_weight}\n")
