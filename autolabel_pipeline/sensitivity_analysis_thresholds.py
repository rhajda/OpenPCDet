
# Import libraries
from easydict import EasyDict
import pathlib
import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from main_autolabel import main_pseudo_label, load_config
from evaluate_labels import main_evaluate_labels

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


def plot_graphs():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    bar_width = 0.01
    # Subplot 1: list_map_car
    axes[0].bar(grid_threshold_confidence_car_cyclist_pedestrian, list_map_car, width=0.005)
    axes[0].set_ylabel('AP_R40@0.7 in %')
    axes[0].set_xticks(grid_threshold_confidence_car_cyclist_pedestrian)
    axes[0].set_title('Car')
    axes[0].set_ylim(0, 100)
    # Subplot 2: list_map_pedestrian
    axes[1].bar(grid_threshold_confidence_car_cyclist_pedestrian, list_map_pedestrian, width=0.005)
    axes[1].set_ylabel('AP_R40@0.5 in %')
    axes[1].set_xticks(grid_threshold_confidence_car_cyclist_pedestrian)
    axes[1].set_title('Pedestrian')
    axes[1].set_ylim(0, 100)
    # Subplot 3: list_map_cyclist
    axes[2].bar(grid_threshold_confidence_car_cyclist_pedestrian, list_map_cyclist, width=0.005)
    axes[2].set_xlabel('Confidence threshold')
    axes[2].set_ylabel('AP_R40@0.5 in %')
    axes[2].set_xticks(grid_threshold_confidence_car_cyclist_pedestrian)
    axes[2].set_title('Cyclist')
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    plot_filename = 'AP-conf_NMS_' + str(threshold_nms)
    plot_file_path = os.path.join(folder_path, plot_filename + '.svg')
    plt.savefig(plot_file_path, format='svg')
    plt.close()



def sensitivity_analysis_nms():

    # loop over THRESHOLD_NMS_OVERLAP and then loop over THRESHOLDS per class.
    grid_threshold_nms_overlap = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    grid_threshold_confidence_car_cyclist_pedestrian = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


    columns = ['thres_conf', 'nms_overlap']
    for category in cfg.PIPELINE.CLASSES:
        columns.extend([f"{category}_AP", f"{category}_TP", f"{category}_FP", f"{category}_FN"])
    df = pd.DataFrame(columns=columns)

    for threshold_nms in grid_threshold_nms_overlap:
        cfg['PIPELINE']['NMS_VOTING']['THRESHOLDS']['THRESHOLD_NMS_OVERLAP'] = threshold_nms

        for threshold_confidence in grid_threshold_confidence_car_cyclist_pedestrian:
            print('threshold_nms: ', threshold_nms, '; threshold_confidence: ', threshold_confidence)
            cfg['PIPELINE']['NMS_VOTING']['THRESHOLDS']['THRESHOLD_CONFIDENCE_CAR'] = threshold_confidence
            cfg['PIPELINE']['NMS_VOTING']['THRESHOLDS']['THRESHOLD_CONFIDENCE_CYCLIST'] = threshold_confidence
            cfg['PIPELINE']['NMS_VOTING']['THRESHOLDS']['THRESHOLD_CONFIDENCE_PEDESTRIAN'] = threshold_confidence

            main_pseudo_label(cfg, BATCH_SIZE_VOTING=10000, START_AT_CHECKPOINT=False, START_FRAME=7400)
            _, mAP3d_R40, confusion_matrix = main_evaluate_labels(cfg, cfg.DATA.PATH_GROUND_TRUTHS, cfg.DATA.PATH_PSEUDO_LABELS.PSEUDO_LABELS_NMS)

            new_row = { 'thres_conf': threshold_confidence,
                        'nms_overlap': threshold_nms,
                        'Car_AP': float(mAP3d_R40[0][0][0]),
                        'Car_TP': int(confusion_matrix[0][0.7]['confusion_mat']['TP']),
                        'Car_FP': int(confusion_matrix[0][0.7]['confusion_mat']['FP']),
                        'Car_FN': int(confusion_matrix[0][0.7]['confusion_mat']['FN']),
                        'Pedestrian_AP': float(mAP3d_R40[1][0][0]),
                        'Pedestrian_TP': int(confusion_matrix[1][0.5]['confusion_mat']['TP']),
                        'Pedestrian_FP': int(confusion_matrix[1][0.5]['confusion_mat']['FP']),
                        'Pedestrian_FN': int(confusion_matrix[1][0.5]['confusion_mat']['FN']),
                        'Cyclist_AP': float(mAP3d_R40[2][0][0]),
                        'Cyclist_TP': int(confusion_matrix[2][0.5]['confusion_mat']['TP']),
                        'Cyclist_FP': int(confusion_matrix[2][0.5]['confusion_mat']['FP']),
                        'Cyclist_FN': int(confusion_matrix[2][0.5]['confusion_mat']['FN']) }

            # Append the new row to the DataFrame
            df = df.append(new_row, ignore_index=True)
            df.to_csv("/home/autolabel_pipeline/evaluation/sensitivity_NMS.csv", index=False)

    print("done")

def sensitivity_analysis_majority():

    def vote_and_evaluate(cfg, df, thres_iou, thres_3, thres_2, thres_1):
        print("thres_iou: ", thres_iou, "thres_3: ", thres_3, "thres_2: ", thres_2, "thres_1: ", thres_1)

        if True:
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_IOU'] = thres_iou

            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_3_BBOXES']['CONFIDENCE_CAR'] = thres_3
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_3_BBOXES'][
                'CONFIDENCE_CYCLIST'] = thres_3
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_3_BBOXES'][
                'CONFIDENCE_PEDESTRIAN'] = thres_3

            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_2_BBOXES']['CONFIDENCE_CAR'] = thres_2
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_2_BBOXES'][
                'CONFIDENCE_CYCLIST'] = thres_2
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_2_BBOXES'][
                'CONFIDENCE_PEDESTRIAN'] = thres_2

            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_1_BBOX']['CONFIDENCE_CAR'] = thres_1
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_1_BBOX']['CONFIDENCE_CYCLIST'] = thres_1
            cfg['PIPELINE']['MAJORITY_VOTING']['THRESHOLDS']['THRESHOLD_1_BBOX'][
                'CONFIDENCE_PEDESTRIAN'] = thres_1

            main_pseudo_label(cfg, BATCH_SIZE_VOTING=10000, START_AT_CHECKPOINT=False, START_FRAME=7430)
            _, mAP3d_R40, confusion_matrix = main_evaluate_labels(cfg, cfg.DATA.PATH_GROUND_TRUTHS,
                                                                  cfg.DATA.PATH_PSEUDO_LABELS.PSEUDO_LABELS_MAJORITY)

            new_row = {'thres_iou': thres_iou,
                       'thres_3': thres_3,
                       'thres_2': thres_2,
                       'thres_1': thres_1,
                       'Car_AP': float(mAP3d_R40[0][0][0]),
                       'Car_TP': int(confusion_matrix[0][0.7]['confusion_mat']['TP']),
                       'Car_FP': int(confusion_matrix[0][0.7]['confusion_mat']['FP']),
                       'Car_FN': int(confusion_matrix[0][0.7]['confusion_mat']['FN']),
                       'Pedestrian_AP': float(mAP3d_R40[1][0][0]),
                       'Pedestrian_TP': int(confusion_matrix[1][0.5]['confusion_mat']['TP']),
                       'Pedestrian_FP': int(confusion_matrix[1][0.5]['confusion_mat']['FP']),
                       'Pedestrian_FN': int(confusion_matrix[1][0.5]['confusion_mat']['FN']),
                       'Cyclist_AP': float(mAP3d_R40[2][0][0]),
                       'Cyclist_TP': int(confusion_matrix[2][0.5]['confusion_mat']['TP']),
                       'Cyclist_FP': int(confusion_matrix[2][0.5]['confusion_mat']['FP']),
                       'Cyclist_FN': int(confusion_matrix[2][0.5]['confusion_mat']['FN'])}

            # Append the new row to the DataFrame
            df = df.append(new_row, ignore_index=True)
            df.to_csv("/home/autolabel_pipeline/evaluation/sensitivity_MAJORITY.csv", index=False)

            return df

    columns = ['thres_iou', 'thres_3', 'thres_2', 'thres_1']
    for category in cfg.PIPELINE.CLASSES:
        columns.extend([f"{category}_AP", f"{category}_TP", f"{category}_FP", f"{category}_FN"])
    df = pd.DataFrame(columns=columns)
    print("Columns: ", columns)


    TYPE_1 = False
    if TYPE_1:

        grid_threshold_iou = [0.4]
        grid_threshold_3_bbox_group = [0.6, 0.7, 0.8]
        # grid_threshold_3_bbox_group = [0.6, 0.7, 0.8, 0.9, 0.98]
        # grid_threshold_3_bbox_group = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

        count = 0
        for thres_iou in grid_threshold_iou:

            for thres_3 in grid_threshold_3_bbox_group:
                grid_threshold_2_bbox_group = [elem for elem in grid_threshold_3_bbox_group if
                                               elem > thres_3 and elem >= 0.7]

                for thres_2 in grid_threshold_2_bbox_group:
                    grid_threshold_1_bbox_group = [elem for elem in grid_threshold_2_bbox_group if
                                                   elem > thres_2 and elem >= 0.8]

                    for thres_1 in grid_threshold_1_bbox_group:
                        count += 1
                        df = vote_and_evaluate(cfg, df, thres_iou, thres_3, thres_2, thres_1)
        print(count)
        print("done")

    else:
        print("type_2.")

        grid_threshold_iou = [0.3, 0.4, 0.5]
        grid_threshold_3_bbox_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
        grid_threshold_2_bbox_group = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
        grid_threshold_1_bbox_group = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

        count = 0
        for thres_iou in grid_threshold_iou:
            for thres_3 in grid_threshold_3_bbox_group:
                count += 1
                thres_2, thres_1 = 1, 1
                df = vote_and_evaluate(cfg, df, thres_iou, thres_3, thres_2, thres_1)
            for thres_2 in grid_threshold_2_bbox_group:
                count += 1
                thres_3, thres_1 = 1, 1
                df = vote_and_evaluate(cfg, df, thres_iou, thres_3, thres_2, thres_1)
            for thres_1 in grid_threshold_1_bbox_group:
                count += 1
                thres_3, thres_2 = 1, 1
                df = vote_and_evaluate(cfg, df, thres_iou, thres_3, thres_2, thres_1)

        print(count)
        print("done")


if __name__ == "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()


    if cfg.PIPELINE.VOTING_SCHEME == 'NMS':
        print("Sensitivity analysis NMS voting: ")
        sensitivity_analysis_nms()
        exit()

    if cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
        print("Sensitivity analysis MAJORITY voting: ")
        sensitivity_analysis_majority()
        exit()

    else:
        raise ValueError ("autolabel.yaml @ PIPELINE.VOTING_SCHEME is not valid.")
