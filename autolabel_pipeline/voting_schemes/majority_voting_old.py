import numpy as np
import os
import torch
import pandas as pd

# PCDET
from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils




# Function that generates a ndarray encoding the df rows of overlapping bboxes.
def identify_overlapping_bboxes(cfg, df1, df2):
    # Uses euclidian centerpoint distance and then computes IoU.

    # Compute the euclidian distance between all elements of the dataframes to filter out non-overlapping ones.
    distances = []
    for index_df1 in range(len(df1)):
        centerpoint_1 = np.array(df1.iloc[index_df1][2:5].values.astype(float))
        distances_element = []
        for index_df2 in range(len(df2)):
            centerpoint_2 = np.array(df2.iloc[index_df2][2:5].values.astype(float))
            distances_element.append(compute_euclidian_distance(centerpoint_1, centerpoint_2))
        distances.append(distances_element)

    # Compute a threshold that guarantees that there is no intersection between bboxes.
    largest_dim_len = max(df1['dim_len'].max(), df2['dim_len'].max())
    largest_dim_wi = max(df1['dim_wi'].max(), df2['dim_wi'].max())
    largest_bbox_radius = (((largest_dim_wi / 2) ** 2) + ((largest_dim_len / 2) ** 2)) ** 0.5
    centerpoint_distance_threshold = (2 * (largest_bbox_radius * 1.1))

    detected_overlaps = np.array(
        [[1 if val < centerpoint_distance_threshold else 0 for val in inner] for inner in distances])
    detected_overlaps = detected_overlaps.astype(float)

    # for the bounding boxes below the non-overlap threshold, get IoU.
    combinations_to_check = np.argwhere(detected_overlaps == 1)

    #print('possible overlaps: ', "\n", detected_overlaps)

    for i in range(len(combinations_to_check)):
        bbox1 = df1.iloc[combinations_to_check[i][0]]
        bbox2 = df2.iloc[combinations_to_check[i][1]]
        iou = iou_2_df_objects(bbox1, bbox2)
        detected_overlaps[combinations_to_check[i][0]][combinations_to_check[i][1]] = iou

    if cfg.PIPELINE.PRINT_INFORMATION:
        # Set the printing options
        np.set_printoptions(precision=2, suppress=True)
        print("centerpoint_distance_threshold: ", centerpoint_distance_threshold)
        print("Detected overlaps with IoU: ", "\n", detected_overlaps)

    return detected_overlaps


def identify_representative_bboxes(detected_overlaps, df1, df2):

    # get the indeces of overlapping bboxes.
    overlaps_to_check = np.argwhere(detected_overlaps != 0)

    for i in range(len(overlaps_to_check)):

        bbox1 = torch.tensor(np.array(df1.iloc[overlaps_to_check[i][0]][2:9].values.astype(float)))
        bbox2 = torch.tensor(np.array(df2.iloc[overlaps_to_check[i][1]][2:9].values.astype(float)))
        boxes = torch.stack([bbox1, bbox2], dim=0).float().cuda()

        score1 = torch.tensor(np.array(df1.iloc[overlaps_to_check[i][0]][9].astype(float)))
        score2 = torch.tensor(np.array(df2.iloc[overlaps_to_check[i][1]][9].astype(float)))
        scores = torch.stack([score1, score2], dim=0).float()

        representative = iou3d_nms_utils.nms_gpu(boxes, scores, 0.1)

        if cfg.PIPELINE.PRINT_INFORMATION:
            #print("position: ", overlaps_to_check[i])
            # print("boxes: ", boxes)
            # print("scores: ", scores)
            #print("representative: ", representative)
            pass


# Function that computes the euclidian distance between two points.
def compute_euclidian_distance(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2) ** 0.5


# Function that computes  the IoU of two bboxes (pcdet)
def iou_2_df_objects(bbox1, bbox2):

    # write elements ['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z'] from df.iloc[] to torch.tensor
    box1 = torch.tensor(np.array([bbox1[2:9].values.astype(float)]))
    box2 = torch.tensor(np.array([bbox2[2:9].values.astype(float)]))
    iou = box_utils.boxes3d_nearest_bev_iou(box1, box2)
    iou = iou.item()

    return iou



def majority_voting(cfg, df1, df2):

    detected_overlaps = identify_overlapping_bboxes(cfg, df1, df2)

    #identify_representative_bboxes(detected_overlaps, df_ptrcnn, df_ptpillar)

    # Compute intersection over union of two boxes:
    #iou_2_objects = iou_2_df_objects(df_ptrcnn.iloc[0],  df_ptpillar.iloc[0])
    #print("iou_2_objects: ", iou_2_objects)



# additional scripts


# rotate all headings to make heading[0] be on the x-axis.
transformation = 0 - headings[0]
transformed_headings = []

for heading in headings:
    transformed_heading = (heading + transformation) % (2 * np.pi)
    transformed_headings.append(transformed_heading)

print("transformed_headings: ", transformed_headings)



#heading_threshold = np.radians(cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_HEADING)
        #headings = df_group['rot_z'].tolist()

        #reference_heading = headings[0]
        #print(reference_heading)
        #differences = np.abs(np.array(headings[1:]) - reference_heading)
        #print(differences)
        #circular_differences = np.where(differences > np.pi, 2 * np.pi - differences, differences)
        #print(circular_differences)

        # Check if circular differences are within the threshold or 180 degrees +- the threshold
        #heading_check = np.all(np.logical_or(circular_differences <= heading_threshold,
        #                                     circular_differences >= (np.pi - heading_threshold)))

        #heading_majority(df_group, reference_heading)


###  14.06 - from majority voting:
def add_fake_element(df_group):
    ################# debugging:
    # df_group = df_group.drop(df_group.index[-1])
    # df_group = df_group.drop(df_group.index)
    print(df_group, "\n")

    add_elements = False
    if add_elements:
        df_group = df_group.drop(df_group.index)
        # List of dictionaries for new rows
        new_elements = [{'element': 0, 'ID': '000618', 'label': 'Car', 'loc_x': 0.592576,
                         'loc_y': -0.102140, 'loc_z': -0.850193, 'dim_len': 4.324367, 'dim_wi': 1.645006,
                         'dim_ht': 1.538699, 'rot_z': 0, 'score': 0.90},
                        {'element': 2, 'ID': '000618', 'label': 'Car', 'loc_x': 0.592576,
                         'loc_y': -0.102140, 'loc_z': -0.850193, 'dim_len': 4.324367, 'dim_wi': 1.645006,
                         'dim_ht': 1.538699, 'rot_z': 3.27, 'score': 0.90},
                        {'element': 4, 'ID': '000618', 'label': 'Car', 'loc_x': 0.592576,
                         'loc_y': -0.102140, 'loc_z': -0.850193, 'dim_len': 4.324367, 'dim_wi': 1.645006,
                         'dim_ht': 1.538699, 'rot_z': 6, 'score': 0.90}]

        # Append new rows to the DataFrame
        df_group = df_group.append(new_elements, ignore_index=True)
        print(df_group)

        df_group = df_group.drop('element', axis=1)

        print(df_group)
        save_pseudo_labels(cfg, df_group)

    return df_group

    def subfunction_heading(cfg, df_group):
        print("--> subfunction heading.")

        # normalize the headings to be within [0; 2pi]
        def normalize_headings(headings):
            normalized_headings = []
            for heading in headings:
                heading %= 2 * np.pi
                if heading < 0:
                    heading += 2 * np.pi
                normalized_headings.append(heading)

            return normalized_headings

        # Calculate the intervals [X, X + threshold] and [X + π, X + π + threshold]
        def get_valid_heading_intervals(heading, heading_threshold):

            interval1 = np.array([heading, heading + heading_threshold]) % (2 * np.pi)
            interval2 = np.array([heading + np.pi, heading + np.pi + heading_threshold]) % (2 * np.pi)
            intervals_combined = [interval1, interval2]

            intervals = []
            for interval in intervals_combined:
                if interval[0] > interval[1]:
                    first_interval = np.array([interval[0], 2 * np.pi])
                    second_interval = np.array([0, interval[1]])
                    intervals.append(first_interval)
                    intervals.append(second_interval)
                else:
                    intervals.append(interval)

            return intervals

        headings = df_group['rot_z'].tolist()
        heading_threshold = np.radians(cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_HEADING)
        normalized_headings = normalize_headings(headings)
        #print("headings: ", headings)
        #print("normalized_headings: ", normalized_headings)

        for heading in normalized_headings:
            relevant_normalized_headings = normalized_headings.copy()  # Create a copy of the original list
            relevant_normalized_headings.remove(heading)

            intervals = get_valid_heading_intervals(heading, heading_threshold)
            #print("intervals: ", intervals)

            mask_heading = []
            for i in relevant_normalized_headings:
                found_interval = False
                for interval in intervals:
                    if interval[0] <= i <= interval[1]:
                        mask_heading.append(True)
                        found_interval = True
                        break

                if not found_interval:
                    mask_heading.append(False)

            if all(mask_heading):
                return True

        return False


    def subfunction_select_representant(df_group,mask):
        print("--> subfunction representant.")

        # If more than 2 bboxes: split into front/ back heading, from majority select most confident.
        if mask[0] > 2:

            # Find heading majority
            heading_threshold = np.radians(cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_HEADING)

            front_headings = []
            back_headings = []

            headings = df_group['rot_z'].values

            for i in range(len(headings)):
                current_heading = headings[i]
                is_front = True
                for j in range(i + 1, len(headings)):
                    next_heading = headings[j]
                    if abs(current_heading - next_heading) > heading_threshold:
                        is_front = not is_front
                if is_front:
                    front_headings.append(df_group.iloc[i])
                else:
                    back_headings.append(df_group.iloc[i])

            df_group_heading_front = pd.DataFrame(front_headings)
            df_group_heading_back = pd.DataFrame(back_headings)

            print("front: ", df_group_heading_front.shape)
            print("back: ", df_group_heading_back.shape)
            print("front: ", df_group_heading_front)
            print("back: ", df_group_heading_back)


            if df_group_heading_front.shape[0] > df_group_heading_back.shape[0]:
                df_pseudo_label = df_group_heading_front.loc[[df_group_heading_front['score'].idxmax()]].copy()

            elif df_group_heading_front.shape[0] < df_group_heading_back.shape[0]:
                df_pseudo_label = df_group_heading_back.loc[[df_group_heading_back['score'].idxmax()]].copy()

            else:
                df_pseudo_label = df_group.loc[[df_group['score'].idxmax()]].copy()

            print("df_pseudo_label: ", "\n", df_pseudo_label)

