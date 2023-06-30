
#Import libraries
import numpy as np
import os
import torch
import pandas as pd
import networkx as nx

# PCDET
from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Print additional information for debugging
DEBUG_MODE = False
# ADD FAKE ELEMENTS FOR DEBUGGING
ADD_ELEMENTS = False



# Function that adds fake elements to be processed.
def add_fake_elements(cfg, df_all, frame_ID):
    print("__________________________________________________________________")
    print("FAKE ELEMENTS ACTIVE")

    df_all.loc[0, ['loc_x', 'loc_y', 'loc_z']] = [0, 0, 0]
    df_all.loc[1, ['loc_x', 'loc_y', 'loc_z']] = [0, 0, 0]
    df_all.loc[5, ['loc_x', 'loc_y', 'loc_z']] = [1, 0, 0]
    df_all.loc[6, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
    # df_all.loc[8, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
    # df_all.loc[10, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]

    df_all.loc[0, ['dim_len', 'dim_wi', 'dim_ht']] = [2, 1, 1]
    df_all.loc[1, ['dim_len', 'dim_wi', 'dim_ht']] = [2, 1, 1]
    df_all.loc[5, ['dim_len', 'dim_wi', 'dim_ht']] = [2, 1, 1]
    df_all.loc[6, ['dim_len', 'dim_wi', 'dim_ht']] = [2, 1, 1]
    # df_all.loc[8, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
    # df_all.loc[10, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]

    df_all.loc[0, ['label', 'rot_z', 'score']] = ["Car", np.radians(0), 0.96]
    df_all.loc[1, ['label', 'rot_z', 'score']] = ["Car", np.radians(180), 0.96]
    df_all.loc[5, ['label', 'rot_z', 'score']] = ["Car", np.radians(90), 0.9]
    df_all.loc[6, ['label', 'rot_z', 'score']] = ["Car", np.radians(90), 0.9]
    # df_all.loc[8, ['label', 'rot_z', 'score']] = ["Car", np.radians(55), 0.9]
    # df_all.loc[10, ['label', 'rot_z', 'score']] = ["Car", np.radians(30.0001), 0.9]

    df_all = df_all[(df_all['loc_x'] >= 0) & (df_all['loc_x'] <= 10)].reset_index(drop=True)

    print(df_all)
    df_group_save = df_all.copy()
    #save_pseudo_labels(cfg, df_group_save, frame_ID, [False])
    #print("SAVED.")
    print("__________________________________________________________________")

    return df_all

# Function that computes the euclidian distance between two points.
def compute_euclidian_distance(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2) ** 0.5
# Function that computes  the IoU of two bboxes (pcdet)
def iou_2_df_objects(bbox1, bbox2):
    # write elements ['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht', 'rot_z'] from df.iloc[] to torch.tensor
    box1 = torch.tensor(np.array([bbox1[2:9].values.astype(float)]))
    box2 = torch.tensor(np.array([bbox2[2:9].values.astype(float)]))
    iou = box_utils.boxes3d_nearest_bev_iou(box1, box2)
    iou = iou.item()

    return iou


# Function takes bbox dataframes as input and returns groups of overlapping
def get_bbox_groups(cfg, df1, df2, df3, frame_ID):

    # Function that generates a ndarray encoding the df rows of overlapping bboxes.
    def detect_overlapping_bboxes(cfg, df1):
        # Uses euclidian centerpoint distance and then computes IoU.

        # Compute the euclidian distance between all elements of the dataframes to filter out non-overlapping ones.
        distances = []
        for this_index_df1 in range(len(df1)):
            centerpoint_1 = np.array(df1.iloc[this_index_df1][2:5].values.astype(float))
            distances_element = []
            for index_df1 in range(len(df1)):
                centerpoint_2 = np.array(df1.iloc[index_df1][2:5].values.astype(float))
                distances_element.append(compute_euclidian_distance(centerpoint_1, centerpoint_2))
            distances.append(distances_element)

        # Compute a threshold that guarantees that there is no intersection between bboxes.
        largest_dim_len = df1['dim_len'].max()
        largest_dim_wi = df1['dim_wi'].max()
        largest_bbox_radius = (((largest_dim_wi / 2) ** 2) + ((largest_dim_len / 2) ** 2)) ** 0.5
        centerpoint_distance_threshold = (2 * (largest_bbox_radius * 1.1))

        detected_overlaps = np.array(
            [[1 if val < centerpoint_distance_threshold else 0 for val in inner] for inner in distances])
        detected_overlaps = detected_overlaps.astype(float)

        # for the bounding boxes below the non-overlap threshold, get IoU.
        combinations_to_check = np.argwhere(detected_overlaps == 1)

        # print('possible overlaps: ', "\n", detected_overlaps)

        for i in range(len(combinations_to_check)):
            bbox1 = df1.iloc[combinations_to_check[i][0]]
            bbox2 = df1.iloc[combinations_to_check[i][1]]
            iou = iou_2_df_objects(bbox1, bbox2)
            detected_overlaps[combinations_to_check[i][0]][combinations_to_check[i][1]] = iou

        if DEBUG_MODE:
            # Set the printing options
            np.set_printoptions(precision=2, suppress=True)
            print("centerpoint_distance_threshold: ", centerpoint_distance_threshold)
            print("Detected overlaps with IoU: ", "\n", detected_overlaps)

        return detected_overlaps

    # Function identifies all independent groups of overlapping bboxes using networkx graphs
    def identify_independent_bbox_groups(detected_overlaps):

        overlapping_rows = []
        for subarray in detected_overlaps:
            overlapping_rows.append(np.nonzero(subarray)[0].tolist())

        # Create a graph using NetworkX
        graph = nx.Graph()

        # Add nodes to the graph
        graph.add_nodes_from(set(np.concatenate(overlapping_rows)))

        # Iterate over the elements and create edges between connected elements
        for row in overlapping_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[i] != row[j]:
                        graph.add_edge(row[i], row[j])

        # Find connected components in the graph
        connected_components = nx.connected_components(graph)

        # Convert the connected components to a list
        independent_groups = [list(component) for component in connected_components]

        return independent_groups

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("\n", "==> majority_voting: ")

    # Gather all bboxes to one df
    df_all = pd.concat([df1, df2, df3], ignore_index=True).sort_values('label').reset_index(drop=True)

    if ADD_ELEMENTS:
        df_all = add_fake_elements(cfg, df_all, frame_ID)

    if DEBUG_MODE:
        print("df_all: ", "\n", df_all)
    detected_overlaps = detect_overlapping_bboxes(cfg, df_all)
    bbox_groups =  identify_independent_bbox_groups(detected_overlaps)
    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("Detected groups of bboxes: ", "\n", bbox_groups)

    return df_all, detected_overlaps, bbox_groups
# Function performs majority voting on a frame.
def majority_voting(cfg, df1, df2, df3, frame_ID):

    # Convert list to dataframe with bbox info.
    def group_to_df_group(df, group):

        if not group:
            return pd.DataFrame()

        # write group to dataframe
        df_group = df.iloc[group].reset_index(drop=False).rename(columns={"index": "element"})

        # Normalize the headings to [0; 2pi]
        df_group['rot_z'] %= 2 * np.pi
        df_group.loc[df_group['rot_z'] < 0, 'rot_z'] += 2 * np.pi

        return df_group
    # Get number of bboxes in group.
    def subfunction_number_bboxes(df_group):

        if DEBUG_MODE: print("-> subfunction number bboxes.")
        number_bboxes = df_group.shape[0]

        return number_bboxes
    # Check class of group.
    def subfunction_class(df_group):

        if DEBUG_MODE: print("-> subfunction class.")

        classes = df_group['label'].unique()
        if len(classes)== 1:
            return True, None

        else:
            temp_class_bbox_groups = []
            temp_groups = df_group.groupby('label')['element'].apply(list).tolist()
            temp_class_bbox_groups.extend(temp_groups)
            return False, temp_class_bbox_groups
    # Check iou of group.
    def subfunction_iou(cfg, df_group, bbox_iou):

        if DEBUG_MODE: print("-> subfunction iou.")

        combinations = []
        combinations_iou = []
        group = df_group['element'].tolist()

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                combinations.append([group[i], group[j]])

        for combination in combinations:
            combinations_iou.append(bbox_iou[combination[0]][combination[1]])
        iou_dictionary = {tuple(combinations[k]): combinations_iou[k] for k in range(len(combinations))}

        if all(value >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_IOU for value in iou_dictionary.values()):
            return True, None

        else:
            if len(iou_dictionary) > 1:
                graph = nx.Graph()
                for edge, weight in iou_dictionary.items():
                    node1, node2 = edge
                    if weight >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_IOU:
                        graph.add_edge(node1, node2)

                components = list(nx.connected_components(graph))
                isolated_nodes = set()
                for node_pair in iou_dictionary.keys():
                    isolated_nodes.update(node_pair)

                # isolated_nodes keeps only isolated nodes.
                isolated_nodes -= set.union(*components) if components else set()
                for node in isolated_nodes:
                    components.append({node})

                temp_iou_bbox_groups = [list(component) for component in components]

            else:
                if len(iou_dictionary)  == 1:
                    keys = list(iou_dictionary.keys())[0]
                    temp_iou_bbox_groups = [[key] for key in keys]
                    print(temp_iou_bbox_groups)
                else:
                    raise TypeError("No elements in iou_dictionary.")

        return False, temp_iou_bbox_groups
    # Check heading of group.
    def subfunction_heading(cfg, df_group):

        if DEBUG_MODE: print("-> subfunction heading.")

        heading_threshold = np.radians(cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_HEADING)
        headings_dictionary = {}
        for index, row in df_group.iterrows():
            headings_dictionary[row['element']] = {'heading': row['rot_z'], 'similar': []}

        # Check if all headings are similar by being inside threshold intervals.
        for element, heading_data in headings_dictionary.items():
            heading = heading_data['heading']
            relevant_headings_dictionary = headings_dictionary.copy()
            del relevant_headings_dictionary[element]

            # Calculate the intervals [X, X + threshold] and [X + π, X + π + threshold]
            intervals = [np.array([heading, heading + heading_threshold]) % (2 * np.pi),
                         np.array([heading + np.pi, heading + np.pi + heading_threshold]) % (2 * np.pi)]

            mask_heading = []
            for relevant_element, relevant_heading_data in relevant_headings_dictionary.items():
                relevant_heading = relevant_heading_data['heading']
                found_interval = False

                for interval in intervals:
                    # Start of interval bigger than end. --> reverse.
                    if interval[0] > interval[1]:
                        interval_reversed = [interval[1], interval[0]]
                        if not interval_reversed[0] < relevant_heading < interval_reversed[1]:
                            mask_heading.append(True)
                            headings_dictionary[element]['similar'].append(relevant_element)
                            found_interval = True
                            break

                    # Start of interval smaller than end.
                    if interval[0] < interval[1]:
                        if interval[0] <= relevant_heading <= interval[1]:
                            mask_heading.append(True)
                            headings_dictionary[element]['similar'].append(relevant_element)
                            found_interval = True
                            break

                if not found_interval:
                    mask_heading.append(False)

            if all(mask_heading):
                return True, intervals

        # If headings are dissimilar, split them into independent groups.
        temp_heading_bbox_groups = []
        headings_dictionary = {key: value['similar'] for key, value in
                      sorted(headings_dictionary.items(), key=lambda x: len(x[1]['similar']), reverse=True)}

        while headings_dictionary:
            key, values = next(iter(headings_dictionary.items()))
            similar_heading_group = list(np.array([key] + values, dtype=int))
            temp_heading_bbox_groups.append(similar_heading_group)
            headings_dictionary = {key: value for key, value in headings_dictionary.items() if key not in similar_heading_group}

        return False, temp_heading_bbox_groups
    # Check confidence of group.
    def subfunction_confidence(cfg, df_group, mask):

        if DEBUG_MODE: print("-> subfunction confidence.")

        object_type = df_group.loc[0, 'label']
        object_average = df_group['score'].mean()

        if object_type == 'Car':
            # Car; 3 BBOXES
            if mask[0] >=3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_3_BBOXES.CONFIDENCE_CAR:
                    return object_average
                else: return False
            # Car; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_2_BBOXES.CONFIDENCE_CAR:
                    return object_average
                else:  return False
            # Car; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_1_BBOX.CONFIDENCE_CAR:
                    return object_average
                else: return False

        if object_type == 'Cyclist':

            # Cyclist; 3 BBOXES
            if mask[0] >= 3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_3_BBOXES.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False
            # Cyclist; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_2_BBOXES.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False
            # Cyclist; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_1_BBOX.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False

        if object_type == 'Pedestrian':

            # Pedestrian; 3 BBOXES
            if mask[0] >= 3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_3_BBOXES.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
            # Pedestrian; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_2_BBOXES.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
            # Pedestrian; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLDS.THRESHOLD_1_BBOX.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
    # Get representative from group.
    def subfunction_select_representative(df_group,heading_intervals):

        if DEBUG_MODE: print("-> subfunction representative.")

        # identify the majority of similar bbox headings by assigning relative headings.
        interval1 = heading_intervals[0]
        interval2 = heading_intervals[1]
        #print(interval1, interval2)

        df_group['relative_heading'] = None
        for i in range (len(df_group)):
            heading = df_group['rot_z'][i]

            # Headings A and B are used for "front" and "back".
            if interval1[0] > interval1[1]:
                interval_reversed = [interval1[1], interval1[0]]
                if not interval_reversed[0] < heading < interval_reversed[1]:
                    df_group.at[i, 'relative_heading'] = 'A'

            if interval1[0] < interval1[1]:
                if interval1[0] <= heading <= interval1[1]:
                    df_group.at[i, 'relative_heading'] = 'A'

            if interval2[0] > interval2[1]:
                interval_reversed = [interval2[1], interval2[0]]
                if not interval_reversed[0] < heading < interval_reversed[1]:
                    df_group.at[i, 'relative_heading'] = 'B'

            if interval2[0] < interval2[1]:
                if interval2[0] <= heading <= interval2[1]:
                    df_group.at[i, 'relative_heading'] = 'B'


        #df_group = df_group.drop(2)
        if DEBUG_MODE: print(df_group)

        # Find the most frequent value(s)
        most_frequent_heading = df_group['relative_heading'].value_counts()
        majority_values = most_frequent_heading[most_frequent_heading == most_frequent_heading.max()].index
        #print(majority_values)

        # Check if there is a single majority value
        if len(majority_values) == 1:
            df_majority = df_group[df_group['relative_heading'] == majority_values[0]]
        else:
            df_majority = df_group.copy()

        df_single_pseudo_label = df_majority[df_majority['score'] == df_majority['score'].max()]

        # Select one random row if multiple rows have the same max score
        if len(df_single_pseudo_label) > 1:
            df_single_pseudo_label = df_single_pseudo_label.sample(n=1)
            FLAG_HIGHLY_UNCERTAIN_PSEUDO_LABEL[0] = True

        return df_single_pseudo_label



    # Vote a representative from single bbox in group.
    def flow_manager_single_element(cfg, mask, df_group):

        # subfunction_confidence returns the mean confidence if above threshold, else False.
        mask.append(subfunction_confidence(cfg, df_group, mask))
        if not mask[1]:
            if cfg.PIPELINE.PRINT_INFORMATION: print("GROUP REJECTED.")
            return pd.DataFrame()

        # Remove unnecessary columns of pseudo label.
        df_single_pseudo_label = df_group

        return df_single_pseudo_label
    # Vote a representative from multiple bboxes in group.
    def flow_manager_multiple_elements(cfg, mask, df_group, bbox_iou):

        # subfunction_class (True same class, False not.)
        subfunction_class_boolean, temp_class_bbox_groups = subfunction_class(df_group)
        mask.append(subfunction_class_boolean)
        if not mask[1]:
            return False, temp_class_bbox_groups

        # subfunction_iou (True IoUs >= threshold, False not.)
        subfunction_iou_boolean, temp_iou_bbox_groups = subfunction_iou(cfg, df_group, bbox_iou)
        mask.append(subfunction_iou_boolean)
        if not mask[2]:
            return False, temp_iou_bbox_groups

        # subfunction_heading (Intervals including all similar headings, False if dissimilar.)
        subfunction_heading_boolean, heading_intervals = subfunction_heading(cfg, df_group)
        mask.append(subfunction_heading_boolean)
        if not mask[3]:
            temp_heading_bbox_groups = heading_intervals
            return False, temp_heading_bbox_groups

        # subfunction_confidence: (Mean confidence if > threshold, False if not.)
        mask.append(subfunction_confidence(cfg, df_group, mask))
        if not mask[4]:
            if cfg.PIPELINE.PRINT_INFORMATION: print("GROUP REJECTED.")
            return True, pd.DataFrame()

        # subfunction_select_representative returns the most confident representative from the majority heading.
        df_single_pseudo_label = subfunction_select_representative(df_group, heading_intervals)
        df_single_pseudo_label = df_single_pseudo_label.drop(['relative_heading'], axis=1).reset_index(
            drop=True)
        df_single_pseudo_label.at[0, 'score'] = mask[4]

        return True, df_single_pseudo_label



    # Processes all sets of overlapping bounding boxes.
    def main_queue(df_all, current_group):

        df_group = group_to_df_group(df_all, current_group)
        if df_group.empty:
            return pd.DataFrame()

        if cfg.PIPELINE.PRINT_INFORMATION:
            print(df_group)

        mask = []
        mask.append(subfunction_number_bboxes(df_group))
        if mask[0] > 1:
            if DEBUG_MODE: print("flow_manager_multiple_elements: ")
            FLAG_MAIN_QUEUE, df_single_pseudo_label = flow_manager_multiple_elements(cfg, mask, df_group, bbox_iou)

            if not FLAG_MAIN_QUEUE:
                temp_bbox_group = df_single_pseudo_label
                df_temp_pseudo_label = temp_queue(df_all, temp_bbox_group, bbox_iou)
                return df_temp_pseudo_label

        else:
            if DEBUG_MODE: print("flow_manager_single_element: ")
            df_single_pseudo_label = flow_manager_single_element(cfg, mask, df_group)

        return df_single_pseudo_label
    # Processes bounding box groups with that do not pass main_queue checks.
    def temp_queue(df_all, temp_bbox_group, bbox_iou):
        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
            print("__________________________________________________________________")
            print("TEMP QUEUE. ")

        df_representatives = pd.DataFrame()
        while temp_bbox_group:
            temp_mask = []
            current_temp_group = temp_bbox_group.pop(0)
            df_temp_group = group_to_df_group(df_all, current_temp_group)

            if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
                print("Voting on temp_group: ", current_temp_group)
                print(df_temp_group)

            temp_mask.append(subfunction_number_bboxes(df_temp_group))
            if temp_mask[0] > 1:
                if DEBUG_MODE: print("temp flow_manager_multiple_elements: ")
                FLAG_MAIN_QUEUE, df_representative = flow_manager_multiple_elements(cfg, temp_mask, df_temp_group, bbox_iou)

                if not FLAG_MAIN_QUEUE:
                    temp_temp_bbox_group = df_representative
                    temp_bbox_group.extend(temp_temp_bbox_group)
                else:
                    if not df_representative.empty:
                        df_representative['weight'] = temp_mask[0]
                        df_representatives = pd.concat((df_representatives, df_representative), axis=0).reset_index(
                        drop=True)

            else:
                if DEBUG_MODE: print("temp flow_manager_single_element: ")
                df_representative = flow_manager_single_element(cfg, temp_mask, df_temp_group)
                if not df_representative.empty:
                    df_representative['weight'] = temp_mask[0]
                    df_representatives = pd.concat((df_representatives, df_representative), axis=0).reset_index(
                        drop=True)

        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE: print("Set df_representatives: ", "\n", df_representatives)

        if df_representatives.empty:
            if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
                print("__________________________________________________________________")
            return pd.DataFrame()

        else:

            df_representatives = df_representatives.sort_values(by='weight', ascending=False).reset_index(drop=True)
            df_rep_pseudo_labels = pd.DataFrame()
            while not df_representatives.empty:
                overlapping_representatives_iou = np.where(bbox_iou[df_representatives.at[0, 'element']] != 0)[0]
                df_rep_pseudo_label = df_representatives[df_representatives['element'].isin(overlapping_representatives_iou)].copy()
                df_representatives = df_representatives[~df_representatives['element'].isin(overlapping_representatives_iou)].reset_index(drop=True)
                # select best bbox representative. --> weight
                df_rep_pseudo_label = df_rep_pseudo_label[df_rep_pseudo_label['weight'] == df_rep_pseudo_label['weight'].max()]
                # select best bbox representative. --> score
                df_rep_pseudo_label = df_rep_pseudo_label[df_rep_pseudo_label['score'] == df_rep_pseudo_label['score'].max()]
                # Select one random row if multiple rows have the same max score
                if len(df_rep_pseudo_label) > 1:
                    df_rep_pseudo_label = df_rep_pseudo_label.sample(n=1)
                    FLAG_HIGHLY_UNCERTAIN_PSEUDO_LABEL[0] = True

                df_rep_pseudo_labels = pd.concat((df_rep_pseudo_labels, df_rep_pseudo_label), axis=0).reset_index(drop=True)

            if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
                print("\n","Final df_rep_pseudo_labels: ", "\n", df_rep_pseudo_labels)
                print("__________________________________________________________________")

            df_rep_pseudo_labels = df_rep_pseudo_labels.drop(['weight'], axis=1).reset_index(drop=True)

            return df_rep_pseudo_labels



    # Define a flag that labels highly uncertain pseudo-labels to be saved separately.
    FLAG_HIGHLY_UNCERTAIN_PSEUDO_LABEL = [False]

    if df1.empty and df2.empty and df3.empty:
        save_pseudo_labels(cfg, pd.DataFrame(), frame_ID, FLAG_HIGHLY_UNCERTAIN_PSEUDO_LABEL)
        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
            print("No bbox proposals. Saved empty frame", frame_ID)
        return

    df_all, bbox_iou, bbox_groups = get_bbox_groups(cfg, df1, df2, df3, frame_ID)
    if len(bbox_groups) == 0:
        raise TypeError("No bounding box groups but elements in df_all. Check out error.")

    else:
        df_pseudo_labels = pd.DataFrame()
        while bbox_groups:
            current_group = bbox_groups.pop(0)
            if cfg.PIPELINE.PRINT_INFORMATION:
                print("\n", "Voting: ", current_group)

            df_single_pseudo_label = main_queue(df_all, current_group)
            df_pseudo_labels = pd.concat((df_pseudo_labels, df_single_pseudo_label), axis=0).reset_index(drop=True)

        if not df_pseudo_labels.empty:
            df_pseudo_labels = df_pseudo_labels.drop(['element'], axis=1).reset_index(drop=True)

            if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
                print("\n", "\n", "df_pseudo_labels: ", "\n", df_pseudo_labels)
        else:
            if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
                print("\n", "\n", "df_pseudo_labels: ", "\n", "empty dataframe")

        save_pseudo_labels(cfg, df_pseudo_labels, frame_ID, FLAG_HIGHLY_UNCERTAIN_PSEUDO_LABEL)
        if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
            print("Pseudo-labels saved for frame ", frame_ID)


# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(cfg, df_pseudo_labels, frame_ID, flag_uncertain_label):

    # Save pseudo-labels as csv to folder.
    if not os.path.exists(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS):
        os.makedirs(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS)

    # Switch paths for flag_uncertain_label True/ False.
    if not flag_uncertain_label[0]:
        path_to_save = cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS
    else:
        path_to_save = os.path.join(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS, 'control' )
        # Create path if non-existent
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

    # Save pseudo-label to regular folder
    csv_filename = frame_ID + '.csv'
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_to_save, csv_filename), index=False, header=False)

