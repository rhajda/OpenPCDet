
#Import libraries
import numpy as np
import os
import torch
import pandas as pd
import networkx as nx




# PCDET
from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(cfg, df_pseudo_labels):
    # Save pseudo-labels as csv to folder.
    if not os.path.exists(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS):
        os.makedirs(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS)

    csv_filename = df_pseudo_labels.iloc[0, 0] + '.csv'
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(cfg.PIPELINE.MAJORITY_VOTING.PATH_SAVE_PSEUDO_LABELS, csv_filename),
                                        index=False, header=False)

# Function takes bbox dataframes as input and returns groups of overlapping
def get_bbox_groups(cfg, df1, df2, df3):

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

        if cfg.PIPELINE.PRINT_INFORMATION:
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


    # Print additional information when True.
    DEBUG_MODE = False

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("\n", "--> majority_voting triggered: ")

    # Gather all bboxes to one df
    df_all = pd.concat([df1, df2, df3], ignore_index=True).sort_values('label').reset_index(drop=True)

    # debugging
    ########################################################################
    add_elements = True
    if add_elements:
        print("FAKE ELEMENTS ACTIVE")

        df_all.loc[0, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
        #df_all.loc[1, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
        df_all.loc[5, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
        #df_all.loc[6, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
        #df_all.loc[8, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]
        #df_all.loc[10, ['loc_x', 'loc_y', 'loc_z']] = [2, 0, 0]

        df_all.loc[0, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
        #df_all.loc[1, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
        df_all.loc[5, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
        #df_all.loc[6, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
        #df_all.loc[8, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]
        #df_all.loc[10, ['dim_len', 'dim_wi', 'dim_ht']] = [3, 1.5, 1.5]

        df_all.loc[0, ['label', 'rot_z']] = ["Car", np.radians(0)]
        #df_all.loc[1, ['label', 'rot_z']] = ["Car", np.radians(5)]
        df_all.loc[5, ['label', 'rot_z']] = ["Car", np.radians(40)]
        #df_all.loc[6, ['label', 'rot_z']] = ["Car", np.radians(15)]
        #df_all.loc[8, ['label', 'rot_z']] = ["Car", np.radians(20)]
        #df_all.loc[10, ['label', 'rot_z']] = ["Car", np.radians(30.0001)]

        #df_all = df_all[(df_all['loc_x'] >= 0) & (df_all['loc_x'] <= 10)].reset_index(drop=True)


        print(df_all)
        df_group_save = df_all.copy()
        save_pseudo_labels(cfg, df_group_save)
        print("SAVED.")
    ########################################################################

    if DEBUG_MODE:
        print("df_all: ", "\n", df_all)
    detected_overlaps = detect_overlapping_bboxes(cfg, df_all)
    bbox_groups =  identify_independent_bbox_groups(detected_overlaps)
    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("Detected groups of bboxes: ", "\n", bbox_groups)

    return df_all, detected_overlaps, bbox_groups



# Function performs majority voting on a frame.
def majority_voting(cfg, df1, df2, df3):

    # Convert list to dataframe with bbox info.
    def group_to_df_group(df, group):

        if not group:
            print("empty group. Exit. Implement discard or error.")
            exit()
        # write group to dataframe
        df_group = df.iloc[group].reset_index(drop=False).rename(columns={"index": "element"})

        # Normalize the headings to [0; 2pi]
        df_group['rot_z'] %= 2 * np.pi
        df_group.loc[df_group['rot_z'] < 0, 'rot_z'] += 2 * np.pi

        return df_group
    # Get number of bboxes in group.
    def subfunction_number_bboxes(df_group):
        print("--> subfunction number bboxes.")
        number_bboxes = df_group.shape[0]

        return number_bboxes
    # Check class of group.
    def subfunction_class(df_group):
        print("--> subfunction class.")

        classes = df_group['label'].unique()
        if len(classes)== 1:

            return True

        else:
            print("__________________________________________________________________")
            temp_bbox_groups = []
            temp_groups = df_group.groupby('label')['element'].apply(list).tolist()
            temp_bbox_groups.extend(temp_groups)
            print(temp_bbox_groups)

            return False
    # Check iou of group.
    def subfunction_iou(cfg, df_group, bbox_iou):
        print("--> subfunction iou.")

        combinations = []
        combinations_iou = []
        group = df_group['element'].tolist()

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                combinations.append([group[i], group[j]])

        for combination in combinations:
            combinations_iou.append(bbox_iou[combination[0]][combination[1]])

        iou_dictionary = {tuple(combinations[k]): combinations_iou[k] for k in range(len(combinations))}
        print(iou_dictionary)

        if all(value >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_IOU for value in iou_dictionary.values()):
            return True

        else:
            print("IoU below threshold. Implement IoU Split.")
            if len(iou_dictionary) > 1:

                graph = nx.Graph()
                for edge, weight in iou_dictionary.items():
                    node1, node2 = edge
                    if weight >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_IOU:
                        graph.add_edge(node1, node2)

                components = list(nx.connected_components(graph))
                isolated_nodes = set()
                for node_pair in iou_dictionary.keys():
                    isolated_nodes.update(node_pair)

                isolated_nodes -= set.union(*components)
                for node in isolated_nodes:
                    components.append({node})

                output = [list(component) for component in components]
                print(output)

            else:
                if len(iou_dictionary)  == 1:
                    keys = list(iou_dictionary.keys())[0]
                    output = [[key] for key in keys]
                    print(output)
                else:
                    raise TypeError("No elements in iou_dictionary.")

        return False
    # Check heading of group.
    def subfunction_heading(cfg, df_group):
        print("--> subfunction heading.")

        heading_threshold = np.radians(cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_HEADING)
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
                return intervals

        # If headings are dissimilar, split them into independent groups.
        output = []
        headings_dictionary = {key: value['similar'] for key, value in
                      sorted(headings_dictionary.items(), key=lambda x: len(x[1]['similar']), reverse=True)}

        while headings_dictionary:
            key, values = next(iter(headings_dictionary.items()))
            similar_heading_group = list(np.array([key] + values, dtype=int))
            output.append(similar_heading_group)
            headings_dictionary = {key: value for key, value in headings_dictionary.items() if key not in similar_heading_group}

        print(output)


        return False
    # Check confidence of group.
    def subfunction_confidence(cfg, df_group, mask):
        print("--> subfunction confidence.")

        object_type = df_group.loc[0, 'label']
        object_average = df_group['score'].mean()

        if object_type == 'Car':
            # Car; 3 BBOXES
            if mask[0] >=3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_3_BBOXES.CONFIDENCE_CAR:
                    return object_average
                else: return False
            # Car; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_2_BBOXES.CONFIDENCE_CAR:
                    return object_average
                else:  return False
            # Car; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_1_BBOX.CONFIDENCE_CAR:
                    return object_average
                else: return False

        if object_type == 'Cyclist':

            # Cyclist; 3 BBOXES
            if mask[0] >= 3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_3_BBOXES.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False
            # Cyclist; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_2_BBOXES.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False
            # Cyclist; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_1_BBOX.CONFIDENCE_CYCLIST:
                    return object_average
                else: return False

        if object_type == 'Pedestrian':

            # Pedestrian; 3 BBOXES
            if mask[0] >= 3:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_3_BBOXES.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
            # Pedestrian; 2 BBOXES
            if mask[0] == 2:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_2_BBOXES.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
            # Pedestrian; 1 BBOX
            if mask[0] == 1:
                if object_average >= cfg.PIPELINE.MAJORITY_VOTING.THRESHOLD_1_BBOX.CONFIDENCE_PEDESTRIAN:
                    return object_average
                else: return False
    # Get representative from group.
    def subfunction_select_representative(df_group,mask):
        print("--> subfunction representant.")

        # identify the majority of similar bbox headings by assigning relative headings.
        interval1 = mask[3][0]
        interval2 = mask[3][1]
        #print(interval1, interval2)

        df_group['relative_heading'] = None
        for i in range (len(df_group)):
            heading = df_group['rot_z'][i]

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
        print(df_group)

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

        return df_single_pseudo_label


    # Vote a representative from single bbox in group.
    def flow_manager_single_element(cfg, mask, df_group):

        # subfunction_confidence returns the mean confidence if above threshold, else False.
        mask.append(subfunction_confidence(cfg, df_group, mask))
        if not mask[1]:
            print("GROUP REJECTED.")
            df_empty_pseudo_label = df_group.iloc[0:0].drop(['element'], axis=1).copy()
            return df_empty_pseudo_label

        # Remove unnecessary columns of pseudo label.
        df_single_pseudo_label = df_group.drop(['element'], axis=1).reset_index(drop=True)

        return df_single_pseudo_label

    # Vote a representative from multiple bboxes in group.
    def flow_manager_multiple_elements(cfg, mask, df_group, bbox_iou):

        # subfunction_class (True same class, False not.)
        mask.append(subfunction_class(df_group))
        if not mask[1]:
            print("Trigger temp queue. Classes not matching. ")
            return False

        # subfunction_iou (True IoUs >= threshold, False not.)
        mask.append(subfunction_iou(cfg, df_group, bbox_iou))
        if not mask[2]:
            print("Trigger temp queue. IoU not matching.")
            return False

        # subfunction_heading (Intervals including all similar headings, False if dissimilar.)
        mask.append(subfunction_heading(cfg, df_group))
        if not mask[3]:
            print("Trigger temp queue. Headings not matching.")
            return False

        # subfunction_confidence: (Mean confidence if > threshold, False if not.)
        mask.append(subfunction_confidence(cfg, df_group, mask))
        if not mask[4]:
            print("GROUP REJECTED.")
            return

        # subfunction_select_representative returns the most confident representative from the majority heading.
        df_single_pseudo_label = subfunction_select_representative(df_group, mask)

        # Remove unnecessary columns of pseudo label.
        df_single_pseudo_label = df_single_pseudo_label.drop(['relative_heading', 'element'], axis=1).reset_index(drop=True)
        df_single_pseudo_label.at[0, 'score'] = mask[4]

        return df_single_pseudo_label


    def flow_manager_temp_queue(cfg, mask, df_group_mismatch, bbox_iou, mismatch_type):
        print("__________________________________________________________________")

        temp_bbox_groups = []

        print("flow_manager_temp_queue: ", mismatch_type)

        if mismatch_type == "class":
            print(df_group_mismatch)
            temp_groups = df_group_mismatch.groupby('label')['element'].apply(list).tolist()
            temp_bbox_groups.extend(temp_groups)


        if mismatch_type == "iou":
            print()

        if mismatch_type == "heading":
            print()

        print(temp_bbox_groups)
        print("__________________________________________________________________")
        return



    def main_queue(bbox_groups):

        df_pseudo_labels = pd.DataFrame()
        while bbox_groups:

            current_group = bbox_groups.pop(0)
            df_group = group_to_df_group(df_all, current_group)
            print("\n", "Voting on group: ", current_group)
            print(df_group)

            # mask: ["number bboxes", "class", "IoU", "heading", "overall confidence"]
            mask = []
            mask.append(subfunction_number_bboxes(df_group))
            if mask[0] > 1:
                print("flow_manager_multiple_elements: ")
                df_single_pseudo_label = flow_manager_multiple_elements(cfg, mask, df_group, bbox_iou)
                #print("flow_manager_multiple_elements output: ", "\n", df_single_pseudo_label)
            else:
                print("flow_manager_single_element: ")
                df_single_pseudo_label = flow_manager_single_element(cfg, mask, df_group)
                #print("flow_manager_single_elements output: ", "\n", df_single_pseudo_label)

            break

            df_pseudo_labels = pd.concat((df_pseudo_labels, df_single_pseudo_label), axis=0).reset_index(drop=True)

        print("\n", "\n", "df_pseudo_labels: ", "\n", df_pseudo_labels)





    # get the bboxes, bbox IoUs, independent bbox groups
    df_all, bbox_iou, bbox_groups = get_bbox_groups(cfg, df1, df2, df3)

    # Trigger the main queue.
    print("\n", "bbox_groups: ", bbox_groups)
    main_queue(bbox_groups)

