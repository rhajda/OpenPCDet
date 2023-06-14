
import numpy as np
import pandas as pd

def find_overlapping_bboxes(df):
    bboxes = df[['loc_x', 'loc_y', 'loc_z', 'dim_len', 'dim_wi', 'dim_ht']].values

    overlaps = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bbox1 = bboxes[i]
            bbox2 = bboxes[j]

            # Calculate the coordinates of each bounding box
            bbox1_coords = calculate_bbox_coordinates(bbox1)
            bbox2_coords = calculate_bbox_coordinates(bbox2)

            # Check for overlap in each dimension (x, y, z)
            if (bbox1_coords[0][1] >= bbox2_coords[0][0] and bbox1_coords[0][0] <= bbox2_coords[0][1] and
                    bbox1_coords[1][1] >= bbox2_coords[1][0] and bbox1_coords[1][0] <= bbox2_coords[1][1] and
                    bbox1_coords[2][1] >= bbox2_coords[2][0] and bbox1_coords[2][0] <= bbox2_coords[2][1]):
                overlaps.append([i, j])

    return np.array(overlaps)

def calculate_bbox_coordinates(bbox):
    loc_x, loc_y, loc_z, dim_len, dim_wi, dim_ht = bbox

    # Calculate the coordinates of the bounding box
    x_min = loc_x - dim_len / 2
    x_max = loc_x + dim_len / 2
    y_min = loc_y - dim_wi / 2
    y_max = loc_y + dim_wi / 2
    z_min = loc_z - dim_ht / 2
    z_max = loc_z + dim_ht / 2

    return [(x_min, x_max), (y_min, y_max), (z_min, z_max)]


def majority_voting(cfg, df1, df2, df3):

    print("--> test voting")

    # Prints additional information when True.
    DEBUG_MODE = True

    if cfg.PIPELINE.PRINT_INFORMATION or DEBUG_MODE:
        print("\n", "--> majority_voting triggered: ")

    # Gather all bboxes to one df
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    df_all = df_all.sort_values('label').reset_index(drop=True)

    overlaps = find_overlapping_bboxes(df_all)

    print(overlaps)