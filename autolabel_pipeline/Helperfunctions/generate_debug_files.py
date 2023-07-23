
import pandas as pd
import pathlib
import os
# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]




# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(df_pseudo_labels, frame_ID):

    path_debug_groundtruth = "/home/autolabel_pipeline/debug/debug_gt"
    path_debug_predictions = "/home/autolabel_pipeline/debug/debug_dt"

    # Create path if non-existent
    if not os.path.exists(path_debug_groundtruth):
        os.makedirs(path_debug_groundtruth)

    # Create path if non-existent
    if not os.path.exists(path_debug_predictions):
        os.makedirs(path_debug_predictions)

    # Save pseudo-label to regular folder
    csv_filename = frame_ID + '.csv'
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_groundtruth, csv_filename), index=False, header=False)
    df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_predictions, csv_filename), index=False, header=False)






numbers = [str(num).zfill(6) for num in range(50)]


for number in numbers:
    # Create the dataframe
    data = {
        'ID': [number],
        'label': ['Car'],
        'loc_x': [10],
        'loc_y': [1],
        'loc_z': [1],
        'dim_length': [5],
        'dim_width': [2],
        'dim_height': [1.5],
        'rotation_y': [0],
        'score': [0.2]
    }

    df = pd.DataFrame(data)

    save_pseudo_labels(df, number)



