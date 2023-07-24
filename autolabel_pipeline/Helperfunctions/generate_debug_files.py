
import pandas as pd
import pathlib
import os
# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]




# Function saves a dataframe with pseudo labels to csv.
def save_pseudo_labels(df_pseudo_labels, frame_ID, str_type):

    csv_filename = frame_ID + '.csv'
    path_debug_groundtruth = "/home/autolabel_pipeline/debug/debug_gt"
    path_debug_predictions = "/home/autolabel_pipeline/debug/debug_dt"

    if str_type == "GT":
        # Create path if non-existent
        if not os.path.exists(path_debug_groundtruth):
            os.makedirs(path_debug_groundtruth)

        df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_groundtruth, csv_filename), index=False,
                                            header=False)

    elif str_type == "DT":
        # Create path if non-existent
        if not os.path.exists(path_debug_predictions):
            os.makedirs(path_debug_predictions)

        df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_predictions, csv_filename), index=False, header=False)

    elif str_type == "GTDT":

        if not os.path.exists(path_debug_groundtruth):
            os.makedirs(path_debug_groundtruth)
        if not os.path.exists(path_debug_predictions):
            os.makedirs(path_debug_predictions)

        df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_groundtruth, csv_filename), index=False,
                                            header=False)
        df_pseudo_labels.iloc[:, 1:].to_csv(os.path.join(path_debug_predictions, csv_filename), index=False,
                                            header=False)





def automatic():
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

        save_pseudo_labels(df, number, "GTDT")

def manual(int_number):
    number = str(int_number).zfill(6)

    data_dt = {'ID': [number], 'label': ['Car'], 'loc_x': [10], 'loc_y': [1], 'loc_z': [1],
        'dim_length': [5], 'dim_width': [2], 'dim_height': [1.5], 'rotation_y': [0],'score': [0.8]}

    data_gt = {'ID': [number, number], 'label': ['Pedestrian', 'Car'], 'loc_x': [10, 2], 'loc_y': [1, 1], 'loc_z': [1, 0],
               'dim_length': [5, 5], 'dim_width': [2, 2], 'dim_height': [1.5, 1.5], 'rotation_y': [0, 0],
               'score': [0, 0]}


    df_dt = pd.DataFrame(data_dt)
    df_gt = pd.DataFrame(data_gt)

    save_pseudo_labels(df_gt, number, "GT")
    save_pseudo_labels(df_dt, number, "DT")



if __name__ == "__main__":

    FLAG_MANUAL_MODE = True

    if FLAG_MANUAL_MODE:
        # number for csv name, GT or DT
        manual(0)
    else:
        automatic()

    print("done")