

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict
import pandas as pd
import numpy as np
import pathlib
import numba
import yaml
import os

# Import visualization functions
from base_functions import load_config, autolabel_path_manager


def class_info(this_class, data, path_project_evaluation):

    this_class_data = data[data['class'] == this_class]
    print(this_class_data)
    this_class_data_boxplot = this_class_data[['dim_height', 'dim_width', 'dim_length']]

    mean_dim_height = this_class_data_boxplot['dim_height'].mean()
    mean_dim_width = this_class_data_boxplot['dim_width'].mean()
    mean_dim_length = this_class_data_boxplot['dim_length'].mean()
    print(f"Mean dim_height: {mean_dim_height}")
    print(f"Mean dim_width: {mean_dim_width}")
    print(f"Mean dim_length: {mean_dim_length}")

    # Boxplot
    dim_height_list = this_class_data_boxplot['dim_height'].tolist()
    dim_width_list = this_class_data_boxplot['dim_width'].tolist()
    dim_length_list = this_class_data_boxplot['dim_length'].tolist()


    data_list = [[dim_height_list, dim_width_list, dim_length_list]]
    title_list = [f'KITTI 100% Training Set BBoxes: {this_class}']
    names_list = [['Height', 'Width', 'Length']]
    xlabel_list = ['Dimensions in Meter']

    my_figsize = [(7, 3)]


    for i, (data, names, xlabel, title) in enumerate(zip(data_list, names_list, xlabel_list, title_list)):
        fig, ax = plt.subplots(figsize=my_figsize[i])

        ax.boxplot(data, vert=False)
        ax.set_yticks(range(1, len(names) + 1))
        ax.set_yticklabels(names)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(False)
        ax.set_xlim(0, 14)

        plt.tight_layout()

        plot_filename = 'boxplot_KITTI_train_' + str(this_class) + '.svg'
        file_path = os.path.join(path_project_evaluation, plot_filename)
        plt.savefig(file_path, format='svg')
        plt.close()
        print(f"SAVED SUCCESSFULLY: {plot_filename}")
        return

def scatter_plot(this_class, data, path_project_evaluation):
    this_class_data = data[data['class'] == this_class]
    print(this_class_data)
    this_class_data_boxplot = this_class_data[['loc_x', 'loc_y', 'loc_z']]


    # Boxplot
    loc_x_list = this_class_data_boxplot['loc_x'].tolist()
    loc_y_list = this_class_data_boxplot['loc_y'].tolist()

    # Create a jointplot
    joint = sns.jointplot(data=pd.DataFrame({'Center point Location X in m': loc_x_list, 'Center point Location Y in m': loc_y_list}),
                          x='Center point Location Y in m', y='Center point Location X in m',
                          marginal_kws=dict(bins=8), s=40)

    # Get the jointplot's axes
    ax = joint.ax_joint

    # Set the x and y axis limits
    ax.set_xlim(-50, 55)
    ax.set_ylim(0, 95)

    # Calculate the percentage of points in each x and y bin
    hist_counts_x, x_bins = np.histogram(loc_y_list, bins=8)
    hist_counts_y, y_bins = np.histogram(loc_x_list, bins=8)
    total_points = len(loc_x_list)
    percentages_x = (hist_counts_x / total_points) * 100
    percentages_y = (hist_counts_y / total_points) * 100

    # Add text annotations to each bin for both x and y
    for i in range(8):
        ax.annotate(f'{percentages_x[i]:.1f}%', xy=(x_bins[i], 95), ha='left', va='top', rotation=60)
        ax.annotate(f'{percentages_y[i]:.1f}%', xy=(55, y_bins[i]), ha='right', va='bottom')


    plot_filename = 'jointplot_NMS_KITTI_it2_' + str(this_class) + '.svg'
    file_path = os.path.join(path_project_evaluation, plot_filename)
    plt.savefig(file_path, format='svg')
    plt.close()
    print(f"SAVED SUCCESSFULLY: {plot_filename}")


def get_dataset_infos(path_imagesets, path_project_evaluation):

    data = pd.DataFrame(columns=['class', 'dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z'])

    # Open the text file and get frame-IDs
    with open(path_imagesets, 'r') as file:
        file_contents = [line.strip() for line in file.readlines()]
    #file_contents = ['000000', '000003', '000007']

    print(len(file_contents))



    count = 0
    for labelfile in file_contents:

        if count >= 10000:
            break

        label_file = os.path.join(cfg.DATA.PATH_GROUND_TRUTHS, (labelfile + '.txt'))
        with open(label_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            elements = line.split()
            row_data = {
                'class': elements[0],
                'dim_height': elements[8],
                'dim_width': elements[9],
                'dim_length': elements[10],
                'loc_x': elements[11],
                'loc_y': elements[12],
                'loc_z': elements[13]}
            data = data.append(row_data, ignore_index=True)

        count += 1

    # Count the number of rows for each class
    class_counts = data['class'].value_counts()

    print(class_counts)

    exit()

    columns_to_convert = ['dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z']
    data[columns_to_convert] = data[columns_to_convert].astype(float)
    print(data)

    # Car
    class_info('Car',data, path_project_evaluation)
    #scatter_plot('Car', data, path_project_evaluation)
    # Ped
    class_info('Pedestrian', data, path_project_evaluation)
    #scatter_plot('Pedestrian', data, path_project_evaluation)
    # Cycl.
    class_info('Cyclist', data, path_project_evaluation)
    #scatter_plot('Cyclist', data, path_project_evaluation)

def get_pseudo_labels_to_df(path_project_evaluation):

    data = pd.DataFrame()
    pseudo_label_list = []

    for file_name in os.listdir(path_manager.get_path("path_pseudo_labels_nms")):
        if file_name.endswith('.csv'):
            base_name = os.path.splitext(file_name)[0]
            pseudo_label_list.append(base_name)
    print(len(pseudo_label_list))


    count = 0
    for labelfile in pseudo_label_list:
        if count >= 100000:
            break

        label_file = os.path.join(path_manager.get_path("path_pseudo_labels_nms"), (labelfile + '.csv'))

        # Check if the CSV file is empty before attempting to read it
        if os.path.getsize(label_file) == 0:
            print("here")
            continue
        # Read the CSV file using pandas without specifying column names
        df = pd.read_csv(label_file, header=None)

        # Extract specific columns for each row and append them to the main DataFrame
        for index, row in df.iterrows():
            row_data = {
                'class': row[0],
                'loc_x': row[1],
                'loc_y': row[2],
                'loc_z': row[3],
                'dim_length': row[4],
                'dim_width': row[5],
                'dim_height': row[6]
            }
            data = data.append(row_data, ignore_index=True)
        count +=1
    columns_to_convert = ['dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z']
    data[columns_to_convert] = data[columns_to_convert].astype(float)
    print(data)

    # Car
    class_info('Car', data, path_project_evaluation)
    #scatter_plot('Car', data, path_project_evaluation)
    # Ped
    #class_info('Pedestrian', data, path_project_evaluation)
    # scatter_plot('Pedestrian', data, path_project_evaluation)
    # Cycl.
    #class_info('Cyclist', data, path_project_evaluation)
    # scatter_plot('Cyclist', data, path_project_evaluation)


    return




if __name__ == "__main__":

    # Load EasyDict to access parameters.
    cfg = load_config()
    path_manager = autolabel_path_manager(cfg)

    path_project_evaluation = os.path.join(path_manager.get_path("path_project_data"), "evaluation", 'datasets')
    os.makedirs(path_project_evaluation, exist_ok=True)


    # FOR GT
    GT_plots = False
    if GT_plots:
        #path_imagesets = os.path.join(path_manager.get_path("path_project_dataset"), "original/val.txt")
        path_imagesets = os.path.join(path_manager.get_path("path_project_dataset"), "ImageSets_KITTI_full/train.txt")
        #path_imagesets = os.path.join(path_manager.get_path("path_project_dataset"), "original/train.txt")
        get_dataset_infos(path_imagesets, path_project_evaluation)

    # FOR PSEUDO-LABELS
    PL_plots =  True
    if PL_plots:
        get_pseudo_labels_to_df(path_project_evaluation)




