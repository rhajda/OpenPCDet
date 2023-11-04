

# Import libraries
import numpy as np
import pathlib
import os
import random


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


# Define a custom sorting function to sort the file names as 6-digit numbers
def custom_sort(file_name):
    # Extract and convert the number part to an integer
    number_part = file_name.split('.')[0]
    return int(number_part)

def load_txt_file(file_path):
    my_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            my_list.append(line)
    return my_list

def reduce_waymo_in_kitti_format_dataset(file_path, file_path_save, num_elements):

    my_list = load_txt_file(file_path)
    print("my_list: ", len(my_list))
    random.shuffle(my_list)

    elements = my_list[:num_elements]
    elements_sorted = sorted(elements, key=custom_sort)
    print("elements_sorted: ", len(elements_sorted))
    print(elements_sorted)

    # Write the filenames to the output file
    with open(file_path_save, 'w') as output_file:
        for filename in elements_sorted:
            output_file.write(filename + '\n')

    print(f"File names written to '{file_path_save}'")

def update_labels(val_file_path_save, train_file_path_save, test_file_path_save, file_labels):

    val_list = load_txt_file(val_file_path_save)
    train_list = load_txt_file(train_file_path_save)
    print("val_list: ", len(val_list))
    print("train_list: ", len(train_list))

    test_list = val_list + train_list
    test_list_sorted = sorted(test_list, key=custom_sort)
    print("test_list_sorted: ", len(test_list_sorted))

    # Write the filenames to the output file
    with open(test_file_path_save, 'w') as output_file:
        for filename in test_list_sorted:
            output_file.write(filename + '\n')
    print(f"File names written to '{test_file_path_save}'")


    label_files = [filename for filename in os.listdir(file_labels) if filename.endswith('.txt')]
    label_files = [os.path.splitext(filename)[0] for filename in label_files]
    label_files_sorted = sorted(label_files, key=custom_sort)
    print("label_files_sorted: ", len(label_files_sorted))

    elements_not_in_test = set(label_files_sorted) - set(test_list_sorted)
    print("elements_not_in_test: ", len(elements_not_in_test))

    # Remove labels in label_2 folder that are not needed anymore.
    for element in elements_not_in_test:
        filename_to_remove = element + '.txt'
        file_to_remove = os.path.join(file_labels, filename_to_remove)
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
        else:
            print(f"File {filename_to_remove} does not exist.")



if __name__ == "__main__":

    file_ImageSets = '/home/data/autolabel_waymo/ImageSets_KITTI_full'
    file_labels = '/home/data/autolabel_waymo/training/label_2'

    val_file_path = os.path.join(file_ImageSets, 'val.txt')
    val_file_path_save = os.path.join(file_ImageSets, 'val_new.txt')
    train_file_path = os.path.join(file_ImageSets, 'train.txt')
    train_file_path_save = os.path.join(file_ImageSets, 'train_new.txt')
    test_file_path_save = os.path.join(file_ImageSets, 'test_new.txt')

    reduce_waymo_in_kitti_format_dataset(val_file_path, val_file_path_save, 500)
    reduce_waymo_in_kitti_format_dataset(train_file_path, train_file_path_save, 2000)
    update_labels(val_file_path_save, train_file_path_save, test_file_path_save, file_labels)



