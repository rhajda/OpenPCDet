
# Import libraries
import numpy as np
import pathlib
import os


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    val_file = '/home/data/autolabel_waymo/ImageSets_KITTI_full'
    val_file_path = os.path.join(val_file, 'val.txt')

    folder_path = '/home/data/autolabel_waymo/training/label_2'
    output_directory = '/home/data/autolabel_waymo/ImageSets_KITTI_full'
    output_file_path = os.path.join(output_directory, 'train.txt')


    val_list = []
    with open(val_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            val_list.append(line + ".txt")
    print("val_list: ", len(val_list))

    os.makedirs(output_directory, exist_ok=True)


    txt_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]
    print("All files: ", len(txt_files))


    train_files = [file for file in txt_files if file not in val_list]

    # sort the numbers
    # Define a custom sorting function to sort the file names as 6-digit numbers
    def custom_sort(file_name):
        # Extract and convert the number part to an integer
        number_part = file_name.split('.')[0]
        return int(number_part)


    # Sort the file names using the custom sorting function
    sorted_train_files = sorted(train_files, key=custom_sort)

    ####


    print("sorted_train_files: ", len(sorted_train_files))


    # Write the filenames to the output file
    with open(output_file_path, 'w') as output_file:
        for filename in sorted_train_files:
            file_name, file_extension = os.path.splitext(filename)
            output_file.write(file_name + '\n')

    print(f"File names written to '{output_file_path}'")

