
# Import libraries
import numpy as np
import os


# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    folder_path = '/home/data/waymo_open_dataset/label_2'
    output_directory = '/home/data/waymo_open_dataset/ImageSets_KITTI_full'
    output_file_path = os.path.join(output_directory, 'train.txt')

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all .txt files in the specified folder
    txt_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]

    # Write the filenames to the output file
    with open(output_file_path, 'w') as output_file:
        for filename in txt_files:
            file_name, file_extension = os.path.splitext(filename)
            output_file.write(file_name + '\n')

    print(f"File names written to '{output_file_path}'")