
# Import libraries
import pathlib
import numpy as np

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]


def read_file_to_list(file_path):
    data_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data_list.append(line.strip())
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error while reading the file: {e}")

    return data_list


if __name__ == "__main__":

    train_file_og = "/home/data/autolabel_waymo/ImageSets_KITTI_full/train.txt"
    working_directory = "/home/data/autolabel_waymo/ImageSets_KITTI_full/"
    split_ratio = 0.2


    frames_train_og = [int(frame) for frame in read_file_to_list(train_file_og)]

    num_train_elements = int(len(frames_train_og) * split_ratio)
    np.random.shuffle(frames_train_og)

    train = frames_train_og[:num_train_elements]
    train.sort()
    train = [str(num).zfill(6) for num in train]

    pseudo_label = frames_train_og[num_train_elements:]
    pseudo_label.sort()
    pseudo_label = [str(num).zfill(6) for num in pseudo_label]

    print("Train:", len(train))
    print("Pseudo Label:", len(pseudo_label))


    # Save train list as train.txt
    with open(working_directory + "train_.txt", "w") as train_file:
        for item in train:
            train_file.write("%s\n" % item)

    # Save pseudo-label list as pseudo-label.txt
    with open(working_directory + "pseudo_label_.txt", "w") as pseudo_label_file:
        for item in pseudo_label:
            pseudo_label_file.write("%s\n" % item)
