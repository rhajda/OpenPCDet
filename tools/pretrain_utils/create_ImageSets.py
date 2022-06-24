import numpy as np
import os
import sys


try:
    dir = sys.argv[1]
except IndexError:
    dir = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_noise_002/training/label_2"
split_train_valtest = 2/3
split_val_test = 1/2
step = 5

total_files = min(30000, len(os.listdir(dir)))
print(f"Total files {total_files}")

# Distribution of the data
train_size = int(split_train_valtest * total_files)
val_size = int((1 - split_train_valtest) * split_val_test * total_files)
test_size = int((1 - split_train_valtest) * (1 - split_val_test) * total_files)
print(f"Undownsampled training size ({train_size / total_files * 100} %  of total size): {train_size}")
print(f"Undownsampled validation size ({val_size / total_files * 100} % of total size): {val_size}")
print(f"Undownsampled testing size ({test_size / total_files * 100} % of total size): {test_size}")

train_list = [str(idx).zfill(6) for idx in np.arange(0, train_size, step)]
validation_list = [str(idx).zfill(6) for idx in np.arange(int(train_list[-1]) + step, train_size + val_size, step)]
testing_list = [str(idx).zfill(6) for idx in np.arange(int(validation_list[-1]) + step, total_files, step)]

print(f"Number of train samples (downsampled by factor {step}): {len(train_list)}")
print(f"Number of validation samples (downsampled by factor {step}): {len(validation_list)}")
print(f"Number of testing samples (downsampled by factor {step}): {len(testing_list)}")

# Training
with open('train.txt', 'w') as temp_file:
    for item in train_list:
        temp_file.write("%s\n" % item)

# Validation
with open('val.txt', 'w') as temp_file:
    for item in validation_list:
        temp_file.write("%s\n" % item)

# Testing
with open('test.txt', 'w') as temp_file:
    for item in testing_list:
        temp_file.write("%s\n" % item)
