import numpy as np
import os
import sys


try:
    dir = sys.argv[1]
except IndexError:
    dir = "/mnt/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/sim_noise_002/training/label_2"
dist = 0.8
step = 5

initial_count = min(35000, len(os.listdir(dir)))
print(f"Total files {initial_count}")

# Distribution of the data
train_size = dist * initial_count
train_size = int(train_size)
print(f"Undownsampled training size (80% of total size) {train_size}")
print(f"Undownsampled validation size (20% of total size) {initial_count - train_size}")

train_list = [str(idx).zfill(6) for idx in np.arange(0, train_size, step)]
validation_list = [str(idx).zfill(6) for idx in np.arange(int(train_list[-1]) + step, initial_count, step)]

print(f"Number of train samples (downsampled by factor {step}): {len(train_list)}")
print(f"Number of validation samples (downsampled by factor {step}): {len(validation_list)}")

#Training
with open('train.txt', 'w') as temp_file:
    for item in train_list:
        temp_file.write("%s\n" % item)

#Validation
with open('val.txt', 'w') as temp_file:
    for item in validation_list:
        temp_file.write("%s\n" % item)
