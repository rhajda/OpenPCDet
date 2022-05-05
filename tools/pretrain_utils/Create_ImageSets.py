import numpy as np
import os

initial_count = 0
dir = r"Z:\13_Vegas_Challenge\03_Data\02_Real\00_extracted_data\for_training\real\label_2"
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        initial_count += 1
print(initial_count)

# Distribution of the data
dist = 0.8
train_size = dist * initial_count
train_size = int(train_size+0.5)
print(train_size)


train_list = list(np.arange(0,train_size,5))
validation_list = list(np.arange(train_size, initial_count,5))

#Training
with open('train.txt', 'w') as temp_file:
    for item in train_list:
        temp_file.write("%s\n" % item)

#Validation
with open('validation.txt', 'w') as temp_file:
    for item in validation_list:
        temp_file.write("%s\n" % item)