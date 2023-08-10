import pickle

# Specify the path to your pickle file
pickle_file_path = '/home/data/autolabel/kitti_dbinfos_train.pkl'

print(pickle_file_path)
# Read the contents of the pickle file
with open(pickle_file_path, 'rb') as f:
    content = pickle.load(f)

# Print the contents
print(content)

