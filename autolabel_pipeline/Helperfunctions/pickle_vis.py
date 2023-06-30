import pickle

# Specify the path to your pickle file
pickle_file_path = '/home/output/home/tools/cfgs/autolabel_models/pointpillar/default/eval/eval_with_train/epoch_74/val/result.pkl'

print(pickle_file_path)
# Read the contents of the pickle file
with open(pickle_file_path, 'rb') as f:
    content = pickle.load(f)

# Print the contents
print(content[0])

