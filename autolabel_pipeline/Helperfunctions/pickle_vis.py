import pickle

# Specify the path to your pickle file
pickle_file_path = '/home/output/home/tools/cfgs/autolabel_models/pointrcnn/default/eval/eval_all_default/default/epoch_75/val/glob_feat_75.pkl'

print(pickle_file_path)

# Read the contents of the pickle file
with open(pickle_file_path, 'rb') as f:
    content = pickle.load(f)

# Print the contents
print(content)

