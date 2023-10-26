import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP


# Directory path and file names
directory = "/home/sebastian/Code/pointnet-autoencoder-pytorch/output/20230905-131612/glob_feats"
real_file = "glob_feats_real_down.pkl"
sim_file = "glob_feats_sim_down.pkl"
real_file = "glob_feats_real.pkl"
sim_file = "glob_feats_sim.pkl"

# Define functions to load and concatenate pickle files
def load_and_concatenate_pickles(directory, real_file, sim_file):
    real_path = os.path.join(directory, real_file)
    sim_path = os.path.join(directory, sim_file)

    # Load the real pickle file
    with open(real_path, 'rb') as f:
        real_data = np.asarray(pickle.load(f)).reshape(-1,512)
        real_label = np.zeros(len(real_data))

    # Load the sim pickle file
    with open(sim_path, 'rb') as f:
        sim_data = np.asarray(pickle.load(f)).reshape(-1,512)
        sim_label = np.ones(len(sim_data))

    # Concatenate the data along the first dimension
    concatenated_data = np.concatenate((real_data, sim_data), axis=0)
    concatenated_label = np.concatenate((real_label, sim_label), axis=0)

    return concatenated_data, concatenated_label

# Call the function and get the concatenated data
concatenated_data, concatenated_label = load_and_concatenate_pickles(directory, real_file, sim_file)

for perplexity in [10]:
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity, method="barnes_hut", verbose=True, n_iter=1000)
    tsne_result = tsne.fit_transform(concatenated_data)

    reducer = UMAP(n_neighbors=3)
    umap_result = reducer.fit_transform(concatenated_data)

    scatter = []
    fig, ax = plt.subplots(figsize=(12, 12))
    for label in [0, 1]:
        x = umap_result[concatenated_label == label][:, 0]
        y = umap_result[concatenated_label == label][:, 1]
        color = "r" if label == 0 else "b"
        scatter.append(ax.scatter(x, y, color=color, marker="x"))

plt.show()

print()