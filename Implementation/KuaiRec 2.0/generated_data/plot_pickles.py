import pickle
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "Bound": "history_bound.pickle",
    "MAE Recall": "history_mae_recall.pickle",
    "MAE AUC": "history_mae_auc.pickle"
}

# Load data from pickles
data = {}
for key, path in file_paths.items():
    with open(path, 'rb') as handle:
        data[key] = pickle.load(handle)

# Plotting each metric separately
for key, values in data.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(values)), values, label=key, linestyle='-', marker='o')
    plt.xlabel('Number of Partitions')
    plt.ylabel(key)
    plt.title(f'{key} History')
    plt.legend()
    plt.grid(True)
    plt.show()
