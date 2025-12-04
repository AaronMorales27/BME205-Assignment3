import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.stats import mode
# Number of clusters
k = 30
# Load our dataset
# 1355 dog samples, each with 784 SNP features.
# clades serve as the true labels
X = np.load("dogs_X.npy") # shape: (samples: 1355, features: 178)
clades = np.load("dogs_clades.npy", allow_pickle=True) # ?shape: (1355, )?

# Heirarchal linkage matrix
# Ward's method minimizes within-cluster variance at each merge step
Z = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    truncate_mode='lastp', # truncate tree into p clusters(30)
    p=k, show_leaf_counts=True # Show leaf counts
    # leaf_rotation=90.,
    # leaf_font_size=10.,
)
plt.title(f"Truncated Hierarchical Dendrogram (k={k})")
plt.xlabel("Cluster")
plt.ylabel("Distance")
plt.savefig("Dogs_dendrogram_truncated.png", dpi=300, bbox_inches='tight')
plt.close()

# store clustered labels for truncated dendrogram
# fcluster converts the hierarchical linkage (Z) into flat cluster labels.
labels = fcluster(Z, t=k, criterion='maxclust')

# Compute clustering error by comparing labels to clades
def clustering_error(true_labels, cluster_labels):
    """
    Computes total number of samples that do NOT belong
    to the majority true class within each cluster.

    Args:
        true_labels (array): Ground truth class (clades)
        cluster_labels (array): Cluster assignments from fcluster

    Returns:
        int: Total number of misclassified samples
    """
    # unique_clusters holds the unique cluster labels in the dendrogram
    unique_clusters = np.unique(cluster_labels)
    error = 0

    # For each unique cluster in the dendrogram
    for cluster in unique_clusters:
        # Find indices of samples belonging to this cluster
        indices = np.where(cluster_labels == cluster)[0]
        cluster_true = true_labels[indices]  # True clades of those those samples

        # Find the majority clade (most common true label)
        values, counts = np.unique(cluster_true, return_counts=True)
        majority_label = values[np.argmax(counts)]

        # Count how many samples in this cluster do NOT match that majority
        error += np.sum(cluster_true != majority_label)

    return error

error = clustering_error(clades, labels)
print(f"k={k}, ERROR={error}")
'''
from sklearn.decomposition import PCA
import pandas as pd
X_reduced = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='tab20', s=10)
plt.title("PCA projection colored by cluster assignment")
plt.show()

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=pd.factorize(clades)[0], cmap='tab20', s=10)
plt.title("PCA projection colored by true clades")
plt.show()
'''