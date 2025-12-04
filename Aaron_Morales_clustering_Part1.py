import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, required=True, help='Number of clusters')
parser.add_argument('--max_iter', type=int, default=300)
parser.add_argument('--tol', type=float, default=1e-4)

args = parser.parse_args()

k = args.k # k holds the number of clusters for the script
max_iter = args.max_iter
tol = args.tol

np.random.seed(42)  # for reproducibility
# X == CONSTANT??
# Load our MNIST dataset(784 pixel feature columns w/ 6000 labeled numbers)
X = np.load('MNIST_X_subset.npy', allow_pickle=True)  # shape (6000, 784)
y = np.load('MNIST_y_subset.npy', allow_pickle=True)  # shape (6000,)

# Ensure float for means, preserve original range (MNIST is usually 0-255)
X = X.astype(np.float64)
n_samples, n_features = X.shape


# compute squared Euclidean distances for each n samples and k clusters by the formula
# |X - C|^2 = |X|^2 + |C|^2 - 2 X * C(DOT)
# squared distances suffice for argmin and checking movement if you compare squared shifts
def squared_distances(X, centroids):
    # X: (n_samples, n_feautures)
    # centroids: (k, n feautures)

    # Returns: a df (n_samples, k clusters) where each value is the distance to kth centroid

    # This step squares the sample-feature values, and sums across our feature columns
    # Creates a new arrays suitable for dot product
    X_sq = np.sum(X ** 2, axis = 1)[:, np.newaxis] # (n_samples, 1)
    C_sq = np.sum(centroids ** 2, axis=1)[np.newaxis, :]# (1, k)

    X_dot_C = X.dot(centroids.T) # Transpose for easy dot prod calc
    d_sq = X_sq + C_sq - 2 * X_dot_C
    # floating point rounding distances might come out as tiny negative numbers (like -1e-12)
    return np.maximum(d_sq, 0.0)


def k_means(X, k, max_iter, tol):
    # Initialize centroids
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices].copy()  # shape (k, 784 features)
    
    for iter in range(max_iter):
        # Assignment step(closest centroid for each sample)
        d_sq = squared_distances(X, centroids) # (n_samples, k)
        # assign label to hold min centroid index
        # a 1D array holding index values like 0,1,2...,k
        labels = np.argmin(d_sq, axis = 1) # (n_samples, )

        # Update new centroid position
        # Line iterates each centroid->k, boolean masks array X
        # , selecting indices in X where centroids = j
        # (Builds an array of only cluster j identify sample-features)
        # Sum down each column for each feature to get mean for each feature of j identifying samples
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        # subtract the two (k, features) matrices to determine
        # Holds the largest norm for any k centroids(how much they moved)
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        if shift < tol:
            # print(f"Converged after {iter+1} iterations (max shift={shift:.6f})")
            break

        centroids = new_centroids
        # TROUBLE SHOOT
        '''
        if iter % 5 == 0:
            cluster_sizes = np.bincount(labels, minlength=k)
            print(f"Iteration {iter}: cluster sizes = {cluster_sizes}")
        '''

    return centroids, labels

def clustering_error(labels, y, k):
    # Note
    # y = true label identity
    # labels = cluster assigned identity
    total_error = 0
    for j in range(k):
        cluster_labels = y[labels == j] # boolean mask collects indices of y where predicted labels is j

        if len(cluster_labels) == 0: # empty cluster
            continue

        majority_label = np.bincount(cluster_labels).argmax()

        # Count how many samples in the cluster do NOT match the majority label
        errors = np.sum(cluster_labels != majority_label)
        total_error += errors

    return total_error

def save_centroid_images(centroids, k):
    """
    centroids : np.ndarray (array of centroid vectors, shape (k, 784))
    k : int (Number of clusters)
    """
    # Each centroid is a 1D vector of 784 pixels → reshape to 28x28
    fig, axes = plt.subplots(1, k, figsize=(1.5*k, 1.5))
    
    for i, ax in enumerate(axes):
        centroid_img = centroids[i].reshape(28, 28)
        ax.imshow(centroid_img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'C{i}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'centroids_k{k}.png', dpi=150)
    plt.close(fig)

# RUN #####
centroids, labels = k_means(X, k, max_iter, tol)
error = clustering_error(labels, y, k)
'''
cluster_sizes = np.bincount(labels, minlength=k)
print("np.bincount output:", cluster_sizes)

print(f"Labels range: min={labels.min()}, max={labels.max()}, unique={len(np.unique(labels))}")
'''

# Save centroids visualization
save_centroid_images(centroids, k)

# Print the required formatted output
print(f"k={k}, ERROR={error}")

