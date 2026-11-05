import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("lab9/ic272_lab9_customers.csv")
X = df[["x_km", "y_km"]].to_numpy(float)
print(X.shape) # (300, 2)
def random_init(X, K, seed):
    """K distinct data points, chosen at random, as starting centroids."""
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return X[idx]
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=14)
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
plt.show()


def pairwise_distances(A, B):
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def assign_clusters(X, centroids):
    distances = pairwise_distances(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, centroids):
    new_centroids = centroids.copy()

    for k in range(len(centroids)):
        points = X[labels == k]

        if len(points) > 0:
            new_centroids[k] = np.mean(points, axis=0)

    return new_centroids


def distortion(X, labels, centroids):
    J = 0

    for k in range(len(centroids)):
        points = X[labels == k]
        J += np.sum((points - centroids[k])**2)

    return J


def kmeans(X, init_centroids, max_iter=100):
    centroids = init_centroids.copy()

    for i in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    labels = assign_clusters(X, centroids)
    J = distortion(X, labels, centroids)

    return centroids, labels, J, i + 1


# Task 1
init_centroids = random_init(X, 3, 42)

centroids, labels, J, iterations = kmeans(X, init_centroids)

print("Centroids:\n", centroids)
print("Distortion:", J)
print("Iterations:", iterations)


# Task 2
K = 6
results = []

for seed in range(10):
    init = random_init(X, K, seed)
    c, labels, J, it = kmeans(X, init)

    results.append([seed, c, labels, J, it, init])

    print(seed, "Distortion:", round(J, 4), "Iterations:", it)


best = min(results, key=lambda x: x[3])
worst = max(results, key=lambda x: x[3])

print("Best seed:", best[0], "Distortion:", best[3])
print("Worst seed:", worst[0], "Distortion:", worst[3])


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, result, title in zip(
    axes, [best, worst], ["Best Clustering", "Worst Clustering"]
):
    labels = result[2]
    centroids = result[1]

    for k in range(K):
        points = X[labels == k]
        ax.scatter(points[:, 0], points[:, 1], s=14)

    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker="X", color="black", s=150)

    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()


# Task 3
def best_of_kmeans(X, K, n_init=10):
    best_J = float("inf")
    best = None

    for seed in range(n_init):
        init = random_init(X, K, seed)
        c, labels, J, it = kmeans(X, init)

        if J < best_J:
            best_J = J
            best = (c, labels, J, it)

    return best


distortions = []

for K in range(1, 11):
    c, labels, J, it = best_of_kmeans(X, K, 10)
    distortions.append(J)

    print("K:", K, "Distortion:", round(J, 4))


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Distortion")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid()
plt.show()
