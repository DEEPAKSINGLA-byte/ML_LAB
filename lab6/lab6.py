import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("lab6/Lab6_vibrations.csv")
features = ["vib_x", "vib_y", "machine_hours"]
X = df[features].to_numpy(float)
y = df["faulty"].to_numpy()
idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
print("train:", X_train.shape, " test:", X_test.shape) 

plt.scatter(df['vib_x'],df['vib_y'],c=df['faulty'])
for name, s in zip(features, X_train.std(axis=0)):
    print(f"{name:15s}{s:12.3f}")

def pairwise_distance(X1,X2):
    diff=X1[:,None,:]-X2[None,:,:]
    squared_distances=np.sum(diff**2,axis=2)
    return np.sqrt(squared_distances)
def knn_predict(X_train, y_train, X_query, k):
    D=pairwise_distance(X_query,X_train)
    nearest_idx=np.argsort(D,axis=1)[:,:k]
    nearest_labels=y_train[nearest_idx]
    predictions=(nearest_labels.mean(axis=1)>0.5).astype(int)
    return predictions
y_pred_train=knn_predict(X_train,y_train,X_train,k=1)
print("All predictions correct:",
      np.array_equal(y_pred_train, y_train))


def standardise(X, mu=None, sigma=None):
    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)

    Z = (X - mu) / sigma
    return Z, mu, sigma
Z_train, mu, sigma = standardise(X_train)
Z_test, _, _ = standardise(X_test, mu, sigma)


# 1. k-NN on raw features
y_pred_raw = knn_predict(
    X_train, y_train, X_test, k=7
)

acc_raw = np.mean(y_pred_raw == y_test)

# 2. k-NN on standardised features
y_pred_std = knn_predict(
    Z_train, y_train, Z_test, k=7
)

acc_std = np.mean(y_pred_std == y_test)

print("Raw feature accuracy:", acc_raw)
print("Standardised accuracy:", acc_std)
print("Majority-class baseline: 0.55")


def k_fold_cv_knn(X, y, k, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    fold_indices = np.array_split(indices, folds)
    accuracy = []

    for i in range(folds):
        val = fold_indices[i]
        tr = np.concatenate([
            fold_indices[j]
            for j in range(folds)
            if j != i
        ])

        X_tr, y_tr = X[tr], y[tr]
        X_val, y_val = X[val], y[val]

        # Fit standardisation on this fold's training data only
        Z_tr, mu, sigma = standardise(X_tr)
        Z_val, _, _ = standardise(X_val, mu, sigma)

        y_pred = knn_predict(
            Z_tr, y_tr, Z_val, k
        )

        accuracy.append(np.mean(y_pred == y_val))

    return np.mean(accuracy)


k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75]

cv_scores = []

for k in k_values:
    score = k_fold_cv_knn(
        X_train,
        y_train,
        k=k,
        folds=5,
        seed=0
    )
    cv_scores.append(score)

# Print the results
print("k\tCV Accuracy")
for k, score in zip(k_values, cv_scores):
    print(f"{k}\t{score:.4f}")

# Find the best k
best_idx = np.argmax(cv_scores)
best_k = k_values[best_idx]

print("\nBest k:", best_k)
print("Best CV accuracy:", round(cv_scores[best_idx], 4))


# Evaluate the chosen k, k=1, and k=75
for k in [best_k, 1, 75]:
    y_pred = knn_predict(
        Z_train, y_train, Z_test, k
    )

    test_accuracy = np.mean(y_pred == y_test)

    print(f"k = {k:2d}, Test accuracy = {test_accuracy:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(k_values, cv_scores, marker='o')

plt.xscale('log')
plt.xlabel('Number of neighbours (k)')
plt.ylabel('5-fold CV accuracy')
plt.title('Choosing k for k-NN')
plt.xticks(k_values, labels=k_values)
plt.grid(True, alpha=0.3)

plt.show()
