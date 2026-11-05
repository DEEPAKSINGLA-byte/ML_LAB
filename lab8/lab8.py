
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lab8/ic272_lab8_loans.csv")

X = df[["income_k", "utilisation"]].to_numpy(float)
emp = df["employment"].to_numpy()
y = df["defaulted"].to_numpy(int)

idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]

X_train, y_train, emp_train = X[tr], y[tr], emp[tr]
X_test, y_test, emp_test = X[te], y[te], emp[te]

p = np.full(400, 0.01)
print(np.prod(p))
print(np.sum(np.log(p)))


def standardise(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    return X_train_std, X_test_std, mu, sigma


def gaussian_pdf(x, mu, var):
    coefficient = 1 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-0.5 * ((x - mu) ** 2) / var)
    return coefficient * exponent


def fit_gaussian_nb(X, y):
    classes = np.array([0, 1])
    n_samples, n_features = X.shape

    priors = np.zeros(2)
    means = np.zeros((2, n_features))
    variances = np.zeros((2, n_features))

    for c in classes:
        X_c = X[y == c]

        priors[c] = len(X_c) / len(X)
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0)

    return priors, means, variances


A = np.array([160, 170, 165], dtype=float)
B = np.array([150, 155, 145], dtype=float)

X_example = np.concatenate([A, B]).reshape(-1, 1)
y_example = np.array([0, 0, 0, 1, 1, 1])

priors_ex, means_ex, variances_ex = fit_gaussian_nb(
    X_example, y_example
)

x = 162

p_x_given_A = gaussian_pdf(
    x, means_ex[0, 0], variances_ex[0, 0]
)

p_x_given_B = gaussian_pdf(
    x, means_ex[1, 0], variances_ex[1, 0]
)

p_A_given_x = (
    p_x_given_A * priors_ex[0]
    / (
        p_x_given_A * priors_ex[0]
        + p_x_given_B * priors_ex[1]
    )
)

print("P(162 | A):", p_x_given_A)
print("P(162 | B):", p_x_given_B)
print("P(A | 162):", p_A_given_x)


def predict_proba_nb(X, priors, means, variances):
    n_samples, n_features = X.shape
    log_score = np.zeros((n_samples, 2))

    for c in [0, 1]:
        log_prior = np.log(priors[c])

        log_density = (
            -0.5 * np.log(2 * np.pi * variances[c])
            - (X - means[c]) ** 2 / (2 * variances[c])
        )

        log_score[:, c] = log_prior + np.sum(
            log_density, axis=1
        )

    p_default = 1 / (
        1 + np.exp(log_score[:, 0] - log_score[:, 1])
    )

    return p_default


def roc_curve_scratch(y_true, y_prob):
    thresholds = np.r_[
        np.inf, np.sort(np.unique(y_prob))[::-1]
    ]

    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    order = np.argsort(fpr)

    return fpr[order], tpr[order]


def auc_trapezoid(fpr, tpr):
    area = 0.0

    for i in range(1, len(fpr)):
        width = fpr[i] - fpr[i - 1]
        height = (tpr[i] + tpr[i - 1]) / 2
        area += width * height

    return area


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def design_matrix(X):
    return np.column_stack((np.ones(X.shape[0]), X))


def log_likelihood(X, y, theta):
    A = design_matrix(X)
    h = sigmoid(A @ theta)
    eps = 1e-12

    ll = np.sum(
        y * np.log(h + eps)
        + (1 - y) * np.log(1 - h + eps)
    )

    return ll


def fit_logistic(X, y, lr=0.1, n_iters=2000):
    A = design_matrix(X)
    m = len(y)

    theta = np.zeros(A.shape[1])
    history = []

    for i in range(n_iters):
        h = sigmoid(A @ theta)
        gradient = (A.T @ (y - h)) / m
        theta = theta + lr * gradient

        history.append(log_likelihood(X, y, theta))

    return theta, history


priors, means, variances = fit_gaussian_nb(
    X_train, y_train
)

y_prob_nb = predict_proba_nb(
    X_test, priors, means, variances
)

X_train_std, X_test_std, mu, sigma = standardise(
    X_train, X_test
)

w, history = fit_logistic(X_train_std, y_train)

X_test_design = design_matrix(X_test_std)
y_prob_lr = sigmoid(X_test_design @ w)

fpr_nb, tpr_nb = roc_curve_scratch(y_test, y_prob_nb)
fpr_lr, tpr_lr = roc_curve_scratch(y_test, y_prob_lr)

auc_nb = auc_trapezoid(fpr_nb, tpr_nb)
auc_lr = auc_trapezoid(fpr_lr, tpr_lr)

print("Gaussian Naive Bayes AUC:", auc_nb)
print("Logistic Regression AUC:", auc_lr)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_nb,
    tpr_nb,
    label=f"Gaussian Naive Bayes (AUC = {auc_nb:.3f})"
)

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {auc_lr:.3f})"
)

plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Gaussian Naive Bayes vs Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
