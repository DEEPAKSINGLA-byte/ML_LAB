import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/home/deepak/ML_LAB/lab4/ic272_lab4_rent.csv')

for column in df.columns:
    print(column, "max =", max(df[column]), "min =", min(df[column]))

def train_test_split_scratch(X, y, test_frac, seed):

    rng = np.random.default_rng(seed)

    indices = rng.permutation(X.shape[0])

    X = X[indices]
    y = y[indices]

    split = int(X.shape[0] * test_frac)

    X_test = X[:split]
    X_train = X[split:]

    y_test = y[:split]
    y_train = y[split:]

    return X_train, X_test, y_train, y_test
def fit_simple_lr(x, y):

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = 0
    denominator = 0

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    w = numerator / denominator

    b = y_mean - w * x_mean

    return w, b
X = df['area'].values
y = df['rent'].values

X_train, X_test, y_train, y_test = train_test_split_scratch(
    X,
    y,
    test_frac=0.2,
    seed=0
)


w, b = fit_simple_lr(X_train, y_train)

print("\nRegression coefficients:")
print("w =", round(w, 3))
print("b =", round(b, 3))



x_line = np.linspace(X_test.min(), X_test.max(), 100)

y_line = w * x_line + b

plt.scatter(
    X_test,
    y_test,
    label="Test data"
)

plt.plot(
    x_line,
    y_line,
    label="Fitted line"
)


# Axis labels
plt.xlabel("Area")
plt.ylabel("Rent")

# Legend
plt.legend()

# Display plot
plt.show()