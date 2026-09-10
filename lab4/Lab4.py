import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('lab4/ic272_lab4_rent.csv')

def train_test_split_scratch(X,y,test_frac,seed):
    rng=np.random.default_rng(seed)
    indices=rng.permutation(X.shape[0])
    split=int(X.shape[0]*test_frac)
    X=X[indices]
    Y=y[indices]
    X_train=X[split:]
    Y_train=Y[split:]
    X_test=X[:split]
    Y_test=Y[:split]
    return X_train,Y_train,X_test,Y_test
X=df['area'].values
Y=df['rent'].values
X_train, Y_train, X_test, Y_test = train_test_split_scratch(X, Y, test_frac=0.2, seed=42)
def fit_simple_lr(x,y):
    n=len(x)
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    numerator=np.sum((x-x_mean)*(y-y_mean))
    denominator=np.sum((x-x_mean)**2)
    slope=numerator/denominator
    intercept=y_mean-slope*x_mean
    return slope,intercept
slope, intercept = fit_simple_lr(X_train, Y_train)

plt.scatter(X_train,Y_train)
x_range=np.linspace(X_train.min(),X_train.max(),100)
y_range=slope*x_range+intercept
plt.plot(x_range,y_range,color='red')
plt.show()