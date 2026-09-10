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
X_train, Y_train, X_test, Y_test = train_test_split_scratch(X, Y, test_frac=0.2, seed=0)
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


#TASK 2

def design_matrix(X):
    x=np.ones(X.shape[0])
    A=np.column_stack((x,X))
    return A
def fit_normal_equation(X,y):
    A=design_matrix(X)
    theta=np.linalg.solve(A.T @ A,A.T @ y)
    return theta
X = df[['area', 'rooms', 'age', 'dist_km', 'floors']].values
Y = df['rent'].values

theta=fit_normal_equation(X,Y)
print(theta)


#TASK 3
def loss(theta,X,y):
    A=design_matrix(X)
    n=A.shape[0]
    loss=(1/n)*np.sum((A@theta-y)**2)
    return loss
def fit_gradient_descent(X,y,lr,n_iters):
    A=design_matrix(X)
    theta=np.zeros(A.shape[1])
    n=A.shape[0]
    loss_list=[loss(theta,X,y)]
    for i in range(n_iters):
        theta=theta-(2/n)*lr*A.T@(A@theta-y)
        loss_list.append(loss(theta,X,y))
    return theta,loss_list
X = df[['area', 'rooms', 'age', 'dist_km', 'floors']].values
Y = df['rent'].values

X_train, Y_train, X_test, Y_test = train_test_split_scratch(
    X, Y, test_frac=0.2, seed=0
)
theta,loss_list=fit_gradient_descent(X_train,Y_train,lr=0.005,n_iters=20000)
plt.plot(loss_list)
plt.show()