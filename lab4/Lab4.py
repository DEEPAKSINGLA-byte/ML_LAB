import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('lab4/ic272_lab4_rent.csv')

#TASK 1
for column in df.columns:
    print(column, "min =", df[column].min(), "max =", df[column].max())
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
print("w =", round(slope, 3))
print("b =", round(intercept, 3))
plt.scatter(X_test, Y_test)
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
X_train, Y_train, X_test, Y_test = train_test_split_scratch(
    X, Y, test_frac=0.2, seed=0
)
theta_ne=fit_normal_equation(X_train,Y_train)
print(theta_ne.round(3))


#TASK 3,X,y
def loss(theta,A,y):
    return 1/A.shape[0]*np.sum((A@theta-y)**2)
def fit_gradient_descent(X,y,lr,n_iters):
    A=design_matrix(X)
    theta=np.zeros(A.shape[1])
    loss_list=[]
    for i in range(n_iters):
        theta=theta-lr*(2/A.shape[0]*A.T@(A@theta-y))
        loss_list.append(loss(theta,A,y))
    return theta,loss_list

X = df[['area', 'rooms', 'age', 'dist_km', 'floors']].values
Y = df['rent'].values

X_train, Y_train, X_test, Y_test = train_test_split_scratch(
    X, Y, test_frac=0.2, seed=0
)
theta,loss_list=fit_gradient_descent(X_train,Y_train,lr=0.005,n_iters=20000)
print(np.max(np.abs(theta-theta_ne)))
plt.plot(loss_list)
plt.show()

lr=[0.001,0.005,0.01]
losses_list=[]
losses_list.append(fit_gradient_descent(X_train,Y_train,lr=0.001,n_iters=20000)[1])
losses_list.append(fit_gradient_descent(X_train,Y_train,lr=0.005,n_iters=20000)[1])
losses_list.append(fit_gradient_descent(X_train,Y_train,lr=0.01,n_iters=20000)[1])
plt.plot(losses_list[0],label='lr=0.001')
plt.plot(losses_list[1],label='lr=0.005')
plt.plot(losses_list[2],label='lr=0.01')
plt.yscale('log')
plt.legend()
plt.show()

#TASK 4

def rmse(y_true,y_pred):
    return np.sqrt(1/y_pred.shape[0]*np.sum((y_true-y_pred)**2))
def r2_score_scratch(y_true,y_pred):
    den=np.sum((y_true-np.mean(y_true))**2)
    num=np.sum((y_true-y_pred)**2)
    return 1-num/den
y_pred_test = design_matrix(X_test) @ theta
y_pred_train = design_matrix(X_train) @ theta

print('Test RMSE:', round(rmse(Y_test, y_pred_test), 4))
print('Test R2:', round(r2_score_scratch(Y_test, y_pred_test), 4))
print('Train R2:', round(r2_score_scratch(Y_train, y_pred_train), 4))
plt.scatter(y_pred_test, Y_test-y_pred_test)
plt.axhline(y=0)
plt.show()