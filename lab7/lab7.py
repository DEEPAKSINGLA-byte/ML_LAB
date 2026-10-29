import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/home/deepak/ML_LAB/lab7/lab7_loans.csv")
features = ["income_k", "utilisation"]
X = df[features].to_numpy(float)
y = df["defaulted"].to_numpy(int)
idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
print("train:", X_train.shape, " test:", X_test.shape)

def fit_normal_equation(X,y):
    A=design_matrix(X)
    theta=np.linalg.solve(A.T @ A,A.T @ y)
    return theta
def confusion_matrix(y_true,y_pred):
    TP=np.sum((y_true==1)&(y_pred==1))
    TN=np.sum((y_true==0) & (y_pred==0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    accuracy=(TP+TN)/len(y_true)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=(2*precision*recall)/(precision+recall)
    return accuracy,precision,recall,f1
def standardise(X_train,X_test):
    mu=np.mean(X_train,axis=0)
    sigma=np.std(X_train,axis=0)
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    return X_train_std,X_test_std,mu,sigma
def design_matrix(X):
    return np.column_stack((np.ones(X.shape[0]),X))
def sigmoid(z):
    return 1/(1+np.exp(-z))
def log_likelihood(X,y,theta):
    A=design_matrix(X)
    h=sigmoid(A @ theta)
    eps=1e-12
    ll=np.sum(y*np.log(h+eps)+(1-y)*np.log(1-h+eps))
    return ll
Z_train,Z_test,mu,sigma=standardise(X_train,X_test)
theta_zero=np.zeros(Z_train.shape[1]+1)
ll_total=log_likelihood(Z_train,y_train,theta_zero)
ll_per_example = ll_total / len(y_train)

print("Total log-likelihood:", ll_total)
print("Log-likelihood per example:", ll_per_example)

def fit_logistic(X,y,lr=0.1,n_iters=2000):
    A=design_matrix(X)
    m=len(y)
    theta=np.zeros(A.shape[1])
    history=[]
    for i in range(n_iters):
        h=sigmoid(A @ theta)
        gradient=(A.T @ (y-h))/m
        theta=theta+lr*gradient
        history.append(log_likelihood(X,y,theta))
    return theta,history
theta, history = fit_logistic(
    Z_train,
    y_train,
    lr=0.1,
    n_iters=2000
)

print("Learned parameters:", theta)
print("Initial log-likelihood:", log_likelihood(
    Z_train, y_train, np.zeros(len(theta))
))
print("Final log-likelihood:", history[-1])

plt.plot(range(1,len(history)+1),history)
plt.show()

A_test=design_matrix(Z_test)
y_prob=sigmoid(A_test@theta)

y_pred_05=(y_prob >=0.5).astype(int)

metrics_05=confusion_matrix(y_test,y_pred_05)
print("Threshold = 0.5")
print("Accuracy: ", metrics_05[0])
print("Precision:", metrics_05[1])
print("Recall:   ", metrics_05[2])
print("F1 score: ", metrics_05[3])

y_pred_03 = (y_prob >= 0.3).astype(int)

metrics_03 = confusion_matrix(y_test, y_pred_03)

print("\nThreshold = 0.3")
print("Accuracy: ", metrics_03[0])
print("Precision:", metrics_03[1])
print("Recall:   ", metrics_03[2])
print("F1 score: ", metrics_03[3])