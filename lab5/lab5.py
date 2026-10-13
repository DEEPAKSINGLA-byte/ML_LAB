import numpy as np
import pandas as pd
df = pd.read_csv("lab5/ic272_lab5_rent.csv")
features = [c for c in df.columns if c != "rent"]
X = df[features].to_numpy(float)
y = df["rent"].to_numpy(float)
idx = np.random.default_rng(0).permutation(len(y))
tr, te = idx[:40], idx[40:]
X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]
print("train:", X_train.shape, " test:", X_test.shape) # (40, 13) (260, 13)
def design_matrix(X): # from Lab 4
    return np.c_[np.ones(len(X)), X]
def fit_normal_equation(X, y): # from Lab 4
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ y)
#TASK 1
def standardise(X_train,X_test):
    mean=np.mean(X_train,axis=0)
    std=np.std(X_train,axis=0)
    X_train_std=(X_train-mean)/std
    X_test_std=(X_test-mean)/std
    return mean,std,X_train_std,X_test_std
mean,std,X_train_std,X_test_std=standardise(X_train,X_test)
print("X_train_std",np.std(X_train_std,axis=0))
print("X_test_std",np.std(X_test_std,axis=0))
#TASK2
def fit_ridge(X,y,lam):
    A=design_matrix(X)
    I=np.eye(A.shape[1])
    I[0,0]=0
    return np.linalg.solve(A.T @ A + lam*I,A.T@y)
lam=[0,1,10,100]
for l in lam:
    coeff=fit_ridge(X_train_std,y_train,l)
    print("coeff",coeff)
    print(np.sum(np.abs(coeff)<1e-8))
#TASK 3 
def k_fold_csv(X,y,lam,k=5,seed=0):
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(y))
    folds=np.array_split(indices,k)
    e=[]
    for i in range(k):
        val_idx=folds[i]
        X_val=X[val_idx]
        y_val=y[val_idx]
        tr=np.concatenate([folds[j] for j in range(k) if j!=i])
        Xtr=X[tr]
        ytr=y[tr]
        coeff=fit_ridge(Xtr,ytr,lam)
        A=design_matrix(X_val)
        y_pred=A @ coeff
        rmse=(np.mean((y_pred-y_val)**2))**0.5
        e.append(rmse)
    return np.mean(e)
lam=[0,0.1,0.3,1,3,10,30,100]
error=[]
eo=[]
for l in lam:
    error.append((l,k_fold_csv(X_train_std,y_train,l)))
    eo.append(k_fold_csv(X_train_std,y_train,l))
bl=lam[np.argmin(eo)]
print(error)
coeff=fit_ridge(X_train_std,y_train,0)
y_pred_0=design_matrix(X_test_std)@coeff

test_rmse_0 = np.sqrt(
    np.mean((y_test - y_pred_0)**2)
)
coeff_b=fit_ridge(X_train_std,y_train,bl)
y_pred=design_matrix(X_test_std) @ coeff_b
test_rmse_best = np.sqrt(
    np.mean((y_test - y_pred)**2)
)
print("Test RMSE at lambda=0:", test_rmse_0)
print("Test RMSE at best lambda:", test_rmse_best)
