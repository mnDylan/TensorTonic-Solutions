import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _binaryCE(y, p):
    BCEloss = - (y * np.log(p) + (1 - y) * log(1 - p)).mean()
    return BCEloss

def _gradient(X, y, w, b):
    """
    Gradient of BCE loss w.r.t. w and b.
    """
    m = X.shape[0]
    z = X @ w + b
    p = _sigmoid(z)

    diff = p - y  

    dw = (X.T @ diff) / m
    db = np.mean(diff)
    return dw, db


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for step in range(steps):
        dw, db = _gradient(X, y, w, b)
        w -= lr * dw
        b -= lr * db

    return w, b