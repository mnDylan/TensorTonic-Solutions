import numpy as np
from math import erf, sqrt

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x)
    erf_vec = np.vectorize(erf)
    result = 0.5 * x * (1 + erf_vec(x / sqrt(2)))
    return result 