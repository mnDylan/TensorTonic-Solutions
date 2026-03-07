import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    result = []
    for xi in x: 
        leaky_relu = np.where(xi >= 0, xi, alpha * xi)
        result.append(leaky_relu)
    result = np.asarray(result)
    return result 