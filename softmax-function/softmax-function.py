import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x) 
    output = []

    if x.ndim == 1:
        z = x - np.max(x)
        exp_z = np.exp(z)
        result = exp_z / np.sum(exp_z)
        output = result

    elif x.ndim == 2:
        z = x - np.max(x, axis=1, keepdims=True)
        exp_z = np.exp(z)
        result = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        output = result

    else:
        raise ValueError("Input must be a 1D or 2D array.")

    return output
    