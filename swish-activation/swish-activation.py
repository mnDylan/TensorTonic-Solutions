import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x)
    output = x * 1 / (1 + np.exp(-x))
    return output 