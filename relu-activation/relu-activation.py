import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.asarray(x)
    result = np.maximum(0, x)
    return result 