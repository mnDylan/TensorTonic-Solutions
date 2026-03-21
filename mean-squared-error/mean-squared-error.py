import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    
    if len(y_pred) != len(y_true):
        raise ValueError("Len of y_pred and y_true must match")

    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = float)

    loss = np.mean((y_pred - y_true)**2)

    return loss 