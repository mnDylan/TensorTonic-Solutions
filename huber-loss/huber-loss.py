import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = float)
    delta = np.asarray(delta)
    
    error = y_true - y_pred
    L1 = delta * (np.abs(error) - 0.5*delta) 
    L2 = 0.5 * error**2 

    loss = np.where(np.abs(error) >= delta, L1, L2)

    return np.mean(loss) 