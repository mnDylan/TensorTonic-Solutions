import numpy as np
import math 

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    pt = np.where(targets == 1, predictions, 1 - predictions)
    loss = - alpha * (1 - pt)**gamma * np.log(pt)

    return loss.mean()