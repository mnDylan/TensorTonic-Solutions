import numpy as np 
import math 
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """

    len_x = len(x)
    result = []
    for i in range(len_x): 
        if x[i] > 0: 
            result_activate = x[i] 
            result.append(result_activate)
        elif alpha >= 0: 
            result_activate = alpha * (math.exp(x[i]) -1)
            result.append(result_activate)
        else:
            print('Please input valit alpha >= 0 ')
    return result 
            
