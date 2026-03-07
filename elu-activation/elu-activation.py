import numpy as np 
import math 
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []
    for xi in x: 
        if xi > 0: 
            result_activate = xi
            result.append(result_activate)
        elif alpha >= 0: 
            result_activate = alpha * (math.exp(xi) -1)
            result.append(result_activate)
        else:
            print('Please input valit alpha >= 0 ')
    return result 
            
