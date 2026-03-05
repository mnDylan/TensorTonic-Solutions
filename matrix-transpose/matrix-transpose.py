import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    n_rows = len(A)
    n_cols = len(A[0])
    
    At = []
    for c in range(n_cols):
        row = []
        for r in range(n_rows):
            row.append(A[r][c])
        At.append(row)
        result = np.array(At)
    return result

