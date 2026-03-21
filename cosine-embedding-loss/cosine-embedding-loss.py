import numpy as np 


def cosine_similarity(a, b):
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cosine = dot / (norm_a * norm_b)
    return cosine 

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cosine = cosine_similarity(x1, x2)

    L1 = 1 - cosine 
    L2 = max(0, cosine - margin)

    loss = L1 if label == 1 else L2

    return loss
    

    