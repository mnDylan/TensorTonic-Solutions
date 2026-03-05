import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    
    # --- Force floating dtype to avoid integer truncation ---
    param = np.asarray(param, dtype=np.float64)
    grad  = np.asarray(grad,  dtype=np.float64)
    m     = np.asarray(m,     dtype=np.float64)
    v     = np.asarray(v,     dtype=np.float64)


    # Step 1 Update first moment (momentum)
    m_new = beta1 * m + (1.0 - beta1) * grad
 
    # Step 2 Update second moment (adaptive rate)
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Step 3 Bias-correct both movements 
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)

    # Step 4 Update parameter 
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m_new, v_new
    
    
