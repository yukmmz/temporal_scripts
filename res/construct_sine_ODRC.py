# ODRC construct sine reservoir (Python version)
import numpy as np
from scipy import sparse
from param_ODRC import *

def construct_sine_ODRC():
    print('constructing a reservoir:')
    WIn = input_weight_amp / np.sqrt(numIn) * np.random.randn(numUnits, numIn)
    WFb = feedback_weight_amp / np.sqrt(numOut) * np.random.randn(numUnits, numOut)
    WOsc = osc_weight_amp / np.sqrt(numOsc) * np.random.randn(numUnits, numOsc)

    W_mask = np.random.rand(numUnits, numUnits)
    W_mask[W_mask <= p_connect] = 1
    W_mask[W_mask < 1] = 0
    W = np.random.randn(numUnits, numUnits) * scale
    W = W * W_mask
    np.fill_diagonal(W, 0)
    W_sparse = sparse.csr_matrix(W)

    f = fmin + (fmax - fmin) * np.random.rand(numOsc)
    phi = 2 * np.pi * np.random.rand(numOsc)
    pos = np.arange(n_steps_test)
    Osc = np.array([np.sin(2 * np.pi * f[k] * pos / 1000 + phi[k]) for k in range(numOsc)])
    # Osc[:, n_steps:n_steps_test] = 0  # interval部分は0

    return WIn, WFb, WOsc, W_sparse, Osc
