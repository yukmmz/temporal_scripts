import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

# 必要なパラメータは外部から渡す想定
# input_weight_amp, numIn, numUnits_osc, numOsc, numUnits, feedback_weight_amp, numOut,
# osc_weight_amp, scale, p_connect, scale_osc, check_duration, reset_duration, tau_osc, dt, resample_threshold, n_steps_test

def construct_neural_ODRC(
    input_weight_amp, numIn, numUnits_osc, numOsc, numUnits, feedback_weight_amp, numOut,
    osc_weight_amp, scale, p_connect, scale_osc, check_duration, reset_duration, tau_osc, dt, resample_threshold, n_steps_test
):
    print('constructing RNN modules:')

    # input connections WIn
    WIn_osc = input_weight_amp / np.sqrt(numIn) * np.random.randn(numUnits_osc, numIn, numOsc)
    WIn = input_weight_amp / np.sqrt(numIn) * np.random.randn(numUnits, numIn)

    # feedback connections WFb
    WFb = feedback_weight_amp / np.sqrt(numOut) * np.random.randn(numUnits, numOut)

    WOsc = osc_weight_amp / np.sqrt(numOsc) * np.random.randn(numUnits, numOsc)

    # recurrent connections W
    W_mask = np.random.rand(numUnits, numUnits)
    W_mask[W_mask <= p_connect] = 1
    W_mask[W_mask < 1] = 0
    W = np.random.randn(numUnits, numUnits) * scale
    W = W * W_mask
    np.fill_diagonal(W, 0)  # set self-connections to zero
    W_sparse = sparse.csr_matrix(W)

    W_osc = np.zeros((numUnits_osc, numUnits_osc, numOsc))
    Resample = np.zeros(numOsc)

    # resampling W
    for k in range(numOsc):
        if k % max(1, round(numOsc / 10)) == 0:
            print(f'  oscillator {k+1}/{numOsc}')
        while True:
            W_mask = np.random.rand(numUnits_osc, numUnits_osc)
            W_mask[W_mask <= p_connect] = 1
            W_mask[W_mask < 1] = 0
            W_osc_k = np.random.randn(numUnits_osc, numUnits_osc) * scale_osc
            W_osc_k = W_osc_k * W_mask
            np.fill_diagonal(W_osc_k, 0)
            W_osc[:, :, k] = W_osc_k
            W_osc_sparse = sparse.csr_matrix(W_osc_k)

            # initial conditions
            Xv = 1 * (2 * np.random.rand(numUnits_osc) - 1)
            X = np.tanh(Xv)
            OutOsc_history = np.zeros(check_duration)

            for i in range(check_duration):
                Input = 1 if i < reset_duration else 0
                Xv_current = W_osc_sparse.dot(X) + WIn_osc[:, :, k].dot(np.full(numIn, Input))
                Xv = Xv + ((-Xv + Xv_current) / tau_osc) * dt
                X = np.tanh(Xv)
                OutOsc_history[i] = X[0]

            Out_min = np.min(OutOsc_history[reset_duration+1000:check_duration])
            Out_max = np.max(OutOsc_history[reset_duration+1000:check_duration])
            if (Out_max - Out_min) > resample_threshold:
                break
            else:
                Resample[k] = 1
    print(f"  {int(np.sum(Resample))} oscillators were resampled.")

    # oscillators
    Osc = np.zeros((numOsc, n_steps_test))

    return {
        'WIn_osc': WIn_osc,
        'WIn': WIn,
        'WFb': WFb,
        'WOsc': WOsc,
        'W_sparse': W_sparse,
        'W_osc': W_osc,
        'Osc': Osc,
        'Resample': Resample
    }
