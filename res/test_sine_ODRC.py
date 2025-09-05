# ODRC test (Python version)
import numpy as np
from param_ODRC import *

def test_sine_ODRC(W, WIn, WFb, WOsc, Osc, WOut, input_pattern, target_Out, start_train_n, end_train_n):
    print('testing:')
    Out_test_history = np.zeros((numOut, n_steps_test, n_test_loops))
    OutUnits_test_history = np.zeros((n_steps_test, n_test_loops, numUnits))
    R2_test = np.zeros((numOut, n_test_loops))
    Error_test = np.zeros((numOut, n_test_loops))
    for j in range(n_test_loops):
        print(f'  loop: {j+1}/{n_test_loops}  ', end='')
        Xv = 1 * (2 * np.random.rand(numUnits, 1) - 1)
        X = np.tanh(Xv)
        Xv_current = Xv.copy()
        Out = WOut @ X[:numOutUnits].flatten()
        for i in range(n_steps_test):
            Input = input_pattern[:, i].reshape(-1, 1)  # (numIn, 1)
            osc_input = Osc[:, i].reshape(-1, 1)        # (numOsc, 1)
            Out_vec = Out.reshape(-1, 1)                # (numOut, 1)
            Xv_current = W @ X + WIn @ Input + WFb @ Out_vec + WOsc @ osc_input
            Xv = Xv + ((-Xv + Xv_current) / tau) * dt
            X = np.tanh(Xv)
            OutUnits_test_history[i, j, :] = X.ravel()
            Out = WOut @ X[:numOutUnits].flatten()
            Out_test_history[:, i, j] = Out
        for n in range(numOut):
            R = np.corrcoef(Out_test_history[n, start_train_n:end_train_n, j], target_Out[n, start_train_n:end_train_n])
            R2_test[n, j] = R[0, 1] ** 2
            print(f'R^2({n+1})={R2_test[n, j]:.3f}, ', end='')
        print('')
        print('              ', end='')
        for n in range(numOut):
            Error_test[n, j] = np.sqrt(np.mean((Out_test_history[n, start_train_n:end_train_n, j] - target_Out[n, start_train_n:end_train_n]) ** 2))
            print(f'MSE({n+1})={Error_test[n, j]:.3f}, ', end='')
        print('')
    return R2_test, Error_test, Out_test_history, OutUnits_test_history
