import numpy as np
import torch

def test_neural_ODRC(
    W_osc, WIn_osc, WIn, WFb, WOsc, W, WOut, numUnits_osc, numOsc, numUnits, numOut, numOutUnits,
    input_pattern, target_Out, n_test_loops, n_steps_test, start_train_n, end_train_n,
    tau_osc, tau, dt, noise_amp
):
    print('testing:')
    OutUnits_test_history = np.zeros((n_steps_test, n_test_loops, numUnits))
    Out_test_history = np.zeros((numOut, n_steps_test, n_test_loops))
    Osc = np.zeros((numOsc, n_steps_test))
    R2_test = np.zeros((numOut, n_test_loops))
    Error_test = np.zeros((numOut, n_test_loops))

    for j in range(n_test_loops):
        print(f'  loop: {j+1}/{n_test_loops}  ', end='')
        Xv_osc = 1 * (2 * np.random.rand(numUnits_osc, numOsc) - 1)
        X_osc = np.tanh(Xv_osc)
        Xv_current_osc = Xv_osc.copy()
        Out_osc = X_osc[0, :].copy()

        Xv = 1 * (2 * np.random.rand(numUnits, 1) - 1)
        X = np.tanh(Xv)
        Xv_current = Xv.copy()
        Out = WOut @ X[:numOutUnits].flatten()

        for i in range(n_steps_test):
            Input = input_pattern[:, i]
            if i <= n_steps_test:
                for k in range(numOsc):
                    Xv_current_osc[:, k] = W_osc[:, :, k] @ X_osc[:, k] + WIn_osc[:, :, k] @ Input
                Xv_osc = Xv_osc + ((-Xv_osc + Xv_current_osc) / tau_osc) * dt
                X_osc = np.tanh(Xv_osc)
                Out_osc = X_osc[0, :].copy()
            else:
                Out_osc = np.zeros(numOsc)

            Xv_current = W @ X + WIn @ Input + WFb @ Out + WOsc @ Out_osc
            Xv = Xv + ((-Xv + Xv_current) / tau) * dt
            X = np.tanh(Xv)

            OutUnits_test_history[i, j, :] = X.flatten()
            Out = WOut @ X[:numOutUnits].flatten()
            Osc[:, i] = Out_osc
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
    return R2_test, Error_test, Out_test_history, OutUnits_test_history, Osc
