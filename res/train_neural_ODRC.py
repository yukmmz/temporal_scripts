import numpy as np
import torch

def train_neural_ODRC(
    W_osc, WIn_osc, WIn, WFb, WOsc, W, numUnits_osc, numOsc, numUnits, numOut, numOutUnits,
    input_pattern, target_Out, n_learn_loops, n_steps, start_train_n, end_train_n, learn_every,
    tau_osc, tau, dt, noise_amp, delta
):
    print('training readout:')
    WOut = np.zeros((numOut, numOutUnits))
    P = np.eye(numOutUnits) / delta
    OutUnits_learn_history = np.zeros((n_steps, n_learn_loops, numUnits))
    Out_learn_history = np.zeros((numOut, n_steps, n_learn_loops))
    Osc = np.zeros((numOsc, n_steps))
    R2_learn = np.zeros((numOut, n_learn_loops))
    Error_learn = np.zeros((numOut, n_learn_loops))

    for j in range(n_learn_loops):
        print(f'  loop: {j+1}/{n_learn_loops}  ', end='')
        Xv_osc = 1 * (2 * np.random.rand(numUnits_osc, numOsc) - 1)
        X_osc = np.tanh(Xv_osc)
        Xv_current_osc = Xv_osc.copy()
        Out_osc = X_osc[0, :].copy()

        Xv = 1 * (2 * np.random.rand(numUnits, 1) - 1)
        X = np.tanh(Xv)
        Xv_current = Xv.copy()
        Out = WOut @ X[:numOutUnits].flatten()

        train_window = 0
        for i in range(n_steps):
            Input = input_pattern[:, i]
            for k in range(numOsc):
                Xv_current_osc[:, k] = W_osc[:, :, k] @ X_osc[:, k] + WIn_osc[:, :, k] @ Input
            Xv_osc = Xv_osc + ((-Xv_osc + Xv_current_osc) / tau_osc) * dt
            X_osc = np.tanh(Xv_osc)
            Out_osc = X_osc[0, :].copy()

            Xv_current = W @ X + WIn @ Input + WFb @ Out + WOsc @ Out_osc
            Xv = Xv + ((-Xv + Xv_current) / tau) * dt
            X = np.tanh(Xv)

            OutUnits_learn_history[i, j, :] = X.flatten()
            Out = WOut @ X[:numOutUnits].flatten()

            if i == start_train_n:
                train_window = 1
            if i == end_train_n:
                train_window = 0

            if (train_window == 1) and (i % learn_every == 0):
                error = target_Out[:, i] - Out
                P_old = P.copy()
                P_old_X = P_old @ X[:numOutUnits].flatten()
                den = 1 + X[:numOutUnits].flatten().T @ P_old_X
                P = P_old - np.outer(P_old_X, P_old_X) / den
                WOut = WOut + np.outer(error, P_old_X / den)

            Osc[:, i] = Out_osc
            Out_learn_history[:, i, j] = Out

        for n in range(numOut):
            R = np.corrcoef(Out_learn_history[n, start_train_n:end_train_n, j], target_Out[n, start_train_n:end_train_n])
            R2_learn[n, j] = R[0, 1] ** 2
            print(f'R^2({n+1})={R2_learn[n, j]:.3f}, ', end='')
        print('')
        print('              ', end='')
        for n in range(numOut):
            Error_learn[n, j] = np.sqrt(np.mean((Out_learn_history[n, start_train_n:end_train_n, j] - target_Out[n, start_train_n:end_train_n]) ** 2))
            print(f'MSE({n+1})={Error_learn[n, j]:.3f}, ', end='')
        print('')
    return WOut, R2_learn, Error_learn, Out_learn_history, OutUnits_learn_history, Osc
