# ODRC plot (Python version)
import matplotlib.pyplot as plt

def plot_ODRC(time_axis, start_train, Out_learn_history, n_learn_loops, lwidth2, fsize, OutUnits_learn_history, Osc, numOsc, n_steps, Out_test_history, n_test_loops, OutUnits_test_history):
    # Learning
    plt.figure(1)
    plt.subplot(5,1,1)
    for j in range(n_learn_loops):
        plt.plot(time_axis - start_train, Out_learn_history[0, :len(time_axis), j], linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylabel('FORCE Outputs', fontsize=fsize)
    plt.xlabel('time (ms)')

    plt.subplot2grid((5,1), (1,0), rowspan=2)
    for j in range(n_learn_loops):
        for k in range(min(10, OutUnits_learn_history.shape[2])):
            plt.plot(time_axis - start_train, OutUnits_learn_history[:len(time_axis), j, k] + 2 * k, linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylabel('FORCE Dyamics', fontsize=fsize)
    plt.xlabel('time (ms)')

    plt.subplot2grid((5,1), (3,0), rowspan=2)
    for j in range(n_learn_loops):
        for k in range(min(10, numOsc)):
            plt.plot(time_axis - start_train, Osc[k, :len(time_axis)] + 2 * k, linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylabel('Module outputs', fontsize=fsize)
    plt.xlabel('time (ms)')

    # Testing
    plt.figure(2)
    plt.subplot(5,1,1)
    for j in range(n_test_loops):
        plt.plot(time_axis - start_train, Out_test_history[0, :n_steps, j], linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylim([-1, 1])
    plt.ylabel('Module outputs', fontsize=fsize)

    plt.subplot2grid((5,1), (1,0), rowspan=2)
    for j in range(n_test_loops):
        for k in range(min(10, OutUnits_test_history.shape[2])):
            plt.plot(time_axis - start_train, OutUnits_test_history[:n_steps, j, k] + 2 * k, linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylabel('FORCE dynamics', fontsize=fsize)

    plt.subplot2grid((5,1), (3,0), rowspan=2)
    for j in range(n_test_loops):
        for k in range(min(10, numOsc)):
            plt.plot(time_axis - start_train, Osc[k, :n_steps] + 2 * k, linewidth=lwidth2)
    plt.xlim([time_axis[0] - start_train, time_axis[-1] - start_train])
    plt.ylabel('Module outputs', fontsize=fsize)
    plt.show()
