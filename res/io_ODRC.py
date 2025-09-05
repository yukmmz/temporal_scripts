# ODRC input/output pattern generation (Python version)
import numpy as np
from scipy.stats import norm
from param_ODRC import *

start_pulse_n = int(start_pulse / dt)
reset_duration_n = int(reset_duration / dt)
start_train_n = int(start_train / dt)
end_train_n = int(end_train / dt)
discard_n = int(discard / dt)
start_n = int(start_train / dt)
chaos_n = n_steps_test + discard_n - start_n

input_pattern = np.zeros((numIn, n_steps_test))
input_pattern[0, start_pulse_n:(start_pulse_n + reset_duration_n)] = input_pulse_value

# ターゲット出力（例：Gaussian bell curve）
peak_time = start_train + interval
bell = norm.pdf(time_axis, peak_time, peak_width)
target_Out = ready_level_timing + ((peak_level_timing - ready_level_timing) / np.max(bell)) * bell

target_Out = target_Out.reshape(1, -1)  # (numOut, n_steps_test)

# 履歴用配列
Out_learn_history = np.zeros((numOut, n_steps, n_learn_loops))
OutUnits_learn_history = np.zeros((n_steps, n_learn_loops, numUnits))
Out_test_history = np.zeros((numOut, n_steps_test, n_test_loops))
OutUnits_test_history = np.zeros((n_steps_test, n_test_loops, numUnits))
