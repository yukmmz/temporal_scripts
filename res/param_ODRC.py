# ODRC parameters (Python version)
import numpy as np

# Main parameters
interval = 20000
interval_test = 0

numOsc = 10
fmin = 0.1
fmax = 1
numUnits = 400
numOutUnits = numUnits

g = 1.5
feedback_weight_amp = 3.0
osc_weight_amp = 0.5
numOut = 1

PLOT = True

# Recurrent neural networks
p_connect = 0.1
scale = g / np.sqrt(p_connect * numUnits)
numIn = 1
tau = 10.0

# Neural oscillators
numUnits_osc = 100
g_osc = 1.2
scale_osc = g_osc / np.sqrt(p_connect * numUnits_osc)
tau_osc = 20
check_duration = 5000
resample_threshold = 0.5

# Input weight parameter
input_weight_amp = 5.0

# Training & loops
learn_every = 2
n_learn_loops = 10 # originally, 10
n_test_loops = 10

delta = 10.0

# Input parameters
input_pulse_value = 2.0
start_pulse = 200
reset_duration = 50

# Training duration
start_train = start_pulse + reset_duration
end_train = start_train + interval + 200

# Output parameters
ready_level = -0.8
peak_level = 0.8
ready_level_timing = 0.2
peak_level_timing = 1
discard = 3000
peak_width = 30

dt = 1
tmax = end_train
n_steps = int(tmax / dt)
n_steps_test = n_steps + interval_test
time_axis = np.arange(0, tmax, dt)
time_axis_test = np.arange(0, n_steps_test, dt)

# Drawing
lwidth = 2
lwidth2 = 1.5
fsize = 10
