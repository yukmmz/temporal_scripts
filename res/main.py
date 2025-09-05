
# ODRC main (Python version)
# Sinusoidal ODRCの一連の流れがそのまま実行できます。
# パラメータを変更したい場合は param_ODRC.py の該当変数を書き換えてください。
# 例: numUnits, numOsc, interval, feedback_weight_amp など

import numpy as np
from param_ODRC import *
from io_ODRC import input_pattern, target_Out, start_train_n, end_train_n
from construct_sine_ODRC import construct_sine_ODRC
from train_sine_ODRC import train_sine_ODRC
from test_sine_ODRC import test_sine_ODRC
from plot_ODRC import plot_ODRC

# --- ネットワーク構築 ---
WIn, WFb, WOsc, W_sparse, Osc = construct_sine_ODRC()

# --- 学習 ---
WOut, R2_learn, Error_learn, Out_learn_history, OutUnits_learn_history = train_sine_ODRC(
    W_sparse, WIn, WFb, WOsc, Osc, input_pattern, target_Out, start_train_n, end_train_n)

# --- テスト ---
R2_test, Error_test, Out_test_history, OutUnits_test_history = test_sine_ODRC(
    W_sparse, WIn, WFb, WOsc, Osc, WOut, input_pattern, target_Out, start_train_n, end_train_n)

# --- プロット ---
if PLOT:
    plot_ODRC(time_axis, start_train, Out_learn_history, n_learn_loops, lwidth2, fsize,
              OutUnits_learn_history, Osc, numOsc, n_steps, Out_test_history, n_test_loops, OutUnits_test_history)

# --- パラメータ変更方法 ---
# 例: リザバーのユニット数を変更したい場合
# param_ODRC.py の numUnits = 400 などを編集してください。
# 他のパラメータも同様に param_ODRC.py で設定できます。
