from fastsst import SingularSpectrumTransformation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data_and_score(raw_data, score):
    f,ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(raw_data); ax[0].set_title("raw data")
    ax[1].plot(score,"r"); ax[1].set_title("score")


sst = SingularSpectrumTransformation(win_length=30)

"""
note:
 - data must be 1d np.ndarray
 - the first run takes a few seconds for jit compling
"""
x0 = np.sin(2*np.pi*1*np.linspace(0,10,1000))
x1 = np.sin(2*np.pi*2*np.linspace(0,10,1000))
x2 = np.sin(2*np.pi*8*np.linspace(0,10,1000))
x = np.hstack([x0, x1, x2])
x +=  + np.random.rand(x.size)
score = SingularSpectrumTransformation(win_length=60, order=60, lag=10).score_offline(x)
plot_data_and_score(x, score)
plt.show()

x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
x = np.hstack([x0, x1, x2])
x +=  np.random.rand(x.size)
score = SingularSpectrumTransformation(win_length=50).score_offline(x)
plot_data_and_score(x, score)
plt.show()

current_file_path = os.path.abspath(__file__)
signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                           signal_name)
signal = np.load(signal_path)[2, :1000]
fs = 100

x = signal
score = SingularSpectrumTransformation(win_length=60, order=60, lag=10).score_offline(x)
plot_data_and_score(x, score)
plt.show()