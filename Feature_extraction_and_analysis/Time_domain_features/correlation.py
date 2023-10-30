import numpy as np
import scipy
import matplotlib.pyplot as plt

def cal_corrcoef(signal1, signal2):
    """
    Description:
        To get the correlate coefficient

    Input:
        Two signal with same length

    Return:
        The correlate coefficient
    """
    return np.corrcoef(signal1, signal2)[0,1]

def cal_serial_corr(signal, lag):
    """
    Description:
        To get the serial correlate coefficient

    Input:
        One signal and the lag which means how much it delays

    Return:
        The serial correlate coefficient
    """
    signal1 = signal[lag:]
    signal2 = signal[:len(signal)-lag]
    return np.corrcoef(signal1, signal2)[0,1]

def cal_autocorr(signal, plot = False):
    """
    Description:
        To get the auto correlate coefficient

    Input:
        One signal

    Return:
        The serial correlate coefficient with different lag which is from 0 to len(wave)//2
    """
    lags = range(len(signal)//2)
    corrs = [cal_serial_corr(signal, lag) for lag in lags]
    if plot:
        plt.plot(lags, corrs)
        plt.show()
    return lags, corrs


if __name__ == "__main__":
    signal = np.load("../../data/sim_100_0.1_90_140_train.npy")[10,:1000]
    signal1 = signal[100:200]
    signal2 = signal[150:250]

    print(cal_corrcoef(signal1,signal2))
    print(cal_serial_corr(signal, 51))
    print(cal_autocorr(signal, True))