import tftb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import os

def get_wvd(signal, fs, T):
    """
    Description:
        Analyze the time-frequency characteristics of a signal using the Wigner-Ville Transform (WVT) and visualize the results.

    Params:
        signal (numpy.ndarray): The input signal.
        fs (float): The sampling frequency of the signal.
        T (float): Time of the signal

    Returns:
        tfr_wvd (numpy.ndarray): The time-frequency representation (WVD) of the signal.
        t_wvd (numpy.ndarray): Time values corresponding to the WVD.
        f_wvd (numpy.ndarray): Normalized frequency values corresponding to the WVD.
    """

    t = np.arange(0, T, 1.0 / fs)

    # Doing the WVT
    wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=t)
    tfr_wvd, t_wvd, f_wvd = wvd.run()


    return tfr_wvd, t_wvd, f_wvd


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100
    T = 10

    tfr_wvd, t_wvd, f_wvd = get_wvd(signal,fs, T)

    plt.pcolormesh(t_wvd, f_wvd, tfr_wvd)
    plt.colorbar()
    plt.show()



