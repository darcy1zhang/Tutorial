import numpy as np
import matplotlib.pyplot as plt
from tfmean import *

def wspec_pl(wspec, nvoice):
    """
    Description:
        Displays normalized log of wavelet spectrum.

    Parameters:
        wspec (numpy.ndarray): The wavelet spectrum.
        nvoice (int): The number of voices.

    Returns:
        None
    """
    epsilon = 1e-8
    log_wspec = np.log(wspec + epsilon) / np.log(2 ** (2 / nvoice))
    # print(log_wspec)
    plt.plot(np.arange(len(log_wspec)), log_wspec)
    plt.title("log(wavelet spectrum)")
    plt.xlabel("log(scale)")
    plt.ylabel("V(a)")
    plt.show()

if __name__ == "__main__":
    from cwt import *
    import os

    noctave = 5
    nvoice = 3
    w0 = 2 * np.pi

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    cwt_result = cwt(signal, noctave, nvoice, w0)
    mean_fre = tfmean(cwt_result, plot=False)
    # print(mean_fre)
    wspec_pl(mean_fre, 2)
