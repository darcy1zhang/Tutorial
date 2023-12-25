import numpy as np
import matplotlib.pyplot as plt


def hurst_est(wspec, range, nvoice, plot=True):
    """Estimate Hurst exponent from wavelet spectrum

    Args:
        wspec (np.ndarray): Wavelet spectrum, shape (n_scale,)
        range (tuple): Range of scales to use
        nvoice (int): Number of scales per octave
        plot (bool): If True, plot regression line

    Returns:
        hurst (float): Estimated Hurst exponent
    """

    # Select scales in given range
    start, end = range
    scales = 2 ** (np.arange(start, end) / nvoice)
    wspec = wspec[start:end]

    # Log-log regression
    pos_idx = wspec > 0
    log_scales = np.log2(scales[pos_idx])
    log_wspec = np.log2(wspec[pos_idx])
    regr = np.polyfit(log_scales, log_wspec, 1)
    hurst = regr[0]

    if plot:
        plt.loglog(scales, wspec, 'bo')
        plt.loglog(scales, 2 ** regr[1] * scales ** regr[0], 'r-')
        plt.xlabel('Scales');
        plt.ylabel('Wavelet spectrum')
        plt.show()

    return hurst


if __name__ == "__main__":
    from cwt import *
    import os
    from tfmean import *

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

    hurst = hurst_est(mean_fre, (1,15), 3)