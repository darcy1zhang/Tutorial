import numpy as np
import pywt
import matplotlib.pyplot as plt

def cwt(data, sampling_rate, wavename, totalscal=256):
    """
    Description:
        Perform Continuous Wavelet Transform (CWT) on a given signal.

    Parameters:
        data (numpy.ndarray): Input signal.
        sampling_rate (float): Sampling rate of the signal.
        wavename (str): Name of the wavelet function to use.
        totalscal (int, optional): Length of the scale sequence for wavelet transform (default is 256).

    Returns:
        cwtmatr (numpy.ndarray): The CWT matrix representing the wavelet transform of the input signal.
        frequencies (numpy.ndarray): The frequencies corresponding to the CWT matrix rows.
    """

    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    cwtmatr, frequencies = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)

    return cwtmatr, frequencies


if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100
    t = np.arange(0, 10, 1.0/fs)

    wavename = "morl"

    cwtmatr, frequencies = cwt(signal, fs, wavename)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, signal)
    plt.xlabel(u"time(s)")
    plt.title(u"Time Spectrum")

    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    # plt.plot(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()
