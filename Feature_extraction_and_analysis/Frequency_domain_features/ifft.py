import numpy as np
import matplotlib.pyplot as plt
from fft import perform_fft

def perform_ifft(fft_result):
    """
    Description:
        Perform Inverse Fast Fourier Transform (IFFT) on a given FFT result.

    Params:
        fft_result (numpy.ndarray): The FFT result.

    Return:
        numpy.ndarray: Array of complex numbers representing the IFFT result.
    """

    ifft_result = np.fft.ifft(fft_result)
    return ifft_result

if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    fft_result, frequencies = perform_fft(signal,fs)
    ifft_result = perform_ifft(fft_result)

    plt.subplot(2,1,1)
    plt.plot(signal)
    plt.subplot(2,1,2)
    plt.plot(ifft_result.real)
    plt.show()