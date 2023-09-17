from scipy import signal
import numpy as np


def vwt(input, scale, w0=2 * np.pi):
    """
    Description:
        Compute Morlet's wavelet transform at one scale.

    Params:
        input:Input signal (1D array)
        scale:Scale at which the wavelet transform is computed
        w0:Center frequency of the wavelet

    Returns:
        output:1D complex array containing wavelet transform at one scale
    """

    # Construct Morlet wavelet kernel
    wavelet_kernel = np.exp(1j * w0 * np.arange(len(input))) * \
                     np.exp(-np.square(np.arange(len(input))) / scale)

    # Compute wavelet transform
    output = signal.convolve(input, wavelet_kernel, mode='same')

    return output

if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal1 = np.load(signal_path)[2, :1000]
    fs = 100


    result = vwt(signal1, 0.02)

    print(result.shape)