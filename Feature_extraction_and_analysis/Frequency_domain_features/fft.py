import numpy as np
import matplotlib.pyplot as plt

def perform_fft(signal, sampling_rate):
    """
    Description:
        Perform Fast Fourier Transform (FFT) on a given signal.

    Params:
        signal (numpy.ndarray): The input signal.
        sampling_rate (float): The sampling rate of the signal.

    Return:
        numpy.ndarray: Array of complex numbers representing the FFT result.
        numpy.ndarray: Array of corresponding frequency values.
    """

    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_result = np.fft.fft(signal)

    return fft_result, frequencies


if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100
    time = np.arange(0, 10, 1 / fs)

    fft_result, frequencies = perform_fft(signal, fs)

    plt.figure(figsize=(8, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.grid()
    plt.show()
