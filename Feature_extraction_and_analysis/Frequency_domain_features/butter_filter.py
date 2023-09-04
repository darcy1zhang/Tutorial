import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Description:
        Design a bandpass Butterworth filter.

    Params:
        lowcut (float): Lower cutoff frequency of the bandpass filter in Hz.
        highcut (float): Upper cutoff frequency of the bandpass filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Order of the filter (default is 5).

    Returns:
        numpy.ndarray: Numerator coefficients of the filter.
        numpy.ndarray: Denominator coefficients of the filter.
    """

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    """
    Description:
        Design a lowpass Butterworth filter.

    Params:
        cutoff (float): Cutoff frequency of the lowpass filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Order of the filter (default is 5).

    Returns:
        numpy.ndarray: Numerator coefficients of the filter.
        numpy.ndarray: Denominator coefficients of the filter.
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    """
    Description:
        Design a highpass Butterworth filter.

    Params:
        cutoff (float): Cutoff frequency of the highpass filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Order of the filter (default is 5).

    Returns:
        numpy.ndarray: Numerator coefficients of the filter.
        numpy.ndarray: Denominator coefficients of the filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_filter(data, b, a):
    """
    Description:
        Apply a digital IIR filter to the input data.

    Params:
        data (numpy.ndarray): Input data to be filtered.
        b (numpy.ndarray): Numerator coefficients of the filter.
        a (numpy.ndarray): Denominator coefficients of the filter.

    Returns:
        numpy.ndarray: Filtered output data.
    """
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    lowcut = 0.5
    highcut = 2.5
    cutoff_lowpass = 10
    cutoff_highpass = 2

    b_bandpass, a_bandpass = butter_bandpass(lowcut, highcut, fs, order=6)
    b_lowpass, a_lowpass = butter_lowpass(cutoff_lowpass, fs, order=6)
    b_highpass, a_highpass = butter_highpass(cutoff_highpass, fs, order=6)

    plt.subplot(2,1,1)
    plt.plot(signal)

    filtered = butter_filter(signal, b_lowpass, a_lowpass)

    plt.subplot(2,1,2)
    plt.plot(filtered)
    plt.show()
