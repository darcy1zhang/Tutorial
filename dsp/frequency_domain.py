import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import butter, lfilter
from scipy.signal import welch
from scipy.stats import entropy
from scipy.stats import kurtosis, skew
from numpy.fft import fft, ifft, fftfreq


def my_fft(signal, fs):
    """
    Description:
        Get the spectrum of the input signal
    Args:
        signal: input signal
        fs: sampling rate
    Returns:
        The spectrum of the input, containing the freq of x-axis and the mag of the y-axis. The mag is complex.
    """
    l = len(signal)
    mag = fft(signal)
    freq = fftfreq(l, 1 / fs)
    mag = mag / l * 2

    return freq, mag


def my_ifft(mag):
    """
    Description:
        Use the mag of my_fft to recover the original signal
    Args:
        mag: Output of my_fft
    Returns:
        The recovered original signal
    """
    mag = mag / 2 * len(mag)
    x = ifft(mag)

    return x



def extract_spectral_power(signal, fs, frequency_band):
    """
    Description:
        Extract the spectral power of a signal within a specified frequency band.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
        frequency_band (tuple): Frequency band of interest (lowcut, highcut) in Hz.

    Returns:
        float: Spectral power within the specified frequency band.
    """

    f, Pxx = welch(signal, fs=fs)
    mask = (f >= frequency_band[0]) & (f <= frequency_band[1])
    spectral_power = np.sum(Pxx[mask])
    return spectral_power

def extract_peak_frequency(signal, fs):
    """
    Description:
        Extract the frequency with the highest spectral amplitude (peak frequency).

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Frequency with the highest spectral amplitude (peak frequency).
    """

    f, Pxx = welch(signal, fs=fs)
    peak_frequency = f[np.argmax(Pxx)]
    return peak_frequency


def extract_power_spectral_density(signal, fs):
    """
    Description:
        Extract the power spectral density (PSD) of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        numpy.ndarray: Frequency vector.
        numpy.ndarray: Power spectral density values.
    """

    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

def extract_spectral_entropy(signal, fs, num_segments=10):
    """
    Description:
        Extract the spectral entropy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
        num_segments (int, optional): Number of segments for entropy calculation.

    Returns:
        float: Spectral entropy value.
    """

    f, Pxx = welch(signal, fs=fs)
    segment_size = len(f) // num_segments
    segment_entropies = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment_Pxx = Pxx[start_idx:end_idx]
        segment_entropies.append(entropy(segment_Pxx))

    spectral_entropy = np.mean(segment_entropies)
    return spectral_entropy


def extract_spectral_kurtosis(signal, fs):
    """
    Description:
        Extract the spectral kurtosis of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Spectral kurtosis value.
    """

    f, Pxx = welch(signal, fs=fs)
    spectral_kurtosis = kurtosis(Pxx)
    return spectral_kurtosis

def extract_spectral_skewness(signal, fs):
    """
    Description:
        Extract the spectral skewness of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Spectral skewness value.
    """

    f, Pxx = welch(signal, fs=fs)
    spectral_skewness = skew(Pxx)
    return spectral_skewness

def extract_mean_spectral_energy(signal, fs):
    """
    Description:
        Extract the mean spectral energy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Mean spectral energy value.
    """

    f, Pxx = welch(signal, fs=fs)
    mean_spectral_energy = np.mean(Pxx)
    return mean_spectral_energy

def extract_spectral_power(signal, fs, frequency_band):
    """
    Description:
        Extract the spectral power of a signal within a specified frequency band.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
        frequency_band (tuple): Frequency band of interest (lowcut, highcut) in Hz.

    Returns:
        float: Spectral power within the specified frequency band.
    """

    f, Pxx = welch(signal, fs=fs)
    mask = (f >= frequency_band[0]) & (f <= frequency_band[1])
    spectral_power = np.sum(Pxx[mask])
    return spectral_power

def extract_peak_frequency(signal, fs):
    """
    Description:
        Extract the frequency with the highest spectral amplitude (peak frequency).

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Frequency with the highest spectral amplitude (peak frequency).
    """

    f, Pxx = welch(signal, fs=fs)
    peak_frequency = f[np.argmax(Pxx)]
    return peak_frequency


def extract_power_spectral_density(signal, fs):
    """
    Description:
        Extract the power spectral density (PSD) of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        numpy.ndarray: Frequency vector.
        numpy.ndarray: Power spectral density values.
    """

    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

def extract_spectral_entropy(signal, fs, num_segments=10):
    """
    Description:
        Extract the spectral entropy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
        num_segments (int, optional): Number of segments for entropy calculation.

    Returns:
        float: Spectral entropy value.
    """

    f, Pxx = welch(signal, fs=fs)
    segment_size = len(f) // num_segments
    segment_entropies = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment_Pxx = Pxx[start_idx:end_idx]
        segment_entropies.append(entropy(segment_Pxx))

    spectral_entropy = np.mean(segment_entropies)
    return spectral_entropy


def extract_spectral_kurtosis(signal, fs):
    """
    Description:
        Extract the spectral kurtosis of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Spectral kurtosis value.
    """

    f, Pxx = welch(signal, fs=fs)
    spectral_kurtosis = kurtosis(Pxx)
    return spectral_kurtosis

def extract_spectral_skewness(signal, fs):
    """
    Description:
        Extract the spectral skewness of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Spectral skewness value.
    """

    f, Pxx = welch(signal, fs=fs)
    spectral_skewness = skew(Pxx)
    return spectral_skewness

def extract_mean_spectral_energy(signal, fs):
    """
    Description:
        Extract the mean spectral energy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Mean spectral energy value.
    """

    f, Pxx = welch(signal, fs=fs)
    mean_spectral_energy = np.mean(Pxx)
    return mean_spectral_energy


def DCT_synthesize(amps, fs, ts):
    """
    Description:
        Synthesize a mixture of cosines with given amps and fs.

    Input:
        amps: amplitudes
        fs: frequencies in Hz
        ts: times to evaluate the signal

    Returns:
        wave array
    """
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    ys = np.dot(M, amps)
    return ys

def DCT_analyze(ys, fs, ts):
    """
    Description:
        Analyze a mixture of cosines and return amplitudes.

    Input:
        ys: wave array
        fs: frequencies in Hz
        ts: time when the signal was evaluated

    returns:
        vector of amplitudes
    """
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    amps = np.dot(M, ys) / 2
    return amps

def DCT_iv(ys):
    """
    Description:
        Computes DCT-IV.

    Input:
        wave array

    returns:
        vector of amplitudes
    """
    N = len(ys)
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    amps = np.dot(M, ys) / 2
    return amps


def inverse_DCT_iv(amps):
    return DCT_iv(amps) * 2



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





    amps = np.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    ys = DCT_synthesize(amps, fs, ts)

    amps2 = DCT_iv(ys)
    print('amps', amps)
    print('amps2', amps2)
