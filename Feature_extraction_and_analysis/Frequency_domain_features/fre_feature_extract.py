import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100


    # Extract spectral kurtosis
    spectral_kurtosis = extract_spectral_kurtosis(signal, fs)
    print("Spectral Kurtosis:", spectral_kurtosis)

    # Extract spectral skewness
    spectral_skewness = extract_spectral_skewness(signal, fs)
    print("Spectral Skewness:", spectral_skewness)

    # Extract mean spectral energy
    mean_spectral_energy = extract_mean_spectral_energy(signal, fs)
    print("Mean Spectral Energy:", mean_spectral_energy)


    # Extract power spectral density
    frequencies, psd = extract_power_spectral_density(signal, fs)
    print("Frequencies:", frequencies)
    print("Power Spectral Density:", psd)
    plt.plot(frequencies, psd)
    plt.show()

    # Extract spectral entropy
    spectral_entropy = extract_spectral_entropy(signal, fs)
    print("Spectral Entropy:", spectral_entropy)


    # Extract spectral power in a frequency band
    frequency_band = (4, 6)  # Specify the frequency band of interest
    spectral_power = extract_spectral_power(signal, fs, frequency_band)
    print("Spectral Power:", spectral_power)

    # Extract peak frequency
    peak_frequency = extract_peak_frequency(signal, fs)
    print("Peak Frequency:", peak_frequency)
