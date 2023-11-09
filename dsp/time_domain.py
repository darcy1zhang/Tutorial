import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import os
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d
import json
import tsfel
from scipy.fft import fft,ifft


def cal_corrcoef(signal1, signal2):
    """
    Description:
        To get the correlate coefficient

    Input:
        Two signal with same length

    Return:
        The correlate coefficient
    """
    return np.corrcoef(signal1, signal2)[0,1]

def cal_serial_corr(signal, lag):
    """
    Description:
        To get the serial correlate coefficient

    Input:
        One signal and the lag which means how much it delays

    Return:
        The serial correlate coefficient
    """
    signal1 = signal[lag:]
    signal2 = signal[:len(signal)-lag]
    return np.corrcoef(signal1, signal2)[0,1]

def cal_autocorr(signal, plot = False):
    """
    Description:
        To get the auto correlate coefficient

    Input:
        One signal

    Return:
        The serial correlate coefficient with different lag which is from 0 to len(wave)//2
    """
    lags = range(len(signal)//2)
    corrs = [cal_serial_corr(signal, lag) for lag in lags]
    if plot:
        plt.plot(lags, corrs)
        plt.show()
    return lags, corrs


def analytic_signal(x):
    """
    Description:
        Get the analytic version of the input signal
    Args:
        x: input signal which is a real-valued signal
    Returns:
        The analytic version of the input signal which is a complex-valued signal
    """
    N = len(x)
    X = fft(x,N)
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2*np.ones(N//2-1)
    h[N//2] = 1
    Z = X*h
    z = ifft(Z,N)
    return z

def hilbert_transform(x):
    """
    Description:
        Get the hilbert transformation of the input signal
    Args:
        x: a real-valued singal
    Returns:
        Return the result of hilbert transformation which is the imaginary part of the analytic signal
    """
    z = analytic_signal(x)
    return z.imag

def envelope_hilbert(signal, fs):
    """
    Description:
        Analyzes a signal using the Hilbert Transform to extract envelope and phase information.

    Params:
        signal (array-like): The input signal to be analyzed.
        fs (float): The sampling frequency of the input signal.

    Returns:
        inst_amplitude (array-like): The instantaneous amplitude of the signal envelope.
        inst_freq (array-like): The instantaneous frequency of the signal.
        inst_phase (array-like): The instantaneous phase of the signal.
        regenerated_carrier (array-like): The regenerated carrier signal from the instantaneous phase.
    """

    z= hilbert(signal) #form the analytical signal
    inst_amplitude = np.abs(z) #envelope extraction
    inst_phase = np.unwrap(np.angle(z))#inst phase
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs #inst frequency

    #Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(inst_phase)

    return inst_amplitude, inst_freq, inst_phase, regenerated_carrier


def update_array(a, data_tmp):
    """
    Description:
        Update an array 'a' by removing elements based on the pattern in 'data_tmp'.

    Params:
        a (numpy.ndarray): The input array to be updated.
        data_tmp (numpy.ndarray): The data array used for comparison.

    Returns:
        updated_array (numpy.ndarray): The updated array after removing elements.
    """
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a


def fpeak(signal):
    """
    Description:
        Detect peaks in a signal and perform linear interpolation to obtain an envelope.

    Params:
        signal (numpy.ndarray): The input signal.
        t (numpy.ndarray): The corresponding time values for the signal.

    Returns:
        peaks (numpy.ndarray): An array containing the indices of the detected peaks.
    """
    t = np.arange(len(signal))
    # find all peaks in th signal
    peak_indices, _ = find_peaks(signal)

    # interpolate the peaks to form the envelope
    t_peaks = t[peak_indices]
    peak_values = signal[peak_indices]
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    # find the peaks of envelope
    peaks2, _ = find_peaks(envelope, distance=10)

    # remove wrong peaks
    peaks2 = update_array(peaks2, signal)

    # make sure the first peak is the higher peak
    if len(peaks2) > 1:
        if (signal[peaks2[1]] > signal[peaks2[0]]):
            peaks2 = np.delete(peaks2, 0)

    # make sure the number of peaks is even
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    return peaks2

def tsfel_feature(signal, fs = 100):

    with open("../json/all_features.json", 'r') as file:
        cgf_file = json.load(file)

    # cgf_file = tsfel.get_features_by_domain("temporal")

    features = tsfel.time_series_features_extractor(cgf_file, signal, fs=fs, window_size=len(signal),
                                                    features_path="../utils/my_features.py").values.flatten()

    return features


if __name__ == "__main__":
    signal = np.load("../../data/sim_100_0.1_90_140_train.npy")[10,:1000]
    signal1 = signal[100:200]
    signal2 = signal[150:250]

    print(cal_corrcoef(signal1,signal2))
    print(cal_serial_corr(signal, 51))
    print(cal_autocorr(signal, True))

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    inst_amplitude, inst_freq, inst_phase, regenerated_carrier = envelope_hilbert(signal, fs)

    window_size = 3  # Adjust the window size as needed
    smoothed_envelope = np.convolve(inst_amplitude, np.ones(window_size) / window_size, mode='same')

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.plot(smoothed_envelope, 'r')  # overlay the extracted envelope
    plt.title('Modulated signal and extracted envelope')
    plt.xlim(0, 200)
    plt.xlabel('n')
    plt.ylabel('x(t) and |z(t)|')
    plt.subplot(2, 1, 2)
    plt.plot(regenerated_carrier)
    plt.title('Extracted carrier or TFS')
    plt.xlabel('n')
    plt.ylabel('cos[\omega(t)]')
    plt.show()



    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")
    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data",
                               signal_name)
    signal = np.load(signal_path)[0, :1000]

    fs = 100
    # t = np.linspace(0, 10, 10 * fs)

    peaks = fpeak(signal)
    t = np.arange(len(signal))
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.plot(t[peaks], signal[peaks], 'o', color='green')
    plt.show()

