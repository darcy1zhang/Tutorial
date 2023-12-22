import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from scipy.signal import butter, lfilter, hilbert, chirp, welch, find_peaks
from scipy.stats import entropy, kurtosis, skew
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
from numpy.fft import fft, ifft, fftfreq
import os
import json
import tsfel
from sklearn.metrics import mean_squared_error
import pywt
import tftb
# import chirplet
import ssqueezepy as sq
from pylab import (arange, flipud, linspace, cos, pi, log, hanning,
                   ceil, log2, floor, empty_like, fft, ifft, fabs, exp, roll, convolve)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastsst import SingularSpectrumTransformation



# Time Domain
## Analytic Signal and Hilbert Transform
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
    X = fft(x, N)
    h = np.zeros(N)
    h[0] = 1
    h[1:N // 2] = 2 * np.ones(N // 2 - 1)
    h[N // 2] = 1
    Z = X * h
    z = ifft(Z, N)
    return z

def hilbert_transform(x):
    """
    Description:
        Get the hilbert transformation of the input signal
    Args:
        x: a real-valued singal
    Returns:
        Return the result of hilbert transformation which is the imaginary part of the analytic signal. It is a
        real-valued number.
    """
    z = analytic_signal(x)
    return z.imag

## Peak Detection
### Peak of Peak Algorithm
def get_peaks(signal):
    """
    Description:
        Detect peaks in a signal and perform linear interpolation to obtain an envelope.

    Params:
        signal (numpy.ndarray): The input signal.

    Returns:
        peaks (numpy.ndarray): An array containing the indices of the detected peaks.
    """
    t = np.arange(len(signal))
    peak_indices, _ = find_peaks(signal) # find all peaks in th signal

    # interpolate the peaks to form the envelope
    t_peaks = t[peak_indices]
    peak_values = signal[peak_indices]
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2, _ = find_peaks(envelope, distance=10) # find the peaks of envelope
    peaks2 = update_array(peaks2, signal) # remove wrong peaks

    # make sure the first peak is the higher peak
    if len(peaks2) > 1:
        if (signal[peaks2[1]] > signal[peaks2[0]]):
            peaks2 = np.delete(peaks2, 0)

    # make sure the number of peaks is even
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    return peaks2

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

## Envelope Extraction
### Envelope from Peaks
def envelope_from_peaks(signal):
    """
    Description
        Interpolation the peaks to get the envelope of the input signal. The algorithm is only suitable for the signal
        with a lot of noise
    Args:
        signal: The input signal
    Returns:
        envelope: The envelope of the input signal
    """
    t = np.arange(len(signal))
    peak_indices, _ = find_peaks(signal)

    # interpolate the peaks to form the envelope
    t_peaks = t[peak_indices]
    peak_values = signal[peak_indices]
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    return envelope

### Average Envelope
def average_envelope(signal, window_length):
    """
    Description:
        Use the average window to get the envelope
    Args:
        signal: input signal
        window_length: the length of the average window
    Returns:
        envelope: the envelope of the input signal
    """
    weights = np.ones(window_length) / window_length
    envelope = np.convolve(np.abs(signal), weights, mode='valid')
    padding = (window_length - 1) // 2
    envelope = np.concatenate([np.zeros(padding), envelope, np.zeros(padding)])
    return envelope

### Envelope and Phase Extraction using Hilbert Transform
def inst_amplitude(signal):
    """
    Description:
        Use hilbert transformation to compute the instantaneous amplitude or the envelope of the input signal
    Args:
        signal: input signal
    Returns:
        The instantaneous amplitude or the envelope of the signal
    """
    z = analytic_signal(signal)
    return np.abs(z)

def inst_phase(signal):
    """
    Description:
        Use hilbert transformation to compute the instantaneous phase of the input signal
    Args:
        signal: input signal
    Returns:
        instantaneous phase
    """
    z = analytic_signal(signal)
    return np.unwrap(np.angle(z))

def inst_freq(signal, fs):
    """
    Description:
        Use hilbert transformation to compute the instantaneous temporal frequency of the input signal
    Args:
        signal: input signal
        fs: frequency of sampling of input signal
    Returns:
        the instantaneous temporal frequency
    """
    inst_phase_sig = inst_phase(signal)
    return np.diff(inst_phase_sig) / (2 * np.pi) * fs

## Singular Spectrum Transform (SST)
def sst(signal, win_length):
    """
    Description:
        It is a change point detection algorithm
    Args:
        signal: the input signal
        win_length: window length of Hankel matrix
    Returns:
        score: an array measuring the degree of change
    """
    sst = SingularSpectrumTransformation(win_length=win_length)
    score = sst.score_offline(signal)
    return score

## Time Domain Feature
### Petrosian Fractal Dimension (PFD)
def pfd(signal):
    """
    Description:
        It calculates the fractal dimension of a signal to describe its complexity and irregularity. A higher Petrosian
        Fractal Dimension value indicates a more complex signal.
    Args:
        signal: The input signal
    Returns:
        The value of pfd
    """
    diff = np.diff(signal)
    n_zero_crossings = np.sum(diff[:-1] * diff[1:] < 0)
    pfd = np.log10(len(signal)) / (
                np.log10(len(signal)) + np.log10(len(signal) / (len(signal) + 0.4 * n_zero_crossings)))
    return pfd

# Frequency Domain
## Fast Fourier Transform (FFT)
def my_fft(signal, fs):
    """
    Description:
        Get the spectrum of the input signal
    Args:
        signal: input signal
        fs: sampling rate
    Returns:
        The spectrum of the input, containing the freq of x-axis and the mag of the y-axis. The mag is complex number.
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
        The recovered original signal. It is complex-valued.
    """
    mag = mag / 2 * len(mag)
    x = ifft(mag)
    return x

## Frequency Domain Feature
### Power Spectral Density (PSD)
def psd(signal, fs):
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

# Time-Frequency Domain
## Short Time Fourier Transform (STFT)
def my_stft(signal, fs, plot=False, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False,
            return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    """
    Description:
        Compute the Linear Spectrogram of a signal using Short-time Fourier Transform (STFT).

    Params:
        signal (numpy.ndarray): The input signal.
        fs (int): The sample rate of the signal.
        nperseg (int, optional): The size of the analysis window in samples. Default is 256.
        The other parameters are seldom used.

    Returns:
        freqs (numpy.ndarray): The frequency values in Hz.
        times (numpy.ndarray): The time values in seconds.
        spectrogram (numpy.ndarray): The computed linear spectrogram.
    """
    f, t, Z = scipy.signal.stft(signal, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, boundary, padded,
                                axis, scaling)
    if plot:
        plt.pcolormesh(t, f, np.abs(Z))
        plt.show()
    return f, t, Z

## Wavelet Analysis
### Mexican Hat Wavelet
def mexican_hat_wavelet(sigma, length):
    """
    Description:
        Generate the mexican hat wavelet. It is the second derivative of the Gaussian function.
    Args:
        sigma: It has the same meaning in the Gaussian function
        length: length of the wavelet
    Returns:
        The mexican hat wavelet
    """
    t = np.linspace(-int(length / 2), length / 2, length * 10)
    psi = 1 / (np.sqrt(2 * np.pi) * np.power(sigma, 3)) * np.exp(-np.power(t, 2) / (2 * np.power(sigma, 2))) * (
                (np.power(t, 2) / np.power(sigma, 2)) - 1)
    return psi

### Morlet Wavelet
def morlet_wavelet(length, sigma, a=5):
    """
    Description:
        Generate the morlet wavelet which value is complex.
    Args:
        length: Length of the wavelet.
        sigma: Scaling parameter that affects the width of the window.
        a: Modulation parameter. Default is 5
    Returns:
        The morlet wavelet which is complex-valued.
    """
    morlet_wav = scipy.signal.morlet2(length, sigma, a)
    return morlet_wav

### Continues Wavelet Transform (CWT)

def my_cwt(signal, scales, wavelet, fs, show=False):
    """
    Description:
        Compute the cwt of the input signal
    Args:
        signal: input signal
        scales: the scales of wavelet, we can use pywt.scale2frequency to convert them to corresponding frequency
        wavelet: the type of the wavelet, there are "morl", "mexh" and so on. You can use
            wavlist = pywt.wavelist(kind='continuous') to get the available wavelet
        fs: the sampling frequency
        show: whether to show the result
    Returns:
        cofficient: the result of cwt. The length of y-axis depends on scales and length of x-axis depends on length of
            input signal
        frequencies: the corresponding frequencies to  scales
    """
    freq = pywt.scale2frequency(wavelet, scales) * fs
    if freq[0] > fs / 2:
        raise ValueError("The intended frequency is too high, please increase the lowest number of scales")
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1 / fs)
    if show:
        plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(signal) / fs, frequencies[-1], frequencies[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.show()
    return coefficients, frequencies

## Polynomial Chirplet Transform (PCT)
### Chirplet Transform
def chirplet_transform(signal, show=False):
    """
    Description:
        Generate the chirplet_trainsform of the input signal
    Args:
        signal: Input signal
        show: whether to show the result of the chirplet transform
    Returns:
        The result of the chirplet transform
    """
    chirps = FCT()
    ct_matrix = chirps.compute(signal)
    if show:
        plt.title("chirplet transform")
        plt.imshow(ct_matrix, aspect="auto")
    return ct_matrix

## Wigner Ville Distribution (WVD)
def my_wvd(signal, show=False):
    """
    Description:
        Analyze the time-frequency characteristics of a signal using the Wigner-Ville Transform (WVT) and visualize the results.

    Params:
        signal (numpy.ndarray): The input signal.
        show: whether to plot the result
    Returns:
        tfr_wvd (numpy.ndarray): The time-frequency representation (WVD) of the signal.
        t_wvd (numpy.ndarray): Time values corresponding to the WVD.
        f_wvd (numpy.ndarray): Normalized frequency values corresponding to the WVD.
    """
    wvd = tftb.processing.WignerVilleDistribution(signal)
    tfr_wvd, t_wvd, f_wvd = wvd.run()
    if show:
        wvd.plot(kind="contourf", scale="log")
    return tfr_wvd, t_wvd, f_wvd

## SynchroSqueezing Transform (SST)
def sst_stft(signal, fs, window, nperseg=256, show=False, n_fft=None, hop_len=1, modulated=True, ssq_freqs=None,
             padtype='reflect', squeezing='sum', gamma=None, preserve_transform=None, dtype=None, astensor=True,
             flipud=False, get_w=False, get_dWx=False):
    """
    Description:
        Synchrosqueezed Short-Time Fourier Transform.
    Args:
        signal: the input signal
        fs: frequency of sampling
        window: type of the window
        nperseg: Length of each segment
        show: whether to show the result
        n_fft: length of fft
        The other parameters are seldom used.
    Returns:
        Tx: Synchrosqueezed STFT of `x`, of same shape as `Sx`.
        Sx: STFT of `x`
        ssq_freqs: Frequencies associated with rows of `Tx`.
        Sfs: Frequencies associated with rows of `Sx` (by default == `ssq_freqs`).
    """
    Tx, Sx, ssq_freqs, Sfs = sq.ssq_stft(signal, window=window, win_len=nperseg, fs=fs, n_fft=n_fft)
    if show:
        plt.subplot(2, 1, 1)
        plt.title("STFT of Input signal")
        plt.imshow(np.abs(Sx), aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Synchrosqueezed STFT of Input signal")
        plt.imshow(np.abs(Tx), aspect="auto")
        plt.tight_layout()
        plt.show()
    return Tx, Sx, ssq_freqs, Sfs

def sst_cwt(signal, wavelet, scales, nv, fs, gamma=None, show=False):
    """
    Description:
        Synchrosqueezed Continuous Wavelet Transform
    Args:
        signal: input of signal
        wavelet: the type of mother wavelet
        scales: how to scale the output, log or linear
        nv: number of voices
        fs: sampling frequency
        gamma: CWT phase threshold
        show: whether to show the result
    Returns:
        Tx: Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        Wx: Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
        ssq_freqs: Frequencies associated with rows of `Tx`.
        scales: Scales associated with rows of `Wx`.
    """
    Tx, Wx, ssq_freqs, scales = sq.ssq_cwt(x=signal, wavelet=wavelet, scales=scales, nv=nv, fs=fs, gamma=gamma)
    if show:
        plt.subplot(2, 1, 1)
        plt.imshow(np.abs(Wx), aspect='auto', extent=[0, len(signal) / fs, ssq_freqs[-1], ssq_freqs[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.subplot(2, 1, 2)
        plt.imshow(np.abs(Tx), aspect='auto', extent=[0, len(signal) / fs, ssq_freqs[-1], ssq_freqs[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Synchrosqueezed Continuous Wavelet Transform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()
    return Tx, Wx, ssq_freqs, scales











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


def sine_wave(length_seconds, sampling_rate, frequencies, func="sin", add_noise=0, plot=False):
    """
    Generate a n-D array, `length_seconds` seconds signal at `sampling_rate` sampling rate.
    Cited from https://towardsdatascience.com/hands-on-signal-processing-with-python-9bda8aad39de

    Args:
        length_seconds : float
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal, `3.5` for a 3.5-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies : 1 or 2 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
            2 dimension python list, i.e. [[5, 12, 15],[1]], to generate a signal with 2 channels, where the second channel containing 1-Hz signal
        func : string, optional, default: sin
            The periodic function to generate signal, either `sin` or `cos`
        add_noise : float, optional, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean, optional, default: False
            Plot the generated signal

    Returns:
        signal : n-d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    Usage:
        >>> s = generate_sine_waves(length_seconds=4,
        >>>     sampling_rate=100,
        >>>     frequencies=[2],
        >>>     plot=True
        >>> )
        >>>
        >>> s = generate_sine_waves(length_seconds=4,
        >>>     sampling_rate=100,
        >>>     frequencies=[1,2],
        >>>     func="cos",
        >>>     add_noise=0.5,
        >>>     plot=True
        >>> )
        >>>
        >>> s = generate_sine_waves(length_seconds=3.5,
        >>>     sampling_rate=100,
        >>>     frequencies=[[1,2],[1],[2]],
        >>>     plot=True
        >>> )
    """

    frequencies = np.array(frequencies, dtype=object)
    assert len(frequencies.shape) == 1 or len(frequencies.shape) == 2, "frequencies must be 1d or 2d python list"

    expanded = False
    if isinstance(frequencies[0], int):
        frequencies = np.expand_dims(frequencies, axis=0)
        expanded = True

    sampling_rate = int(sampling_rate)
    npnts = int(sampling_rate * length_seconds)  # number of time samples
    time = np.arange(0, npnts) / sampling_rate
    signal = np.zeros((frequencies.shape[0], npnts))

    for channel in range(0, frequencies.shape[0]):
        for fi in frequencies[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2 * np.pi * fi * time)
            else:
                signal[channel] = signal[channel] + np.sin(2 * np.pi * fi * time)

        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2 * (signal[channel] - min) / (max - min)) - 1

    if add_noise:
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies.shape[0], npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.title(
            'Signal with sampling rate of ' + str(sampling_rate) + ', lasting ' + str(length_seconds) + '-seconds')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    if expanded:
        signal = signal[0]

    return signal


def triangle_wave(t, width=1):
    """
    Return a periodic sawtooth or triangle waveform.

    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        Time.
    width : array_like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the sawtooth waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 1, 500)
    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))

    """
    t, w = np.asarray(t), np.asarray(width)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (np.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))

    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (np.pi * (wsub + 1) - tsub) / (np.pi * (1 - wsub)))
    return y


def square_wave(t, duty=0.5):
    """
    Return a periodic square-wave waveform.

    The square wave has a period ``2*pi``, has value +1 from 0 to
    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
    the interval [0,1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        The input time array.
    duty : array_like, optional
        Duty cycle.  Default is 0.5 (50% duty cycle).
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the square waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 1, 500, endpoint=False)
    >>> plt.plot(t, signal.square(2 * np.pi * 5 * t))
    >>> plt.ylim(-2, 2)

    A pulse-width modulated sine wave:

    >>> plt.figure()
    >>> sig = np.sin(2 * np.pi * t)
    >>> pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(t, sig)
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(t, pwm)
    >>> plt.ylim(-1.5, 1.5)

    """
    t, w = np.asarray(t), np.asarray(duty)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = np.mod(t, 2 * np.pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    np.place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    np.place(y, mask3, -1)
    return y


def chirp_wave(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    See Also
    --------
    sweep_poly

    Notes
    -----
    There are four options for the `method`.  The following formulas give
    the instantaneous frequency (in Hz) of the signal generated by
    `chirp()`.  For convenience, the shorter names shown below may also be
    used.

    linear, lin, li:

        ``f(t) = f0 + (f1 - f0) * t / t1``

    quadratic, quad, q:

        The graph of the frequency f(t) is a parabola through (0, f0) and
        (t1, f1).  By default, the vertex of the parabola is at (0, f0).
        If `vertex_zero` is False, then the vertex is at (t1, f1).  The
        formula is:

        if vertex_zero is True:

            ``f(t) = f0 + (f1 - f0) * t**2 / t1**2``

        else:

            ``f(t) = f1 - (f1 - f0) * (t1 - t)**2 / t1**2``

        To use a more general quadratic function, or an arbitrary
        polynomial, use the function `scipy.signal.sweep_poly`.

    logarithmic, log, lo:

        ``f(t) = f0 * (f1/f0)**(t/t1)``

        f0 and f1 must be nonzero and have the same sign.

        This signal is also known as a geometric or exponential chirp.

    hyperbolic, hyp:

        ``f(t) = f0*f1*t1 / ((f0 - f1)*t + f1*t1)``

        f0 and f1 must be nonzero.

    Examples
    --------
    The following will be used in the examples:

    >>> import numpy as np
    >>> from scipy.signal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt

    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = np.linspace(0, 10, 1500)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(t, w)
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()

    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `scipy.signal.spectrogram`.
    We'll use a 4 second interval sampled at 7200 Hz.

    >>> fs = 7200
    >>> T = 4
    >>> t = np.arange(0, int(T*fs)) / fs

    We'll use this function to plot the spectrogram in each example.

    >>> def plot_spectrogram(title, w, fs):
    ...     ff, tt, Sxx = spectrogram(w, fs=fs, nperseg=256, nfft=576)
    ...     fig, ax = plt.subplots()
    ...     ax.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r',
    ...                   shading='gouraud')
    ...     ax.set_title(title)
    ...     ax.set_xlabel('t (sec)')
    ...     ax.set_ylabel('Frequency (Hz)')
    ...     ax.grid(True)
    ...

    Quadratic chirp from 1500 Hz to 250 Hz
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=T, method='quadratic')
    >>> plot_spectrogram(f'Quadratic Chirp, f(0)=1500, f({T})=250', w, fs)
    >>> plt.show()

    Quadratic chirp from 1500 Hz to 250 Hz
    (vertex of the parabolic curve of the frequency is at t=T):

    >>> w = chirp(t, f0=1500, f1=250, t1=T, method='quadratic',
    ...           vertex_zero=False)
    >>> plot_spectrogram(f'Quadratic Chirp, f(0)=1500, f({T})=250\\n' +
    ...                  '(vertex_zero=False)', w, fs)
    >>> plt.show()

    Logarithmic chirp from 1500 Hz to 250 Hz:

    >>> w = chirp(t, f0=1500, f1=250, t1=T, method='logarithmic')
    >>> plot_spectrogram(f'Logarithmic Chirp, f(0)=1500, f({T})=250', w, fs)
    >>> plt.show()

    Hyperbolic chirp from 1500 Hz to 250 Hz:

    >>> w = chirp(t, f0=1500, f1=250, t1=T, method='hyperbolic')
    >>> plot_spectrogram(f'Hyperbolic Chirp, f(0)=1500, f({T})=250', w, fs)
    >>> plt.show()

    """
    # 'phase' is computed in _chirp_phase, to make testing easier.
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
    # Convert  phi to radians.
    phi *= np.pi / 180
    return np.cos(phase + phi)


def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    """
    Calculate the phase used by `chirp` to generate its output.

    See `chirp` for a description of the arguments.

    """
    t = np.asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    if method in ['linear', 'lin', 'li']:
        beta = (f1 - f0) / t1
        phase = 2 * np.pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            phase = 2 * np.pi * (f0 * t + beta * t ** 3 / 3)
        else:
            phase = 2 * np.pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            phase = 2 * np.pi * f0 * t
        else:
            beta = t1 / np.log(f1 / f0)
            phase = 2 * np.pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            # Degenerate case: constant frequency.
            phase = 2 * np.pi * f0 * t
        else:
            # Singular point: the instantaneous frequency blows up
            # when t == sing.
            sing = -f1 * t1 / (f0 - f1)
            phase = 2 * np.pi * (-sing * f0) * np.log(np.abs(1 - t / sing))

    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                         " or 'hyperbolic', but a value of %r was given."
                         % method)

    return phase


def white_noise(length_seconds, sampling_rate, plot=False):
    r"""
    Generate white noise signal.

    Args:
        length_seconds : float
            Duration of the white noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        plot : boolean, optional, default: False
            Plot the generated white noise signal.

    Returns:
        signal : 1-D ndarray
            Generated white noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> wn = generate_white_noise(length_seconds=5,
        >>>                           sampling_rate=44100,
        >>>                           plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples
    white_noise = np.random.normal(0, 1, npnts)

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, white_noise)
        plt.title('White Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return white_noise


def band_limited_white_noise(length_seconds, sampling_rate, frequency_range, plot=False):
    r"""
    Generate band-limited white noise signal.

    Args:
        length_seconds : float
            Duration of the band-limited white noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        frequency_range : tuple or list of two floats
            Frequency range (start, end) of the band-limited noise in Hz.
        plot : boolean, optional, default: False
            Plot the generated band-limited white noise signal.

    Returns:
        signal : 1-D ndarray
            Generated band-limited white noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> blwn = generate_band_limited_white_noise(length_seconds=5,
        >>>                                        sampling_rate=44100,
        >>>                                        frequency_range=(20, 2000),
        >>>                                        plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples
    frequency_start, frequency_end = frequency_range

    # Generate chirp-like signal to simulate band-limited white noise
    t = np.arange(0, npnts) / sampling_rate
    signal = np.cumsum(np.random.normal(0, 1, npnts))
    signal /= np.max(np.abs(signal))
    signal *= np.sin(2 * np.pi * np.linspace(frequency_start, frequency_end, npnts))

    if plot:
        plt.plot(t, signal)
        plt.title('Band-Limited White Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return signal


def pink_noise(length_seconds, sampling_rate, plot=False):
    r"""
    Generate pink noise signal.

    Args:
        length_seconds : float
            Duration of the pink noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        plot : boolean, optional, default: False
            Plot the generated pink noise signal.

    Returns:
        signal : 1-D ndarray
            Generated pink noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> pn = generate_pink_noise(length_seconds=5,
        >>>                           sampling_rate=44100,
        >>>                           plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate pink noise using numpy and cumulative sum
    pink_noise = np.cumsum(np.random.normal(0, 1, npnts))
    pink_noise /= np.max(np.abs(pink_noise))

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, pink_noise)
        plt.title('Pink Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return pink_noise


def brown_noise(length_seconds, sampling_rate, plot=False):
    r"""
    Generate brown noise signal (also known as red noise).

    Args:
        length_seconds : float
            Duration of the brown noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        plot : boolean, optional, default: False
            Plot the generated brown noise signal.

    Returns:
        signal : 1-D ndarray
            Generated brown noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> bn = generate_brown_noise(length_seconds=5,
        >>>                           sampling_rate=44100,
        >>>                           plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate brown noise using numpy and cumulative sum
    brown_noise = np.cumsum(np.random.normal(0, 1, npnts))
    brown_noise -= np.mean(brown_noise)
    brown_noise /= np.max(np.abs(brown_noise))

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, brown_noise)
        plt.title('Brown Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return brown_noise





def impulsive_noise(length_seconds, sampling_rate, probability, plot=False):
    r"""
    Generate impulsive noise signal (also known as salt-and-pepper noise).

    Args:
        length_seconds : float
            Duration of the impulsive noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        probability : float
            Probability of occurrence of impulsive noise (between 0 and 1).
        plot : boolean, optional, default: False
            Plot the generated impulsive noise signal.

    Returns:
        signal : 1-D ndarray
            Generated impulsive noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> inoise = generate_impulsive_noise(length_seconds=5,
        >>>                                    sampling_rate=44100,
        >>>                                    probability=0.05,
        >>>                                    plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate impulsive noise using numpy with given probability
    impulsive_noise = np.random.choice([-1, 1], size=npnts, p=[probability / 2, 1 - probability / 2])

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, impulsive_noise)
        plt.title('Impulsive Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return impulsive_noise


def click_noise(length_seconds, sampling_rate, position, amplitude=1.0, plot=False):
    r"""
    Generate click noise signal.

    Args:
        length_seconds : float
            Duration of the click noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        position : float
            Position of the click in seconds.
        amplitude : float, optional, default: 1.0
            Amplitude of the click.
        plot : boolean, optional, default: False
            Plot the generated click noise signal.

    Returns:
        signal : 1-D ndarray
            Generated click noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> cnoise = generate_click_noise(length_seconds=5,
        >>>                                 sampling_rate=44100,
        >>>                                 position=2.0,
        >>>                                 amplitude=0.5,
        >>>                                 plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate click noise using numpy
    click_noise = np.zeros(npnts)
    click_sample = int(sampling_rate * position)
    if 0 <= click_sample < npnts:
        click_noise[click_sample] = amplitude

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, click_noise)
        plt.title('Click Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return click_noise


def transient_noise_pulses(length_seconds, sampling_rate, num_pulses, pulse_duration, amplitude_range, plot=False):
    r"""
    Generate transient noise pulses signal.

    Args:
        length_seconds : float
            Duration of the transient noise pulses signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        num_pulses : int
            Number of pulses to generate.
        pulse_duration : float
            Duration of each pulse in seconds.
        amplitude_range : tuple or list of two floats
            Range of amplitudes (min, max) for the pulses.
        plot : boolean, optional, default: False
            Plot the generated transient noise pulses signal.

    Returns:
        signal : 1-D ndarray
            Generated transient noise pulses signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> tnpulses = generate_transient_noise_pulses(length_seconds=5,
        >>>                                           sampling_rate=44100,
        >>>                                           num_pulses=5,
        >>>                                           pulse_duration=0.01,
        >>>                                           amplitude_range=(0.5, 1.0),
        >>>                                           plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples
    pulse_samples = int(sampling_rate * pulse_duration)

    # Generate transient noise pulses using numpy
    signal = np.zeros(npnts)
    for _ in range(num_pulses):
        pulse_amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        pulse_start = np.random.randint(0, npnts - pulse_samples)
        signal[pulse_start:pulse_start + pulse_samples] += pulse_amplitude

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, signal)
        plt.title('Transient Noise Pulses')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return signal


def thermal_noise(length_seconds, sampling_rate, noise_density, plot=False):
    r"""
    Generate thermal noise signal.

    Args:
        length_seconds : float
            Duration of the thermal noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        noise_density : float
            Noise power density in watts/Hz (e.g., -174 dBm/Hz for room temperature).
        plot : boolean, optional, default: False
            Plot the generated thermal noise signal.

    Returns:
        signal : 1-D ndarray
            Generated thermal noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> tnoise = generate_thermal_noise(length_seconds=5,
        >>>                                  sampling_rate=44100,
        >>>                                  noise_density=-174,
        >>>                                  plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Calculate the noise variance from noise density
    noise_variance = 10 ** (noise_density / 10)

    # Generate thermal noise using numpy
    thermal_noise = np.random.normal(0, np.sqrt(noise_variance / 2), npnts)

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, thermal_noise)
        plt.title('Thermal Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return thermal_noise


def shot_noise(length_seconds, sampling_rate, rate, plot=False):
    r"""
    Generate shot noise signal.

    Args:
        length_seconds : float
            Duration of the shot noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        rate : float
            Average rate of occurrence of events (e.g., photons hitting a detector).
        plot : boolean, optional, default: False
            Plot the generated shot noise signal.

    Returns:
        signal : 1-D ndarray
            Generated shot noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> snoise = generate_shot_noise(length_seconds=5,
        >>>                               sampling_rate=44100,
        >>>                               rate=1000,
        >>>                               plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate shot noise using numpy
    time_intervals = np.random.exponential(scale=1 / rate, size=npnts)
    event_times = np.cumsum(time_intervals)
    signal = np.zeros(npnts)
    for event_time in event_times:
        sample_index = int(event_time * sampling_rate)
        if 0 <= sample_index < npnts:
            signal[sample_index] += 1

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, signal)
        plt.title('Shot Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return signal


def flicker_noise(length_seconds, sampling_rate, exponent, amplitude=1.0, plot=False):
    r"""
    Generate flicker noise (I/f noise) signal.

    Args:
        length_seconds : float
            Duration of the flicker noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        exponent : float
            Exponent of the flicker noise power spectrum.
        amplitude : float, optional, default: 1.0
            Amplitude of the flicker noise.
        plot : boolean, optional, default: False
            Plot the generated flicker noise signal.

    Returns:
        signal : 1-D ndarray
            Generated flicker noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> fn = generate_flicker_noise(length_seconds=5,
        >>>                             sampling_rate=44100,
        >>>                             exponent=0.5,
        >>>                             amplitude=0.1,
        >>>                             plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples

    # Generate flicker noise using numpy and power-law scaling
    freqs = np.fft.fftfreq(npnts, d=1 / sampling_rate)
    power_spectrum = 1 / np.abs(freqs) ** exponent
    spectrum = np.sqrt(power_spectrum) * np.exp(1j * np.angle(np.fft.fft(np.random.normal(0, 1, npnts))))
    flicker_noise = np.real(np.fft.ifft(spectrum))

    # Normalize and apply amplitude
    flicker_noise /= np.max(np.abs(flicker_noise))
    flicker_noise *= amplitude

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, flicker_noise)
        plt.title('Flicker Noise (I/f Noise)')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return flicker_noise


def burst_noise(length_seconds, sampling_rate, burst_rate, burst_duration, amplitude_range, plot=False):
    r"""
    Generate burst noise signal.

    Args:
        length_seconds : float
            Duration of the burst noise signal in seconds.
        sampling_rate : int
            The sampling rate of the signal.
        burst_rate : float
            Average rate of occurrence of burst events (e.g., bursts of interference).
        burst_duration : float
            Duration of each burst in seconds.
        amplitude_range : tuple or list of two floats
            Range of amplitudes (min, max) for the bursts.
        plot : boolean, optional, default: False
            Plot the generated burst noise signal.

    Returns:
        signal : 1-D ndarray
            Generated burst noise signal, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> bnoise = generate_burst_noise(length_seconds=5,
        >>>                                sampling_rate=44100,
        >>>                                burst_rate=0.1,
        >>>                                burst_duration=0.2,
        >>>                                amplitude_range=(0.5, 1.0),
        >>>                                plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples
    burst_samples = int(sampling_rate * burst_duration)

    # Generate burst noise using numpy
    signal = np.zeros(npnts)
    burst_times = np.random.exponential(scale=1 / burst_rate, size=npnts)
    burst_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], npnts)
    current_sample = 0
    for i in range(npnts):
        if current_sample <= i < current_sample + burst_samples:
            signal[i] = burst_amplitudes[i]
        if i >= current_sample + burst_samples:
            current_sample += int(burst_times[i] * sampling_rate)

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, signal)
        plt.title('Burst Noise')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return signal


def atmospheric_noise(length_seconds, sampling_rate, frequency_range, plot=False):
    r"""
    Generate atmospheric noise (sferics) simulation using random noise.

    Args:
        length_seconds : float
            Duration of the atmospheric noise simulation in seconds.
        sampling_rate : int
            The sampling rate of the simulation.
        frequency_range : tuple or list of two floats
            Frequency range of the atmospheric noise simulation (start, end) in Hz.
        plot : boolean, optional, default: False
            Plot the generated atmospheric noise simulation.

    Returns:
        signal : 1-D ndarray
            Generated atmospheric noise simulation, a numpy array of length `sampling_rate * length_seconds`.
    Usage:
        >>> atmospheric_noise = generate_atmospheric_noise(length_seconds=5,
        >>>                                                sampling_rate=44100,
        >>>                                                frequency_range=(0, 20000),
        >>>                                                plot=True)
    """
    npnts = int(sampling_rate * length_seconds)  # Number of time samples
    frequency_start, frequency_end = frequency_range

    # Generate random noise within the specified frequency range
    t = np.arange(0, npnts) / sampling_rate
    signal = np.random.normal(0, 1, npnts)
    signal *= np.sin(2 * np.pi * np.linspace(frequency_start, frequency_end, npnts))

    if plot:
        time = np.arange(0, npnts) / sampling_rate
        plt.plot(time, signal)
        plt.title('Atmospheric Noise (Sferics) Simulation')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    return signal


def echo_effect(signal, delay_seconds, attenuation, plot=False):
    r"""
    Generate an echo effect by delaying and attenuating the input signal.

    Args:
        signal : 1-D ndarray
            Input signal to which the echo effect is applied.
        delay_seconds : float
            Delay time for the echo in seconds.
        attenuation : float
            Attenuation factor for the echo (e.g., 0.5 for -6 dB attenuation).
        plot : boolean, optional, default: False
            Plot the original and echoed signals.

    Returns:
        echoed_signal : 1-D ndarray
            Output signal with the applied echo effect.
    Usage:
        >>> original_signal = np.random.normal(0, 1, 44100)
        >>> echoed_signal = generate_echo_effect(original_signal,
        >>>                                      delay_seconds=0.5,
        >>>                                      attenuation=0.5,
        >>>                                      plot=True)
    """
    npnts = len(signal)  # Number of time samples

    # Generate the echoed signal using delay and attenuation
    delay_samples = int(delay_seconds * len(signal))
    echoed_signal = np.zeros(npnts + delay_samples)
    echoed_signal[delay_samples:] = signal
    echoed_signal[:npnts] += attenuation * signal

    if plot:
        time = np.arange(0, len(echoed_signal)) / len(echoed_signal)
        plt.plot(time, signal, label='Original Signal')
        plt.plot(time, echoed_signal, label='Echoed Signal')
        plt.title('Echo Effect Simulation')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    return echoed_signal


def load_scg(noise_level, train_or_test: str):
    """
    Load SCG (SeismoCardioGram) data with specified noise level and training/testing mode.

    Args:
        noise_level (float): The level of noise in the data (0, 0.1, or 0.8).
        train_or_test (str): Either 'train' or 'test' mode.

    Returns:
        signals (numpy.ndarray): Loaded SCG signals.
        labels (numpy.ndarray): Loaded labels (ID + Time + H + R + S + D).
        duration (int): Duration of the data in seconds (10 s).
        fs (int): Sampling frequency of the data (100 Hz).
    """
    # Check if the provided train_or_test is valid
    if train_or_test.lower() not in ['train', 'test']:
        raise ValueError("Please make sure it is either 'train' or 'test'!")

    # Check if the provided noise_level is valid
    if noise_level not in [0, 0.1, 0.8]:
        raise ValueError("Now, we only support noise levels 0, 0.1, and 0.8")

    n_samples, S_start, S_end = 0, 0, 0

    # Set the number of samples and range based on train_or_test mode
    if train_or_test.lower() == 'train':
        n_samples = 100
        S_start, S_end = 90, 140
    elif train_or_test.lower() == 'test':
        n_samples = 100
        S_start, S_end = 140, 180

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    filepath = 'sim_{}_{}_{}_{}_{}.npy'.format(n_samples, noise_level, S_start, S_end, train_or_test)

    # Combine the current file's directory and the constructed file path
    file_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), 'data', filepath)

    # Load data from the constructed file path
    data = np.load(file_path)

    # Split loaded data into signals and labels
    signals, labels = data[:, :1000], data[:, 1000:]

    # Set duration and sampling frequency
    duration = 10
    fs = 100

    return signals, labels, duration, fs


def load_scg_template(noise_level, train_or_test: str):
    if train_or_test.lower() not in ['train', 'test']:
        raise ValueError("Please make sure it is either 'train' or 'test'!")

    # Check if the provided noise_level is valid
    if noise_level not in [0.1]:
        raise ValueError("Now, we only support noise levels 0.1")

    n_samples, S_start, S_end = 0, 0, 0
    # Set the number of samples and range based on train_or_test mode

    S_start, S_end = 90, 180
    if train_or_test.lower() == 'train':
        n_samples = 5000
    elif train_or_test.lower() == 'test':
        n_samples = 3000
    filepath = 'sim_{}_0.1_{}_{}_{}_template.npy'.format(n_samples, S_start, S_end, train_or_test)

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Combine the current file's directory and the constructed file path
    file_path = os.path.join(os.path.dirname(current_file_path), 'classification_S', filepath)

    # Load data from the constructed file path
    data = np.load(file_path, allow_pickle=True)

    signals = []
    labels = []

    for piece in data:
        piece_np = np.array(piece)
        signals.append(piece_np[:-6])
        labels.append(piece_np[-6:])

    # Set duration and sampling frequency
    duration = 10
    fs = 100

    return signals, labels, duration, fs


def cal_corrcoef(signal1, signal2):
    """
    Description:
        To get the correlate coefficient

    Input:
        Two signal with same length

    Return:
        The correlate coefficient
    """
    return np.corrcoef(signal1, signal2)[0, 1]


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
    signal2 = signal[:len(signal) - lag]
    return np.corrcoef(signal1, signal2)[0, 1]


def cal_autocorr(signal, plot=False):
    """
    Description:
        To get the auto correlate coefficient

    Input:
        One signal

    Return:
        The serial correlate coefficient with different lag which is from 0 to len(wave)//2
    """
    lags = range(len(signal) // 2)
    corrs = [cal_serial_corr(signal, lag) for lag in lags]
    if plot:
        plt.plot(lags, corrs)
        plt.show()
    return lags, corrs






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

    z = hilbert(signal)  # form the analytical signal
    inst_amplitude = np.abs(z)  # envelope extraction
    inst_phase = np.unwrap(np.angle(z))  # inst phase
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs  # inst frequency

    # Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(inst_phase)

    return inst_amplitude, inst_freq, inst_phase, regenerated_carrier


def get_template(signal):
    """
    Description:
        use cluster method to get the template
    Args:
        signal: the periodic signal
    Returns:
        The template of the periodic signal
    """

    peaks2 = get_peaks(signal)

    avg_index = (peaks2[::2] + peaks2[1::2]) // 2

    # 使用这些平均数作为x的下标，将x切割成多个部分
    splits = np.split(signal, avg_index)

    max_length = max(len(split) for split in splits)

    # 补充每个部分使其长度相等
    padded_splits = [np.pad(split, (0, max_length - len(split))) for split in splits]

    # 将这些部分堆叠成一个二维数组
    stacked_array = np.vstack(padded_splits)
    stacked_array = np.delete(stacked_array, 0, axis=0)

    class PulseClustering:
        def __init__(self, threshold):
            self.threshold = threshold
            self.clusters = []

        def fit(self, pulses):
            for pulse in pulses:
                if not self.clusters:  # 如果聚类为空，创建第一个聚类
                    self.clusters.append([pulse])
                else:
                    for cluster in self.clusters:
                        center_pulse = np.mean(cluster, axis=0)  # 计算聚类中心
                        rmse = np.sqrt(mean_squared_error(center_pulse, pulse))  # 计算RMSE
                        if rmse < self.threshold:  # 如果RMSE低于阈值，将脉冲添加到聚类中
                            cluster.append(pulse)
                            break
                    else:  # 如果脉冲与现有的所有聚类的中心的RMSE都高于阈值，创建新的聚类
                        self.clusters.append([pulse])

        def get_clusters(self):
            return self.clusters

    threshold = 0.000005  # 这是一个选择的阈值

    clustering = PulseClustering(threshold)
    clustering.fit(stacked_array)
    clusters = clustering.get_clusters()

    num_pulses_per_cluster = [len(cluster) for cluster in clusters]

    max_cluster = max(clusters, key=len)

    # 计算最大聚类的平均脉冲
    average_pulse = np.mean(max_cluster, axis=0)
    return average_pulse














def tsfel_feature(signal, fs=100):
    with open("../json/all_features.json", 'r') as file:
        cgf_file = json.load(file)

    # cgf_file = tsfel.get_features_by_domain("temporal")

    features = tsfel.time_series_features_extractor(cgf_file, signal, fs=fs, window_size=len(signal),
                                                    features_path="../utils/my_features.py").values.flatten()

    return features


# Below is the needed function for chirplet transform
class FCT:
    """
    Attributes :
        _duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
        _num_octaves : the number of octaves
        _num_chirps_by_octave : the number of chirps by octave
        _polynome_degree : degree of the polynomial function
        _end_smoothing : define the size the output of the signal
        _samplerate : samplerate of the signal

    """

    def __init__(self,
                 duration_longest_chirplet=1,
                 num_octaves=5,
                 num_chirps_by_octave=10,
                 polynome_degree=0,
                 end_smoothing=0.001,
                 sample_rate=22050):
        """
        Args:
            duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
            num_octaves : the number of octaves
            num_chirps_by_octave : the number of chirps by octave
            polynome_degree : degree of the polynomial function
            end_smoothing : define the size the output of the signal
            sample_rate : samplerate of the signal
        """
        self._duration_longest_chirplet = duration_longest_chirplet

        self._num_octaves = num_octaves

        self._num_chirps_by_octave = num_chirps_by_octave

        self._polynome_degree = polynome_degree

        self._end_smoothing = end_smoothing

        # Samplerate of the signal. Has to be defined in advance.
        self._samplerate = sample_rate

        self._chirps = self.__init_chirplet_filter_bank()

    def __init_chirplet_filter_bank(self):
        """generate all the chirplets based on the attributes

        Returns :
            The bank of chirplets
        """
        num_chirps = self._num_octaves * self._num_chirps_by_octave

        # create a list of coefficients based on attributes
        lambdas = 2.0 ** (1 + arange(num_chirps) / float(self._num_chirps_by_octave))

        # Low frequencies for a signal
        start_frequencies = (self._samplerate / lambdas) / 2.0

        # high frequencies for a signal
        end_frequencies = self._samplerate / lambdas

        durations = 2.0 * self._duration_longest_chirplet / flipud(lambdas)

        chirplets = list()
        for low_frequency, high_frequency, duration in zip(start_frequencies, end_frequencies, durations):
            chirplets.append(Chirplet(self._samplerate, low_frequency, high_frequency, duration, self._polynome_degree))
        return chirplets

    @property
    def time_bin_duration(self):
        """
        Return :
            The time bin duration

        """
        return self._end_smoothing * 10

    def compute(self, input_signal):
        """compute the FCT on the given signal
        Args :
            input_signal : Array of an audio signal

        Returns :
            The Fast Chirplet Transform of the given signal

        """
        # keep the real length of the signal
        size_data = len(input_signal)

        nearest_power_2 = 2 ** (size_data - 1).bit_length()

        # find the best power of 2
        # the signal must not be too short

        while nearest_power_2 <= self._samplerate * self._duration_longest_chirplet:
            nearest_power_2 *= 2

        # pad with 0 to have the right length of signal

        data = np.lib.pad(input_signal, (0, nearest_power_2 - size_data), 'constant', constant_values=0)

        # apply the fct to the adapted length signal

        chirp_transform = apply_filterbank(data, self._chirps, self._end_smoothing)

        # resize the signal to the right length

        chirp_transform = resize_chirps(size_data, nearest_power_2, chirp_transform)

        return chirp_transform


def resize_chirps(size_data, size_power_2, chirps):
    """Resize the matrix of chirps to the length of the signal
    Args:
        size_data : number of samples of the audio signal
        size_power_2 : number of samples of the signal to apply the FCT
        chirps : the signal to resize
    Returns :
        Chirps with the correct length
    """
    size_chirps = len(chirps)
    ratio = size_data / size_power_2
    size = int(ratio * len(chirps[0]))

    resize_chirps = np.zeros((size_chirps, size))
    for i in range(0, size_chirps):
        resize_chirps[i] = chirps[i][0:size]
    return resize_chirps


class Chirplet:
    """chirplet class
    Attributes:
        _min_frequency : lowest frequency where the chirplet is applied
        _max_frequency : highest frequency where the chirplet is applied
        _duration : duration of the chirp
        _samplerate : samplerate of the signal
        _polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        _filter_coefficients : coefficients applied to the signal
    """

    def __init__(self, samplerate, min_frequency, max_frequency, sigma, polynome_degree):

        """
        Args :
            samplerate : samplerate of the signal
            min_frequency : lowest frequency where the chirplet is applied
            max_frequency : highest frequency where the chirplet is applied
            duration : duration of the chirp
            polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        """
        self._min_frequency = min_frequency

        self._max_frequency = max_frequency

        self._duration = sigma / 10

        self._samplerate = samplerate

        self._polynome_degree = polynome_degree

        self._filter_coefficients = self.calcul_coefficients()

    def calcul_coefficients(self):
        """calculate coefficients for the chirplets
        Returns :
            apodization coeeficients
        """
        num_coeffs = linspace(0, self._duration, int(self._samplerate * self._duration))

        if self._polynome_degree:
            temp = (self._max_frequency - self._min_frequency)
            temp /= ((
                                 self._polynome_degree + 1) * self._duration ** self._polynome_degree) * num_coeffs ** self._polynome_degree + self._min_frequency
            wave = cos(2 * pi * num_coeffs * temp)
        else:
            temp = (self._min_frequency * (self._max_frequency / self._min_frequency) ** (
                        num_coeffs / self._duration) - self._min_frequency)
            temp *= self._duration / log(self._max_frequency / self._min_frequency)
            wave = cos(2 * pi * temp)

        coeffs = wave * hanning(len(num_coeffs)) ** 2

        return coeffs

    def smooth_up(self, input_signal, thresh_window, end_smoothing):
        """generate fast fourier transform from a signal and smooth it
        Params :
            input_signal : audio signal
            thresh_window : relative to the size of the windows
            end_smoothing : relative to the length of the output signal
        Returns :
            fast Fourier transform of the audio signal applied to a specific domain of frequencies
        """
        windowed_fft = build_fft(input_signal, self._filter_coefficients, thresh_window)
        return fft_smoothing(fabs(windowed_fft), end_smoothing)


def apply_filterbank(input_signal, chirplets, end_smoothing):
    """generate list of signal with chirplets
    Params :
        input_signal : audio signal
        chirplets : the chirplet bank
        end_smoothing : relative to the length of the output signal
    Returns :
        fast Fourier transform of the signal to all the frequency domain
    """
    fast_chirplet_transform = list()

    for chirplet in chirplets:
        chirp_line = chirplet.smooth_up(input_signal, 6, end_smoothing)
        fast_chirplet_transform.append(chirp_line)

    return np.array(fast_chirplet_transform)


def fft_smoothing(input_signal, sigma):
    """smooth the fast transform Fourier
    Params :
        input_signal : audio signal
        sigma : relative to the length of the output signal
    Returns :
        a shorter and smoother signal

    """
    size_signal = input_signal.size

    # shorten the signal
    new_size = int(floor(10.0 * size_signal * sigma))
    half_new_size = new_size // 2

    fftx = fft(input_signal)

    short_fftx = []
    for ele in fftx[:half_new_size]:
        short_fftx.append(ele)

    for ele in fftx[-half_new_size:]:
        short_fftx.append(ele)

    apodization_coefficients = generate_apodization_coeffs(half_new_size, sigma, size_signal)

    # apply the apodization coefficients
    short_fftx[:half_new_size] *= apodization_coefficients
    short_fftx[half_new_size:] *= flipud(apodization_coefficients)

    realifftxw = ifft(short_fftx).real
    return realifftxw


def generate_apodization_coeffs(num_coeffs, sigma, size):
    """generate apodization coefficients
    Params :
        num_coeffs : number of coefficients
        sigma : relative to the length of the output signal
        size : size of the signal
    Returns :
        apodization coefficients

    """
    apodization_coefficients = arange(num_coeffs)
    apodization_coefficients = apodization_coefficients ** 2
    apodization_coefficients = apodization_coefficients / (2 * (sigma * size) ** 2)
    apodization_coefficients = exp(-apodization_coefficients)
    return apodization_coefficients


def fft_based(input_signal, filter_coefficients, boundary=0):
    """applied fft if the signal is too short to be splitted in windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        boundary : manage the bounds of the signal
    Returns :
        audio signal with application of fast Fourier transform
    """
    num_coeffs = filter_coefficients.size
    half_size = num_coeffs // 2

    if boundary == 0:  # ZERO PADDING
        input_signal = np.lib.pad(input_signal, (half_size, half_size), 'constant', constant_values=0)
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size - num_coeffs), 'constant',
                                         constant_values=0)
        newx = ifft(fft(input_signal) * fft(filter_coefficients))
        return newx[num_coeffs - 1:-1]

    elif boundary == 1:  # symmetric
        input_signal = np.concatenate(
            [flipud(input_signal[:half_size]), input_signal, flipud(input_signal[half_size:])])
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size - num_coeffs), 'constant',
                                         constant_values=0)
        newx = ifft(fft(input_signal) * fft(filter_coefficients))
        return newx[num_coeffs - 1:-1]

    else:  # periodic
        return roll(ifft(fft(input_signal) * fft(filter_coefficients, input_signal.size)), -half_size).real


def build_fft(input_signal, filter_coefficients, threshold_windows=6, boundary=0):
    """generate fast transform fourier by windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        threshold_windows : calcul the size of the windows
        boundary : manage the bounds of the signal
    Returns :
        fast Fourier transform applied by windows to the audio signal

    """
    num_coeffs = filter_coefficients.size
    # print(n,boundary,M)
    half_size = num_coeffs // 2
    signal_size = input_signal.size
    # power of 2 to apply fast fourier transform
    windows_size = 2 ** ceil(log2(num_coeffs * (threshold_windows + 1)))
    number_of_windows = floor(signal_size // windows_size)

    if number_of_windows == 0:
        return fft_based(input_signal, filter_coefficients, boundary)

    windowed_fft = empty_like(input_signal)
    # pad with 0 to have a size in a power of 2
    windows_size = int(windows_size)

    zeropadding = np.lib.pad(filter_coefficients, (0, windows_size - num_coeffs), 'constant', constant_values=0)

    h_fft = fft(zeropadding)

    # to browse the whole signal
    current_pos = 0

    # apply fft to a part of the signal. This part has a size which is a power
    # of 2
    if boundary == 0:  # ZERO PADDING

        # window is half padded with since it's focused on the first half
        window = input_signal[current_pos:current_pos + windows_size - half_size]
        zeropaddedwindow = np.lib.pad(window, (len(h_fft) - len(window), 0), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)

    elif boundary == 1:  # SYMMETRIC
        window = np.concatenate(
            [flipud(input_signal[:half_size]), input_signal[current_pos:current_pos + windows_size - half_size]])
        x_fft = fft(window)

    else:
        x_fft = fft(input_signal[:windows_size])

    windowed_fft[:windows_size - num_coeffs] = (ifft(x_fft * h_fft)[num_coeffs - 1:-1]).real

    current_pos += windows_size - num_coeffs - half_size
    # apply fast fourier transofm to each windows
    while current_pos + windows_size - half_size <= signal_size:
        x_fft = fft(input_signal[current_pos - half_size:current_pos + windows_size - half_size])
        # Suppress the warning, work on the real/imagina
        windowed_fft[current_pos:current_pos + windows_size - num_coeffs] = (
        ifft(x_fft * h_fft)[num_coeffs - 1:-1]).real
        current_pos += windows_size - num_coeffs
    # print(countloop)
    # apply fast fourier transform to the rest of the signal
    if windows_size - (signal_size - current_pos + half_size) < half_size:

        window = input_signal[current_pos - half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size - (signal_size - current_pos + half_size))),
                                      'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = roll(ifft(x_fft * h_fft), half_size)[
                                     half_size:half_size + windowed_fft.size - current_pos].real
        windowed_fft[-half_size:] = convolve(input_signal[-num_coeffs:], filter_coefficients, 'same')[-half_size:]
    else:

        window = input_signal[current_pos - half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size - (signal_size - current_pos + half_size))),
                                      'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = ifft(x_fft * h_fft)[
                                     num_coeffs - 1:num_coeffs + windowed_fft.size - current_pos - 1].real

    return windowed_fft


# chirplet transform function ends here












# # 定义Chirplet函数
# def chirplet(alpha, beta, gamma, t):
#     return np.exp(1j * (alpha * t ** 2 + beta * t + gamma))
#
#
# # 定义PCT计算函数
# def polynomial_chirplet_transform(signal, alpha, beta, gamma):
#     n = len(signal)
#     t = np.arange(n)
#     pct_result = np.zeros(n, dtype=complex)
#
#     for i in range(n):
#         t_shifted = t - t[i]
#         chirplet_function = chirplet(alpha, beta, gamma, t_shifted)
#         pct_result[i] = np.sum(signal * chirplet_function)
#
#     return np.abs(pct_result)  # 提取振幅信息

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
