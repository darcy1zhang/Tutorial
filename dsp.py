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
## Template of SCG
def get_template(signal, threshold=0.000005):
    """
    Description:
        use cluster method to get the template
    Args:
        signal: the periodic signal
    Returns:
        The template of the periodic signal
    """
    peaks2 = get_peaks(signal)
    peaks2 = peaks2[1:-1]
    avg_index = (peaks2[::2] + peaks2[1::2]) // 2  # Use the mid of peaks as segment point
    splits = np.split(signal, avg_index)

    # Use the longest length as the length of template
    max_length = max(len(split) for split in splits)
    padded_splits = [np.pad(split, (0, max_length - len(split))) for split in splits]

    # Stack the segments
    stacked_array = np.vstack(padded_splits)
    stacked_array = np.delete(stacked_array, 0, axis=0)

    class PulseClustering:
        def __init__(self, threshold):
            self.threshold = threshold
            self.clusters = []

        def fit(self, pulses):
            for pulse in pulses:
                if not self.clusters:
                    self.clusters.append([pulse])
                else:
                    for cluster in self.clusters:
                        center_pulse = np.mean(cluster, axis=0)  # Use average to get the middle of the cluster
                        rmse = np.sqrt(mean_squared_error(center_pulse, pulse))  # Calculate RMSE distance
                        # If the distance between new signal and middle of cluster is less than shreshold, add it into
                        # the cluster
                        if rmse < self.threshold:
                            cluster.append(pulse)
                            break
                    # If the distance between new singal and middles of existing clusters is greater than shreshold,
                    # create a new cluster
                    else:
                        self.clusters.append([pulse])

        def get_clusters(self):
            return self.clusters

    clustering = PulseClustering(threshold)
    clustering.fit(stacked_array)
    clusters = clustering.get_clusters()
    num_pulses_per_cluster = [len(cluster) for cluster in clusters]
    max_cluster = max(clusters, key=len)
    average_pulse = np.mean(max_cluster, axis=0)  # Calculate the average of max cluster
    return average_pulse

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
