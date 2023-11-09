import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import convolve
import scipy.signal
import tftb
import os

def get_wvd(signal, fs, T):
    """
    Description:
        Analyze the time-frequency characteristics of a signal using the Wigner-Ville Transform (WVT) and visualize the results.

    Params:
        signal (numpy.ndarray): The input signal.
        fs (float): The sampling frequency of the signal.
        T (float): Time of the signal

    Returns:
        tfr_wvd (numpy.ndarray): The time-frequency representation (WVD) of the signal.
        t_wvd (numpy.ndarray): Time values corresponding to the WVD.
        f_wvd (numpy.ndarray): Normalized frequency values corresponding to the WVD.
    """

    t = np.arange(0, T, 1.0 / fs)

    # Doing the WVT
    wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=t)
    tfr_wvd, t_wvd, f_wvd = wvd.run()


    return tfr_wvd, t_wvd, f_wvd

def STFT(signal, sample_rate, window_size=512, overlap=0.5, window = "hann"):
    """
    Description:
        Compute the Linear Spectrogram of a signal using Short-time Fourier Transform (STFT).

    Params:
        signal (numpy.ndarray): The input signal.
        sample_rate (int): The sample rate of the signal.
        window_size (int, optional): The size of the analysis window in samples. Default is 512.
        overlap (float, optional): The overlap between successive windows, as a fraction of the window size. Default is 0.5.

    Returns:
        freqs (numpy.ndarray): The frequency values in Hz.
        times (numpy.ndarray): The time values in seconds.
        spectrogram (numpy.ndarray): The computed linear spectrogram.
    """

    # Compute the Linear Spectrogram using scipy.signal.spectrogram
    frequencies, times, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, window='hann', nperseg=window_size, noverlap=int(overlap * window_size))

    return frequencies, times, 10 * np.log10(Sxx)  # Convert to dB for better visualization


# 定义Chirplet函数
def chirplet(alpha, beta, gamma, t):
    return np.exp(1j * (alpha * t ** 2 + beta * t + gamma))


# 定义PCT计算函数
def polynomial_chirplet_transform(signal, alpha, beta, gamma):
    n = len(signal)
    pct_result = np.zeros(n, dtype=complex)

    for i in range(n):
        t_shifted = t - t[i]
        chirplet_function = chirplet(alpha, beta, gamma, t_shifted)
        pct_result[i] = np.sum(signal * chirplet_function)

    return np.abs(pct_result)  # 提取振幅信息

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

    sample_rate = 100
    t = np.arange(0, 10, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

    # 设置Chirplet参数
    alpha = 0.1
    beta = 0.5
    gamma = 0.0

    # 执行PCT
    pct_result = polynomial_chirplet_transform(signal, alpha, beta, gamma)

    # 绘制PCT结果
    plt.figure(figsize=(12, 6))

    # 信号图
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # PCT结果图
    plt.subplot(2, 1, 2)
    plt.plot(t, pct_result)
    plt.title('Polynomial Chirplet Transform Result')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()