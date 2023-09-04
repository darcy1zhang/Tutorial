import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def compute_linear_spectrogram(signal, sample_rate, window_size=512, overlap=0.5, window = "hann"):
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



if __name__ == "__main__":
    import numpy as np
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data", signal_name)
    signal = np.load(signal_path)[2,:1000]
    fs = 100

    freqs, times, spectrogram = compute_linear_spectrogram(signal, fs, 10)


    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, freqs, spectrogram, shading='auto')
    plt.title('Linear Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.show()
