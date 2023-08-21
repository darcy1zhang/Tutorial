#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, \
    exp, cos, sin, polyval, polyint

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_white_noise(length_seconds, sampling_rate, plot=False):
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_band_limited_white_noise(length_seconds, sampling_rate, frequency_range, plot=False):
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_pink_noise(length_seconds, sampling_rate, plot=False):
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

#%%
def generate_brown_noise(length_seconds, sampling_rate, plot=False):
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


#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_impulsive_noise(length_seconds, sampling_rate, probability, plot=False):
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

#%%
def generate_click_noise(length_seconds, sampling_rate, position, amplitude=1.0, plot=False):
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_transient_noise_pulses(length_seconds, sampling_rate, num_pulses, pulse_duration, amplitude_range, plot=False):
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_thermal_noise(length_seconds, sampling_rate, noise_density, plot=False):
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_shot_noise(length_seconds, sampling_rate, rate, plot=False):
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
    time_intervals = np.random.exponential(scale=1/rate, size=npnts)
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


#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_flicker_noise(length_seconds, sampling_rate, exponent, amplitude=1.0, plot=False):
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
    freqs = np.fft.fftfreq(npnts, d=1/sampling_rate)
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

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_burst_noise(length_seconds, sampling_rate, burst_rate, burst_duration, amplitude_range, plot=False):
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
    burst_times = np.random.exponential(scale=1/burst_rate, size=npnts)
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


#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_atmospheric_noise(length_seconds, sampling_rate, frequency_range, plot=False):
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

#%% Echo and Multi-path Reflections
import numpy as np
import matplotlib.pyplot as plt

def generate_echo_effect(signal, delay_seconds, attenuation, plot=False):
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




#%%
if __name__ == "__main__":

    # wn = generate_white_noise(length_seconds=5, sampling_rate=44100, plot=True)
    blwn = generate_band_limited_white_noise(length_seconds=5, sampling_rate=44100, frequency_range=(20, 2000), plot=True)

# %%
