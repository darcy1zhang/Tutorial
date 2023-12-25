from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os


# DSP Functions
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



# Singal Generation

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


# Other Functions

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
    file_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), '../data', filepath)

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