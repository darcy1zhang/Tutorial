#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, \
    exp, cos, sin, polyval, polyint

#%% function script
def generate_sine_waves(length_seconds, sampling_rate, frequencies, func="sin", add_noise=0, plot=False):
    r"""
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
    npnts = int(sampling_rate*length_seconds)  # number of time samples
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros((frequencies.shape[0],npnts))

    for channel in range(0,frequencies.shape[0]):
        for fi in frequencies[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2*np.pi*fi*time)
            else:
                signal[channel] = signal[channel] + np.sin(2*np.pi*fi*time)

        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2*(signal[channel]-min)/(max-min))-1

    if add_noise:
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.title('Signal with sampling rate of '+str(sampling_rate)+', lasting '+str(length_seconds)+'-seconds')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    if expanded:
        signal = signal[0]

    return signal

#%%
def generate_triangle_waves(t, width=1):
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
    t, w = asarray(t), asarray(width)
    w = asarray(w + (t - t))
    t = asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    place(y, mask1, nan)

    # take t modulo 2*pi
    tmod = mod(t, 2 * pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    tsub = extract(mask2, tmod)
    wsub = extract(mask2, w)
    place(y, mask2, tsub / (pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))

    mask3 = (1 - mask1) & (1 - mask2)
    tsub = extract(mask3, tmod)
    wsub = extract(mask3, w)
    place(y, mask3, (pi * (wsub + 1) - tsub) / (pi * (1 - wsub)))
    return y


#%%
def generate_square_waves(t, duty=0.5):
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
    t, w = asarray(t), asarray(duty)
    w = asarray(w + (t - t))
    t = asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    place(y, mask1, nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = mod(t, 2 * pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    place(y, mask3, -1)
    return y


#%%
def generate_chirp_waves(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
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
    phi *= pi / 180
    return cos(phase + phi)

def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    """
    Calculate the phase used by `chirp` to generate its output.

    See `chirp` for a description of the arguments.

    """
    t = asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    if method in ['linear', 'lin', 'li']:
        beta = (f1 - f0) / t1
        phase = 2 * pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            phase = 2 * pi * (f0 * t + beta * t ** 3 / 3)
        else:
            phase = 2 * pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            phase = 2 * pi * f0 * t
        else:
            beta = t1 / log(f1 / f0)
            phase = 2 * pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            # Degenerate case: constant frequency.
            phase = 2 * pi * f0 * t
        else:
            # Singular point: the instantaneous frequency blows up
            # when t == sing.
            sing = -f1 * t1 / (f0 - f1)
            phase = 2 * pi * (-sing * f0) * log(np.abs(1 - t/sing))

    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                         " or 'hyperbolic', but a value of %r was given."
                         % method)

    return phase

#%% main script
if __name__ == '__main__':

    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, \
    exp, cos, sin, polyval, polyint

    t = np.linspace(0, 10, 1500)

    # data = generate_sine_waves(length_seconds=10,
    #          sampling_rate=100,
    #          frequencies=[1],
    #         #  add_noise=0.5,
    #          plot=False
    #      )
    
    # data = generate_triangle_waves(
    #     t = 2 * np.pi * 5 * t,
    #     width = 0.5
    # )

    # data = generate_square_waves(
    #     2 * np.pi * 5 * t
    # )

    data = generate_chirp_waves(
    t, f0=6, f1=1, t1=10, method='linear')

    print(data.shape)
    plt.figure(figsize=(20, 4))
    plt.title("simulated sine waves")
    # plt.plot (data1)
    # plt.plot (data2)
    plt.plot (data)
    plt.show()
# %%
