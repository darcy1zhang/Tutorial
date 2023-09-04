import math
import random

import numpy as np
import scipy

from ..signal import signal_distort, signal_resample
import matplotlib.pyplot as plt


def scg_simulate(
        duration=10, length=None, sampling_rate=100, noise=0.01, heart_rate=60, heart_rate_std=1, respiratory_rate=15,
        systolic=120, diastolic=80, method="simple", random_state=None
):
    """Simulate an scg/EKG signal.

    Generate an artificial (synthetic) scg signal of a given duration and sampling rate using either
    the scgSYN dynamical model (McSharry et al., 2003) or a simpler model based on Daubechies wavelets
    to roughly approximate cardiac cycles.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute). The default is 70. Note that for the
        scgSYN method, random fluctuations are to be expected to mimick a real heart rate. These
        fluctuations can cause some slight discrepancies between the requested heart rate and the
        empirical heart rate, especially for shorter signals.
    heart_rate_std : int
        Desired heart rate standard deviation (beats per minute).
    method : str
        The model used to generate the signal. Can be 'simple' for a simulation based on Daubechies
        wavelets that roughly approximates a single cardiac cycle. If 'scgsyn' (default), will use an
        advanced model desbribed `McSharry et al. (2003) <https://physionet.org/content/scgsyn/>`_.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    array
        Vector containing the scg signal.

    Examples
    ----------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> scg1 = nk.scg_simulate(duration=10, method="simple")
    >>> scg2 = nk.scg_simulate(duration=10, method="scgsyn")
    >>> pd.DataFrame({"scg_Simple": scg1,
    ...               "scg_Complex": scg2}).plot(subplots=True) #doctest: +ELLIPSIS
    array([<AxesSubplot:>, <AxesSubplot:>], dtype=object)

    See Also
    --------
    rsp_simulate, eda_simulate, ppg_simulate, emg_simulate


    References
    -----------
    - McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A. (2003). A dynamical model for
    generating synthetic electrocardiogram signals. IEEE transactions on biomedical engineering, 50(3), 289-294.
    - https://github.com/diarmaidocualain/scg_simulation

    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Run appropriate method

    if method.lower() in ["simple", "daubechies"]:
        # print("method is:", method)
        scg = _scg_simulate_daubechies(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate,
            respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic
        )
    # else:
    #     # print("method is:", method)
    #     approx_number_beats = int(np.round(duration * (heart_rate / 60)))
    #     scg = _scg_simulate_scgsyn(
    #         sfscg=sampling_rate,
    #         N=approx_number_beats,
    #         Anoise=0,
    #         hrmean=heart_rate,
    #         hrstd=heart_rate_std,
    #         lfhfratio=0.5,
    #         sfint=sampling_rate,
    #         ti=(-70, -15, 0, 15, 100),
    #         ai=(1.2, -5, 30, -7.5, 0.75),
    #         bi=(0.25, 0.1, 0.1, 0.1, 0.4),
    #     )
    #     # Cut to match expected length
    #     scg = scg[0:length]

    # Add random noise
    if noise > 0:
        scg = signal_distort(
            scg,
            sampling_rate=sampling_rate,
            noise_amplitude=noise,
            noise_frequency=[5, 10, 100],
            noise_shape="laplace",
            random_state=random_state,
            silent=True,
        )

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return scg


# =============================================================================
# Daubechies
# =============================================================================
def _scg_simulate_daubechies(duration=10, length=None, sampling_rate=100, heart_rate=70, respiratory_rate=15,
                             systolic=120, diastolic=80):
    """Generate an artificial (synthetic) scg signal of a given duration and sampling rate.

    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/scg_simulation>`_.

    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # # print(p)
    # # min_p = 9 max_p = 34
    # cardiac_s = scipy.signal.wavelets.daub(int(p))
    # cardiac_d = scipy.signal.wavelets.daub(int(p)) * (diastolic/systolic)
    # print(f"cardiac_s: {len(cardiac_s)}, cardiac_d: {len(cardiac_d)}")

    # cardiac_s = scipy.signal.wavelets.daub(int(systolic/10)) * int(math.sqrt(pow(systolic,2)+pow(heart_rate,2)))
    # # print("cardiac_s:", len(cardiac_s))

    # cardiac_d = scipy.signal.wavelets.daub(int(diastolic/10)) * int(math.sqrt(pow(diastolic,2)+pow(heart_rate,2))*0.3)
    # print("cardiac_d:", len(cardiac_d))

    # Add the gap after the pqrst when the heart is resting.
    # cardiac = np.concatenate([cardiac, np.zeros(10)])
    # cardiac = np.concatenate([cardiac_s, cardiac_d])

    cardiac_length = int(100 * sampling_rate / heart_rate)  # sampling_rate #
    ind = random.randint(17, 34)
    cardiac_s = scipy.signal.wavelets.daub(ind)
    cardiac_d = scipy.signal.wavelets.daub(ind) * 0.3 * diastolic / 80  # change height to 0.3
    cardiac_s = scipy.signal.resample(cardiac_s, 100)
    cardiac_d = scipy.signal.resample(cardiac_d, 100)
    cardiac_s = cardiac_s[0:40]
    distance = 180 - systolic  # systolic 81-180
    # distance = cardiac_length - len(cardiac_s) - len(cardiac_d) - systolic # here 140 = 40 (cardiac_s) + 100 (cardiac_d) as below
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    # cardiac = scipy.signal.resample(cardiac, 100) # fix every cardiac length to 100
    cardiac = scipy.signal.resample(cardiac, cardiac_length)  # fix every cardiac length to 1000/heart_rate

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    scg = np.tile(cardiac, num_heart_beats)

    # Change amplitude
    # scg = scg * 10
    # scg = scg * 10

    # Resample
    scg = signal_resample(
        scg, sampling_rate=int(len(scg) / 10), desired_length=length, desired_sampling_rate=sampling_rate
    )

    # max_peak = max(scg)
    # peak_threshold = max_peak/s_d + 0.1
    # peaks, _ = scipy.signal.find_peaks(scg, height=peak_threshold)

    ### add rr
    num_points = duration * sampling_rate
    x_space = np.linspace(0, 1, num_points)
    seg_fre = respiratory_rate / (60 / duration)
    seg_amp = max(scg) * 0.00001
    rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
    # scg *= rr_component
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)
    # plt.plot(rr_component * 1000)
    # plt.scatter(peaks, scg[peaks], c = 'r')

    # #modeified rr component
    # for i in range(len(scg)):
    #     if scg[i] > 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)
    #     elif scg[i] < 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)

    scg *= (rr_component + 2 * seg_amp)
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)

    # import matplotlib.pyplot as plt
    # # plt.plot(rr_component,'r')
    # plt.plot(scg)
    # plt.show()

    # import pdb; pdb.set_trace()
    return scg


# =============================================================================
# scgSYN
# =============================================================================
def _scg_simulate_scgsyn(
        sfscg=256,
        N=256,
        Anoise=0,
        hrmean=60,
        hrstd=1,
        lfhfratio=0.5,
        sfint=512,
        ti=(-70, -15, 0, 15, 100),
        ai=(1.2, -5, 30, -7.5, 0.75),
        bi=(0.25, 0.1, 0.1, 0.1, 0.4),
):
    """This function is a python translation of the matlab script by `McSharry & Clifford (2013)

    <https://physionet.org/content/scgsyn>`_.

    Parameters
    ----------
    % Operation uses the following parameters (default values in []s):
    % sfscg: scg sampling frequency [256 Hertz]
    % N: approximate number of heart beats [256]
    % Anoise: Additive uniformly distributed measurement noise [0 mV]
    % hrmean: Mean heart rate [60 beats per minute]
    % hrstd: Standard deviation of heart rate [1 beat per minute]
    % lfhfratio: LF/HF ratio [0.5]
    % sfint: Internal sampling frequency [256 Hertz]
    % Order of extrema: (P Q R S T)
    % ti = angles of extrema (in degrees)
    % ai = z-position of extrema
    % bi = Gaussian width of peaks

    Returns
    -------
    array
        Vector containing simulated scg signal.

#    Examples
#    --------
#    >>> import matplotlib.pyplot as plt
#    >>> import neurokit2 as nk
#    >>>
#    >>> s = _scg_simulate_scgsynth()
#    >>> x = np.linspace(0, len(s)-1, len(s))
#    >>> num_points = 4000
#    >>>
#    >>> num_points = min(num_points, len(s))
#    >>> plt.plot(x[:num_points], s[:num_points]) #doctest: +SKIP
#    >>> plt.show() #doctest: +SKIP

    """

    if not isinstance(ti, np.ndarray):
        ti = np.array(ti)
    if not isinstance(ai, np.ndarray):
        ai = np.array(ai)
    if not isinstance(bi, np.ndarray):
        bi = np.array(bi)

    ti = ti * np.pi / 180

    # Adjust extrema parameters for mean heart rate
    hrfact = np.sqrt(hrmean / 60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * bi
    ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2]) * ti

    # Check that sfint is an integer multiple of sfscg
    q = np.round(sfint / sfscg)
    qd = sfint / sfscg
    if q != qd:
        raise ValueError(
            "Internal sampling frequency (sfint) must be an integer multiple of the scg sampling frequency"
            " (sfscg). Your current choices are: sfscg = " + str(sfscg) + " and sfint = " + str(sfint) + "."
        )

    # Define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and respiratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # Calculate time scales for rr and total output
    sfrr = 1
    trr = 1 / sfrr
    rrmean = 60 / hrmean
    n = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = _scg_simulate_rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n)

    # Upsample rr time series from 1 Hz to sfint Hz
    rr = signal_resample(rr0, sampling_rate=1, desired_sampling_rate=sfint)

    # Make the rrn time series
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tscg = 0
    i = 0
    while i < len(rr):
        tscg += rr[i]
        ip = int(np.round(tscg / dt))
        rrn[i:ip] = rr[i]
        i = ip
    Nt = ip

    # Integrate system using fourth order Runge-Kutta
    x0 = np.array([1, 0, 0.04])

    # tspan is a tuple of (min, max) which defines the lower and upper bound of t in ODE
    # t_eval is the list of desired t points for ODE
    # in Matlab, ode45 can accepts both tspan and t_eval in one argument
    Tspan = [0, (Nt - 1) * dt]
    t_eval = np.linspace(0, (Nt - 1) * dt, Nt)

    # as passing extra arguments to derivative function is not supported yet in solve_ivp
    # lambda function is used to serve the purpose
    result = scipy.integrate.solve_ivp(
        lambda t, x: _scg_simulate_derivsscgsyn(t, x, rrn, ti, sfint, ai, bi), Tspan, x0, t_eval=t_eval
    )
    X0 = result.y

    # downsample to required sfscg
    X = X0[:, np.arange(0, X0.shape[1], q).astype(int)]

    # Scale signal to lie between -0.4 and 1.2 mV
    z = X[2, :].copy()
    zmin = np.min(z)
    zmax = np.max(z)
    zrange = zmax - zmin
    z = (z - zmin) * 1.6 / zrange - 0.4

    # include additive uniformly distributed measurement noise
    eta = 2 * np.random.uniform(len(z)) - 1
    return z + Anoise * eta  # Return signal


def _scg_simulate_derivsscgsyn(t, x, rr, ti, sfint, ai, bi):
    ta = math.atan2(x[1], x[0])
    r0 = 1
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2) / r0

    ip = np.floor(t * sfint).astype(int)
    w0 = 2 * np.pi / rr[min(ip, len(rr) - 1)]
    # w0 = 2*np.pi/rr[ip[ip <= np.max(rr)]]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * x[0] - w0 * x[1]
    dx2dt = a0 * x[1] + w0 * x[0]

    # matlab rem and numpy rem are different
    # dti = np.remainder(ta - ti, 2*np.pi)
    dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1 * (x[2] - zbase)

    dxdt = np.array([dx1dt, dx2dt, dx3dt])
    return dxdt


def _scg_simulate_rrprocess(
        flo=0.1, fhi=0.25, flostd=0.01, fhistd=0.01, lfhfratio=0.5, hrmean=60, hrstd=1, sfrr=1, n=256
):
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1 ** 2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2 ** 2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate((Hw[0: int(n / 2)], Hw[int(n / 2) - 1:: -1]))
    Sw = (sfrr / 2) * np.sqrt(Hw0)

    ph0 = 2 * np.pi * np.random.uniform(size=int(n / 2 - 1))
    ph = np.concatenate([[0], ph0, [0], -np.flipud(ph0)])
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    xstd = np.std(x)
    ratio = rrstd / xstd
    return rrmean + x * ratio  # Return RR