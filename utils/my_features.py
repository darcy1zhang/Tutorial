import numpy as np
from tsfel import set_domain
from Peaks_detection_and_envelope import fpeak, update_array

@set_domain("domain", "temporal")
def D12(signal):
    """Horizontal distance from high peak to low peak.

    Parameters
    ----------
    signal:
        The time series to calculate the feature of.
    Returns
    -------
    float
        Horizontal distance from high peak to low peak
    """

    peaks = fpeak(signal)
    if len(peaks) < 2:
        return None

    return (peaks[::2]-peaks[1::2]) / len(peaks) / 2

@set_domain("domain", "temporal")
def D21(signal):
    """Horizontal distance from low peak to high peak.

    Parameters
    ----------
    signal:
        The time series to calculate the feature of.
    Returns
    -------
    float
        Horizontal distance from low peak to high peak
    """

    peaks = fpeak(signal)
    if len(peaks) > 1:
        peaks = np.delete(peaks, 0)
        peaks = np.delete(peaks, len(peaks)-1)
    if len(peaks) < 2:
        return None

    return (peaks[::2]-peaks[1::2]) / len(peaks) / 2

@set_domain("domain", "temporal")
def P1(signal):
    """Height of high peak.

    Parameters
    ----------
    signal:
        The time series to calculate the feature of.
    Returns
    -------
    float
        Height of high peak
    """

    peaks = fpeak(signal)
    if len(peaks) < 2:
        return None

    return np.mean(signal[peaks[::2]])

@set_domain("domain", "temporal")
def P2(signal):
    """Height of low peak.

    Parameters
    ----------
    signal:
        The time series to calculate the feature of.
    Returns
    -------
    float
        Height of low peak
    """

    peaks = fpeak(signal)
    if len(peaks) < 2:
        return None

    return np.mean(signal[peaks[1::2]])

