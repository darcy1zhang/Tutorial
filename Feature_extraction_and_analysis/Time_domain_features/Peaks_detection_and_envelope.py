import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d

#%%
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
    peaks2,_ = find_peaks(envelope, distance = 10)

    # remove wrong peaks
    peaks2 = update_array(peaks2, signal)

    # make sure the first peak is the higher peak
    if len(peaks2) > 1:
        if(signal[peaks2[1]]>signal[peaks2[0]]):
            peaks2 = np.delete(peaks2, 0)

    # make sure the number of peaks is even
    if len(peaks2)%2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)
    
    return peaks2


#%% main script
if __name__ == '__main__':
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    from scipy.signal import argrelextrema, find_peaks
    from scipy.interpolate import interp1d
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")
    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data", signal_name)
    signal = np.load(signal_path)[0, :1000]

    fs = 100
    # t = np.linspace(0, 10, 10 * fs)

    peaks = fpeak(signal)
    t = np.arange(len(signal))
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.plot(t[peaks], signal[peaks], 'o', color = 'green')
    plt.show()

# %%
