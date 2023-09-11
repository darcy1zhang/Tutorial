import numpy as np
import matplotlib.pyplot as plt

def smoothts(ts, windowsize):
    """
    Description:
        Smooth a time series by averaging over a window.

    Params:
        ts: Time series data.
        windowsize: Length of the smoothing window.

    Returns:
        Smoothed time series (1D array).
    """
    sigsize = len(ts)
    sts = np.zeros(sigsize)

    sts = sts.reshape(sigsize, 1)
    ts = ts.reshape(sigsize, 1)



    for i in range(sigsize):
        start_idx = max(0, i - windowsize // 2)
        end_idx = min(sigsize, i + windowsize // 2 + 1)
        sts[i] = np.mean(ts[start_idx:end_idx])

    return sts.flatten()  # Return the smoothed time series as a 1D array

if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    windowsize = 3  # Smoothing window size
    smoothed_ts = smoothts(signal, windowsize)
    print("Smoothed Time Series:", smoothed_ts)

    plt.subplot(2,1,1)
    plt.plot(signal)
    plt.subplot(2,1,2)
    plt.plot(smoothed_ts)
    plt.show()
