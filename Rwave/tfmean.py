import numpy as np
import matplotlib.pyplot as plt

def tfmean(input, plot=True):
    """
    Description:
        Compute the mean of time-frequency representation frequency by frequency.

    Params:
        input (numpy.ndarray): Time-frequency transform (output of cwt or cgt).
        plot (bool, optional): If set to True, displays the values of the energy as a function of the scale (or frequency).
                               Default is True.

    Returns:
        numpy.ndarray: 1D array containing the noise estimate.
    """
    # Calculate the mean along the frequency axis (axis 0)
    input = np.transpose(input)
    mean_frequency = np.mean(input, axis=0)

    if plot:
        # Plot the mean frequency values as a function of the scale (or frequency)
        plt.plot(mean_frequency)
        plt.xlabel('Scale (or Frequency)')
        plt.ylabel('Mean Frequency')
        plt.title('Mean Frequency by Frequency')
        plt.grid(True)
        plt.show()

    return mean_frequency

if __name__ == "__main__":
    from cwt import *
    import os

    noctave = 5
    nvoice = 3
    w0 = 2 * np.pi

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    cwt_result = cwt(signal, noctave, nvoice, w0)


    mean_fre = tfmean(cwt_result, plot=True)


    print(mean_fre)