import numpy as np
import matplotlib.pyplot as plt



def tfvar(input, plot=True):
    """
    Description:
        Compute the variance of time-frequency representation frequency by frequency.

    Params:
        input (numpy.ndarray): Time-frequency transform (output of cwt or cgt) with shape (15, 1000).
        plot (bool, optional): If set to True, displays the values of the energy as a function of the scale (or frequency).
                               Default is True.

    Returns:
        numpy.ndarray: 1D array containing the variance of the 15 frequency values.
    """
    # Calculate the variance along the frequency axis (axis 0)
    input = np.transpose(input)
    variance_frequency = np.var(input, axis=0)

    if plot:
        # Define the 15 frequency values (you may adjust this based on your specific frequency range)
        frequencies = np.arange(15)

        # Plot the variance of frequency values as a function of frequency
        plt.plot(frequencies, variance_frequency)
        plt.xlabel('Frequency Index')
        plt.ylabel('Variance of Frequency')
        plt.title('Variance of Frequency by Frequency')
        plt.grid(True)
        plt.show()

    return variance_frequency


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


    var = tfvar(cwt_result, plot=True)


    print(var)