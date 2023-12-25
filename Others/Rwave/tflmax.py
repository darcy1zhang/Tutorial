import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def tflmax(input, plot=True):
    """
    Description:
        Computes the local maxima of the modulus of a time-frequency transform.

    Params:
        input: Time-frequency transform (real 2D array).
        plot: If set to True, displays the local maxima on the graphic device (default is True).

    Returns:
        maxima: Values of the local maxima (2D array).
    """
    # Find local maxima in the input using scipy's maximum_filter
    neighborhood_size = 3  # Size of the local neighborhood for maximum search
    maxima = ndi.maximum_filter(input, size=neighborhood_size)


    # Create a mask to identify local maxima
    is_local_max = (input == maxima)
    print(is_local_max.shape)

    if plot:
        # Display the local maxima on a graphic device
        plt.imshow(is_local_max, aspect="auto", cmap='gray')
        plt.title('Local Maxima')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    return maxima

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


    local_maxima = tflmax(cwt_result, plot=True)


    print("Local Maxima Values:")
    print(local_maxima)
