import numpy as np
import matplotlib.pyplot as plt
from cwt import *

def cwtimage(input):
    """
    Description:
        Continuous wavelet transform display. Converts the output (modulus or argument)
        of cwtpolar to a 2D array and displays it on the graphic device.

    Params:
        input (numpy.ndarray): 3D array containing a continuous wavelet transform (output of cwtpolar).

    Returns:
        numpy.ndarray: 2D array.
    """
    # Get dimensions of the input 3D array
    sigsize, noctave, nvoice = input.shape

    # Create a 2D array for the output, initialized with zeros
    output = np.zeros((sigsize, noctave * nvoice))

    # Rearrange the data from the input 3D array into the output 2D array
    for i in range(noctave):
        k = i * nvoice
        for j in range(nvoice):
            output[:, k + j] = input[:, i, j]

    # Display the output as an image on the graphic device
    plt.imshow(output, aspect='auto', cmap='jet')
    plt.xlabel("Scale")
    plt.ylabel("Time")
    plt.title("Wavelet Transform Display")
    plt.colorbar()
    plt.show()

    return output

if __name__ == "__main__":
    import os
    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100
    noctave = 5
    nvoice = 3
    w0 = 2 * np.pi
    twoD = True
    plot = True

    cwt_result = cwt(signal, noctave, nvoice, w0, twoD, plot)

    output = cwtimage(cwt_result)