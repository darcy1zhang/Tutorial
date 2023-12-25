import matplotlib.pyplot as plt
import numpy as np
from cwt import *


def cwtpolar(cwt, threshold=0.0):
    """
    Description:
        Continuous wavelet transform conversion: Converts one of the possible outputs of CWT to modulus and phase.

    Params:
        cwt (numpy.ndarray): 3D array containing a continuous wavelet transform (output of CWT, with twoD=False).
        threshold (float, optional): The phase is forced to -π if the modulus is less than threshold (default is 0.0).

    Returns:
        output1 (numpy.ndarray): Modulus.
        output2 (numpy.ndarray): Phase.
    """
    # Get dimensions of the input CWT
    sigsize, noctave, nvoice = cwt.shape

    # Initialize output arrays for modulus and phase
    output1 = np.zeros((sigsize, noctave, nvoice))
    output2 = np.zeros((sigsize, noctave, nvoice))

    # Compute modulus and phase for each scale and voice
    for i in range(noctave):
        for j in range(nvoice):
            # Calculate modulus using the absolute value of the complex number
            output1[:, i, j] = np.abs(cwt[:, i, j])

            # Calculate phase using the arctan2 function
            output2[:, i, j] = np.arctan2(np.imag(cwt[:, i, j]), np.real(cwt[:, i, j]))

    # Set phase to -π for modulus values less than the threshold
    ma = np.max(output1)
    rel = threshold * ma
    output2[output1 < rel] = -np.pi

    return output1, output2

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
    twoD = False
    plot = True

    cwt_result = cwt(signal, noctave, nvoice, w0, twoD, plot)

    output1, output2 = cwtpolar(cwt_result, 0)

    print(output1)
    print(output2)
