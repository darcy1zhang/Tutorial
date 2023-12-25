import numpy as np
import matplotlib.pyplot as plt

def tfgmax(input, plot=True):
    """
    Description:
        Computes the maxima of the modulus of a continuous wavelet transform.

    Params:
        input: Wavelet transform (output of the function cwt).
        plot: If set to TRUE, displays the values of the energy as a function of the scale (default is TRUE).

    Returns:
        output: Values of the maxima (1D array).
        pos: Positions of the maxima (1D array).
    """
    # Calculate the modulus of the wavelet transform
    modulus = np.abs(input)


    # Find maxima for each fixed value of the time variable
    maxima = np.max(modulus, axis=0)
    maxima_positions = np.argmax(modulus, axis=0)

    if plot:
        # Display values of energy as a function of scale
        plt.plot(maxima, label='Max Energy')
        plt.xlabel('Scale')
        plt.ylabel('Energy')
        plt.title('Energy vs. Scale')
        plt.legend()
        plt.show()

    return maxima, maxima_positions

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

    maxima_values, maxima_positions = tfgmax(cwt_result, plot=True)

    print("Maxima Values:", maxima_values)
    print("Maxima Positions:", maxima_positions)
