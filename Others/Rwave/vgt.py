import numpy as np
from scipy import signal


def vgt(input, frequency, scale, plot=False):
    """
    Description:
        Calculate continuous Gabor transform of input signal at given frequency.

    Params:
        input (ndarray): Input signal array
        frequency (float): Frequency value
        scale (float): Window size
        plot (bool): Whether to plot the real part

    Returns:
        output (ndarray): Complex array containing Gabor transform with real and imaginary parts
        """

    gabor_kernel = np.multiply(np.exp(-scale * np.square(np.arange(input.size))),
                               np.exp(1j * 2 * np.pi * frequency * np.arange(input.size)))
    output = signal.convolve(input, gabor_kernel, mode='same')

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(np.real(output))
        plt.title('Real part of Gabor transform')
        plt.show()

    return output


if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal1 = np.load(signal_path)[2, :1000]
    fs = 100


    result = vgt(signal1, 20, 0.02, True)

    print(result.shape)
