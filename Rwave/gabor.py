import numpy as np
import matplotlib.pyplot as plt

def gabor(sigsize, location, frequency, scale):
    """
    Description:
        Generate a Gabor function for the given location, frequency, and scale parameters.

    Params:
        sigsize: Length of the Gabor function.
        location: Position of the Gabor function.
        frequency: Frequency of the Gabor function.
        scale: Size parameter for the Gabor function.

    Returns:
        Complex 1D array of size sigsize representing the Gabor function.
    """
    x = np.arange(sigsize) - location
    # Calculate the real and imaginary parts of the Gabor function
    real_part = np.exp(-(x**2) / (2 * scale**2)) * np.cos(2 * np.pi * frequency * x)
    imag_part = np.exp(-(x**2) / (2 * scale**2)) * np.sin(2 * np.pi * frequency * x)
    gabor_function = real_part + 1j * imag_part

    return gabor_function

if __name__ == "__main__":
    sigsize = 1024
    location = 512
    frequency = 2 * np.pi
    scale = 20
    gabor = gabor(sigsize, location, frequency, scale)
    plt.plot(gabor.real)
    plt.show()

