import numpy as np
import matplotlib.pyplot as plt

def morlet(sigsize, location, scale, w0=2 * np.pi):
    """
    Description:
        Compute a Morlet wavelet at the specified time and scale in the time-scale plane.

    Params:
        sigsize: Length of the output wavelet.
        location: Time location of the wavelet.
        scale: Scale of the wavelet.
        w0: Central frequency of the wavelet (default is 2 * pi).

    Returns:
        Complex 1D array representing the values of the Morlet wavelet at the specified time and scale.
    """
    t = np.arange(sigsize) - location
    # Calculate the real and imaginary parts of the Morlet wavelet
    real_part = np.exp(-(t**2) / (2 * scale**2)) * np.cos(w0 * t)
    imag_part = np.exp(-(t**2) / (2 * scale**2)) * np.sin(w0 * t)
    morlet_wavelet = real_part + 1j * imag_part

    return morlet_wavelet


if __name__ == "__main__":
    sigsize = 1024
    location = 512
    scale = 20
    morlet = morlet(sigsize, location, scale)
    plt.plot(morlet.real)
    plt.show()


