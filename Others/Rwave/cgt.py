import numpy as np
import matplotlib.pyplot as plt


def cgt(input, nvoice, freqstep, scale=1, plot=True):
    """
    Description:
        Computes the continuous Gabor transform with Gaussian window.

    Params:
        input (numpy.ndarray): Input signal (possibly complex-valued).
        nvoice (int): Number of frequencies for which the Gabor transform is to be computed.
        freqstep (float, optional): Sampling rate for the frequency axis. Default is 1/nvoice.
        scale (float, optional): Size parameter for the window. Default is 1.
        plot (bool, optional): Logical variable set to True to display the modulus
            of the continuous Gabor transform on the graphic device. Default is True.

    Returns:
        numpy.ndarray: Continuous (complex) Gabor transform (2D array).
    """
    # Check if freqstep is within the Nyquist limit
    if freqstep >= (1 / nvoice):
        raise ValueError("freqstep must be less than 1/nvoice to avoid aliasing.")

    # Get the size of the input signal
    signal_size = len(input)

    # Initialize the Gabor transform result matrix
    gabor_transform = np.zeros((signal_size, nvoice), dtype=complex)

    # Generate the Gabor window
    window = np.exp(
        -0.5 * ((np.arange(-scale * signal_size / 2, scale * signal_size / 2) / (scale * signal_size / 4)) ** 2))

    # Compute the Gabor transform for each frequency
    for k in range(nvoice):
        # Compute the frequency for this Gabor transform
        freq = k * freqstep

        # Generate the complex sinusoidal kernel
        kernel = np.exp(1j * 2 * np.pi * freq * np.arange(signal_size))

        # Apply the Gabor window to the input signal
        gabor_input = input * window

        # Compute the Gabor transform for this frequency
        gabor_transform[:, k] = np.fft.ifft(np.fft.fft(gabor_input) * np.fft.fft(kernel))

    # Plot the modulus of the continuous Gabor transform
    if plot:
        plt.imshow(np.abs(gabor_transform).T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Modulus')
        plt.title('Continuous Gabor Transform')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    return gabor_transform


if __name__ == "__main__":
    import numpy as np
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100


    nvoice = 64
    freqstep = 1 / (nvoice + 1)
    scale = 1
    plot = True

    gabor_transform = cgt(signal, nvoice, freqstep, scale, plot)
    print(gabor_transform.shape)
