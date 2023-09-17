import numpy as np
import matplotlib.pyplot as plt
import pywt

def WV(input_signal, nvoice, freqstep=1.0, plot=True):
    """
    Description:
        Compute the Wigner-Ville transform using PyWavelets library.

    Params:
        input_signal (array-like): Input signal for which the Wigner-Ville transform is calculated.
        nvoice (int): Number of frequency bands for the CWT.
        freqstep (float, optional): Sampling rate for the frequency axis, which is 1/nvoice.
        plot (bool, optional): If set to True, displays the modulus of CWT on the graphic device. Default is True.

    Returns:
        wv_transform (numpy.ndarray): Complex-valued Wigner-Ville transform of the
        input signal. The dimensions of the array are (nvoice, len(input_signal)).
        The transform represents the time-frequency distribution of the signal's energy.
    """
    coeffs, _ = pywt.cwt(input_signal, np.arange(1, nvoice + 1), 'gaus1')

    # Calculate the Wigner-Ville transform
    wv_transform = np.abs(coeffs) ** 2

    if plot:
        # Display the modulus of CWT
        plt.imshow(wv_transform, extent=[0, len(input_signal), 0, nvoice * freqstep],
                   aspect='auto', cmap='viridis', origin='lower')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Wigner-Ville Transform')
        plt.colorbar(label='Magnitude')
        plt.show()

    return wv_transform



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

    wv = WV(signal, 50, 1/50, True)
