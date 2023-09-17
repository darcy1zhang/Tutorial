import pywt
import numpy as np
import matplotlib.pyplot as plt

def cwtTh(input, noctave, nvoice=1, moments=2, twoD=True, plot=True):
    """
    Description:
        Compute the continuous wavelet transform with (complex-valued) Cauchy’s wavelet using PyWavelets.

    Params:
        input (numpy.ndarray): Input signal (possibly complex-valued).
        noctave (int): Number of powers of 2 for the scale variable.
        nvoice (int, optional): Number of scales between 2 consecutive powers of 2 (default is 1).
        moments (int): Number of vanishing moments.
        twoD (bool, optional): If set to True, organizes the output as a 2D array
                              (signal_size X nb_scales). If not, a 3D array (signal_size X noctave X nvoice) is returned.
        plot (bool, optional): If set to True, displays the modulus of the continuous wavelet transform on the graphic device.

    Returns:
        numpy.ndarray: Continuous (complex) wavelet transform.
    """
    scales = np.logspace(0, noctave, num=noctave * nvoice, base=2)
    coefs, frequencies = pywt.cwt(input, scales, 'cgau2', sampling_period=1.0 / w0)

    if plot:
        plt.imshow(np.abs(coefs), aspect='auto', cmap='jet')
        plt.xlabel("Time")
        plt.ylabel("Scale")
        plt.title("Cauchy's Wavelet Transform Modulus")
        plt.colorbar()
        plt.show()

    return np.abs(coefs)


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

    result = cwtTh(signal, noctave, nvoice, moments=2)


    print(result.shape)
