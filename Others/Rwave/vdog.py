import numpy as np
import pywt

def vDOG(input, scale, moments):
    """
    Description:
        Compute DOG wavelet transform at one scale

    Params:
        input (np.ndarray): Input signal (1D array)
        scale (float): Scale to compute transform
        moments (int): Number of vanishing moments

    Returns:
        coef (np.ndarray): 1D complex array with wavelet transform at given scale
    """

    coefs,fre = pywt.cwt(input, [scale], f'gaus{moments}')

    return coefs

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

    result = vDOG(signal, 5, 2)

    print(result.shape)
