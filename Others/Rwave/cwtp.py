import numpy as np
import matplotlib.pyplot as plt


import pywt
import numpy as np
import matplotlib.pyplot as plt

def cwtp(input, noctave, nvoice=1, w0=2 * np.pi, twoD=True, plot=False):
    """
    Description:
        Continuous wavelet transform function with phase derivative.
        Computes the continuous wavelet transform with (complex-valued) Morlet wavelet
        and its phase derivative using PyWavelets.

    Params:
        input (numpy.ndarray): Input signal (possibly complex-valued).
        noctave (int): Number of powers of 2 for the scale variable.
        nvoice (int, optional): Number of scales between 2 consecutive powers of 2 (default is 1).
        w0 (float, optional): Central frequency of Morlet wavelet (default is 2*pi).
        twoD (bool, optional): If set to True, organizes the output as a 2D array
                              (signal_size X nb_scales). If not, a 3D array (signal_size X noctave X nvoice) is returned.
        plot (bool, optional): If set to True, displays the modulus of CWT on the graphic device.

    Returns:
        Continuous (complex) wavelet transform and the phase derivative.
        wt: array of complex numbers for the values of the continuous wavelet transform.
        f: array of the same dimensions containing the values of the derivative of the phase
                 of the continuous wavelet transform.
    """
    scales = np.logspace(0, noctave, num=noctave * nvoice, base=2)
    coefs, frequencies = pywt.cwt(input, scales, 'morl', sampling_period=1.0 / w0)

    if plot:
        plt.imshow(coefs, aspect='auto', cmap='jet')
        plt.xlabel("Time")
        plt.ylabel("log(scale)")
        plt.title("Wavelet Transform Modulus")
        plt.colorbar()
        plt.show()


    phase_derivative = np.gradient(np.angle(coefs), axis=1)


    return coefs, phase_derivative



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

    wt, f = cwtp(signal, noctave, nvoice, w0, twoD, plot)


    print(wt.shape)
    print(f.shape)
