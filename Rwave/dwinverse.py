import numpy as np
import pywt


def dwinverse(wt, filtername="haar"):
    """
    Description:
        Invert the dyadic wavelet transform.

    Params:
        wt (numpy.ndarray): Dyadic wavelet transform.
        filtername (str, optional): Filter used for the transformation.

    Returns:
        numpy.ndarray: Reconstructed signal.
    """
    if filtername == "haar":
        wavelet = 'haar'
    else:
        raise ValueError("Unsupported filtername.")

    wt = np.array(wt)
    reconstructed_signal = pywt.waverec([wt], wavelet)

    return reconstructed_signal


if __name__ == "__main__":
    import numpy as np
    from mw import *
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

    plt.subplot(2,1,1)
    plt.plot(signal)


    result = mw(signal, maxresoln=8, filtername="haar", scale=False, plot=False)

    print(result["Wf"].shape)

    recons = dwinverse(result["Wf"], "haar")
    plt.subplot(2,1,2)
    plt.plot(recons)
    plt.show()
