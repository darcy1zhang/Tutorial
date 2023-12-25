import matplotlib.pyplot as plt
import numpy as np

def plotwt(original, psi, phi, maxresoln, scale=False, yaxtype="s"):
    """
    Description:
        Plot the wavelet transform results.

    Params:
        original (array): The original input signal.
        psi (array): The wavelet coefficients, arranged by levels.
        phi (array): The final smooth coefficients.
        maxresoln (int): Number of decomposition levels.
        scale (bool): Whether to scale all plots to the same range.
        yaxtype (string): The yaxis tick type, e.g. 's' for scientific notation.

    Returns:
        None. The function plots the wavelet transform and does not return anything.
    """

    par_orig = plt.rcParams.copy()

    plt.rcParams["figure.figsize"] = [8, 6]

    plt.subplot(maxresoln +2, 1, 1)
    plt.plot(original)
    plt.yticks([])
    plt.axis('off')

    for j in range(1, maxresoln +1):
        plt.subplot(maxresoln +2, 1, j+ 1)
        plt.plot(psi[:, j - 1])
        plt.yticks([])
        plt.axis('off')

    plt.subplot(maxresoln + 2, 1, maxresoln + 2)
    plt.plot(phi)
    plt.yticks([])
    plt.axis('off')

    if scale:
        limit = [min(original, psi.flatten(), phi), max(original, psi.flatten(), phi)]
        for j in range(1, maxresoln + 2):
            plt.subplot(maxresoln + 2, 1, j)
            plt.ylim(limit)


    plt.rcParams.update(par_orig)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    from cwt import *
    import os
    from wpl import *
    from mw import *

    noctave = 5
    nvoice = 3
    w0 = 2 * np.pi

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    result = mw(signal, maxresoln=8, filtername="haar", scale=False, plot=False)
    wpl(result)