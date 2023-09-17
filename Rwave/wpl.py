import matplotlib.pyplot as plt
from plotwt import *

def wpl(dwtrans):
    """
    Description:
        Plot dyadic wavelet transform results.

    Params:
        dwtrans (dict): The dyadic wavelet transform result dictionary, contains:
          - original: The original input signal
          - Wf: Wavelet coefficients
          - Sf: The smooth coefficients
          - maxresoln: Number of decomposition levels
          - np: Length of original signal

    Returns:
        None. The function plots the wavelet transform and does not return anything.
    """
    original = dwtrans['original']
    Wf = dwtrans['Wf']
    Sf = dwtrans['Sf']
    maxresoln = dwtrans['maxresoln']


    plotwt(original, Wf, Sf, maxresoln)



if __name__ == "__main__":
    import numpy as np
    from cwt import *
    import os
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