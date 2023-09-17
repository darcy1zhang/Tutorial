import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from plotresult import *

def ext(wt, scale=False, plot=True):
    """
    Description:
        Compute extrema of the dyadic wavelet transform and optionally plot the results.

    Params:
        wt (dict): A dictionary containing wavelet transform information.
        - 'original' (ndarray): Original signal.
        - 'maxresoln' (int): Number of decomposition levels (resolutions).
        - 'np' (int): Size of the signal.
        - 'Wf' (ndarray): Dyadic wavelet transform coefficients.
        - 'Sf' (ndarray or None): Coarse resolution of the signal.

        scale (bool, optional): A flag indicating if the extrema at each resolution will be plotted at the same scale. Default is False.
        plot (bool, optional): A flag indicating whether to plot the results. Default is True.

    Returns:
        esult (dict): A dictionary containing the following:
        - 'original' (ndarray): Original signal.
        - 'extrema' (ndarray): Extrema representation.
        - 'Sf' (ndarray or None): Coarse resolution of the signal.
        - 'maxresoln' (int): Number of decomposition levels (resolutions).
        - 'np' (int): Size of the signal.
    """

    original = wt['original']
    maxresoln = wt['maxresoln']
    np1 = wt['np']
    wt_coeffs = wt['Wf']

    extrema = find_modulus_maxima(wt_coeffs, maxresoln, np1)


    if plot:
        extrema = np.transpose(extrema)
        plotresult(extrema, original, maxresoln, scale)

    return {
        'original': original,
        'extrema': extrema,
        'Sf': wt['Sf'],
        'maxresoln': maxresoln,
        'np': np
    }

def find_modulus_maxima(wt, resoln, np1):
    extrema = np.zeros(wt.shape)  # 创建与wt相同形状的全零数组用于存储极大值
    np1 = wt.shape[1]
    for j in range(resoln):
        abs_wt = np.abs(wt[j, :])  # 取出一行作为小波系数的绝对值

        extrema[j, 0] = 0.0
        extrema[j, np1 - 1] = 0.0

        for x in range(1, np1 - 1):
            if ((abs_wt[x] > abs_wt[x - 1]) and (abs_wt[x] >= abs_wt[x + 1])) or \
               ((abs_wt[x] > abs_wt[x + 1]) and (abs_wt[x] >= abs_wt[x - 1])):
                extrema[j, x] = wt[j, x]
            else:
                extrema[j, x] = 0.0

    return extrema


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

    result = mw(signal, 8, plot=True)
    ext = ext(result)
