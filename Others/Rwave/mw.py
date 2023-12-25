import numpy as np
import pywt
import matplotlib.pyplot as plt

def mw(inputdata, maxresoln, filtername="haar", scale=False, plot=True):
    """
    Description:
        mw computes the wavelet decomposition using PyWavelets.

    Parameters:
        inputdata: Input data, a 1D array.
        maxresoln: Number of decomposition levels.
        filtername: Name of the wavelet filter (e.g., 'db1' for Haar).
        scale: When True, wavelet transforms at each scale will be plotted with the same scale.
        plot: Indicates whether to plot the wavelet transforms at each scale.

    Returns:
        original: Original signal.
        Wf: Wavelet transform of the signal.
        Sf: Signal at a lower resolution.
        maxresoln: Number of decomposition levels.
        np: Size of the signal.
    """

    # Ensure inputdata is a numpy array
    inputdata = np.array(inputdata)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(inputdata, filtername, level=maxresoln)

    # Extract original signal and coefficients
    original = inputdata

    # Ensure all coefficient arrays are of the same length
    max_len = max(len(c) for c in coeffs[1:])
    Wf = [c.tolist() + [0] * (max_len - len(c)) for c in coeffs[1:]]
    Wf = np.array(Wf)

    Sf = coeffs[-1]

    npoints = len(inputdata)

    if plot:
        # Plot wavelet coefficients
        if scale:
            scales = [len(c) for c in coeffs[1:]]
            scales = np.repeat(scales, max_len)
        else:
            scales = np.arange(max_len)

        # print(np.abs(Wf).shape)

        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(Wf), aspect='auto', cmap='viridis', extent=[0, npoints, maxresoln, 0])
        plt.title('Wavelet Transform')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.colorbar(label='Magnitude')
        plt.show()

    return {
        'original': original,
        'Wf': Wf,
        'Sf': Sf,
        'maxresoln': maxresoln,
        'np': npoints
    }


if __name__ == "__main__":
    import numpy as np
    from cwt import *
    import os
    from tfmean import *

    noctave = 5
    nvoice = 3
    w0 = 2 * np.pi

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100


    result = mw(signal, maxresoln=9, filtername="haar", scale=False, plot=True)
    print(result["np"])

    print("Original Signal Size:", result['np'])
    print("Number of Decomposition Levels:", result['maxresoln'])







