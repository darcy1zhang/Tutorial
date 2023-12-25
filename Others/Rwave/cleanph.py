import numpy as np
import matplotlib.pyplot as plt


def cleanph(tfrep, thresh=0.01, plot=True):
    """
    Description:
        Sets to zero the phase of time-frequency transform when modulus is below a certain value.

    Params:
        tfrep (numpy.ndarray): Continuous time-frequency transform (2D array).
        thresh (float, optional): Relative threshold (default is 0.01). Determines which phases are set to zero
                                  when the modulus is below this threshold.
        plot (bool, optional): If set to True, displays the maxima of the transform on the graphic device.

    Returns:
        numpy.ndarray: Thresholded phase (2D array).
    """
    # Calculate the modulus of the time-frequency transform
    thrmod1 = np.abs(tfrep)
    thrmod2 = np.abs(tfrep)

    # Calculate the threshold value based on the relative threshold
    limit = np.max(thrmod1) * thresh

    # Create masks for phase selection
    thrmod1 = thrmod1 > limit
    thrmod2 = thrmod2 <= limit

    # Threshold the phase values
    output = thrmod1 * np.angle(tfrep) - np.pi * thrmod2

    # Plot if requested
    if plot:
        plt.imshow(output)
        plt.title("Thresholded Phase")
        plt.colorbar()
        plt.show()

    return output


if __name__ == "__main__":
    import os
    import numpy as np
    import scipy.signal

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    frequencies, times, Sxx = scipy.signal.spectrogram(signal, fs=fs, window='hann', nperseg=100,
                                                       noverlap=int(20))

    plt.imshow(Sxx)
    plt.show()

    new_Sxx = cleanph(Sxx)


