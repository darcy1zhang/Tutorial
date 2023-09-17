import numpy as np
import numpy as np


def ridge_detection(tfrep, tfspec=None, iterations=1000, rate=0.001, seed=None):
    """
    Description:
        Ridge detection using crazy climber algorithm

    Params:
        tfrep (np.ndarray): Wavelet or Gabor transform, shape (time, scale)
        tfspec (np.ndarray): Additional potential, shape (scale,)
        iterations (int): Number of iterations
        rate (float): Temperature parameter
        seed (int): Random seed

    Returns:
        beemap (np.ndarray): 2D occupation measure, shape (time, scale)
    """

    # Initialization
    time, scale = tfrep.shape
    if tfspec is None:
        tfspec = np.zeros(scale)
    sqmodulus = np.abs(tfrep) ** 2 - tfspec

    # Crazy climber algorithm
    beemap = np.zeros_like(sqmodulus)
    if seed is not None:
        np.random.seed(seed)
    for i in range(iterations):
        position = np.random.randint(0, time * scale)
        t, s = position // scale, position % scale
        beemap[t, s] += 1
        for _ in range(np.random.randint(3, 8)):
            dt, ds = np.random.randint(-1, 2, size=2)
            nt, ns = t + dt, s + ds
            if 0 <= nt < time and 0 <= ns < scale:
                if sqmodulus[nt, ns] > sqmodulus[t, s]:
                    t, s = nt, ns
                    beemap[t, s] += 1
                else:
                    p = np.exp((sqmodulus[nt, ns] - sqmodulus[t, s]) / rate)
                    if np.random.rand() < p:
                        t, s = nt, ns
                        beemap[t, s] += 1

    return beemap

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    from cwt import *

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

    cwt_result = cwt(signal, noctave, nvoice, w0, twoD, plot)


    beemap = ridge_detection(cwt_result)

    plt.imshow(beemap, cmap='viridis', aspect='auto')
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Occupation Measure (Crazy Climber Algorithm)")
    plt.colorbar()
    plt.show()






