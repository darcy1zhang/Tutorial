import tftb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
# def get_wvd(signal, fs):
#     t = np.arange(0, 10, 1.0 / fs)
#
#     # Doing the WVT
#     wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=t)
#     tfr_wvd, t_wvd, f_wvd = wvd.run()
#     # here t_wvd is the same as our ts, and f_wvd are the "normalized frequencies"
#     # so we will not use them and construct our own.
#
#     # because of how they implemented WVT, the maximum frequency is half of
#     # the sampling Nyquist frequency, so 125 Hz instead of 250 Hz, and the sampling
#     # is 2 * dt instead of dt
#     f_wvd = np.fft.fftshift(np.fft.fftfreq(tfr_wvd.shape[0], d=2 * 1 / fs))
#     df_wvd = f_wvd[1] - f_wvd[0]  # the frequency step in the WVT
#     im = axx[1].imshow(np.fft.fftshift(tfr_wvd, axes=0), aspect='auto', origin='lower',
#                        extent=(t[0] - 1 / fs / 2, t[-1] + 1 / fs / 2,
#                                f_wvd[0] - df_wvd / 2, f_wvd[-1] + df_wvd / 2))
#
#     plt.pcolormesh(t_wvd, f_wvd, tfr_wvd)
#     plt.colorbar()
#     plt.show()
#
#     return f_wvd, df_wvd, im

current_file_path = os.path.abspath(__file__)
signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

signal_path = os.path.join(os.path.dirname(current_file_path), "data",
                           signal_name)
signal = np.load(signal_path)[22, :1000]
fs = 100
t = np.arange(0, 10, 1.0/fs)

# _, _, im = get_wvd(signal,100)

wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=t)
tfr_wvd, t_wvd, f_wvd = wvd.run()
plt.pcolormesh(t_wvd, f_wvd, tfr_wvd)
plt.colorbar()
plt.show()