import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d

#%%
def update_array(a, data_tmp):
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a



def fpeak(signal):
    signal2 = signal

    # 峰值检测
    peak_indices, _ = find_peaks(signal2)  # 返回极大值点的索引

    # 线性插值
    t_peaks = t[peak_indices]  # 极大值点的时间
    peak_values = signal2[peak_indices]  # 极大值点的幅值
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2,_ = find_peaks(envelope, distance = 10)

    peaks2 = update_array(peaks2, signal2)

    if(signal2[peaks2[0]]>signal2[peaks2[1]]):
        peaks2 = np.delete(peaks2, 0)

    if len(peaks2)%2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)
    
    return peaks2


#%% main script
if __name__ == '__main__':
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    from scipy.signal import argrelextrema, find_peaks
    from scipy.interpolate import interp1d
    signal = np.load("data/simu_20000_0.1_90_140_train.npy")[0,:1000]
    fs = 100
    t = np.linspace(0, 10, 10 * fs)

    peaks = fpeak(signal)

    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='原始信号')
    # plt.plot(t, envelope, label='包络', linewidth=2)
    # plt.scatter(t_peaks, peak_values, color='red', label='峰值')
    plt.plot(t[peaks], signal[peaks], 'o', color = 'green')
    plt.legend()
    plt.show()

# %%
