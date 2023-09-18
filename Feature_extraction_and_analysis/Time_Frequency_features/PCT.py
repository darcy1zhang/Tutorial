import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


# 定义Chirplet函数
def chirplet(alpha, beta, gamma, t):
    return np.exp(1j * (alpha * t ** 2 + beta * t + gamma))





# 定义PCT计算函数
def polynomial_chirplet_transform(signal, alpha, beta, gamma):
    n = len(signal)
    pct_result = np.zeros(n, dtype=complex)

    for i in range(n):
        t_shifted = t - t[i]
        chirplet_function = chirplet(alpha, beta, gamma, t_shifted)
        pct_result[i] = np.sum(signal * chirplet_function)

    return np.abs(pct_result)  # 提取振幅信息

if __name__ == "__main__":

    sample_rate = 100
    t = np.arange(0, 10, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

    # 设置Chirplet参数
    alpha = 0.1
    beta = 0.5
    gamma = 0.0

    # 执行PCT
    pct_result = polynomial_chirplet_transform(signal, alpha, beta, gamma)

    # 绘制PCT结果
    plt.figure(figsize=(12, 6))

    # 信号图
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # PCT结果图
    plt.subplot(2, 1, 2)
    plt.plot(t, pct_result)
    plt.title('Polynomial Chirplet Transform Result')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
