import numpy as np


def ridge_icm(tfr, max_iter=100, lam=0.1):
    """Ridge estimation using ICM

    Args:
        tfr (np.ndarray): Time-frequency representation, shape (time, freq)
        max_iter (int): Maximum number of iterations
        lam (float): Regularization parameter

    Returns:
        ridge (np.ndarray): Estimated ridge, shape (time,)
        cost (np.ndarray): Cost function, shape (iter,)
    """

    time, freq = tfr.shape
    ridge = np.zeros(time, dtype=int)
    cost = np.zeros(max_iter)

    for i in range(max_iter):
        new_ridge = ridge.copy()
        cost[i] = _compute_cost(ridge, tfr, lam)
        for t in range(time):
            new_ridge[t] = _update_ridge(ridge, t, tfr, lam)
        if np.all(ridge == new_ridge):
            break
        ridge = new_ridge

    return ridge, cost


def _compute_cost(ridge, tfr, lam):
    """Compute ICM cost function"""
    cost = 0
    for t, f in enumerate(ridge):
        cost += tfr[t, f]
    cost -= lam * (np.diff(ridge) == 0).sum()
    return cost


def _update_ridge(ridge, t, tfr, lam):
    """Update ridge location at time t"""
    freq = ridge[t]
    costs = np.zeros_like(tfr[t])
    costs[ridge == freq] -= lam
    costs[ridge == ridge[t - 1]] += lam
    return np.argmax(costs)


if __name__ == "__main__":
    import numpy as np
    from scipy import signal

    # 生成测试数据
    t = np.linspace(0, 1, 1000)
    f1, f2 = 10, 20
    x = np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)

    # 计算STFT
    f, t, Zxx = signal.stft(x, fs=1000)

    # 脊线估计
    ridge, cost = ridge_icm(np.abs(Zxx), max_iter=100, lam=0.1)

    # 绘制STFT和检测脊线
    import matplotlib.pyplot as plt

    plt.imshow(np.abs(Zxx), aspect='auto', origin='lower')
    plt.plot(t, ridge, 'r-')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()