import numpy as np
import matplotlib.pyplot as plt

def smoothwt(modulus, subrate, flag=False):
    """
    Description:
        Smooth the wavelet (or Gabor) transform in the time direction.

    Params:
        modulus: Time-Frequency representation (real valued).
        subrate: Length of smoothing window.
        flag: If set to TRUE, subsample the representation (default is FALSE).

    Returns:
        2D array containing the smoothed transform.
    """
    rows, cols = modulus.shape  # Get the dimensions of the input representation
    smoothed_transform = np.zeros((rows, cols))  # Initialize the smoothed transform


    if flag:
        # If flag is set to TRUE, subsample the representation
        smoothed_transform = modulus[::subrate, ::subrate]
    else:
        # Apply smoothing to the representation using a moving average window
        for i in range(rows):
            for j in range(cols):
                start_row = max(0, i - subrate // 2)
                end_row = min(rows, i + subrate // 2 + 1)
                start_col = max(0, j - subrate // 2)
                end_col = min(cols, j + subrate // 2 + 1)

                # Calculate the average within the specified window
                smoothed_transform[i, j] = np.mean(modulus[start_row:end_row, start_col:end_col])

    return smoothed_transform

if __name__ == "__main__":
    from cwt import *
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

    cwt_result = cwt(signal, noctave, nvoice, w0)

    # 创建示例的时间-频率表示
    modulus = cwt_result.real

    # 平滑时间-频率表示
    subrate = 5  # 平滑窗口大小
    smoothed_result = smoothwt(modulus, subrate, flag=False)

    # 绘制原始和平滑后的时间-频率表示
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(modulus, aspect='auto', cmap='viridis')
    plt.title("Original Time-Frequency Representation")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed_result, aspect='auto', cmap='viridis')
    plt.title("Smoothed Time-Frequency Representation")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
