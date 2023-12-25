import numpy as np

def adjust_length(signal):
    """
    Description:
        Adjust the length of input data to the nearest power of 2 by zero-padding.

    Params:
        signal (numpy.ndarray or list): Input data to be adjusted.

    Returns:
        new_signal (numpy.ndarray): Signal with adjusted length.
        new_len (int): The new length of the data after adjustment.
    """
    old_signal = np.array(signal)  # Convert inputdata to a NumPy array if it's not already.
    old_len = len(old_signal)
    pow_of_2 = 1
    while 2 * pow_of_2 < old_len:
        pow_of_2 *= 2
    new_len = 2 * pow_of_2

    if old_len == new_len:
        return old_signal, old_len
    else:
        new_signal = np.zeros(new_len)  # Create a new NumPy array filled with zeros.
        new_signal[:old_len] = old_signal
        return new_signal, new_len


if __name__ == "__main__":
    import os
    import numpy as np

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    # Adjust the length of the input data
    adjusted_data, new_length = adjust_length(signal)

    print("Old Length:", len(signal))
    print("New Length:", new_length)
