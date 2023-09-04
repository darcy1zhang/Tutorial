def adjust_length(inputdata):
    """
    Description:
        Adjust the length of input data to the nearest power of 2 by zero-padding.

    Params:
        inputdata (numpy.ndarray or list): Input data to be adjusted.

    Returns:
        adjusted_data (numpy.ndarray): Data with adjusted length.
        new_length (int): The new length of the data after adjustment.
    """
    s = inputdata
    np = len(s)
    pow_of_2 = 1
    while 2 * pow_of_2 < np:
        pow_of_2 *= 2
    new_np = 2 * pow_of_2

    if np == new_np:
        return s, np
    else:
        new_s = [0] * new_np
        new_s[:np] = s
        return new_s, new_np

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
