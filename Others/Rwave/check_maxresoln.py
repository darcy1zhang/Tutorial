def check_maxresoln(maxresoln, np):
    """
    Description:
        Stop when 2^maxresoln is larger than the signal size.

    Params:
        maxresoln (int): Number of decomposition scales.
        np (int): Signal size.

    Raises:
        ValueError: If 2^(maxresoln+1) is greater than np, indicating that maxresoln is too large.

    Returns:
        None
    """
    if 2**(maxresoln + 1) > np:
        raise ValueError("maxresoln is too large for the given signal")

if __name__ == "__main__":
    import os
    import numpy as np

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "data",
                               signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    check_maxresoln(10, len(signal))