import numpy as np
import tsfel
import matplotlib.pyplot as plt
import json

# 目前还有一点问题

def temporal_feature_extract(signal,index,fs, window_size):
    """
    Description:
        Extract temporal features from a signal using Time Series Feature Extraction Library (tsfel).

    Params:
        signal (numpy.ndarray): The input signal.
        index (int): The index of the feature configuration in the configuration file.
        fs (int): The sample rate of the signal.
        window_size (int): The size of the analysis window in samples.

    Returns:
        features (numpy.ndarray): An array containing the extracted temporal features.
    """
    with open('./all_features.json', 'r') as file:
        cfg_file = json.load(file)
    
    new_use_value = "yes" 
    cfg_file["temporal"][list(cfg_file["temporal"].keys())[index]]["use"] = new_use_value


    with open('your_file.json', 'w') as file:
        json.dump(cfg_file, file, indent=4)

    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=fs, window_size=window_size).values.flatten()

    return features

if __name__ == '__main__':
    import numpy as np
    import tsfel
    import matplotlib.pyplot as plt
    import json
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")
    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data", signal_name)
    signal = np.load(signal_path)[0, :1000]
    fs = 100

    t = np.linspace(0, 10, 10 * fs)

    feature = temporal_feature_extract(signal, 3, fs, len(signal))