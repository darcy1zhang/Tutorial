import numpy as np
import tsfel
import matplotlib.pyplot as plt
import json

def temporal_feature_extract(signal,index,fs, window_size):
    with open('./all_features.json', 'r') as file:
        cfg_file = json.load(file)
    
    new_use_value = "yes" 
    data["temporal"][list(data["temporal"].keys())[index]]["use"] = new_use_value


    with open('your_file.json', 'w') as file:
        json.dump(data, file, indent=4)

    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=fs, window_size=window_size).values.flatten()

    return features

if __name__ == '__main__':
    import numpy as np
    import tsfel
    import matplotlib.pyplot as plt
    import json

    signal = np.load("../data/simu_20000_0.1_90_140_train.npy")[0,:1000]
    fs = 100
    t = np.linspace(0, 10, 10 * fs)

    feature = feature(singal, 3, fs, len(signal))