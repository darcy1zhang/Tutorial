import numpy as np
import json
import tsfel
import os


current_file_path = os.path.abspath(__file__)
signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data", signal_name)
signal = np.load(signal_path)[2, :1000]
fs = 100

with open("./all_features.json", 'r') as file:
    cgf_file = json.load(file)


# cgf_file = tsfel.get_features_by_domain("temporal")

features = tsfel.time_series_features_extractor(cgf_file, signal, fs=fs, window_size=len(signal), features_path="my_features.py").values.flatten()
    # .values.flatten()
print(features.shape)
print(features)