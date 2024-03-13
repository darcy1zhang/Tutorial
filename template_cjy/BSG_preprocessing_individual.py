import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def cal_energy_individual(signal, window=32) -> np.ndarray:
    signal_padded = np.pad(signal, [window//2, window//2], 'constant', constant_values=(0, 0))

    squared_signal = np.square(signal_padded)
    start, end = 0, 0
    signal_energy = []
    for i in range(len(signal_padded)-window):
        start = i
        end = i + window
        signal_energy.append(np.sum(squared_signal[start:end]) / window)
    return np.array(signal_energy)

def segmentation_nk2_individual(signal, label, envelope, alpha=0.1, resample_rate=1, left_ratio=0.6, right_ratio=0.6, min_circles=3):
    hr = label[2]

    peaks_temp, _ = find_peaks(envelope, height=np.max(envelope) * alpha, distance=(int(5000*resample_rate) // hr))

    if len(peaks_temp) <= 4:
        peaks_dis_mean = int(6000*resample_rate) // hr
    else:
        peaks_dis_mean = np.int(np.mean(np.diff(peaks_temp)))

    peaks, _ = find_peaks(envelope, height=np.mean(envelope), distance=(int(5000*resample_rate) // hr))

    epoch_start, epoch_end = int(left_ratio * peaks_dis_mean), int(right_ratio * peaks_dis_mean)

    pieces = []
    for i in range(len(peaks)):
        if peaks[i] - epoch_start < 0:
            continue
        if peaks[i] + epoch_end > int(1000*resample_rate):
            break

        start = peaks[i] - epoch_start
        end = peaks[i] + epoch_end
        piece = signal[start: end]
        pieces.append(piece)

    if len(pieces) < min_circles:
        return None

    return pieces

def find_last_zero_position(arr):
    zero_indices = np.where(arr == 0)[0]

    for zero_pos in zero_indices:
        if zero_pos <= 3:
            continue
        if arr[zero_pos-3] == 0 and arr[zero_pos-2] == 0 and arr[zero_pos-1] == 0 and arr[zero_pos+1] != 0:
            return zero_pos
    if len(zero_indices) > 0:
        return zero_indices[-1]
    else:
        return -1

def re_align(pieces):
    padded_pieces = []
    for piece in pieces:
        padded_pieces.append(np.pad(piece, (35, 35), 'constant'))

    padded_pieces = np.array(padded_pieces)

    for i in range(len(padded_pieces)):
        time_series2 = padded_pieces[i]
        ccf_results = 0

        for j in range(len(padded_pieces)):
            if i == j:
                continue
            time_series1 = padded_pieces[j]
            ccf_result = np.correlate(time_series1 - np.mean(time_series1), time_series2 - np.mean(time_series2), mode='full') / (np.std(time_series1) * np.std(time_series2) * len(time_series1))
            ccf_results += ccf_result

        lags = np.arange(-len(time_series1) + 1, len(time_series2))
        max_correlation_index = np.argmax(ccf_results)
        optimal_lag = lags[max_correlation_index]

        if optimal_lag > 30:
            optimal_lag = 30
        elif optimal_lag < -30:
            optimal_lag = -30
        new_time_series2 = np.roll(time_series2, optimal_lag)
        padded_pieces[i] = new_time_series2


    zero_start_max, zero_end_max = 0, 0
    for piece in padded_pieces:
        zero_start = find_last_zero_position(piece)
        zero_end = find_last_zero_position(piece[::-1])

        zero_start_max = max(zero_start, zero_start_max)
        zero_end_max = max(zero_end, zero_end_max)
    try:
        # realigned_all_pieces.append(np.array(padded_pieces)[:, zero_start_max:-zero_end_max])
        if zero_end_max == 0:
            return padded_pieces[:, zero_start_max:]
        else:
            return padded_pieces[:, zero_start_max:-zero_end_max]
    except IndexError:
        print(len(padded_pieces))
        print(padded_pieces.shape)
        print(len(padded_pieces[0]))
        print(zero_start_max)
        print(zero_end_max)
        for piece in padded_pieces:
            plt.plot(piece)
        plt.show()
        raise IndexError("IndexError")

def filter_by_template(realigned_pieces, ratio=0.1, energy_threshold=0.15):
    template, good_realigned_pieces = QualityAssess4Template(realigned_pieces)

    if template is None:
        return None

    energy = np.square(template)
    length = len(template)
    side_length = int(length * ratio)
    all_energy = np.sum(energy)
    left_energy = np.sum(energy[:side_length])
    right_energy = np.sum(energy[-side_length:])

    if left_energy > all_energy * energy_threshold:
        return None
    elif right_energy > all_energy * energy_threshold:
        return None

    return good_realigned_pieces, template

def QualityAssess4Template(pieces, corrcoef_threshold=0.65):
    mean_template = np.mean(pieces, 0)

    coefs = []
    for piece in pieces:
        coef = np.corrcoef(mean_template, piece)[0, 1]
        coefs.append(coef)

    coefs_mean = np.mean(coefs)
    coefs_np = np.array(coefs)

    if coefs_mean < corrcoef_threshold:
        return None, None
    else:
        good_coefs_ids = np.where(coefs_np > corrcoef_threshold)[0]
        good_pieces = pieces[good_coefs_ids]
        good_mean_template = np.mean(good_pieces, 0)
        return good_mean_template, good_pieces

def processs_data_individual(signal, label, resample_rate=1):
    """"
    output:
    if the quality of this signal is okay, processs_data_individual will return 3 types of data
    energies: the energy envelope of original signal
    temp[0]: good heartbeat circles could be used for template extraction. shape: [good_heartbeat_number, heartbeat_length]
    temp[1]: template which is calculated by taking averaging of temp[0]. shape: [heartbeat_length]

    if the quality of this signal is too bad to extract template, processs_data_individual will return (energies, None, None)

    there are lots of variable, but the default value is okay.
    """

    # get the energy envelope
    energies = cal_energy_individual(signal, window=int(48*resample_rate))

    # segment the signal by energy envelope, pieces_energy includes good heartbeats and bad heartbeats
    pieces_energy = segmentation_nk2_individual(signal, label, energies, alpha=0.1, resample_rate=resample_rate)

    # if there are few good heartbeats in one piece of signal, pieces_energy is None
    if pieces_energy is None:
        return energies, None, None

    # heartbeats have some phase variation. we need to use linear phase shift to adjust it.
    realigned_pieces = re_align(pieces_energy[:-1])

    # we use realigned heartbeats to extract template, if the generated template's quality is bad, temp is None.
    temp = filter_by_template(realigned_pieces)
    if temp is None:
        return energies, None, None
    else:
        # successfully!
        return energies, temp[0], temp[1]