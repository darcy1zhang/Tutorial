from typing import TypedDict, Union
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import acf
from scipy.signal import resample_poly
import os
import pandas as pd
# from tsfresh import extract_features
from scipy.signal import hilbert, savgol_filter, periodogram

def wavelet_denoise(data, wave, Fs=None, n_decomposition=None):
    a = data
    w = wave
    ca = []
    cd = []
    rec_a = []
    rec_d = []
    freq_range = []

    for i in range(n_decomposition):
        if i == 0:
            freq_range.append(Fs/2)
        freq_range.append(Fs/2/(2**(i+1)))
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec = pywt.waverec(coeff_list, w)
        rec_a.append(rec)
        # ax3[i].plot(Fre, FFT_y1)
        # print(max_freq)

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    return rec_a, rec_d


def signal_quality_assessment(
        x, n_decomposition, Fs, n_lag, denoised_method='DWT'):
    x = (x - np.mean(x)) / np.std(x)

    if denoised_method == 'DWT':
        rec_a, rec_d = wavelet_denoise(data=x, wave='db7', Fs=Fs, n_decomposition=n_decomposition)
        min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4]))  # len(rec_a[-1]) len(rec_d[-5])
        denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][:min_len]  # + rec_a[-1][:min_len]

    index = 0
    window_size = 100
    z = hilbert(denoised_sig)  # form the analytical signal
    envelope = np.abs(z)

    # checkpoint3, mode = 'nearest'
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope)) / np.std(smoothed_envelope)

    # de-trend
    trend = savgol_filter(smoothed_envelope, 201, 2, mode='nearest')
    smoothed_envelope = smoothed_envelope - trend
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope)) / np.std(smoothed_envelope)

    acf_x = acf(smoothed_envelope, nlags=n_lag)
    acf_x = acf_x - np.mean(acf_x)

    # 获得acf_x的频域数据，包括freq, power
    f, Pxx_den = periodogram(acf_x, fs=100, nfft=1024)
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)

    # acf_x的每个窗口的平均值
    sig_means = []
    index = 0
    window_size = 100
    while index + window_size < len(acf_x):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size

    if np.std(sig_means) < 0.1 and 0.6 < frequency < 2.5 and power > 0.1:
        peak_ids, _ = find_peaks(acf_x, height=np.mean(acf_x))
        time_diff = peak_ids[1:] - peak_ids[:-1]

        # check if the peaks is periodic
        candidates = []
        for peak_id in peak_ids:
            if peak_id > 51 and peak_id < 167:
                candidates.append(peak_id)

        if len(candidates) > 1:
            sorted_ids = sorted(range(len(acf_x[candidates])), key=lambda k: acf_x[candidates][k])
            height = acf_x[candidates][sorted_ids[-1]]
            if (height - acf_x[candidates][sorted_ids[-2]] > 0.2) or (
                    height - acf_x[candidates][sorted_ids[-2]] > 0.5 * height):
                median_hr = candidates[sorted_ids[-1]]
            elif sorted_ids[-2] > sorted_ids[-1]:
                median_hr = candidates[sorted_ids[-1]]
            elif len(candidates) > 2:
                median_hr = candidates[np.argmax(acf_x[candidates])]
            else:
                median_hr = candidates[sorted_ids[-2]]
        elif len(candidates) == 1:
            median_hr = candidates[0]
        else:
            median_hr = np.median(time_diff)

        frequency = 1 / (median_hr / 100)
        if len(peak_ids) > 3:
            res = ['good data', np.std(sig_means), frequency, power]
        else:
            res = ['bad data', np.std(sig_means), frequency, power]
    else:
        res = ['bad data', np.std(sig_means), frequency, power]
    return res, denoised_sig


def zscore_2d(matrix):
    matrix_mean = matrix.mean(1, keepdims=True)
    matrix_std = matrix.std(1, keepdims=True)
    return (matrix - matrix_mean) / matrix_std


def load_RealData(dataset_path_train=None, dataset_path_test=None):
    if dataset_path_train is None:
        dataset_path_train = '../../../Data/RealData/all_back_train.npy'
    if dataset_path_test is None:
        dataset_path_test = '../../../Data/RealData/all_back_test.npy'

    train_data, test_data = np.load(dataset_path_train), np.load(dataset_path_test)
    signals_train, labels_train = train_data[:, :1000], train_data[:, 1000:]
    signals_test, labels_test = test_data[:, :1000], test_data[:, 1000:]

    print(signals_train.shape, labels_test.shape)

    return labels_train, labels_test, zscore_2d(signals_train), zscore_2d(signals_test)


class MyDict(TypedDict):
    key1: np.ndarray
    key2: np.ndarray
    key3: Union[None, np.ndarray]


def pack(signals, labels, ids=None):
    return {'signals': signals, 'labels': labels, 'ids': ids}

def unpack(package):
    return package['signals'], package['labels'], package['ids']

def pack_choice(package, num):
        return package['signals'][:, :num], package['labels'][:, :num], package['ids'][:, :num]


def segmentation_nk2(package: MyDict, envelopes, alpha, resample_rate, show=False):
    signals, labels, ids = unpack(package)

    all_pieces = []
    ids_output = []
    cnt4filter = []

    number = 0

    for cnt, index in enumerate(ids):
        signal = signals[cnt]
        envelope = envelopes[cnt]
        hr = labels[cnt, 2]

        peaks_temp, _ = find_peaks(envelope, height=np.max(envelope) * 0.15, distance=(int(5500*resample_rate) // hr))

        if len(peaks_temp) <= 4:
            peaks_dis_mean = int(6000*resample_rate) // hr
        else:
            peaks_dis_mean = np.int(np.mean(np.diff(peaks_temp)))

        peaks, _ = find_peaks(envelope, height=np.max(envelope) * alpha, distance=(int(5000*resample_rate) // hr))

        if show:
            plt.figure(figsize=(12, 3))
            plt.plot(envelope)
            plt.plot(signal)
            plt.scatter(peaks, envelope[peaks])
            plt.show()

        # epoch_start, epoch_end = int(0.45 * peaks_dis_mean), int(0.5 * peaks_dis_mean)
        epoch_start, epoch_end = int(0.6 * peaks_dis_mean), int(0.6 * peaks_dis_mean)

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

        if len(pieces) < 3:
            number += 1
            continue

        ids_output.append(index)
        cnt4filter.append(cnt)
        all_pieces.append(pieces)

    signals_output = signals[cnt4filter]
    labels_output = labels[cnt4filter]
    ids_output = np.array(ids_output)
    package_output = pack(signals_output, labels_output, ids_output)
    print(f'filter out {number} signals')

    return package_output, all_pieces


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

def re_align(all_pieces):
    realigned_all_pieces = []

    for cnt, pieces in enumerate(all_pieces):
        # print(cnt, len(pieces))
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
                realigned_all_pieces.append(padded_pieces[:, zero_start_max:])
            else:
                realigned_all_pieces.append(padded_pieces[:, zero_start_max:-zero_end_max])
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

    return realigned_all_pieces

def QualityAssess4Template(pieces):
    mean_template = np.mean(pieces, 0)

    coefs = []
    for piece in pieces:
        coef = np.corrcoef(mean_template, piece)[0, 1]
        coefs.append(coef)

    coefs_mean = np.mean(coefs)
    coefs_np = np.array(coefs)

    if coefs_mean < 0.65:
        return None, None
    else:
        good_coefs_ids = np.where(coefs_np > 0.65)[0]
        good_pieces = pieces[good_coefs_ids]
        good_mean_template = np.mean(good_pieces, 0)
        return good_mean_template, good_pieces

def denoise(package: MyDict) -> MyDict:
    signals_input, labels_input, ids_input = unpack(package)

    ids = []
    signals_output = []

    for cnt, BCG in enumerate(signals_input):
        res, denoised_sig = signal_quality_assessment(x=BCG, n_decomposition=6, Fs=100, n_lag=len(BCG)//2)
        if res[0] == 'good data':
            ids.append(cnt)
            signals_output.append(resample_poly(denoised_sig, 1000, len(denoised_sig)))
            # print(len(denoised_sig))

    print(len(ids))

    labels_output = labels_input[ids]

    return pack(np.array(signals_output), np.array(labels_output), np.array(ids))

# # energy
def cal_energy(package, window=32, show=False) -> np.ndarray:
    signals, _, _ = unpack(package)
    energies = []
    for signal in signals:

        signal_padded = np.pad(signal, [window//2, window//2], 'constant', constant_values=(0, 0))

        squared_signal = np.square(signal_padded)

        start, end = 0, 0
        signal_energy = []
        for i in range(len(signal_padded)-window):
            start = i
            end = i + window
            signal_energy.append(np.sum(squared_signal[start:end]) / window)

        energies.append(np.array(signal_energy))
        if show:
            plt.figure(figsize=(12, 3))
            plt.plot(signal)
            plt.plot(signal_energy)
            plt.show()
    return np.array(energies)


def filter_by_template(package, realigned_all_pieces):
    signals, labels, ids = unpack(package)

    cnts = []
    new_realigned_all_pieces = []
    templates = []

    for cnt, index in enumerate(ids):
        realigned_pieces = realigned_all_pieces[cnt]
        template, good_realigned_pieces = QualityAssess4Template(realigned_pieces)

        if template is None:
            continue

        energy = np.square(template)
        length = len(template)
        ratio = 0.1
        side_length = int(length * ratio)
        all_energy = np.sum(energy)
        left_energy = np.sum(energy[:side_length])
        right_energy = np.sum(energy[-side_length:])

        if left_energy > all_energy * 0.075:
            continue
        elif right_energy > all_energy * 0.075:
            continue
        else:
            cnts.append(cnt)
            new_realigned_all_pieces.append(good_realigned_pieces)
            templates.append(template)

    print(len(cnts))
    return pack(signals[cnts], labels[cnts], ids[cnts]), new_realigned_all_pieces, templates

def process_data(package_input, resample_rate=1):

    energies = cal_energy(package_input, window=int(48*resample_rate))
    filtered_package, all_pieces_energy = segmentation_nk2(package_input, energies, 0.15, resample_rate, show=False)
    realigned_all_pieces = re_align(all_pieces_energy)

    return filter_by_template(filtered_package, realigned_all_pieces)


