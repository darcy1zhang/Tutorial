import warnings
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from scipy.signal import welch, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
from numpy.fft import fft, ifft, fftfreq
from sklearn.metrics import mean_squared_error
import pywt
import tftb
from tqdm import tqdm
import ssqueezepy as sq
from pylab import (arange, flipud, linspace, cos, pi, log, hanning,
                   ceil, log2, floor, empty_like, fft, ifft, fabs, exp, roll, convolve)
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastsst import SingularSpectrumTransformation
plt.rcParams['figure.figsize'] = [10, 3]

# Time Domain
## Generate SCG   
def scg_simulate(**kwargs):
    """
    Description:
        The main function to generate a synthetic SCG dataset
    Args:
        num_rows: default = 1
            Number of samples in the dataset
        duration: default = 10 
            Length of signal
        sampling_rate: default = 100
            Sampling rate of signal
        heart_rate: default = (50,150)
            The range of heart rate
        add_respiratory: default = True
            Whether to add respiratory
        respiratory_rate: default = (10,30)
            The range of the respiratory_rate
        systolic: default = (90,140)
            The range of the systolic
        diastolic: default = (80,100)
            The range of the diastolic
        pulse_type: default = "db"
            Type of wavelet to form a basic waveform of SCG
        noise_type: default = ["basic"]
            Type of added noise
        noise_shape: default = "laplace"
            Shape of the basic noise
        noise_amplitude: default = 0.1
            Amplitude of basic noise
        noise_frequency: default = [5,10,100]
            Frequency of basic noise
        powerline_amplitude: default = 0
            Amplitude of the powerline noise (relative to the standard deviation of the signal)
        powerline_frequency: default = 50
            Frequency of the powerline noise 
        artifacts_amplitude: default = 0
            Amplitude of the artifacts (relative to the standard deviation of the signal)
        artifacts_frequency: default = 100
            Frequency of the artifacts
        artifacts_number: default = 5
            Number of artifact bursts. The bursts have a random duration between 1 and 10% of the signal duration
        artifacts_shape: default = "laplace"
            Shape of the artifacts
        n_echo: default = 3
            Number of echo repetitions to add
        attenuation_factor: default = [0.1, 0.05, 0.02]
            List of attenuation factors for each echo
        delay_factor: default = [5] * 3
            List of delay factors (in samples) for each echo
        random_state: default = None
            Seed for the random number generator. Keep it fixed for reproducible results
        silent: default = False
            Whether or not to display warning messages
        data_file: default = "./data.npy"
            The path to generate the dataset
    Returns:
        A synthetic SCG dataset in the specified path
    """
    args = {
        'num_rows' : 1,
        'duration' : 10, 
        'sampling_rate' : 100,
        'heart_rate' : (50,150),
        'add_respiratory' : True,
        'respiratory_rate' : (10,30),
        'systolic' : (90,140),
        'diastolic' : (60,100),
        'pulse_type' : "db",
        'noise_type' : ["basic"],
        'noise_shape' : "laplace",
        'noise_amplitude' : 0.1,
        'noise_frequency' : [5,10,100],
        'powerline_amplitude' : 0,
        'powerline_frequency' : 50,
        'artifacts_amplitude' : 0,
        'artifacts_frequency' : 100,
        'artifacts_number' : 5,
        'artifacts_shape' : "laplace",
        'n_echo' : 3, 
        'attenuation_factor' : [0.1, 0.05, 0.02],
        'delay_factor' : [5] * 3,
        'random_state' : None,
        'silent' : False,
        'data_file' : "./data.npy"
    }

    args.update(kwargs)
    simulated_data = []

    for ind in tqdm(range(args['num_rows'])):
        heart_rate = random.randint(args['heart_rate'][0], args['heart_rate'][1])
        respiratory_rate = random.randint(args['respiratory_rate'][0], args['respiratory_rate'][1])

        systolic = random.randint(args['systolic'][0], args['systolic'][1])
        diastolic = random.randint(args['diastolic'][0], args['diastolic'][1])

        print('hr:', heart_rate, 'rr:', respiratory_rate, 
              'sp:', systolic, 'dp:', diastolic)
       
        data = _scg_simulate(
            duration = args['duration'], 
            sampling_rate = args['sampling_rate'], 
            heart_rate = heart_rate,  
            add_respiratory = args['add_respiratory'],
            respiratory_rate = respiratory_rate, 
            systolic = systolic, 
            diastolic = diastolic, 
            pulse_type = args['pulse_type'], 
            noise_type  =  args['noise_type'],
            noise_shape =  args['noise_shape'],
            noise_amplitude =  args['noise_amplitude'],
            noise_frequency = args['noise_frequency'],
            powerline_amplitude = args['powerline_amplitude'],
            powerline_frequency = args['powerline_frequency'],
            artifacts_amplitude = args['artifacts_amplitude'],
            artifacts_frequency = args['artifacts_frequency'],
            artifacts_number = args['artifacts_number'],
            artifacts_shape = args['artifacts_shape'],
            n_echo = args['n_echo'], 
            attenuation_factor = args['attenuation_factor'],
            delay_factor = args['delay_factor'],
            random_state = args['random_state'],
            silent = args['silent']
        )
        ## duration * sampling_rate + 6 size. 6 are [mat_int(here 0 for synthetic data), time_stamp, hr, rr, sbp, dbp]
        simulated_data.append(list(data)+[0]+[ind]+[heart_rate]+[respiratory_rate]+[systolic]+[diastolic])

    simulated_data = np.asarray(simulated_data)
    if args['num_rows'] == 1:
        return simulated_data.flatten()
    else:
        np.save(args['data_file'], simulated_data)
        print(f"{args['data_file']} is generated and saved!")

def _scg_simulate(**kwargs):
    """
    Description:
        Generate a synthetic scg signal of a given duration and sampling rate to roughly approximate cardiac cycles.
    Args:
        duration: length of signal
        sampling_rate: sampling rate of signal
        heart_rate: the range of heart rate
        add_respiratory: whether to add respiratory
        respiratory_rate: value of respiratory rate
        systolic: value of systolic
        diastolic: value of diastolic
        pulse_type: type of wavelet to form a basic waveform of SCG
        noise_type: type of added noise
        noise_shape: shape of the basic noise
        noise_amplitude: amplitude of basic noise
        noise_frequency: frequency of basic noise
        powerline_amplitude: amplitude of the powerline noise (relative to the standard deviation of the signal)
        powerline_frequency: frequency of the powerline noise 
        artifacts_amplitude: amplitude of the artifacts (relative to the standard deviation of the signal)
        artifacts_frequency: frequency of the artifacts
        artifacts_number: number of artifact bursts. The bursts have a random duration between 1 and 10% of the signal duration
        artifacts_shape: shape of the artifacts
        n_echo: number of echo repetitions to add
        attenuation_factor: list of attenuation factors for each echo
        delay_factor: list of delay factors (in samples) for each echo
        random_state: seed for the random number generator. Keep it fixed for reproducible results
        silent: whether or not to display warning messages
    Returns
        scg: a vector of the scg signal.
    """
    args = {
        'duration' : 10, 
        'sampling_rate' : 100, 
        'heart_rate' : 70, 
        'add_respiratory': True,
        'respiratory_rate' : 20, 
        'systolic' : 120, 
        'diastolic' : 80, 
        'pulse_type' : "db", 
        'noise_type'  :  ["basic"],
        'noise_shape' : "laplace",
        'noise_amplitude' : 0,
        'noise_frequency' : [5,10,100],
        'powerline_amplitude' : 0,
        'powerline_frequency' : 50,
        'artifacts_amplitude' : 0,
        'artifacts_frequency' : 100,
        'artifacts_number' : 5,
        'artifacts_shape' : "laplace",
        'n_echo' : 3, 
        'attenuation_factor' : [0.1, 0.05, 0.02],
        'delay_factor' : [5] * 3,
        'random_state' : None,
        'silent' : False
    }

    args.update(kwargs)

    # Seed the random generator for reproducible results
    np.random.seed(args['random_state'])

    scg = _scg_simulate_wavelet(
        duration = args['duration'],
        sampling_rate = args['sampling_rate'],
        heart_rate = args['heart_rate'],
        add_respiratory = args['add_respiratory'],
        respiratory_rate = args['respiratory_rate'],
        systolic = args['systolic'],
        diastolic = args['diastolic'],
        pulse_type = args['pulse_type']
    )

    # Add random noise
    if args['noise_amplitude'] > 0:
        scg = signal_distort(
            signal = scg,
            sampling_rate = args['sampling_rate'],
            noise_type  =  args['noise_type'],
            noise_shape = args['noise_shape'],
            noise_amplitude = args['noise_amplitude'],
            noise_frequency = args['noise_frequency'],
            powerline_amplitude = args['powerline_amplitude'],
            powerline_frequency = args['powerline_frequency'],
            artifacts_amplitude = args['artifacts_amplitude'],
            artifacts_frequency = args['artifacts_frequency'],
            artifacts_number = args['artifacts_number'],
            artifacts_shape = args['artifacts_shape'],
            n_echo = args['n_echo'], 
            attenuation_factor = args['attenuation_factor'],
            delay_factor = args['delay_factor'],
            random_state = args['random_state'],
            silent = args['silent']
        )

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return scg


def _scg_simulate_wavelet(**kwargs):
    """
    Description:
        Generate a synthetic scg signal of given pulse type without noise
    Args:
        duration: length of signal
        sampling_rate: sampling rate of signal
        heart_rate: the range of heart rate
        add_respiratory: whether to add respiratory
        respiratory_rate: value of respiratory rate
        systolic: value of systolic
        diastolic: value of diastolic
        pulse_type: type of wavelet to form a basic waveform of SCG
    Returns:
        scg: a scg signal of given pulse type without noise
    """
    args = {
        'duration' : 10, 
        'sampling_rate' : 100, 
        'heart_rate' : 70, 
        'add_respiratory' : True,
        'respiratory_rate' : 20, 
        'systolic' : 120, 
        'diastolic' : 80, 
        'pulse_type' : "db"
    }

    args.update(kwargs)

    cardiac_length = int(100 * args['sampling_rate'] / args['heart_rate']) 
    
    if args['pulse_type'] == "db":
        ind = random.randint(17, 34) 
        db = pywt.Wavelet(f'db{ind}')
        dec_lo, dec_hi, rec_lo, rec_hi = db.filter_bank
        dec_lo = np.array(dec_lo)[::-1]
        cardiac_s = dec_lo
        cardiac_d = dec_lo * 0.3 * args['diastolic'] / 80 # change height to 0.3
        cardiac_s = scipy.signal.resample(cardiac_s, 100)
        cardiac_d = scipy.signal.resample(cardiac_d, 100)
        
    elif args['pulse_type'] == "mor":
        ind = random.randint(5, 55)
        cardiac_s = scipy.signal.morlet(40,ind/10).real
        cardiac_d = scipy.signal.morlet(40,ind/10).real * 0.3 * args['diastolic'] / 80 # change height to 0.3
        cardiac_s = np.concatenate((cardiac_s,np.zeros(60)))
        cardiac_d = np.concatenate((cardiac_d,np.zeros(60)))

    elif args['pulse_type'] == "ricker":
        ind = random.randint(10, 30)
        cardiac_s = scipy.signal.ricker(40,ind/10)
        cardiac_d = scipy.signal.ricker(40,ind/10)*0.3 * args['diastolic'] / 80 # change height to 0.3
        cardiac_s = np.concatenate((cardiac_s,np.zeros(60)))
        cardiac_d = np.concatenate((cardiac_d,np.zeros(60)))
        
    elif args['pulse_type'] == "sym":
        ind = np.random.randint(12, 20)
        wavelet = pywt.Wavelet(f"sym{ind}")
        dec_lo = wavelet.dec_lo[::-1]
        dec_lo = np.append(dec_lo, np.zeros(20))
        cardiac_s = dec_lo
        cardiac_d = dec_lo * 0.3 * args['diastolic'] / 80 # change height to 0.3
        cardiac_s = scipy.signal.resample(cardiac_s, 100)
        cardiac_d = scipy.signal.resample(cardiac_d, 100)
    
    elif args['pulse_type'] == "coif":
        ind = np.random.randint(5, 17)
        wavelet = pywt.Wavelet(f"coif{ind}")
        dec_lo = wavelet.dec_lo[::-1]
        length = int(0.1 * len(dec_lo))
        dec_lo = dec_lo[length:]
        if len(dec_lo) < 100:
            dec_lo = np.append(dec_lo,np.zeros(100-len(dec_lo)))
        else:
            dec_lo = scipy.signal.resample(dec_lo, 100)
        cardiac_s = dec_lo
        cardiac_d = dec_lo * 0.3 * args['diastolic'] / 80 # change height to 0.3
    
    else:
        raise Exception("The pulse_type contains: db, mor, ricker, sym, coif")

    cardiac_s = cardiac_s[0:40]
    distance = 180 - args['systolic'] 
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    cardiac = scipy.signal.resample(cardiac, cardiac_length) # fix every cardiac length to 1000/heart_rate

    # Caculate the number of beats in capture time period
    num_heart_beats = int(args['duration'] * args['heart_rate'] / 60)

    # Concatenate together the number of heart beats needed
    scg = np.tile(cardiac, num_heart_beats)

    # Resample
    scg = signal_resample(
        scg, 
        sampling_rate = int(len(scg) / 10),
        desired_length = args['sampling_rate'] * args['duration'],
        desired_sampling_rate = args['sampling_rate']
    )
    
    ### add rr
    if args['add_respiratory']:
        num_points = args['duration'] * args['sampling_rate']
        x_space = np.linspace(0,1,num_points)
        seg_fre = args['respiratory_rate'] / (60 / args['duration'])
        seg_amp = max(scg) * 0.00001
        rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
        scg *= (rr_component + 2 * seg_amp)
    else:
        scg *= 0.00001

    return scg

def signal_resample(
    signal,
    desired_length=None,
    sampling_rate=None,
    desired_sampling_rate=None
):
    """
    Description:
        Resample a continuous signal to a different length or sampling rate
    Args:
        signal: signal in the form of a vector of values.
        desired_length: desired length of the signal.
        sampling_rate: original sampling frequency
        desired_sampling_rate : desired sampling frequency
    Returns:
        resampled: a vector containing resampled signal values.
    """
    if desired_length is None:
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))

    # Sanity checks
    if len(signal) == desired_length:
        return signal

    # Resample
    resampled = scipy.ndimage.zoom(signal, desired_length / len(signal))
    
    return resampled


def signal_distort(**kwargs):
    """
    Description:
        Add noise of a given frequency, amplitude and shape to a signal.
    Args:
        signal: signal to distort
        sampling_rate: sampling rate of signal
        noise_type: type of added noise
        noise_shape: shape of the basic noise
        noise_amplitude: amplitude of basic noise
        noise_frequency: frequency of basic noise
        powerline_amplitude: amplitude of the powerline noise (relative to the standard deviation of the signal)
        powerline_frequency: frequency of the powerline noise 
        artifacts_amplitude: amplitude of the artifacts (relative to the standard deviation of the signal)
        artifacts_frequency: frequency of the artifacts
        artifacts_number: number of artifact bursts. The bursts have a random duration between 1 and 10% of the signal duration
        artifacts_shape: shape of the artifacts
        n_echo: number of echo repetitions to add
        attenuation_factor: list of attenuation factors for each echo
        delay_factor: list of delay factors (in samples) for each echo
        random_state: seed for the random number generator. Keep it fixed for reproducible results
        silent: whether or not to display warning messages
    Returns
        distorted: a vector containing the distorted signal
    """
    args = {
        'signal' : None,
        'sampling_rate' : 100,
        'noise_type' : ["basic"],
        'noise_shape' : "laplace",
        'noise_amplitude' : 0.1,
        'noise_frequency' : [5,10,100],
        'powerline_amplitude' : 0,
        'powerline_frequency' : 50,
        'artifacts_amplitude' : 0,
        'artifacts_frequency' : 100,
        'artifacts_number' : 5,
        'artifacts_shape' : "laplace",
        'n_echo' : 3, 
        'attenuation_factor' : [0.1, 0.05, 0.02],
        'delay_factor' : [5] * 3,
        'random_state' : None,
        'silent' : False,
    }

    args.update(kwargs)

    # Seed the random generator for reproducible results.
    np.random.seed(args['random_state'])

    # Make sure that noise_amplitude is a list.
    if isinstance(args['noise_amplitude'], (int, float)):
        noise_amplitude = [args['noise_amplitude']]

    signal_sd = np.std(args['signal'], ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if "basic" in args['noise_type']:
        if min(noise_amplitude) > 0:
            noise += _signal_distort_noise_multifrequency(
                args['signal'],
                signal_sd = signal_sd,
                sampling_rate = args['sampling_rate'],
                noise_amplitude = args['noise_amplitude'],
                noise_frequency = args['noise_frequency'],
                noise_shape = args['noise_shape'],
                silent = args['silent'],
            )
            
    if "resonance" in args['noise_type']:
        noise += _signal_distort_resonance(
            signal = args['signal'], 
            n_echo = args['n_echo'],
            attenuation_factor = args['attenuation_factor'],
            delay_factor = args['delay_factor']
        )
        

    # Powerline noise.
    if "powerline" in args['noise_type']:
        if args['powerline_amplitude'] > 0:
            noise += _signal_distort_powerline(
                signal = args['signal'],
                signal_sd = signal_sd,
                sampling_rate = args['sampling_rate'],
                powerline_frequency = args['powerline_frequency'],
                powerline_amplitude = args['powerline_amplitude'],
                silent = args['silent']
            )
    
    # Artifacts.
    if "artifacts" in args['noise_type']:
        if args['artifacts_amplitude'] > 0:
            noise += _signal_distort_artifacts(
                signal = args['signal'],
                signal_sd = signal_sd,
                sampling_rate = args['sampling_rate'],
                artifacts_frequency = args['artifacts_frequency'],
                artifacts_amplitude = args['artifacts_amplitude'],
                artifacts_number = args['artifacts_number'],
                silent = args['silent']
            )
    
    if "linear_drift" in args['noise_type']:
        noise += _signal_linear_drift(args['signal'])
    
    distorted = args['signal'] + noise

    return distorted

def _signal_distort_resonance(
    signal, n_echo=3, attenuation_factor=[0.1, 0.05, 0.02], delay_factor=[5] * 3
):
    """
    Description:
        Add echo noise to a signal.
    Args:
        signal: input signal to which echo noise will be added.
        n_echo: number of echo repetitions to add.
        attenuation_factor: list of attenuation factors for each echo.
        delay_factor: list of delay factors (in samples) for each echo.
    Returns:
        echo: a vector containing the echo noise
    """

    # Check the types and lengths of attenuation and delay factors
    if not isinstance(attenuation_factor, (list, np.ndarray)):
        raise ValueError("The type of attenuation_factor must be a list or numpy.ndarray")
    if not isinstance(delay_factor, (list, np.ndarray)):
        raise ValueError("The type of delay_factor must be a list or numpy.ndarray")
    if len(attenuation_factor) != n_echo or len(delay_factor) != n_echo:
        raise ValueError("The lengths of attenuation_factor and delay_factor should be equal to n_echo")

    # Create a copy of the original signal
    original_signal = signal.copy()
    echos = np.zeros(shape=original_signal.shape)
    # Iterate over each echo and apply attenuation and delay
    for a_factor, d_factor in zip(attenuation_factor, delay_factor):
        # Apply attenuation to the signal
        attenuation_signal = original_signal * a_factor

        # Shift the attenuated signal to create the echo effect
        attenuation_signal[d_factor:] = attenuation_signal[:-d_factor]
        attenuation_signal[:d_factor] = 0

        # Add the attenuated and delayed signal to the original signal
        echos += attenuation_signal

    return echos

def _signal_linear_drift(signal):

    n_samples = len(signal)
    linear_drift = np.arange(n_samples) * (1 / n_samples)

    return linear_drift


def _signal_distort_artifacts(
    signal,
    signal_sd=None,
    sampling_rate=100,
    artifacts_frequency=100,
    artifacts_amplitude=0,
    artifacts_number=5,
    artifacts_shape="laplace",
    silent=False,
):

    # Generate artifact burst with random onset and random duration.
    artifacts = _signal_distort_noise(
        len(signal),
        sampling_rate=sampling_rate,
        noise_frequency=artifacts_frequency,
        noise_amplitude=artifacts_amplitude,
        noise_shape=artifacts_shape,
        silent=silent,
    )
    if artifacts.sum() == 0:
        return artifacts

    min_duration = int(np.rint(len(artifacts) * 0.001))
    max_duration = int(np.rint(len(artifacts) * 0.01))
    artifact_durations = np.random.randint(min_duration, max_duration, artifacts_number)

    artifact_onsets = np.random.randint(0, len(artifacts) - max_duration, artifacts_number)
    artifact_offsets = artifact_onsets + artifact_durations

    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(artifacts_number):
        artifact_idcs[artifact_onsets[i] : artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        artifacts_amplitude *= signal_sd
    artifacts *= artifacts_amplitude

    return artifacts


def _signal_distort_powerline(
    signal, signal_sd=None, sampling_rate=100, powerline_frequency=50, powerline_amplitude=0, silent=False
):

    duration = len(signal) / sampling_rate
    powerline_noise = signal_simulate(
        duration=duration, sampling_rate=sampling_rate, frequency=powerline_frequency, amplitude=1, silent=silent
    )

    if signal_sd is not None:
        powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise


def _signal_distort_noise_multifrequency(
    signal,
    signal_sd=None,
    sampling_rate=100,
    noise_amplitude=0.1,
    noise_frequency=[5, 10, 100],
    noise_shape="laplace",
    silent=False,
):
    base_noise = np.zeros(len(signal))
    params = listify(noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape)

    for i in range(len(params["noise_amplitude"])):

        freq = params["noise_frequency"][i]
        amp = params["noise_amplitude"][i]
        shape = params["noise_shape"][i]

        if signal_sd is not None:
            amp *= signal_sd

        # Make some noise!
        _base_noise = _signal_distort_noise(
            len(signal),
            sampling_rate=sampling_rate,
            noise_frequency=freq,
            noise_amplitude=amp,
            noise_shape=shape,
            silent=silent,
        )
        base_noise += _base_noise

    return base_noise


def _signal_distort_noise(
    n_samples, sampling_rate=100, noise_frequency=[5, 10, 100], noise_amplitude=0.1, noise_shape="laplace", silent=False
):

    _noise = np.zeros(n_samples)
    # Apply a very conservative Nyquist criterion in order to ensure
    # sufficiently sampled signals.
    nyquist = sampling_rate * 0.1
    if noise_frequency > nyquist:
        if not silent:
            warnings.warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since it cannot be resolved at "
                f" the sampling rate of {sampling_rate} Hz. Please increase "
                f" sampling rate to {noise_frequency * 10} Hz or choose "
                f" frequencies smaller than or equal to {nyquist} Hz.",
                category=NeuroKitWarning
            )
        return _noise
    # Also make sure that at least one period of the frequency can be
    # captured over the duration of the signal.
    duration = n_samples / sampling_rate
    if (1 / noise_frequency) > duration:
        if not silent:
            warnings.warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since its period of {1 / noise_frequency} "
                f" seconds exceeds the signal duration of {duration} seconds. "
                f" Please choose noise frequencies larger than "
                f" {1 / duration} Hz or increase the duration of the "
                f" signal above {1 / noise_frequency} seconds.",
                category=NeuroKitWarning
            )
        return _noise

    noise_duration = int(duration * noise_frequency)

    if noise_shape in ["normal", "gaussian"]:
        _noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        _noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError("NeuroKit error: signal_distort(): 'noise_shape' should be one of 'gaussian' or 'laplace'.")

    if len(_noise) != n_samples:
        _noise = signal_resample(_noise, desired_length=n_samples)
    return _noise

class NeuroKitWarning(RuntimeWarning):
    """
    Description:
        Category for runtime warnings
    """

def listify(**kwargs):
    """
    Description:
        Normalizes the input keyword arguments by converting them into lists of equal length. 
        If an argument is a single value, it is replicated to match the length of the longest 
        input list. If an argument is a list shorter than the longest list, its last element 
        is repeated to achieve the required length.

    Args:
        **kwargs: Variable length keyword arguments. Each can be a single non-list value or a list. 
        Non-list values are treated as single-element lists.

    Returns:
        A dictionary with the original keys and their corresponding values extended to lists of 
        equal length.
    """
    args = kwargs
    maxi = 1

    # Find max length
    for key, value in args.items():
        if isinstance(value, str) is False:
            try:
                if len(value) > maxi:
                    maxi = len(value)
            except TypeError:
                pass

    # Transform to lists
    for key, value in args.items():
        if isinstance(value, list):
            args[key] = _multiply_list(value, maxi)
        else:
            args[key] = _multiply_list([value], maxi)

    return args

def _multiply_list(lst, length):
    q, r = divmod(length, len(lst))
    return q * lst + lst[:r]

## Template of SCG
def get_template(signal, threshold=0.000005):
    """
    Description:
        use cluster method to get the template
    Args:
        signal: the periodic signal
    Returns:
        The template of the periodic signal
    """
    peaks2 = get_peaks(signal)
    peaks2 = peaks2[1:-1]
    avg_index = (peaks2[::2] + peaks2[1::2]) // 2  # Use the mid of peaks as segment point
    splits = np.split(signal, avg_index)

    # Use the longest length as the length of template
    max_length = max(len(split) for split in splits)
    padded_splits = [np.pad(split, (0, max_length - len(split))) for split in splits]

    # Stack the segments
    stacked_array = np.vstack(padded_splits)
    stacked_array = np.delete(stacked_array, 0, axis=0)

    class PulseClustering:
        def __init__(self, threshold):
            self.threshold = threshold
            self.clusters = []

        def fit(self, pulses):
            for pulse in pulses:
                if not self.clusters:
                    self.clusters.append([pulse])
                else:
                    for cluster in self.clusters:
                        center_pulse = np.mean(cluster, axis=0)  # Use average to get the middle of the cluster
                        rmse = np.sqrt(mean_squared_error(center_pulse, pulse))  # Calculate RMSE distance
                        # If the distance between new signal and middle of cluster is less than shreshold, add it into
                        # the cluster
                        if rmse < self.threshold:
                            cluster.append(pulse)
                            break
                    # If the distance between new singal and middles of existing clusters is greater than shreshold,
                    # create a new cluster
                    else:
                        self.clusters.append([pulse])

        def get_clusters(self):
            return self.clusters

    clustering = PulseClustering(threshold)
    clustering.fit(stacked_array)
    clusters = clustering.get_clusters()
    num_pulses_per_cluster = [len(cluster) for cluster in clusters]
    max_cluster = max(clusters, key=len)
    average_pulse = np.mean(max_cluster, axis=0)  # Calculate the average of max cluster
    return average_pulse

## Analytic Signal and Hilbert Transform
def analytic_signal(x):
    """
    Description:
        Get the analytic version of the input signal
    Args:
        x: input signal which is a real-valued signal
    Returns:
        The analytic version of the input signal which is a complex-valued signal
    """
    N = len(x)
    X = fft(x, N)
    h = np.zeros(N)
    h[0] = 1
    h[1:N // 2] = 2 * np.ones(N // 2 - 1)
    h[N // 2] = 1
    Z = X * h
    z = ifft(Z, N)
    return z

def hilbert_transform(x):
    """
    Description:
        Get the hilbert transformation of the input signal
    Args:
        x: a real-valued singal
    Returns:
        Return the result of hilbert transformation which is the imaginary part of the analytic signal. It is a
        real-valued number.
    """
    z = analytic_signal(x)
    return z.imag

## Peak Detection
### Peak of Peak Algorithm
def get_peaks(signal):
    """
    Description:
        Detect peaks in a signal and perform linear interpolation to obtain an envelope.

    Params:
        signal (numpy.ndarray): The input signal.

    Returns:
        peaks (numpy.ndarray): An array containing the indices of the detected peaks.
    """
    t = np.arange(len(signal))
    peak_indices, _ = find_peaks(signal) # find all peaks in th signal

    # interpolate the peaks to form the envelope
    t_peaks = t[peak_indices]
    peak_values = signal[peak_indices]
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2, _ = find_peaks(envelope, distance=10) # find the peaks of envelope
    peaks2 = update_array(peaks2, signal) # remove wrong peaks

    # make sure the first peak is the higher peak
    if len(peaks2) > 1:
        if (signal[peaks2[1]] > signal[peaks2[0]]):
            peaks2 = np.delete(peaks2, 0)

    # make sure the number of peaks is even
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    return peaks2

def update_array(a, data_tmp):
    """
    Description:
        Update an array 'a' by removing elements based on the pattern in 'data_tmp'.

    Params:
        a (numpy.ndarray): The input array to be updated.
        data_tmp (numpy.ndarray): The data array used for comparison.

    Returns:
        updated_array (numpy.ndarray): The updated array after removing elements.
    """
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a

## Envelope Extraction
### Envelope from Peaks
def envelope_from_peaks(signal):
    """
    Description
        Interpolation the peaks to get the envelope of the input signal. The algorithm is only suitable for the signal
        with a lot of noise
    Args:
        signal: The input signal
    Returns:
        envelope: The envelope of the input signal
    """
    t = np.arange(len(signal))
    peak_indices, _ = find_peaks(signal)

    # interpolate the peaks to form the envelope
    t_peaks = t[peak_indices]
    peak_values = signal[peak_indices]
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    return envelope

### Average Envelope
def average_envelope(signal, window_length):
    """
    Description:
        Use the average window to get the envelope
    Args:
        signal: input signal
        window_length: the length of the average window
    Returns:
        envelope: the envelope of the input signal
    """
    weights = np.ones(window_length) / window_length
    envelope = np.convolve(np.abs(signal), weights, mode='valid')
    padding = (window_length - 1) // 2
    envelope = np.concatenate([np.zeros(padding), envelope, np.zeros(padding)])
    return envelope

### Envelope and Phase Extraction using Hilbert Transform
def inst_amplitude(signal):
    """
    Description:
        Use hilbert transformation to compute the instantaneous amplitude or the envelope of the input signal
    Args:
        signal: input signal
    Returns:
        The instantaneous amplitude or the envelope of the signal
    """
    z = analytic_signal(signal)
    return np.abs(z)

def inst_phase(signal):
    """
    Description:
        Use hilbert transformation to compute the instantaneous phase of the input signal
    Args:
        signal: input signal
    Returns:
        instantaneous phase
    """
    z = analytic_signal(signal)
    return np.unwrap(np.angle(z))

def inst_freq(signal, fs):
    """
    Description:
        Use hilbert transformation to compute the instantaneous temporal frequency of the input signal
    Args:
        signal: input signal
        fs: frequency of sampling of input signal
    Returns:
        the instantaneous temporal frequency
    """
    inst_phase_sig = inst_phase(signal)
    return np.diff(inst_phase_sig) / (2 * np.pi) * fs

## Singular Spectrum Transform (SST)
def sst(signal, win_length):
    """
    Description:
        It is a change point detection algorithm
    Args:
        signal: the input signal
        win_length: window length of Hankel matrix
    Returns:
        score: an array measuring the degree of change
    """
    sst = SingularSpectrumTransformation(win_length=win_length)
    score = sst.score_offline(signal)
    return score

## Time Domain Feature
### Petrosian Fractal Dimension (PFD)
def pfd(signal):
    """
    Description:
        It calculates the fractal dimension of a signal to describe its complexity and irregularity. A higher Petrosian
        Fractal Dimension value indicates a more complex signal.
    Args:
        signal: The input signal
    Returns:
        The value of pfd
    """
    diff = np.diff(signal)
    n_zero_crossings = np.sum(diff[:-1] * diff[1:] < 0)
    pfd = np.log10(len(signal)) / (
                np.log10(len(signal)) + np.log10(len(signal) / (len(signal) + 0.4 * n_zero_crossings)))
    return pfd

# Frequency Domain
## Fast Fourier Transform (FFT)
def my_fft(signal, fs):
    """
    Description:
        Get the spectrum of the input signal
    Args:
        signal: input signal
        fs: sampling rate
    Returns:
        The spectrum of the input, containing the freq of x-axis and the mag of the y-axis. The mag is complex number.
    """
    l = len(signal)
    mag = fft(signal)
    freq = fftfreq(l, 1 / fs)
    mag = mag / l
    index = np.argsort(freq)
    freq_sort = freq[index]
    mag_sort = mag[index]
    return freq_sort, mag_sort

def my_ifft(mag):
    """
    Description:
        Use the mag of my_fft to recover the original signal
    Args:
        mag: Output of my_fft
    Returns:
        The recovered original signal. It is complex-valued.
    """
    mag = np.append(mag[int((len(mag)+1)/2):],mag[0:int((len(mag)+1)/2)])
    mag = mag * len(mag)
    x = ifft(mag)
    return x

## Frequency Domain Feature
### Power Spectral Density (PSD)
def psd(signal, fs):
    """
    Description:
        Extract the power spectral density (PSD) of a signal.
    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
    Returns:
        numpy.ndarray: Frequency vector.
        numpy.ndarray: Power spectral density values.
    """
    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

# Time-Frequency Domain
## Short Time Fourier Transform (STFT)
def my_stft(signal, fs, plot=False, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False,
            return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    """
    Description:
        Compute the Linear Spectrogram of a signal using Short-time Fourier Transform (STFT).

    Params:
        signal (numpy.ndarray): The input signal.
        fs (int): The sample rate of the signal.
        nperseg (int, optional): The size of the analysis window in samples. Default is 256.
        The other parameters are seldom used.

    Returns:
        freqs (numpy.ndarray): The frequency values in Hz.
        times (numpy.ndarray): The time values in seconds.
        spectrogram (numpy.ndarray): The computed linear spectrogram.
    """
    f, t, Z = scipy.signal.stft(signal, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, boundary, padded,
                                axis, scaling)
    # print(Z.shape)
    if plot:
        plt.title("STFT of input signal")
        plt.pcolormesh(t, f, np.abs(Z))
        plt.xlabel("Time/S")
        plt.ylabel("Frequency")
        plt.colorbar(label='Magnitude')
        plt.show()
    return f, t, Z

## Wavelet Analysis
### Mexican Hat Wavelet
def mexican_hat_wavelet(sigma, length):
    """
    Description:
        Generate the mexican hat wavelet. It is the second derivative of the Gaussian function.
    Args:
        sigma: It has the same meaning in the Gaussian function
        length: length of the wavelet
    Returns:
        The mexican hat wavelet
    """
    t = np.linspace(-int(length / 2), length / 2, length * 10)
    psi = 1 / (np.sqrt(2 * np.pi) * np.power(sigma, 3)) * np.exp(-np.power(t, 2) / (2 * np.power(sigma, 2))) * (
                (np.power(t, 2) / np.power(sigma, 2)) - 1)
    return psi

### Morlet Wavelet
def morlet_wavelet(length, sigma, a=5):
    """
    Description:
        Generate the morlet wavelet which value is complex.
    Args:
        length: Length of the wavelet.
        sigma: Scaling parameter that affects the width of the window.
        a: Modulation parameter. Default is 5
    Returns:
        The morlet wavelet which is complex-valued.
    """
    morlet_wav = scipy.signal.morlet2(length, sigma, a)
    return morlet_wav

### Continues Wavelet Transform (CWT)
def my_cwt(signal, scales, wavelet, fs, show=False):
    """
    Description:
        Compute the cwt of the input signal
    Args:
        signal: input signal
        scales: the scales of wavelet, we can use pywt.scale2frequency to convert them to corresponding frequency
        wavelet: the type of the wavelet, there are "morl", "mexh" and so on. You can use
            wavlist = pywt.wavelist(kind='continuous') to get the available wavelet
        fs: the sampling frequency
        show: whether to show the result
    Returns:
        cofficient: the result of cwt. The length of y-axis depends on scales and length of x-axis depends on length of
            input signal
        frequencies: the corresponding frequencies to  scales
    """
    freq = pywt.scale2frequency(wavelet, scales) * fs
    if freq[0] > fs / 2:
        raise ValueError("The intended frequency is too high, please increase the lowest number of scales")
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1 / fs)
    if show:
        plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(signal) / fs, frequencies[-1], frequencies[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform')
        plt.xlabel('Time/s')
        plt.ylabel('Frequency')
        plt.show()
    return coefficients, frequencies

## Polynomial Chirplet Transform (PCT)
### Chirplet Transform
def chirplet_transform(signal, show=False):
    """
    Description:
        Generate the chirplet_trainsform of the input signal
    Args:
        signal: Input signal
        show: whether to show the result of the chirplet transform
    Returns:
        The result of the chirplet transform
    """
    chirps = FCT()
    ct_matrix = chirps.compute(signal)
    if show:
        plt.title("chirplet transform")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency")
        plt.imshow(ct_matrix, aspect="auto")
        plt.colorbar(label="Magnitude")
    return ct_matrix


# Below is the needed function for chirplet transform
class FCT:
    """
    Attributes :
        _duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
        _num_octaves : the number of octaves
        _num_chirps_by_octave : the number of chirps by octave
        _polynome_degree : degree of the polynomial function
        _end_smoothing : define the size the output of the signal
        _samplerate : samplerate of the signal

    """

    def __init__(self,
                 duration_longest_chirplet=1,
                 num_octaves=5,
                 num_chirps_by_octave=10,
                 polynome_degree=0,
                 end_smoothing=0.001,
                 sample_rate=22050):
        """
        Args:
            duration_longest_chirplet : duration of the longest chirplet in the bank of chirplets
            num_octaves : the number of octaves
            num_chirps_by_octave : the number of chirps by octave
            polynome_degree : degree of the polynomial function
            end_smoothing : define the size the output of the signal
            sample_rate : samplerate of the signal
        """
        self._duration_longest_chirplet = duration_longest_chirplet

        self._num_octaves = num_octaves

        self._num_chirps_by_octave = num_chirps_by_octave

        self._polynome_degree = polynome_degree

        self._end_smoothing = end_smoothing

        # Samplerate of the signal. Has to be defined in advance.
        self._samplerate = sample_rate

        self._chirps = self.__init_chirplet_filter_bank()

    def __init_chirplet_filter_bank(self):
        """generate all the chirplets based on the attributes

        Returns :
            The bank of chirplets
        """
        num_chirps = self._num_octaves * self._num_chirps_by_octave

        # create a list of coefficients based on attributes
        lambdas = 2.0 ** (1 + arange(num_chirps) / float(self._num_chirps_by_octave))

        # Low frequencies for a signal
        start_frequencies = (self._samplerate / lambdas) / 2.0

        # high frequencies for a signal
        end_frequencies = self._samplerate / lambdas

        durations = 2.0 * self._duration_longest_chirplet / flipud(lambdas)

        chirplets = list()
        for low_frequency, high_frequency, duration in zip(start_frequencies, end_frequencies, durations):
            chirplets.append(Chirplet(self._samplerate, low_frequency, high_frequency, duration, self._polynome_degree))
        return chirplets

    @property
    def time_bin_duration(self):
        """
        Return :
            The time bin duration

        """
        return self._end_smoothing * 10

    def compute(self, input_signal):
        """compute the FCT on the given signal
        Args :
            input_signal : Array of an audio signal

        Returns :
            The Fast Chirplet Transform of the given signal

        """
        # keep the real length of the signal
        size_data = len(input_signal)

        nearest_power_2 = 2 ** (size_data - 1).bit_length()

        # find the best power of 2
        # the signal must not be too short

        while nearest_power_2 <= self._samplerate * self._duration_longest_chirplet:
            nearest_power_2 *= 2

        # pad with 0 to have the right length of signal

        data = np.lib.pad(input_signal, (0, nearest_power_2 - size_data), 'constant', constant_values=0)

        # apply the fct to the adapted length signal

        chirp_transform = apply_filterbank(data, self._chirps, self._end_smoothing)

        # resize the signal to the right length

        chirp_transform = resize_chirps(size_data, nearest_power_2, chirp_transform)

        return chirp_transform


def resize_chirps(size_data, size_power_2, chirps):
    """Resize the matrix of chirps to the length of the signal
    Args:
        size_data : number of samples of the audio signal
        size_power_2 : number of samples of the signal to apply the FCT
        chirps : the signal to resize
    Returns :
        Chirps with the correct length
    """
    size_chirps = len(chirps)
    ratio = size_data / size_power_2
    size = int(ratio * len(chirps[0]))

    resize_chirps = np.zeros((size_chirps, size))
    for i in range(0, size_chirps):
        resize_chirps[i] = chirps[i][0:size]
    return resize_chirps


class Chirplet:
    """chirplet class
    Attributes:
        _min_frequency : lowest frequency where the chirplet is applied
        _max_frequency : highest frequency where the chirplet is applied
        _duration : duration of the chirp
        _samplerate : samplerate of the signal
        _polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        _filter_coefficients : coefficients applied to the signal
    """

    def __init__(self, samplerate, min_frequency, max_frequency, sigma, polynome_degree):

        """
        Args :
            samplerate : samplerate of the signal
            min_frequency : lowest frequency where the chirplet is applied
            max_frequency : highest frequency where the chirplet is applied
            duration : duration of the chirp
            polynome_degree : degree of the polynome to generate the coefficients of the chirplet
        """
        self._min_frequency = min_frequency

        self._max_frequency = max_frequency

        self._duration = sigma / 10

        self._samplerate = samplerate

        self._polynome_degree = polynome_degree

        self._filter_coefficients = self.calcul_coefficients()

    def calcul_coefficients(self):
        """calculate coefficients for the chirplets
        Returns :
            apodization coeeficients
        """
        num_coeffs = linspace(0, self._duration, int(self._samplerate * self._duration))

        if self._polynome_degree:
            temp = (self._max_frequency - self._min_frequency)
            temp /= ((
                                 self._polynome_degree + 1) * self._duration ** self._polynome_degree) * num_coeffs ** self._polynome_degree + self._min_frequency
            wave = cos(2 * pi * num_coeffs * temp)
        else:
            temp = (self._min_frequency * (self._max_frequency / self._min_frequency) ** (
                        num_coeffs / self._duration) - self._min_frequency)
            temp *= self._duration / log(self._max_frequency / self._min_frequency)
            wave = cos(2 * pi * temp)

        coeffs = wave * hanning(len(num_coeffs)) ** 2

        return coeffs

    def smooth_up(self, input_signal, thresh_window, end_smoothing):
        """generate fast fourier transform from a signal and smooth it
        Params :
            input_signal : audio signal
            thresh_window : relative to the size of the windows
            end_smoothing : relative to the length of the output signal
        Returns :
            fast Fourier transform of the audio signal applied to a specific domain of frequencies
        """
        windowed_fft = build_fft(input_signal, self._filter_coefficients, thresh_window)
        return fft_smoothing(fabs(windowed_fft), end_smoothing)


def apply_filterbank(input_signal, chirplets, end_smoothing):
    """generate list of signal with chirplets
    Params :
        input_signal : audio signal
        chirplets : the chirplet bank
        end_smoothing : relative to the length of the output signal
    Returns :
        fast Fourier transform of the signal to all the frequency domain
    """
    fast_chirplet_transform = list()

    for chirplet in chirplets:
        chirp_line = chirplet.smooth_up(input_signal, 6, end_smoothing)
        fast_chirplet_transform.append(chirp_line)

    return np.array(fast_chirplet_transform)


def fft_smoothing(input_signal, sigma):
    """smooth the fast transform Fourier
    Params :
        input_signal : audio signal
        sigma : relative to the length of the output signal
    Returns :
        a shorter and smoother signal

    """
    size_signal = input_signal.size

    # shorten the signal
    new_size = int(floor(10.0 * size_signal * sigma))
    half_new_size = new_size // 2

    fftx = fft(input_signal)

    short_fftx = []
    for ele in fftx[:half_new_size]:
        short_fftx.append(ele)

    for ele in fftx[-half_new_size:]:
        short_fftx.append(ele)

    apodization_coefficients = generate_apodization_coeffs(half_new_size, sigma, size_signal)

    # apply the apodization coefficients
    short_fftx[:half_new_size] *= apodization_coefficients
    short_fftx[half_new_size:] *= flipud(apodization_coefficients)

    realifftxw = ifft(short_fftx).real
    return realifftxw


def generate_apodization_coeffs(num_coeffs, sigma, size):
    """generate apodization coefficients
    Params :
        num_coeffs : number of coefficients
        sigma : relative to the length of the output signal
        size : size of the signal
    Returns :
        apodization coefficients

    """
    apodization_coefficients = arange(num_coeffs)
    apodization_coefficients = apodization_coefficients ** 2
    apodization_coefficients = apodization_coefficients / (2 * (sigma * size) ** 2)
    apodization_coefficients = exp(-apodization_coefficients)
    return apodization_coefficients


def fft_based(input_signal, filter_coefficients, boundary=0):
    """applied fft if the signal is too short to be splitted in windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        boundary : manage the bounds of the signal
    Returns :
        audio signal with application of fast Fourier transform
    """
    num_coeffs = filter_coefficients.size
    half_size = num_coeffs // 2

    if boundary == 0:  # ZERO PADDING
        input_signal = np.lib.pad(input_signal, (half_size, half_size), 'constant', constant_values=0)
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size - num_coeffs), 'constant',
                                         constant_values=0)
        newx = ifft(fft(input_signal) * fft(filter_coefficients))
        return newx[num_coeffs - 1:-1]

    elif boundary == 1:  # symmetric
        input_signal = np.concatenate(
            [flipud(input_signal[:half_size]), input_signal, flipud(input_signal[half_size:])])
        filter_coefficients = np.lib.pad(filter_coefficients, (0, input_signal.size - num_coeffs), 'constant',
                                         constant_values=0)
        newx = ifft(fft(input_signal) * fft(filter_coefficients))
        return newx[num_coeffs - 1:-1]

    else:  # periodic
        return roll(ifft(fft(input_signal) * fft(filter_coefficients, input_signal.size)), -half_size).real


def build_fft(input_signal, filter_coefficients, threshold_windows=6, boundary=0):
    """generate fast transform fourier by windows
    Params :
        input_signal : the audio signal
        filter_coefficients : coefficients of the chirplet bank
        threshold_windows : calcul the size of the windows
        boundary : manage the bounds of the signal
    Returns :
        fast Fourier transform applied by windows to the audio signal

    """
    num_coeffs = filter_coefficients.size
    # print(n,boundary,M)
    half_size = num_coeffs // 2
    signal_size = input_signal.size
    # power of 2 to apply fast fourier transform
    windows_size = 2 ** ceil(log2(num_coeffs * (threshold_windows + 1)))
    number_of_windows = floor(signal_size // windows_size)

    if number_of_windows == 0:
        return fft_based(input_signal, filter_coefficients, boundary)

    windowed_fft = empty_like(input_signal)
    # pad with 0 to have a size in a power of 2
    windows_size = int(windows_size)

    zeropadding = np.lib.pad(filter_coefficients, (0, windows_size - num_coeffs), 'constant', constant_values=0)

    h_fft = fft(zeropadding)

    # to browse the whole signal
    current_pos = 0

    # apply fft to a part of the signal. This part has a size which is a power
    # of 2
    if boundary == 0:  # ZERO PADDING

        # window is half padded with since it's focused on the first half
        window = input_signal[current_pos:current_pos + windows_size - half_size]
        zeropaddedwindow = np.lib.pad(window, (len(h_fft) - len(window), 0), 'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)

    elif boundary == 1:  # SYMMETRIC
        window = np.concatenate(
            [flipud(input_signal[:half_size]), input_signal[current_pos:current_pos + windows_size - half_size]])
        x_fft = fft(window)

    else:
        x_fft = fft(input_signal[:windows_size])

    windowed_fft[:windows_size - num_coeffs] = (ifft(x_fft * h_fft)[num_coeffs - 1:-1]).real

    current_pos += windows_size - num_coeffs - half_size
    # apply fast fourier transofm to each windows
    while current_pos + windows_size - half_size <= signal_size:
        x_fft = fft(input_signal[current_pos - half_size:current_pos + windows_size - half_size])
        # Suppress the warning, work on the real/imagina
        windowed_fft[current_pos:current_pos + windows_size - num_coeffs] = (
        ifft(x_fft * h_fft)[num_coeffs - 1:-1]).real
        current_pos += windows_size - num_coeffs
    # print(countloop)
    # apply fast fourier transform to the rest of the signal
    if windows_size - (signal_size - current_pos + half_size) < half_size:

        window = input_signal[current_pos - half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size - (signal_size - current_pos + half_size))),
                                      'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = roll(ifft(x_fft * h_fft), half_size)[
                                     half_size:half_size + windowed_fft.size - current_pos].real
        windowed_fft[-half_size:] = convolve(input_signal[-num_coeffs:], filter_coefficients, 'same')[-half_size:]
    else:

        window = input_signal[current_pos - half_size:]
        zeropaddedwindow = np.lib.pad(window, (0, int(windows_size - (signal_size - current_pos + half_size))),
                                      'constant', constant_values=0)
        x_fft = fft(zeropaddedwindow)
        windowed_fft[current_pos:] = ifft(x_fft * h_fft)[
                                     num_coeffs - 1:num_coeffs + windowed_fft.size - current_pos - 1].real

    return windowed_fft
# chirplet transform function ends here

## Wigner Ville Distribution (WVD)
def my_wvd(signal, show=False):
    """
    Description:
        Analyze the time-frequency characteristics of a signal using the Wigner-Ville Transform (WVT) and visualize the results.

    Params:
        signal (numpy.ndarray): The input signal.
        show: whether to plot the result
    Returns:
        tfr_wvd (numpy.ndarray): The time-frequency representation (WVD) of the signal.
        t_wvd (numpy.ndarray): Time values corresponding to the WVD.
        f_wvd (numpy.ndarray): Normalized frequency values corresponding to the WVD.
    """
    wvd = tftb.processing.WignerVilleDistribution(signal)
    tfr_wvd, t_wvd, f_wvd = wvd.run()
    if show:
        wvd.plot(kind="contourf", scale="log")
    return tfr_wvd, t_wvd, f_wvd

## SynchroSqueezing Transform (SST)
def sst_stft(signal, fs, window, nperseg=256, show=False, n_fft=None, hop_len=1, modulated=True, ssq_freqs=None,
             padtype='reflect', squeezing='sum', gamma=None, preserve_transform=None, dtype=None, astensor=True,
             flipud=False, get_w=False, get_dWx=False):
    """
    Description:
        Synchrosqueezed Short-Time Fourier Transform.
    Args:
        signal: the input signal
        fs: frequency of sampling
        window: type of the window
        nperseg: Length of each segment
        show: whether to show the result
        n_fft: length of fft
        The other parameters are seldom used.
    Returns:
        Tx: Synchrosqueezed STFT of `x`, of same shape as `Sx`.
        Sx: STFT of `x`
        ssq_freqs: Frequencies associated with rows of `Tx`.
        Sfs: Frequencies associated with rows of `Sx` (by default == `ssq_freqs`).
    """
    Tx, Sx, ssq_freqs, Sfs = sq.ssq_stft(signal, window=window, win_len=nperseg, fs=fs, n_fft=n_fft)
    if show:
        plt.subplot(2, 1, 1)
        plt.title("STFT of Input signal")
        plt.imshow(np.abs(Sx), aspect="auto")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Magnitude")
        plt.subplot(2, 1, 2)
        plt.title("Synchrosqueezed STFT of Input signal")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.imshow(np.abs(Tx), aspect="auto")
        plt.colorbar(label="Magnitude")
        plt.tight_layout()
        plt.show()
    return Tx, Sx, ssq_freqs, Sfs

def sst_cwt(signal, wavelet, scales, nv, fs, gamma=None, show=False):
    """
    Description:
        Synchrosqueezed Continuous Wavelet Transform
    Args:
        signal: input of signal
        wavelet: the type of mother wavelet
        scales: how to scale the output, log or linear
        nv: number of voices
        fs: sampling frequency
        gamma: CWT phase threshold
        show: whether to show the result
    Returns:
        Tx: Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        Wx: Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
        ssq_freqs: Frequencies associated with rows of `Tx`.
        scales: Scales associated with rows of `Wx`.
    """
    Tx, Wx, ssq_freqs, scales = sq.ssq_cwt(x=signal, wavelet=wavelet, scales=scales, nv=nv, fs=fs, gamma=gamma)
    if show:
        plt.subplot(2, 1, 1)
        plt.imshow(np.abs(Wx), aspect='auto', extent=[0, len(signal) / fs, ssq_freqs[-1], ssq_freqs[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform')
        plt.xlabel('Time/s')
        plt.ylabel('Frequency')
        plt.subplot(2, 1, 2)
        plt.imshow(np.abs(Tx), aspect='auto', extent=[0, len(signal) / fs, ssq_freqs[-1], ssq_freqs[0]])
        plt.colorbar(label='Magnitude')
        plt.title('Synchrosqueezed Continuous Wavelet Transform')
        plt.xlabel('Time/s')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    return Tx, Wx, ssq_freqs, scales

def extract_spectral_entropy(signal, fs, num_segments=10):
    """
    Description:
        Extract the spectral entropy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.
        num_segments (int, optional): Number of segments for entropy calculation.

    Returns:
        float: Spectral entropy value.
    """

    f, Pxx = welch(signal, fs=fs)
    segment_size = len(f) // num_segments
    segment_entropies = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment_Pxx = Pxx[start_idx:end_idx]
        segment_entropies.append(entropy(segment_Pxx))

    spectral_entropy = np.mean(segment_entropies)
    return spectral_entropy

def extract_mean_spectral_energy(signal, fs):
    """
    Description:
        Extract the mean spectral energy of a signal.

    Params:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        float: Mean spectral energy value.
    """

    f, Pxx = welch(signal, fs=fs)
    mean_spectral_energy = np.mean(Pxx)
    return mean_spectral_energy

def DCT_synthesize(amps, fs, ts):
    """
    Description:
        Synthesize a mixture of cosines with given amps and fs.

    Input:
        amps: amplitudes
        fs: frequencies in Hz
        ts: times to evaluate the signal

    Returns:
        wave array
    """
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    ys = np.dot(M, amps)
    return ys

def DCT_analyze(ys, fs, ts):
    """
    Description:
        Analyze a mixture of cosines and return amplitudes.

    Input:
        ys: wave array
        fs: frequencies in Hz
        ts: time when the signal was evaluated

    returns:
        vector of amplitudes
    """
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    amps = np.dot(M, ys) / 2
    return amps

def DCT_iv(ys):
    """
    Description:
        Computes DCT-IV.

    Input:
        wave array

    returns:
        vector of amplitudes
    """
    N = len(ys)
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    args = np.outer(ts, fs)
    M = np.cos(np.pi * 2 * args)
    amps = np.dot(M, ys) / 2
    return amps

def inverse_DCT_iv(amps):
    return DCT_iv(amps) * 2

def cal_corrcoef(signal1, signal2):
    """
    Description:
        To get the correlate coefficient

    Input:
        Two signal with same length

    Return:
        The correlate coefficient
    """
    return np.corrcoef(signal1, signal2)[0, 1]

def cal_serial_corr(signal, lag):
    """
    Description:
        To get the serial correlate coefficient

    Input:
        One signal and the lag which means how much it delays

    Return:
        The serial correlate coefficient
    """
    signal1 = signal[lag:]
    signal2 = signal[:len(signal) - lag]
    return np.corrcoef(signal1, signal2)[0, 1]

def cal_autocorr(signal, plot=False):
    """
    Description:
        To get the auto correlate coefficient

    Input:
        One signal

    Return:
        The serial correlate coefficient with different lag which is from 0 to len(wave)//2
    """
    lags = range(len(signal) // 2)
    corrs = [cal_serial_corr(signal, lag) for lag in lags]
    if plot:
        plt.plot(lags, corrs)
        plt.show()
    return lags, corrs