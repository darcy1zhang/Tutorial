import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt

def envelope_hilbert(signal, fs):
    """
    Description:
        Analyzes a signal using the Hilbert Transform to extract envelope and phase information.

    Params:
        signal (array-like): The input signal to be analyzed.
        fs (float): The sampling frequency of the input signal.

    Returns:
        inst_amplitude (array-like): The instantaneous amplitude of the signal envelope.
        inst_freq (array-like): The instantaneous frequency of the signal.
        inst_phase (array-like): The instantaneous phase of the signal.
        regenerated_carrier (array-like): The regenerated carrier signal from the instantaneous phase.
    """

    z= hilbert(signal) #form the analytical signal
    inst_amplitude = np.abs(z) #envelope extraction
    inst_phase = np.unwrap(np.angle(z))#inst phase
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs #inst frequency

    #Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(inst_phase)

    return inst_amplitude, inst_freq, inst_phase, regenerated_carrier


if __name__ == "__main__":
    import os

    current_file_path = os.path.abspath(__file__)
    signal_name = "sim_{}_{}_{}_{}_{}.npy".format(100, 0.1, 90, 140, "train")

    signal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "data", signal_name)
    signal = np.load(signal_path)[2, :1000]
    fs = 100

    inst_amplitude, inst_freq, inst_phase, regenerated_carrier = envelope_hilbert(signal,fs)

    window_size = 3  # Adjust the window size as needed
    smoothed_envelope = np.convolve(inst_amplitude, np.ones(window_size) / window_size, mode='same')

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.plot(smoothed_envelope, 'r')  # overlay the extracted envelope
    plt.title('Modulated signal and extracted envelope')
    plt.xlim(0,200)
    plt.xlabel('n')
    plt.ylabel('x(t) and |z(t)|')
    plt.subplot(2, 1, 2)
    plt.plot(regenerated_carrier)
    plt.title('Extracted carrier or TFS')
    plt.xlabel('n')
    plt.ylabel('cos[\omega(t)]')
    plt.show()