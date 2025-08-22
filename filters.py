
import numpy as np
from scipy.signal import firwin, filtfilt, windows

def highpass_if_needed(neural: np.ndarray, event_functions: str, fs: float) -> np.ndarray:
    if event_functions and ('HighPass' in event_functions or 'high' in event_functions.lower()):
        taps = 81
        cutoff = 0.034
        h = firwin(taps, cutoff, pass_zero=False, window=windows.hann(taps))
        return filtfilt(h, [1.0], neural)
    return neural
