"""Signal conditioning shared by every method."""
import numpy as np
from scipy import signal


def bandpass(x, fs, low=20.0, high=450.0, order=4):
    """Zero-phase Butterworth bandpass (same design as the reference pipeline, SOS form)."""
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def standardize(x):
    """Zero mean, unit variance. Raises ValueError on a constant signal."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        raise ValueError("cannot standardize a constant or non-finite signal")
    return (x - x.mean()) / sd
