"""Known-answer AR signals for the tests (built on emg.synthetic)."""
import numpy as np

from emg.io import FS
from emg.synthetic import ar2_coeffs, simulate_ar  # noqa: F401  (re-exported for tests)


def stationary_ar4(N):
    """Stable AR(4): pole pairs at 0.9*exp(+-j*2pi*80/fs) and 0.85*exp(+-j*2pi*200/fs)."""
    poly = np.poly([
        0.9 * np.exp(2j * np.pi * 80 / FS), 0.9 * np.exp(-2j * np.pi * 80 / FS),
        0.85 * np.exp(2j * np.pi * 200 / FS), 0.85 * np.exp(-2j * np.pi * 200 / FS),
    ]).real
    return np.tile(poly[1:], (N, 1))


def moving_peak_ar2(N, f_start=150.0, f_end=80.0):
    """AR(2) whose resonance sweeps linearly from f_start to f_end Hz over N samples."""
    return ar2_coeffs(np.linspace(f_start, f_end, N))


def rel_err(a, b):
    """Relative L2 error of a against reference b."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)
