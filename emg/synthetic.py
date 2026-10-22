"""Synthetic AR signals with known, possibly time-varying, spectra (tests and calibration)."""
import numpy as np

from .io import FS, Trial


def simulate_ar(A, rng, sigma=1.0):
    """x[n] = -sum_k A[n, k] x[n-k] + e[n], e ~ N(0, sigma^2). A is (N, p)."""
    N, p = A.shape
    x = np.zeros(N)
    e = rng.normal(0.0, sigma, N)
    for n in range(N):
        acc = e[n]
        for k in range(1, min(p, n) + 1):
            acc -= A[n, k - 1] * x[n - k]
        x[n] = acc
    return x


def ar2_coeffs(freq_hz, radius=0.9, fs=FS):
    """AR(2) coefficients (N, 2) with a resonance at `freq_hz` (scalar or (N,) array)."""
    w = 2 * np.pi * np.atleast_1d(np.asarray(freq_hz, dtype=float)) / fs
    return np.column_stack([-2 * radius * np.cos(w), np.full(w.size, radius ** 2)])


def step_trial(f_before=150.0, f_after=80.0, seconds=60.0, step_at=30.0, fs=FS, seed=0):
    """Trial whose AR(2) resonance jumps from f_before to f_after Hz at `step_at` seconds."""
    N = int(seconds * fs)
    t = np.arange(N) / fs
    A = ar2_coeffs(np.where(t < step_at, f_before, f_after), fs=fs)
    x = simulate_ar(A, np.random.default_rng(seed))
    return Trial(t=t, x=x, fs=fs, muscle="SYNTHETIC_STEP", subject=0, trial=0)
