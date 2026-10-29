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


def rep_trial(n_reps=12, rep_s=1.5, conc_s=0.8, rest_s=1.5, f_start=150.0, f_end=100.0,
              ecc_gain=0.4, ecc_shift_hz=40.0, lead_s=2.0, rest_level=0.05, fs=FS, seed=0):
    """Curl-style trial: AR(2) bursts separated by low-level white noise (electrode floor).

    Each rep is a concentric phase (first `conc_s` s, gain 1, resonance f) then an eccentric phase
    (gain `ecc_gain`, resonance f + `ecc_shift_hz`). f falls linearly from f_start to f_end across
    reps. Returns (trial, bounds, conc_stop): bounds (n_reps, 2) the true [start, stop) of each rep
    and conc_stop (n_reps,) the true end of each concentric phase, in samples.
    """
    period = rep_s + rest_s
    N = int(round((lead_s + n_reps * period) * fs))
    starts = np.round((lead_s + period * np.arange(n_reps)) * fs).astype(int)
    bounds = np.column_stack([starts, starts + int(round(rep_s * fs))])
    conc_stop = starts + int(round(conc_s * fs))
    freq = np.full(N, f_start)
    gain = np.zeros(N)
    for (a, b), c, f in zip(bounds, conc_stop, np.linspace(f_start, f_end, n_reps)):
        freq[a:c], gain[a:c] = f, 1.0
        freq[c:b], gain[c:b] = f + ecc_shift_hz, ecc_gain
    rng = np.random.default_rng(seed)
    x = gain * simulate_ar(ar2_coeffs(freq, fs=fs), rng)
    x += rest_level * rng.normal(size=N)
    trial = Trial(t=np.arange(N) / fs, x=x, fs=fs, muscle="SYNTHETIC_REPS", subject=0, trial=0)
    return trial, bounds, conc_stop


def step_trial(f_before=150.0, f_after=80.0, seconds=60.0, step_at=30.0, fs=FS, seed=0):
    """Trial whose AR(2) resonance jumps from f_before to f_after Hz at `step_at` seconds."""
    N = int(seconds * fs)
    t = np.arange(N) / fs
    A = ar2_coeffs(np.where(t < step_at, f_before, f_after), fs=fs)
    x = simulate_ar(A, np.random.default_rng(seed))
    return Trial(t=t, x=x, fs=fs, muscle="SYNTHETIC_STEP", subject=0, trial=0)
