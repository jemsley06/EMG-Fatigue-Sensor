"""Time-varying autoregressive (TVAR) spectral estimation.

Model: x[n] = -sum_{k=1..p} a_k[n] x[n-k] + e[n],  e[n] ~ N(0, sigma2[n]).
"""
import math

import numpy as np

from .io import FS
from .metrics import BAND, mnf

FREQ_GRID = np.arange(BAND[0], BAND[1] + 1.0, 1.0)   # 1 Hz grid over the 20-450 Hz analysis band


# ---------------------------------------------------------------- spectra

def ar_psd(A, sigma2, fs, f):
    """AR power spectrum. A: (T, p) or (p,); sigma2: (T,) or scalar; returns (F, T) or (F,)."""
    A = np.asarray(A, dtype=float)
    single = A.ndim == 1
    A = np.atleast_2d(A)
    sigma2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (A.shape[0],))
    k = np.arange(1, A.shape[1] + 1)
    E = np.exp(-2j * np.pi * np.outer(f, k) / fs)       # (F, p)
    denom = np.abs(1.0 + E @ A.T) ** 2                   # (F, T)
    psd = sigma2[None, :] / denom
    return psd[:, 0] if single else psd


# ---------------------------------------------------------------- order selection

def burg(x, p):
    """Stationary AR(p) fit by Burg's method. Returns (a (p,), sigma2) in the model's sign convention."""
    x = np.asarray(x, dtype=float)
    f = x[1:].copy()
    b = x[:-1].copy()
    a = np.ones(1)
    E = np.dot(x, x) / x.size
    for _ in range(p):
        k = -2.0 * np.dot(f, b) / (np.dot(f, f) + np.dot(b, b))
        a_ext = np.concatenate([a, [0.0]])
        a = a_ext + k * a_ext[::-1]
        E *= 1.0 - k * k
        f, b = (f + k * b)[1:], (b + k * f)[:-1]
    return a[1:], E


def select_order(x, fs=FS, p_range=range(2, 31), p_ref=40, tol_hz=1.0,
                 win_s=4.0, n_windows=5, span_s=60.0):
    """Smallest AR order whose in-band MNF is within `tol_hz` of a high-order (`p_ref`) Burg fit,
    taking the median over `n_windows` windows spread across the first `span_s` seconds.

    BIC is deliberately not used: on 20-450 Hz bandpassed EMG it keeps rewarding poles spent on
    the empty 450-629 Hz stopband and always picks the largest order tried.
    `x` should already be bandpassed and standardized. Falls back to max(p_range) per window.
    """
    n_win = int(round(win_s * fs))
    span = min(len(x), int(round(span_s * fs)))
    if span < n_win:
        raise ValueError(f"signal shorter than one {win_s}s window")

    def fit_mnf(seg, p):
        a, sigma2 = burg(seg, p)
        return float(mnf(ar_psd(a, sigma2, fs, FREQ_GRID), FREQ_GRID))

    best = []
    for s in np.linspace(0, span - n_win, n_windows).astype(int):
        seg = x[s:s + n_win]
        ref = fit_mnf(seg, p_ref)
        best.append(next((p for p in p_range if abs(fit_mnf(seg, p) - ref) <= tol_hz), p_range[-1]))
    return int(np.median(best))
