"""Time-varying autoregressive (TVAR) spectral estimation.

Model: x[n] = -sum_{k=1..p} a_k[n] x[n-k] + e[n],  e[n] ~ N(0, sigma2[n]).
"""
import math

import numpy as np
from numba import njit

from .io import FS
from .metrics import BAND, SpectralTrack, mdf, mnf
from .preprocess import bandpass, standardize

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

# ---------------------------------------------------------------- numba kernels

@njit(cache=True)
def _regressor(x, n, p):
    h = np.empty(p)
    for k in range(p):
        h[k] = -x[n - 1 - k]
    return h


@njit(cache=True)
def _rls_kernel(x, p, lam, delta, alpha, stride):
    N = x.size
    n_out = (N - 1 - p) // stride + 1
    A = np.zeros((n_out, p))
    S2 = np.zeros(n_out)
    theta = np.zeros(p)
    P = delta * np.eye(p)
    s2 = 1.0
    for n in range(p, N):
        h = _regressor(x, n, p)
        Ph = P @ h
        g = Ph / (lam + h @ Ph)
        e = x[n] - h @ theta
        theta += g * e
        P = (P - np.outer(g, Ph)) / lam
        P = 0.5 * (P + P.T)
        s2 = alpha * s2 + (1.0 - alpha) * e * e
        j = n - p
        if j % stride == 0:
            A[j // stride] = theta
            S2[j // stride] = s2
    return A, S2


# ---------------------------------------------------------------- estimators

def _stride(fs, out_dt):
    return max(1, int(round(out_dt * fs)))


def _out_times(n_out, p, stride, fs):
    return (p + stride * np.arange(n_out)) / fs


def tvar_rls(x, p, tau, fs=FS, out_dt=0.05, r_tau=1.0, delta=100.0):
    """Exponentially weighted RLS TVAR. `tau` is the memory in seconds (lambda = exp(-1/(tau*fs))).

    Returns (t_out (T,), A (T, p), sigma2 (T,)), times relative to the first sample.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    stride = _stride(fs, out_dt)
    lam = math.exp(-1.0 / (tau * fs))
    alpha = math.exp(-1.0 / (r_tau * fs))
    A, S2 = _rls_kernel(x, p, lam, delta, alpha, stride)
    return _out_times(len(A), p, stride, fs), A, S2

# ---------------------------------------------------------------- Kalman TVAR

@njit(cache=True)
def _kf_update(x, n, p, q, theta, P, r, alpha):
    """One random-walk Kalman step, in place on theta/P. Returns the updated noise estimate r."""
    for i in range(p):
        P[i, i] += q
    h = _regressor(x, n, p)
    Ph = P @ h
    e = x[n] - h @ theta
    K = Ph / (h @ Ph + r)
    theta += K * e
    P -= np.outer(K, Ph)
    P[:, :] = 0.5 * (P + P.T)
    return alpha * r + (1.0 - alpha) * e * e


@njit(cache=True)
def _kalman_kernel(x, p, q, p0, r0, alpha, stride, M, smooth):
    N = x.size
    n_out = (N - 1 - p) // stride + 1
    A = np.zeros((n_out, p))
    S2 = np.zeros(n_out)
    n_seg = (N - p + M - 1) // M
    ck_theta = np.zeros((n_seg, p))
    ck_P = np.zeros((n_seg, p, p))
    ck_r = np.zeros(n_seg)

    theta = np.zeros(p)
    P = p0 * np.eye(p)
    r = r0
    for n in range(p, N):          # forward filter, checkpoint every M samples
        j = n - p
        if j % M == 0:
            ck_theta[j // M] = theta
            ck_P[j // M] = P
            ck_r[j // M] = r
        r = _kf_update(x, n, p, q, theta, P, r, alpha)
        if j % stride == 0:
            A[j // stride] = theta
            S2[j // stride] = r
    if not smooth:
        return A, S2

    # RTS backward pass, one checkpointed segment at a time (last segment first).
    # Random walk => theta_pred[n+1] = theta_f[n], P_pred[n+1] = P_f[n] + qI.
    I = np.eye(p)
    buf_theta = np.zeros((M, p))
    buf_P = np.zeros((M, p, p))
    theta_s = np.zeros(p)
    for s in range(n_seg - 1, -1, -1):
        start = p + s * M
        stop = min(start + M, N)
        th = ck_theta[s].copy()
        Pc = ck_P[s].copy()
        rr = ck_r[s]
        for n in range(start, stop):   # recompute this segment's filtered states
            rr = _kf_update(x, n, p, q, th, Pc, rr, alpha)
            buf_theta[n - start] = th
            buf_P[n - start] = Pc
        for n in range(stop - 1, start - 1, -1):
            i = n - start
            if n == N - 1:
                theta_s = buf_theta[i].copy()
            else:
                C = np.linalg.solve(buf_P[i] + q * I, buf_P[i]).T   # P_f (P_f + qI)^-1
                theta_s = buf_theta[i] + C @ (theta_s - buf_theta[i])
            j = n - p
            if j % stride == 0:
                A[j // stride] = theta_s
    return A, S2


def tvar_kalman(x, p, q, r=None, fs=FS, out_dt=0.05, r_tau=1.0, p0=10.0, checkpoint=256, smooth=True):
    """Random-walk-coefficient Kalman TVAR with checkpointed RTS smoothing.

    `q` is the per-sample coefficient random-walk variance. `r` fixes the observation noise;
    if None it is tracked as an EMA (time constant `r_tau` s) of squared prediction errors.
    `checkpoint` is the RTS segment length in samples (memory vs recompute; results identical).
    Returns (t_out (T,), A (T, p), sigma2 (T,)), times relative to the first sample.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    stride = _stride(fs, out_dt)
    if r is None:
        r0, alpha = 1.0, math.exp(-1.0 / (r_tau * fs))
    else:
        r0, alpha = float(r), 1.0
    A, S2 = _kalman_kernel(x, p, q, p0, r0, alpha, stride, checkpoint, smooth)
    return _out_times(len(A), p, stride, fs), A, S2

# ---------------------------------------------------------------- pipeline

def run_tvar(trial, method, p, discard_s=1.0, keep_psd=False, **params):
    """bandpass -> standardize -> TVAR -> AR PSD (20-450 Hz, 1 Hz grid) -> MNF/MDF.

    `params` go to the estimator (`q`/`r` for "kalman", `tau` for "rls"). The first `discard_s`
    seconds (estimator start-up) are dropped. Returned times are in the trial's own clock.
    """
    x = standardize(bandpass(trial.x, trial.fs))
    if method == "kalman":
        t, A, s2 = tvar_kalman(x, p, fs=trial.fs, **params)
    elif method == "rls":
        t, A, s2 = tvar_rls(x, p, fs=trial.fs, **params)
    else:
        raise ValueError(f"method must be 'kalman' or 'rls', got {method!r}")
    keep = t >= discard_s
    t, A, s2 = t[keep], A[keep], s2[keep]
    psd = ar_psd(A, s2, trial.fs, FREQ_GRID)
    return SpectralTrack(
        t=trial.t[0] + t, mnf=mnf(psd, FREQ_GRID), mdf=mdf(psd, FREQ_GRID),
        method=f"tvar_{method}", params={"p": p, **params},
        psd=psd if keep_psd else None, f=FREQ_GRID if keep_psd else None,
    )
