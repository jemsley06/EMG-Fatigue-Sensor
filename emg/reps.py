"""Per-rep fatigue: find contraction bursts in the amplitude envelope, then one spectrum per rep.

Continuous tracking (baseline.py, tvar.py) assumes the muscle is active the whole time. With reps,
the rest between them is electrode noise, so MNF is only meaningful inside each burst.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .metrics import mdf, mnf
from .preprocess import bandpass
from .tvar import FREQ_GRID, ar_psd, burg


@dataclass
class RepTable:
    start: np.ndarray   # (R,) s, trial clock
    stop: np.ndarray    # (R,) s
    mnf: np.ndarray     # (R,) Hz
    mdf: np.ndarray     # (R,) Hz
    rms: np.ndarray     # (R,) band-passed amplitude, input units (not standardized)
    params: dict = field(default_factory=dict)

    def relative(self, which="mnf", n_ref=3):
        """`which` per rep as a fraction of the mean of the first `n_ref` reps (1.0 = fresh)."""
        y = getattr(self, which)
        return y / y[:n_ref].mean()


def envelope(x, fs, cutoff=5.0, order=2):
    """Amplitude envelope: full-wave rectify, then zero-phase low-pass at `cutoff` Hz."""
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.abs(x))


def _runs(mask):
    """(R, 2) [start, stop) sample indices of each run of True in a boolean array."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def detect_reps(env, fs, min_rep_s=0.3, min_gap_s=0.2):
    """(R, 2) [start, stop) sample indices of the contraction bursts in envelope `env`.

    A burst shorter than `min_rep_s` is not a rep (a twitch or a cable knock). Two bursts separated
    by less than `min_gap_s` are one rep (the envelope dips mid-movement). The threshold must not
    depend on units: the dataset is in volts (~1e-4) and the MyoWare in ADC counts (~100s).
    """
    # TODO(you): choose the activity threshold, build the boolean mask, then use _runs(mask),
    # merge runs closer than min_gap_s and drop runs shorter than min_rep_s.
    raise NotImplementedError


def rep_spectrum(seg, fs, p):
    """Burg AR(p) spectrum of one rep's samples on FREQ_GRID."""
    if len(seg) < 10 * p:
        raise ValueError(f"rep of {len(seg)} samples is too short for AR({p})")
    a, sigma2 = burg(seg - seg.mean(), p)
    return ar_psd(a, sigma2, fs, FREQ_GRID)


def run_reps(trial, p=14, cutoff=5.0, trim_s=0.1, segments=None, **detect_kw):
    """bandpass -> envelope -> detect reps -> per-rep Burg spectrum -> MNF/MDF/RMS per rep.

    `p` defaults to the biceps order selected in tvar.ipynb. `trim_s` is cut from both ends of each
    rep, where the envelope's smoothing lets rest noise in. `segments` ((R, 2) sample indices)
    skips detection, e.g. for hand-labelled reps.
    """
    x = bandpass(trial.x, trial.fs)
    if segments is None:
        segments = detect_reps(envelope(x, trial.fs, cutoff), trial.fs, **detect_kw)
    segments = np.asarray(segments, dtype=int).reshape(-1, 2)
    trim = int(round(trim_s * trial.fs))
    rows = []
    for a, b in segments:
        core = x[a + trim:b - trim]
        psd = rep_spectrum(core, trial.fs, p)
        rows.append((trial.t[a], trial.t[b - 1], mnf(psd, FREQ_GRID), mdf(psd, FREQ_GRID),
                     np.sqrt(np.mean(core ** 2))))
    cols = np.array(rows, dtype=float).reshape(-1, 5).T
    return RepTable(*cols, params={"p": p, "cutoff": cutoff, "trim_s": trim_s, **detect_kw})
