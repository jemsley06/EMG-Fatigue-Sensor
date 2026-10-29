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
    """One row per rep. Spectral columns and `rms` describe the concentric phase only."""
    start: np.ndarray       # (R,) s, trial clock: rep onset = concentric onset
    conc_stop: np.ndarray   # (R,) s, end of the concentric phase
    stop: np.ndarray        # (R,) s, end of the rep (eccentric included)
    mnf: np.ndarray         # (R,) Hz; NaN if the concentric phase is too short to fit
    mdf: np.ndarray         # (R,) Hz
    rms: np.ndarray         # (R,) band-passed amplitude, input units (not standardized)
    params: dict = field(default_factory=dict)

    @property
    def conc_duration(self):
        return self.conc_stop - self.start

    @property
    def duration(self):
        return self.stop - self.start

    def relative(self, which="mnf", n_ref=3):
        """`which` per rep as a fraction of the mean of the first `n_ref` reps (1.0 = fresh)."""
        y = getattr(self, which)
        return y / np.nanmean(y[:n_ref])

    def slope(self, which="mnf"):
        """Least-squares change of `which` per rep (e.g. Hz/rep), ignoring NaN reps."""
        y = getattr(self, which)
        ok = np.isfinite(y)
        return float(np.polyfit(np.flatnonzero(ok), y[ok], 1)[0])


def envelope(x, fs, cutoff=5.0, order=2):
    """Amplitude envelope: full-wave rectify, then zero-phase low-pass at `cutoff` Hz."""
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.abs(x))


def _runs(mask):
    """(R, 2) [start, stop) sample indices of each run of True in a boolean array."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def detect_reps(env, fs, min_rep_s=0.3, min_gap_s=0.2, frac=0.15, min_contrast=3.0):
    """(R, 2) [start, stop) sample indices of the contraction bursts in envelope `env`.

    The threshold sits `frac` of the way from the rest floor (10th percentile) to the active level
    (99th percentile, so spikes shorter than 1 % of the recording don't set it). It is low so the
    weaker eccentric phase still counts as part of the rep. The threshold is unit-free: the
    dataset is in volts (~1e-4), the MyoWare in ADC counts (~100s). If the active
    level is under `min_contrast` x the floor there are no bursts, only noise, and no reps.
    A burst shorter than `min_rep_s` is not a rep (a twitch or a cable knock). Two bursts separated
    by less than `min_gap_s` are one rep (the envelope dips mid-movement).
    """
    floor, active = np.percentile(env, [10, 99])
    if active < min_contrast * floor:
        return np.empty((0, 2), dtype=int)
    runs = _runs(env > floor + frac * (active - floor))
    if len(runs) > 1:
        split = runs[1:, 0] - runs[:-1, 1] >= min_gap_s * fs
        runs = np.column_stack([runs[np.r_[True, split], 0], runs[np.r_[split, True], 1]])
    return runs[runs[:, 1] - runs[:, 0] >= min_rep_s * fs]


def concentric_stop(env, start, stop, frac=0.6):
    """End (sample) of the concentric phase of the rep env[start:stop].

    Assumes the rep starts concentric (curls; not squats or bench) and that the eccentric phase
    needs less activation, so the concentric phase ends at the last sample still above `frac` x the
    rep's envelope peak. If the eccentric is as strong (near failure) this returns about `stop`.
    """
    seg = env[start:stop]
    return start + int(np.flatnonzero(seg >= frac * seg.max())[-1]) + 1


def rep_spectrum(seg, fs, p):
    """Burg AR(p) spectrum of one phase's samples on FREQ_GRID."""
    if len(seg) < 10 * p:
        raise ValueError(f"segment of {len(seg)} samples is too short for AR({p})")
    a, sigma2 = burg(seg - seg.mean(), p)
    return ar_psd(a, sigma2, fs, FREQ_GRID)


def run_reps(trial, p=14, cutoff=5.0, trim_s=0.1, conc_frac=0.6, segments=None, **detect_kw):
    """bandpass -> envelope -> detect reps -> concentric phase -> Burg spectrum -> MNF/MDF/RMS.

    `p` defaults to the biceps order selected in tvar.ipynb. `trim_s` is cut from both ends of the
    concentric phase, where the envelope's smoothing blurs the boundary. `segments` ((R, 2) rep
    bounds in samples) skips rep detection, e.g. for hand-labelled reps.
    """
    x = bandpass(trial.x, trial.fs)
    env = envelope(x, trial.fs, cutoff)
    if segments is None:
        segments = detect_reps(env, trial.fs, **detect_kw)
    segments = np.asarray(segments, dtype=int).reshape(-1, 2)
    trim = int(round(trim_s * trial.fs))
    rows = []
    for a, b in segments:
        c = concentric_stop(env, a, b, conc_frac)
        core = x[a + trim:c - trim]
        try:
            psd = rep_spectrum(core, trial.fs, p)
            f_mean, f_med = mnf(psd, FREQ_GRID), mdf(psd, FREQ_GRID)
        except ValueError:
            f_mean = f_med = np.nan
        rms = np.sqrt(np.mean(core ** 2)) if len(core) else np.nan
        rows.append((trial.t[a], trial.t[c - 1], trial.t[b - 1], f_mean, f_med, rms))
    cols = np.array(rows, dtype=float).reshape(-1, 6).T
    return RepTable(*cols, params={"p": p, "cutoff": cutoff, "trim_s": trim_s,
                                   "conc_frac": conc_frac, **detect_kw})
