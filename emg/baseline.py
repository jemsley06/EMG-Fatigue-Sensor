"""Reference sliding-window FFT (re-implemented from the dataset's code.ipynb)."""
import numpy as np

from .metrics import SpectralTrack, band_limit, mdf, mnf
from .preprocess import bandpass


def sliding_fft(trial, win_s=4.0, step_s=2.0):
    """Rectangular-window FFT power per window; MNF/MDF over 20-450 Hz; t = window centre."""
    x = bandpass(trial.x, trial.fs)
    x = x - x.mean()
    n_win = int(win_s * trial.fs)
    n_step = int(step_s * trial.fs)
    n = (len(x) - n_win) // n_step + 1
    if n < 1:
        raise ValueError(f"trial shorter than one {win_s}s window")
    starts = np.arange(n) * n_step
    frames = np.stack([x[s:s + n_win] for s in starts], axis=1)        # (n_win, T)
    power = np.abs(np.fft.rfft(frames, axis=0)) ** 2
    f = np.fft.rfftfreq(n_win, d=1.0 / trial.fs)
    power, f = band_limit(power, f)
    return SpectralTrack(
        t=trial.t[starts + n_win // 2], mnf=mnf(power, f), mdf=mdf(power, f),
        method="fft_baseline", params={"win_s": win_s, "step_s": step_s},
    )
