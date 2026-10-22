"""Spectral-track container and frequency metrics shared by every method."""
from dataclasses import dataclass, field

import numpy as np

BAND = (20.0, 450.0)


@dataclass
class SpectralTrack:
    t: np.ndarray                 # (T,) seconds
    mnf: np.ndarray               # (T,) Hz
    mdf: np.ndarray               # (T,) Hz
    method: str
    params: dict = field(default_factory=dict)
    psd: np.ndarray | None = None  # (F, T)
    f: np.ndarray | None = None    # (F,) Hz


def band_limit(psd, f, band=BAND):
    """Keep only rows of `psd` (F, ...) whose frequency lies in [band[0], band[1]]."""
    mask = (f >= band[0]) & (f <= band[1])
    return psd[mask], f[mask]


def mnf(psd, f):
    """Mean frequency. `psd` is (F,) or (F, T); returns scalar or (T,)."""
    psd = np.asarray(psd, dtype=float)
    fw = f if psd.ndim == 1 else f[:, None]
    total = psd.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (fw * psd).sum(axis=0) / total, np.nan)


def mdf(psd, f):
    """Median frequency, linearly interpolated on cumulative power at bin edges.

    Assumes uniform frequency spacing. `psd` is (F,) or (F, T); returns scalar or (T,).
    """
    psd = np.asarray(psd, dtype=float)
    squeeze = psd.ndim == 1
    if squeeze:
        psd = psd[:, None]
    df = f[1] - f[0]
    edges = np.concatenate([[f[0] - df / 2], f + df / 2])
    out = np.full(psd.shape[1], np.nan)
    for j in range(psd.shape[1]):
        cum = np.concatenate([[0.0], np.cumsum(psd[:, j])])
        if cum[-1] <= 0:
            continue
        half = cum[-1] / 2
        i = np.searchsorted(cum, half)  # first edge with cum >= half
        lo, hi = cum[i - 1], cum[i]
        out[j] = edges[i - 1] + (half - lo) / (hi - lo) * df
    return out[0] if squeeze else out
