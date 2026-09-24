"""Scoring spectral tracks for fatigue-trend clarity."""
import numpy as np

from .metrics import SpectralTrack

WHICH = ("mnf", "mdf")


def align_to(track, reference):
    """Sample `track` at `reference`'s time points (linear interpolation; NaN outside its span)."""
    out = {w: np.interp(reference.t, track.t, getattr(track, w), left=np.nan, right=np.nan)
           for w in WHICH}
    return SpectralTrack(t=reference.t.copy(), mnf=out["mnf"], mdf=out["mdf"],
                         method=track.method, params=dict(track.params))


def rise_time(track, step_at, settle_s=10.0, which="mnf"):
    """10-90 % response time (s) of a track to a step change at `step_at`.

    Levels are the medians before `step_at - settle_s` and after `step_at + settle_s`. The 10 % /
    90 % times are the last time below 10 % before, and the first time at/above 90 % after, the
    upward 50 % crossing nearest `step_at` - robust to noise spikes away from the transition.
    Raises ValueError if the step is not resolved within +-settle_s.
    """
    t, y = _xy(track, which)
    pre = np.median(y[t < step_at - settle_s])
    post = np.median(y[t > step_at + settle_s])
    z = (y - pre) / (post - pre)
    win = (t >= step_at - settle_s) & (t <= step_at + settle_s)
    t, z = t[win], z[win]
    ups = np.flatnonzero((z[:-1] < 0.5) & (z[1:] >= 0.5)) + 1
    if ups.size == 0:
        raise ValueError("track never crosses 50 % of the step")
    i50 = ups[np.argmin(np.abs(t[ups] - step_at))]
    below10 = np.flatnonzero(z[:i50] < 0.1)
    above90 = np.flatnonzero(z[i50:] >= 0.9)
    if below10.size == 0 or above90.size == 0:
        raise ValueError(f"step not resolved within +-{settle_s}s of step_at")
    return float(t[i50 + above90[0]] - t[below10[-1]])


def median_rise_time(tracks, step_at, settle_s=10.0, which="mnf"):
    """Median rise time over several tracks (e.g. step signals with different noise seeds),
    skipping tracks where the step is not resolved. NaN if none are."""
    rts = []
    for tr in tracks:
        try:
            rts.append(rise_time(tr, step_at, settle_s, which))
        except ValueError:
            pass
    return float(np.median(rts)) if rts else float("nan")


def match_resolution(rise_times, target):
    """Parameter value whose rise time is closest to `target` on a log scale.

    rise_times: {value: rise time in s}. Values whose rise time could not be measured (None/NaN)
    are ignored.
    """
    ok = {v: rt for v, rt in rise_times.items() if rt is not None and np.isfinite(rt)}
    if not ok:
        raise ValueError("no parameter value has a measurable rise time")
    return min(ok, key=lambda v: abs(np.log(ok[v] / target)))


def _xy(track, which):
    if which not in WHICH:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    t, y = track.t, getattr(track, which)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        raise ValueError(f"need >= 3 finite {which} values, got {ok.sum()}")
    return t[ok], y[ok]


def slope(track, which="mnf"):
    """Least-squares slope in Hz/min."""
    t, y = _xy(track, which)
    return float(np.polyfit(t / 60.0, y, 1)[0])


def trend_snr(track, which="mnf"):
    """|slope * duration| / std(residuals of the linear fit)."""
    t, y = _xy(track, which)
    coef = np.polyfit(t, y, 1)
    resid = y - np.polyval(coef, t)
    return float(abs(coef[0] * (t[-1] - t[0])) / resid.std())


def roughness(track, which="mnf"):
    """Std of first differences (Hz)."""
    _, y = _xy(track, which)
    return float(np.std(np.diff(y)))


def track_stats(track):
    """{'mnf_slope', 'mnf_snr', 'mnf_rough', 'mdf_slope', ...} for one track."""
    return {f"{w}_{name}": fn(track, w)
            for w in WHICH
            for name, fn in (("slope", slope), ("snr", trend_snr), ("rough", roughness))}
