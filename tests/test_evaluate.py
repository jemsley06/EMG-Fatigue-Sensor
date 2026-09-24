import numpy as np
import pytest

from emg.evaluate import (align_to, match_resolution, median_rise_time, rise_time, roughness,
                          slope, track_stats, trend_snr)
from emg.metrics import SpectralTrack


def _track(t, y, method="m", **params):
    return SpectralTrack(t=np.asarray(t, float), mnf=np.asarray(y, float),
                         mdf=np.asarray(y, float) + 1.0, method=method, params=params)


def test_align_to_samples_track_at_reference_times():
    t = np.arange(0, 60, 0.05)
    ref = _track([10.0, 12.5, 100.0], [0.0, 0.0, 0.0], win_s=4.0)
    out = align_to(_track(t, 2.0 * t), ref)
    np.testing.assert_array_equal(out.t, ref.t)
    np.testing.assert_allclose(out.mnf[:2], [20.0, 25.0])
    assert np.isnan(out.mnf[2])                      # outside the track: NaN, not extrapolated
    np.testing.assert_allclose(out.mdf[:2], out.mnf[:2] + 1.0)


def _step(t, step_at, ramp_s, lo=100.0, hi=60.0):
    """Linear ramp from lo to hi over ramp_s seconds centred on step_at."""
    frac = np.clip((t - (step_at - ramp_s / 2)) / ramp_s, 0.0, 1.0)
    return lo + (hi - lo) * frac


def test_rise_time_of_linear_ramp_is_80_percent_of_ramp():
    t = np.arange(0, 60, 0.05)
    assert rise_time(_track(t, _step(t, 30.0, 4.0)), 30.0) == pytest.approx(3.2, abs=0.1)


def test_rise_time_ignores_noise_spikes_away_from_step():
    t = np.arange(0, 60, 0.05)
    y = _step(t, 30.0, 4.0)
    y[(t > 22.0) & (t < 22.2)] = 60.0                # spike to the post-step level, 8 s early
    assert rise_time(_track(t, y), 30.0) == pytest.approx(3.2, abs=0.1)


def test_rise_time_raises_when_step_not_resolved():
    t = np.arange(0, 60, 0.05)
    with pytest.raises(ValueError):
        rise_time(_track(t, _step(t, 30.0, 40.0)), 30.0)   # ramp longer than +-10 s window


def test_median_rise_time_skips_unresolved_tracks():
    t = np.arange(0, 60, 0.05)
    tracks = [_track(t, _step(t, 30.0, r)) for r in (2.0, 4.0, 40.0)]
    assert median_rise_time(tracks, 30.0) == pytest.approx(0.8 * 3.0, abs=0.1)
    assert np.isnan(median_rise_time(tracks[2:], 30.0))


def test_match_resolution_picks_log_closest():
    assert match_resolution({1e-9: 12.0, 1e-8: 4.0, 1e-7: 1.0}, target=3.0) == 1e-8
    assert match_resolution({1: 1.5, 2: 6.0, 3: float("nan"), 4: None}, target=3.0) in (1, 2)
    with pytest.raises(ValueError):
        match_resolution({1: float("nan")}, target=3.0)


def test_slope_snr_roughness_on_known_line():
    t = np.arange(0, 120, 2.0)                        # 2 minutes
    noise = np.tile([0.5, -0.5], t.size // 2)
    tr = _track(t, 100 - 3.0 * t / 60 + noise)        # -3 Hz/min
    assert slope(tr) == pytest.approx(-3.0, abs=0.05)
    assert trend_snr(tr) == pytest.approx(6.0 * (t[-1] - t[0]) / 120 / 0.5, rel=0.05)
    assert roughness(tr) == pytest.approx(np.std(np.diff(100 - 3.0 * t / 60 + noise)))


def test_stats_ignore_nans_and_reject_bad_which():
    t = np.arange(10.0)
    y = 2.0 * t
    y[3] = np.nan
    assert slope(_track(t, y)) == pytest.approx(120.0)
    with pytest.raises(ValueError):
        slope(_track(t, y), which="rms")
    with pytest.raises(ValueError):
        slope(_track(t[:2], y[:2]))


def test_track_stats_keys():
    t = np.arange(10.0)
    s = track_stats(_track(t, t + np.sin(t)))
    assert set(s) == {f"{w}_{k}" for w in ("mnf", "mdf") for k in ("slope", "snr", "rough")}
