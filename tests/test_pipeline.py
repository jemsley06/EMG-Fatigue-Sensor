import numpy as np
import pytest

from emg.baseline import sliding_fft
from emg.io import Trial
from emg.tvar import FREQ_GRID, run_tvar
from tests.synth import FS


def _trial(x, t0=0.0):
    t = t0 + np.arange(x.size) / FS
    return Trial(t=t, x=x, fs=FS, muscle="R_BICEPS_BRACHII", subject=0, trial=5)


def _tone(seconds, freq=100.0):
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * freq * t) + 0.01 * np.random.default_rng(0).normal(size=t.size)


# ---- TVAR pipeline

def test_run_tvar_returns_track_in_trial_time(moving_peak_signal):
    x, _ = moving_peak_signal
    tr = _trial(1e-5 * x, t0=3.0)                 # raw EMG scale, non-zero clock start
    track = run_tvar(tr, "kalman", 4, q=1e-8, keep_psd=True)
    assert track.method == "tvar_kalman" and track.params == {"p": 4, "q": 1e-8}
    assert track.t[0] >= tr.t[0] + 1.0            # first second discarded
    assert track.psd.shape == (FREQ_GRID.size, track.t.size)
    assert np.all((track.mnf > 20) & (track.mnf < 450))
    early = track.mnf[track.t < 20].mean()
    late = track.mnf[track.t > 110].mean()
    assert early > late + 30                       # resonance sweeps 150 -> 80 Hz


def test_run_tvar_drops_psd_by_default(moving_peak_signal):
    track = run_tvar(_trial(moving_peak_signal[0][: int(10 * FS)]), "rls", 4, tau=1.0)
    assert track.psd is None and track.f is None
    assert track.method == "tvar_rls"


def test_run_tvar_rejects_unknown_method(moving_peak_signal):
    with pytest.raises(ValueError):
        run_tvar(_trial(moving_peak_signal[0][:5000]), "burg", 4)


# ---- FFT baseline

def test_baseline_tone_gives_tone_frequency():
    track = sliding_fft(_trial(_tone(20)))
    np.testing.assert_allclose(track.mnf, 100.0, atol=2.0)
    np.testing.assert_allclose(track.mdf, 100.0, atol=0.5)


def test_baseline_window_grid_matches_reference_pipeline():
    track = sliding_fft(_trial(_tone(20)))
    n_win, n_step = int(4 * FS), int(2 * FS)
    assert track.t.size == (int(20 * FS) - n_win) // n_step + 1
    assert track.t[0] == pytest.approx((n_win // 2) / FS)
    assert track.params == {"win_s": 4.0, "step_s": 2.0}
    assert track.method == "fft_baseline"


def test_baseline_too_short_raises():
    with pytest.raises(ValueError):
        sliding_fft(_trial(_tone(3)))
