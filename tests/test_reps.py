import numpy as np
import pytest

from emg.io import FS
from emg.metrics import mnf
from emg.reps import _runs, detect_reps, envelope, run_reps
from emg.synthetic import ar2_coeffs, rep_trial
from emg.tvar import FREQ_GRID, ar_psd


@pytest.fixture(scope="module")
def reps():
    return rep_trial(seed=3)


def _burst_env(pattern, fs=FS, floor=0.05, level=1.0, seed=0):
    """Envelope from [(seconds, active), ...] with a little noise on top."""
    env = np.concatenate([np.full(int(s * fs), level if on else floor) for s, on in pattern])
    return env + 0.01 * np.random.default_rng(seed).normal(size=env.size)


def test_runs_finds_true_runs():
    np.testing.assert_array_equal(_runs([0, 1, 1, 0, 1, 0, 0, 1]), [[1, 3], [4, 5], [7, 8]])
    assert _runs([0, 0]).shape == (0, 2)


def test_envelope_rises_during_burst(reps):
    trial, bounds = reps
    env = envelope(trial.x, trial.fs)
    a, b = bounds[0]
    assert env[a:b].mean() > 5 * env[:a - int(0.5 * FS)].mean()


# ---- detect_reps

def test_detect_reps_finds_every_rep_near_true_bounds(reps):
    trial, bounds = reps
    found = detect_reps(envelope(trial.x, trial.fs), trial.fs)
    assert found.shape == bounds.shape
    assert np.abs(found - bounds).max() < 0.15 * FS


def test_detect_reps_ignores_short_spike():
    env = _burst_env([(2, 0), (1, 1), (1, 0), (0.05, 1), (1, 0), (1, 1), (1, 0)])
    assert len(detect_reps(env, FS)) == 2


def test_detect_reps_merges_brief_dip():
    env = _burst_env([(2, 0), (0.6, 1), (0.08, 0), (0.6, 1), (2, 0)])
    assert len(detect_reps(env, FS)) == 1


def test_detect_reps_does_not_depend_on_units():
    env = _burst_env([(2, 0), (1, 1), (1, 0), (1, 1), (1, 0), (1, 1), (2, 0)])
    np.testing.assert_array_equal(detect_reps(env, FS), detect_reps(1e-4 * env, FS))


def test_detect_reps_finds_nothing_in_rest_only_recording():
    noise = 0.05 * np.random.default_rng(0).normal(size=int(20 * FS))
    assert len(detect_reps(envelope(noise, FS), FS)) == 0


# ---- per-rep spectra (independent of detect_reps)

def test_run_reps_tracks_falling_mnf_per_rep(reps):
    trial, bounds = reps
    table = run_reps(trial, segments=bounds)
    true = [mnf(ar_psd(ar2_coeffs(f)[0], 1.0, FS, FREQ_GRID), FREQ_GRID)
            for f in np.linspace(150.0, 100.0, len(bounds))]
    assert len(table.mnf) == len(bounds)
    assert np.abs(table.mnf - true).max() < 8.0
    assert np.polyfit(np.arange(len(bounds)), table.mnf, 1)[0] < 0
    assert table.relative()[-1] < 0.95


def test_run_reps_rms_is_in_input_units(reps):
    trial, bounds = reps
    small = type(trial)(t=trial.t, x=1e-4 * trial.x, fs=trial.fs, muscle=trial.muscle)
    a, b = run_reps(trial, segments=bounds), run_reps(small, segments=bounds)
    np.testing.assert_allclose(b.rms, 1e-4 * a.rms, rtol=1e-6)
    np.testing.assert_allclose(b.mnf, a.mnf, rtol=1e-6)    # spectrum ignores scale
