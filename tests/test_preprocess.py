import numpy as np
import pytest

from emg.preprocess import bandpass, standardize

FS = 1259.0
T = np.arange(int(10 * FS)) / FS


def _gain(freq):
    x = np.sin(2 * np.pi * freq * T)
    y = bandpass(x, FS)
    mid = slice(int(2 * FS), int(8 * FS))   # ignore filtfilt edge transients
    return y[mid].std() / x[mid].std()


def test_bandpass_passes_in_band_and_rejects_out_of_band():
    assert _gain(100.0) == pytest.approx(1.0, abs=0.02)
    assert _gain(5.0) < 0.01
    assert _gain(580.0) < 0.05


def test_standardize_zero_mean_unit_variance():
    x = 3e-5 * np.random.default_rng(0).normal(size=1000) + 1e-3
    z = standardize(x)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0)


def test_standardize_constant_raises():
    with pytest.raises(ValueError):
        standardize(np.ones(10))
