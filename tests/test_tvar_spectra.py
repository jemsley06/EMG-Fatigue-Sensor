import numpy as np
import pytest
from scipy import signal

from emg.metrics import band_limit
from emg.tvar import FREQ_GRID, ar_psd, burg, select_order
from tests.synth import FS, rel_err


def test_freq_grid_is_1hz_over_analysis_band():
    assert FREQ_GRID[0] == 20.0 and FREQ_GRID[-1] == 450.0
    np.testing.assert_allclose(np.diff(FREQ_GRID), 1.0)


def test_burg_recovers_stationary_coefficients(ar4_signal):
    x, a_true = ar4_signal
    a, sigma2 = burg(x, 4)
    assert rel_err(a, a_true) < 0.01
    assert sigma2 == pytest.approx(1.0, rel=0.05)


def test_ar_psd_matches_welch(ar4_signal):
    x, a_true = ar4_signal
    fw, pw = signal.welch(x, fs=FS, nperseg=1024)
    pw, fw = band_limit(pw, fw)
    pa = ar_psd(a_true, 1.0, FS, fw)
    assert rel_err(pw / pw.sum(), pa / pa.sum()) < 0.15


def test_ar_psd_shapes_and_white_noise():
    A = np.zeros((7, 3))
    assert ar_psd(A, np.ones(7), FS, FREQ_GRID).shape == (FREQ_GRID.size, 7)
    assert ar_psd(A[0], 1.0, FS, FREQ_GRID).shape == (FREQ_GRID.size,)
    np.testing.assert_allclose(ar_psd(A[0], 2.0, FS, FREQ_GRID), 2.0)   # all-zero AR = flat


def test_select_order_finds_low_order_for_ar4(ar4_signal):
    x, _ = ar4_signal
    assert 3 <= select_order(x, FS) <= 8


def test_select_order_falls_back_to_max_order(ar4_signal):
    x, _ = ar4_signal
    assert select_order(x, FS, p_range=range(1, 3), tol_hz=1e-9) == 2


def test_select_order_rejects_short_signal():
    with pytest.raises(ValueError):
        select_order(np.random.default_rng(0).normal(size=100), FS)
