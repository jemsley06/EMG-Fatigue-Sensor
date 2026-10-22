import numpy as np
import pytest

from emg.tvar import _kalman_kernel, tvar_kalman
from tests.synth import FS, rel_err


@pytest.mark.parametrize("kwargs", [
    {}, {"smooth": False}, {"r": 1.0},
], ids=["smoothed", "filtered", "fixed_r"])
def test_kalman_recovers_stationary_coefficients(ar4_signal, kwargs):
    x, a_true = ar4_signal
    t, A, s2 = tvar_kalman(x, 4, q=1e-8, **kwargs)
    burned = t > 10
    assert rel_err(np.median(A[burned], axis=0), a_true) < 0.05
    assert np.median(s2[burned]) == pytest.approx(1.0, rel=0.1)


def test_kalman_smoother_tracks_moving_peak(moving_peak_signal, track_rmse):
    x, A = moving_peak_signal
    rmse_mnf, rmse_mdf = track_rmse(x, A, lambda x: tvar_kalman(x, 2, q=1e-8))
    assert rmse_mnf < 5.0 and rmse_mdf < 5.0


def test_smoothing_beats_filtering(moving_peak_signal, track_rmse):
    x, A = moving_peak_signal
    smooth = track_rmse(x, A, lambda x: tvar_kalman(x, 2, q=1e-9))
    filt = track_rmse(x, A, lambda x: tvar_kalman(x, 2, q=1e-9, smooth=False))
    assert smooth[0] < 0.5 * filt[0]


def test_checkpoint_spacing_does_not_change_smoother(ar4_signal):
    x = ar4_signal[0][:20000]
    args = (x, 4, 1e-7, 10.0, 1.0, 0.999, 10)
    A_small, _ = _kalman_kernel(*args, 256, True)
    A_one, _ = _kalman_kernel(*args, x.size, True)       # single segment = plain RTS
    np.testing.assert_allclose(A_small, A_one, rtol=1e-12, atol=1e-12)


def test_kalman_numba_matches_python(ar4_signal):
    x = ar4_signal[0][: int(2 * FS)]
    args = (x, 4, 1e-6, 10.0, 1.0, 0.999, 10, 64, True)
    for jit_out, py_out in zip(_kalman_kernel(*args), _kalman_kernel.py_func(*args)):
        np.testing.assert_allclose(jit_out, py_out, rtol=1e-9, atol=1e-12)
