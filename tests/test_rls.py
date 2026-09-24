import numpy as np
import pytest

from emg.tvar import _rls_kernel, tvar_rls
from tests.synth import FS, rel_err


def test_rls_recovers_stationary_coefficients(ar4_signal):
    x, a_true = ar4_signal
    t, A, s2 = tvar_rls(x, 4, tau=2.0)
    burned = t > 10
    assert rel_err(np.median(A[burned], axis=0), a_true) < 0.05
    assert np.median(s2[burned]) == pytest.approx(1.0, rel=0.1)


def test_output_grid_is_every_50ms(ar4_signal):
    x, _ = ar4_signal
    t, A, s2 = tvar_rls(x, 4, tau=2.0)
    assert A.shape == (t.size, 4) and s2.shape == (t.size,)
    np.testing.assert_allclose(np.diff(t), round(0.05 * FS) / FS)
    assert t[0] == pytest.approx(4 / FS)


def test_rls_tracks_moving_peak(moving_peak_signal, track_rmse):
    x, A = moving_peak_signal
    rmse_mnf, rmse_mdf = track_rmse(x, A, lambda x: tvar_rls(x, 2, tau=2.0))
    assert rmse_mnf < 8.0 and rmse_mdf < 8.0


def test_rls_numba_matches_python(ar4_signal):
    x = ar4_signal[0][: int(2 * FS)]
    args = (x, 4, 0.999, 100.0, 0.999, 10)
    for jit_out, py_out in zip(_rls_kernel(*args), _rls_kernel.py_func(*args)):
        np.testing.assert_allclose(jit_out, py_out, rtol=1e-9, atol=1e-12)
