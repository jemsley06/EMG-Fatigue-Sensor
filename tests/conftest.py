import numpy as np
import pytest

from tests.synth import FS, moving_peak_ar2, simulate_ar, stationary_ar4


@pytest.fixture(scope="session")
def ar4_signal():
    """60 s of a stationary AR(4) signal and its true coefficients (p,)."""
    N = int(60 * FS)
    A = stationary_ar4(N)
    return simulate_ar(A, np.random.default_rng(0)), A[0]


@pytest.fixture(scope="session")
def moving_peak_signal():
    """120 s AR(2) with a 150 -> 80 Hz resonance sweep, and its true coefficients (N, 2)."""
    N = int(120 * FS)
    A = moving_peak_ar2(N)
    return simulate_ar(A, np.random.default_rng(1)), A


@pytest.fixture(scope="session")
def track_rmse():
    """fit(x) -> (t, A, sigma2); returns (MNF RMSE, MDF RMSE) in Hz against the true AR spectrum."""
    from emg.metrics import mdf, mnf
    from emg.tvar import FREQ_GRID, ar_psd

    def _rmse(x, A_true, fit, burn_in_s=5.0):
        t, A, s2 = fit(x)
        idx = np.round(t * FS).astype(int)
        true_psd = ar_psd(A_true[idx], 1.0, FS, FREQ_GRID)
        est_psd = ar_psd(A, s2, FS, FREQ_GRID)
        burned = t > burn_in_s
        err = lambda fn: np.sqrt(np.mean((fn(est_psd, FREQ_GRID) - fn(true_psd, FREQ_GRID))[burned] ** 2))
        return err(mnf), err(mdf)

    return _rmse
