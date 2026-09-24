import numpy as np
import pytest

from emg.io import FS
from emg.synthetic import ar2_coeffs, simulate_ar, step_trial
from emg.tvar import burg


def test_ar2_coeffs_put_resonance_at_requested_frequency():
    a = ar2_coeffs(100.0)[0]
    poles = np.roots([1.0, *a])
    assert np.abs(poles) == pytest.approx([0.9, 0.9])
    assert np.max(np.angle(poles)) * FS / (2 * np.pi) == pytest.approx(100.0)


def test_simulate_ar_reproduces_coefficients():
    A = np.tile(ar2_coeffs(120.0), (int(30 * FS), 1))
    x = simulate_ar(A, np.random.default_rng(0))
    a, sigma2 = burg(x, 2)
    np.testing.assert_allclose(a, A[0], rtol=0.02)
    assert sigma2 == pytest.approx(1.0, rel=0.05)


def test_step_trial_changes_spectrum_at_step():
    tr = step_trial(seconds=20.0, step_at=10.0)
    half = int(10 * FS)
    before, _ = burg(tr.x[: half], 2)
    after, _ = burg(tr.x[half:], 2)
    np.testing.assert_allclose(before, ar2_coeffs(150.0)[0], atol=0.02)
    np.testing.assert_allclose(after, ar2_coeffs(80.0)[0], atol=0.02)
    assert tr.t.size == tr.x.size == int(20 * FS) and tr.fs == FS
