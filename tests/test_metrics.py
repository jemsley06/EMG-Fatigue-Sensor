import numpy as np
import pytest

from emg.metrics import BAND, SpectralTrack, band_limit, mdf, mnf

F = np.arange(0.0, 101.0)  # 0..100 Hz, 1 Hz spacing


def test_flat_spectrum_mnf_and_mdf_are_band_centre():
    psd = np.ones_like(F)
    assert mnf(psd, F) == pytest.approx(50.0)
    assert mdf(psd, F) == pytest.approx(50.0)


def test_single_tone_mnf_and_mdf_equal_tone_frequency():
    psd = np.zeros_like(F)
    psd[37] = 5.0
    assert mnf(psd, F) == pytest.approx(37.0)
    assert mdf(psd, F) == pytest.approx(37.0)


def test_mdf_interpolates_between_bins():
    psd = np.zeros_like(F)
    psd[[20, 21]] = [1.0, 3.0]    # half the power (2 of 4) reached 1/3 of the way through bin 21
    assert mdf(psd, F) == pytest.approx(20.5 + 1.0 / 3.0)


def test_vectorized_over_time_axis():
    psd = np.stack([np.ones_like(F), np.where(F == 10, 1.0, 0.0)], axis=1)  # (F, 2)
    np.testing.assert_allclose(mnf(psd, F), [50.0, 10.0])
    np.testing.assert_allclose(mdf(psd, F), [50.0, 10.0])


def test_zero_power_gives_nan():
    psd = np.zeros((F.size, 2))
    assert np.all(np.isnan(mnf(psd, F)))
    assert np.all(np.isnan(mdf(psd, F)))


def test_band_limit_keeps_inclusive_band():
    f = np.arange(0.0, 600.0, 1.0)
    psd = np.ones((f.size, 3))
    p2, f2 = band_limit(psd, f)
    assert f2[0] == BAND[0] and f2[-1] == BAND[1]
    assert p2.shape == (f2.size, 3)


def test_spectral_track_defaults():
    tr = SpectralTrack(t=np.zeros(2), mnf=np.zeros(2), mdf=np.zeros(2), method="m")
    assert tr.params == {} and tr.psd is None and tr.f is None
