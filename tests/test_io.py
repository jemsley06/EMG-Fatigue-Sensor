import zipfile

import numpy as np
import pytest

from emg.io import (DEFAULT_DATA_DIR, FS, MUSCLES, TRIAL_TO_MUSCLE, extract_subject, load_myoware,
                    load_trial, trials_for_muscle)

R_HEADER = ["X [s]", "R BICEPS BRACHII: EMG 1 [V]", "X [s].1", "R DELTOID ANTERIOR: EMG 2 [V]",
            "X [s].5", "R DELTOID MEDIUS: EMG 6 [V]", "X [s].6", "R DELTOID POSTERIOR: EMG 7 [V]"]
L_HEADER = ["X [s].2", "L DELTOID ANTERIOR: EMG 3 [V]", "X [s].3", "L BICEPS BRACHII: EMG 4 [V]",
            "X [s].4", "L DELTOID POSTERIOR: EMG 5 [V]", "X [s].7", "L DELTOID MEDIUS: EMG 8 [V]"]


def test_every_trial_maps_to_one_of_eight_muscles():
    assert sorted(TRIAL_TO_MUSCLE) == list(range(1, 13))
    assert len(MUSCLES) == 8
    assert sorted(t for m in MUSCLES for t in trials_for_muscle(m)) == list(range(1, 13))


def test_trials_for_muscle_examples():
    assert trials_for_muscle("R_DELTOID_ANTERIOR") == [1, 9]
    assert trials_for_muscle("L_BICEPS_BRACHII") == [6]


def test_unknown_muscle_and_trial_raise():
    with pytest.raises(KeyError):
        trials_for_muscle("QUADRICEPS")
    with pytest.raises(KeyError):
        load_trial(1, 13)


def _write_csv(path, header, rows):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")


def _rows(n, dt, n_short=None):
    """Numeric rows for an 8-column file; the last `n - n_short` rows have empty L/R biceps cells."""
    rows = []
    for i in range(n):
        t = f"{i * dt}"
        vals = [t, f"{np.sin(i) * 1e-5}"] * 4
        if n_short is not None and i >= n_short:
            vals[2:4] = ["", ""]   # second channel pair ends early (like real files)
        rows.append(vals)
    return rows


@pytest.mark.parametrize("trial, header, col", [(1, R_HEADER, 3), (2, L_HEADER, 1), (6, L_HEADER, 3)])
def test_load_trial_picks_prime_mover_column(tmp_path, trial, header, col):
    (tmp_path / "subject_1").mkdir()
    rows = _rows(50, 1 / FS)
    rows = [[r[i] if i != col else str(1000 + k) for i in range(8)] for k, r in enumerate(rows)]
    _write_csv(tmp_path / "subject_1" / f"trial_{trial}.csv", header, rows)
    tr = load_trial(1, trial, data_dir=tmp_path)
    assert tr.muscle == TRIAL_TO_MUSCLE[trial]
    np.testing.assert_array_equal(tr.x, 1000 + np.arange(50))
    assert tr.fs == pytest.approx(FS, rel=1e-6)
    assert (tr.subject, tr.trial) == (1, trial)


def test_load_trial_drops_trailing_nulls(tmp_path):
    (tmp_path / "subject_1").mkdir()
    _write_csv(tmp_path / "subject_1" / "trial_1.csv", R_HEADER, _rows(50, 1 / FS, n_short=40))
    tr = load_trial(1, 1, data_dir=tmp_path)   # trial 1 = R DELTOID ANTERIOR = columns 2/3
    assert len(tr.x) == len(tr.t) == 40


def test_load_trial_truncates_trailing_zero_row(tmp_path):
    (tmp_path / "subject_1").mkdir()
    rows = _rows(50, 1 / FS) + [["0.0"] * 8]      # real trials 4/6/10/12 end like this
    _write_csv(tmp_path / "subject_1" / "trial_1.csv", R_HEADER, rows)
    tr = load_trial(1, 1, data_dir=tmp_path)
    assert len(tr.t) == 50 and tr.t[-1] == pytest.approx(49 / FS)


def test_load_trial_rejects_wrong_sampling_rate(tmp_path):
    (tmp_path / "subject_1").mkdir()
    _write_csv(tmp_path / "subject_1" / "trial_1.csv", R_HEADER, _rows(50, 1 / 1000.0))
    with pytest.raises(ValueError, match="fs"):
        load_trial(1, 1, data_dir=tmp_path)


def test_missing_subject_dir_hints_to_unzip(tmp_path):
    with pytest.raises(FileNotFoundError, match="unzip"):
        load_trial(2, 1, data_dir=tmp_path)


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "subject_1" / "trial_1.csv").exists(),
                    reason="subject_1 not extracted")
@pytest.mark.parametrize("trial", range(1, 13))
def test_load_real_trial(trial):
    tr = load_trial(1, trial)
    assert tr.muscle == TRIAL_TO_MUSCLE[trial]
    assert len(tr.x) > 10 * FS and np.all(np.isfinite(tr.x))
    assert np.all(np.diff(tr.t) > 0)
    assert tr.fs == pytest.approx(FS, rel=0.01)


def test_extract_subject_unzips_once(tmp_path):
    with zipfile.ZipFile(tmp_path / "subject_3.zip", "w") as z:
        z.writestr("subject_3/trial_1.csv", "original\n")
    d = extract_subject(3, data_dir=tmp_path)
    assert d == tmp_path / "subject_3"
    (d / "trial_1.csv").write_text("edited\n")
    extract_subject(3, data_dir=tmp_path)            # already extracted: must not overwrite
    assert (d / "trial_1.csv").read_text() == "edited\n"


def test_extract_subject_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_subject(4, data_dir=tmp_path)


def _write_myoware(path, t_us, raw=None):
    raw = np.arange(len(t_us)) % 4096 if raw is None else raw
    path.write_text("t_us,raw\n" + "".join(f"{t},{r}\n" for t, r in zip(t_us, raw)))
    return path


def test_load_myoware_reads_counts_and_estimates_fs(tmp_path):
    t_us = 5_000_000 + np.arange(4000) * 500           # 2 kHz, clock not starting at 0
    tr = load_myoware(_write_myoware(tmp_path / "rec.csv", t_us), "R_BICEPS_BRACHII")
    assert tr.fs == pytest.approx(2000.0)
    assert tr.t[0] == 0.0 and tr.t[-1] == pytest.approx(3999 / 2000)
    np.testing.assert_array_equal(tr.x, np.arange(4000) % 4096)
    assert tr.muscle == "R_BICEPS_BRACHII"


def test_load_myoware_rejects_low_fs(tmp_path):
    path = _write_myoware(tmp_path / "rec.csv", np.arange(1000) * 2000)   # 500 Hz
    with pytest.raises(ValueError, match="fs"):
        load_myoware(path, "R_BICEPS_BRACHII")


def test_load_myoware_rejects_dropped_samples(tmp_path):
    t_us = np.arange(4000) * 500
    t_us[2000:] += 500 * 100                            # one gap is fine...
    load_myoware(_write_myoware(tmp_path / "ok.csv", t_us), "R_BICEPS_BRACHII")
    gappy = np.cumsum(np.where(np.arange(4000) % 50 == 0, 1500, 500))   # 2% of intervals are 3x
    with pytest.raises(ValueError, match="gaps"):
        load_myoware(_write_myoware(tmp_path / "bad.csv", gappy), "R_BICEPS_BRACHII")


def test_load_myoware_rejects_non_increasing_time(tmp_path):
    t_us = np.arange(4000) * 500
    t_us[100] = t_us[99]
    with pytest.raises(ValueError, match="increasing"):
        load_myoware(_write_myoware(tmp_path / "rec.csv", t_us), "R_BICEPS_BRACHII")


def test_load_myoware_rejects_wrong_header_and_muscle(tmp_path):
    path = tmp_path / "rec.csv"
    path.write_text("time,value\n0,1\n")
    with pytest.raises(ValueError, match="t_us,raw"):
        load_myoware(path, "R_BICEPS_BRACHII")
    with pytest.raises(KeyError):
        load_myoware(_write_myoware(path, np.arange(10) * 500), "QUADRICEPS")
