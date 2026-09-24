"""Loading trials from sEMG_data/, keyed by muscle."""
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

FS = 1259.0
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "sEMG_data"

TRIAL_TO_MUSCLE = {
    1: "R_DELTOID_ANTERIOR", 9: "R_DELTOID_ANTERIOR",
    2: "L_DELTOID_ANTERIOR", 10: "L_DELTOID_ANTERIOR",
    3: "R_DELTOID_POSTERIOR", 11: "R_DELTOID_POSTERIOR",
    4: "L_DELTOID_POSTERIOR", 12: "L_DELTOID_POSTERIOR",
    5: "R_BICEPS_BRACHII",
    6: "L_BICEPS_BRACHII",
    7: "R_DELTOID_MEDIUS",
    8: "L_DELTOID_MEDIUS",
}
MUSCLES = list(dict.fromkeys(TRIAL_TO_MUSCLE[t] for t in sorted(TRIAL_TO_MUSCLE)))
# muscle label -> prefix of its EMG column header, e.g. "R DELTOID ANTERIOR: EMG 2 [V]"
MUSCLE_TO_COLUMN = {m: m.replace("_", " ") for m in MUSCLES}


@dataclass
class Trial:
    t: np.ndarray
    x: np.ndarray
    fs: float
    muscle: str
    subject: int
    trial: int


def trials_for_muscle(muscle):
    if muscle not in MUSCLE_TO_COLUMN:
        raise KeyError(f"unknown muscle {muscle!r}; expected one of {MUSCLES}")
    return sorted(t for t, m in TRIAL_TO_MUSCLE.items() if m == muscle)


def _pick_columns(header, muscle):
    prefix = MUSCLE_TO_COLUMN[muscle] + ":"
    matches = [i for i, c in enumerate(header) if c.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one column starting {prefix!r}, found {len(matches)} in {header}")
    i = matches[0]
    if i == 0 or not header[i - 1].startswith("X [s]"):
        raise ValueError(f"no time column before {header[i]!r}")
    return header[i - 1], header[i]


def load_trial(subject, trial, data_dir=DEFAULT_DATA_DIR):
    """Load the prime-mover channel of one trial."""
    if trial not in TRIAL_TO_MUSCLE:
        raise KeyError(f"unknown trial {trial}; expected 1-12")
    subject_dir = Path(data_dir) / f"subject_{subject}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"{subject_dir} not found - unzip {subject_dir}.zip into {Path(data_dir)} first"
        )
    path = subject_dir / f"trial_{trial}.csv"
    muscle = TRIAL_TO_MUSCLE[trial]
    header = pl.read_csv(path, n_rows=0).columns
    tcol, xcol = _pick_columns(header, muscle)
    df = pl.read_csv(path, columns=[tcol, xcol]).drop_nulls()
    t = df[tcol].to_numpy().astype(float)
    x = df[xcol].to_numpy().astype(float)
    stop = np.flatnonzero(np.diff(t) <= 0)   # some files end with an all-zero padding row
    if stop.size:
        t, x = t[:stop[0] + 1], x[:stop[0] + 1]
    fs = 1.0 / np.median(np.diff(t))
    if abs(fs - FS) / FS > 0.01:
        raise ValueError(f"{path}: estimated fs {fs:.1f} Hz differs from {FS} Hz by >1%")
    return Trial(t=t, x=x, fs=fs, muscle=muscle, subject=subject, trial=trial)


def extract_subject(subject, data_dir=DEFAULT_DATA_DIR):
    """Unzip <data_dir>/subject_<N>.zip unless subject_<N>/ already exists. Returns the directory."""
    data_dir = Path(data_dir)
    target = data_dir / f"subject_{subject}"
    if target.is_dir():
        return target
    archive = data_dir / f"subject_{subject}.zip"
    if not archive.exists():
        raise FileNotFoundError(f"neither {target} nor {archive} exists")
    with zipfile.ZipFile(archive) as z:
        z.extractall(data_dir)   # archives contain a top-level subject_<N>/ folder
    return target
