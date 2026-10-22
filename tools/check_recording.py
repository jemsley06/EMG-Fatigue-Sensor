"""Sanity-check a MyoWare recording: sampling, ADC range, and whether MNF/MDF drift down.

    python -m tools.check_recording recordings/biceps_01.csv R_BICEPS_BRACHII
"""
import argparse

import numpy as np

from emg.baseline import sliding_fft
from emg.evaluate import track_stats
from emg.io import load_myoware

ADC_MAX = 4095   # ESP32 12-bit


def main():
    ap = argparse.ArgumentParser(description="Sanity-check a MyoWare recording.")
    ap.add_argument("path")
    ap.add_argument("muscle", nargs="?", default="R_BICEPS_BRACHII")
    args = ap.parse_args()

    tr = load_myoware(args.path, args.muscle)   # raises on low fs, gaps, bad timestamps
    clipped = np.mean((tr.x <= 0) | (tr.x >= ADC_MAX))
    print(f"fs {tr.fs:.1f} Hz, {tr.t[-1]:.1f} s, {len(tr.x)} samples")
    print(f"ADC mean {tr.x.mean():.0f} (expect ~mid-range), std {tr.x.std():.1f}, clipped {clipped:.2%}")
    if clipped > 0.001:
        print("  WARNING: clipping distorts the spectrum - check 3.3 V supply and electrode contact")

    track = sliding_fft(tr)
    stats = track_stats(track)
    print(f"{len(track.t)} windows; MNF {track.mnf[0]:.1f} -> {track.mnf[-1]:.1f} Hz, "
          f"MDF {track.mdf[0]:.1f} -> {track.mdf[-1]:.1f} Hz")
    print(f"slope: MNF {stats['mnf_slope']:+.2f} Hz/min, MDF {stats['mdf_slope']:+.2f} Hz/min "
          "(fatigue = negative)")


if __name__ == "__main__":
    main()
