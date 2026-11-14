# EMG-Fatigue-Sensor

Measures muscle fatigue from surface EMG, rep by rep. A MyoWare 2.0 sensor on an ESP32 records the
muscle during a set; the analysis splits the set into reps and reports how each rep's EMG
spectrum shifts as the muscle tires.

## How fatigue shows up in EMG

As a muscle fatigues, its electrical signals travel more slowly along the fibres, so the EMG
spectrum shifts to lower frequencies. Two numbers track that shift:

- **MNF** (mean frequency): the power-weighted average frequency of the spectrum.
- **MDF** (median frequency): the frequency that splits the spectrum's power in half.

Amplitude (**RMS**) is reported alongside them. Falling MNF with rising RMS is fatigue (the same
load needs more motor units); both falling means the lifter eased off.

## Per-rep pipeline (`emg/reps.py`)

```
raw EMG
  -> band-pass 20-450 Hz           removes motion artifact (<20 Hz) and high-frequency noise
  -> envelope                      rectify, low-pass at 5 Hz: activity level over time
  -> detect reps                   envelope bursts above a threshold 15 % of the way from the
                                   rest floor (10th percentile) to the peak (99th percentile)
  -> concentric phase              rep onset to the last point above 60 % of the rep's peak
  -> trim 0.1 s each end           the envelope blurs phase boundaries
  -> Burg AR(14) spectrum          one spectrum per lift
  -> MNF, MDF, RMS per rep
```

Only the **concentric (lifting) phase** is analysed. The lowering phase has a different spectrum,
and the mix of the two would change with tempo. The phase split uses amplitude alone and assumes
each rep starts with the lift, which holds for curls but not squats or bench press.

### Why Burg's method

A lift lasts about 0.5-1 s. An FFT of that little data gives a noisy spectrum, because every
frequency bin is estimated independently. Burg's method fits an autoregressive model (each sample
predicted from the previous 14) by minimising forward and backward prediction error together,
and the spectrum follows from the 14 coefficients. It works only with the samples it has, always
gives a stable model, and gives a smooth spectrum from short records. The implementation is
`burg` in `emg/tvar.py`. A direct comparison against an FFT on real rep recordings has not been
run yet.

Further reading: [MATLAB `arburg`](https://www.mathworks.com/help/signal/ref/arburg.html),
[Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model);
J. P. Burg, *Maximum Entropy Spectral Analysis*, PhD thesis, Stanford, 1975.

### Output

`run_reps(trial)` returns a `RepTable` with one row per rep:

| Field | Meaning |
|---|---|
| `start`, `conc_stop`, `stop` | rep onset, end of the lift, end of the rep (s, recording clock) |
| `conc_duration`, `duration` | lift time and full rep time (s); lifts slow down with fatigue |
| `mnf`, `mdf` | mean and median frequency of the lift (Hz); NaN if the lift is too short to fit |
| `rms` | amplitude of the lift, in the recording's units (not normalised) |
| `relative(which="mnf")` | each rep as a fraction of the first three reps (1.0 = fresh) |
| `slope(which="mnf")` | trend across the set, per rep (e.g. Hz/rep) |

```python
from emg.io import load_myoware
from emg.reps import run_reps

table = run_reps(load_myoware("recordings/biceps_01.csv", "R_BICEPS_BRACHII"))
print(table.mnf, table.relative(), table.slope())
```

The defaults (AR order 14, the 15 % and 60 % thresholds, 0.1 s trim) were chosen on synthetic
data and still need checking against real recordings.

## Continuous tracking (sustained holds)

For holds with no rest, the earlier pipeline tracks MNF/MDF continuously and compares two
estimators in `tvar.ipynb`:

- **Sliding FFT** (`emg/baseline.py`): 4 s windows every 2 s, the dataset's reference method.
- **TVAR** (`emg/tvar.py`): an AR model whose coefficients drift slowly, tracked by a Kalman
  filter and smoothed forward and backward (RTS), or tracked by RLS.

Both are tuned to the same response time before comparing. Results on one subject of the public
dataset are mixed: TVAR gives a clearer trend on the biceps and medial deltoids, the FFT on the
anterior and posterior deltoids. This path is not used for reps.

## Hardware and recording

- `firmware/emg_logger/emg_logger.ino`: ESP32 sketch, MyoWare RAW output on ADC1, 2 kHz sampling,
  `t_us,raw` lines over 921600-baud serial. **The sampling loop is still a TODO.**
- `tools/capture.py`: logs the serial stream to CSV.
- `tools/check_recording.py`: checks sample rate, gaps and clipping, and prints the MNF/MDF trend.

Power the MyoWare from 3.3 V, use an ADC1 pin, and keep the laptop on battery while electrodes
are on a person.

## Web app (planned, not in this repo yet)

```
Browser ──► Next.js API (Vercel) ──► Postgres: recordings, jobs
   │                                       │ trigger: pg_notify
   └── CSV upload ──► Supabase Storage     ▼
                            └────────► Python worker (Fly.io): run_reps -> per-rep results
   ◄── polls every 2 s ── Next.js API ◄── Postgres: per-rep rows
```

1. **Browser**: the user signs in with Supabase Auth, picks a muscle and drops in a recording CSV.
2. **Next.js API**: route handlers on Vercel save the recording row and return a signed upload URL.
3. **Storage**: the browser uploads the CSV straight to Supabase Storage, skipping Vercel's
   4.5 MB request limit.
4. **Job queue**: `/process` adds a jobs row in Postgres; a trigger fires `pg_notify` to wake the
   worker.
5. **Worker**: a Python container on Fly.io claims the job, runs the per-rep analysis and saves
   one row per rep. It runs outside Vercel because the scientific Python stack (numpy, scipy,
   numba with llvmlite: about 360 MB installed) exceeds Vercel's function size limit.
6. **Results**: the UI polls the API every 2 s, then charts MNF and MDF per rep. Row-level
   security keeps each user's data private.

## Layout

```
emg/
  reps.py        per-rep pipeline: envelope, rep detection, concentric phase, Burg spectra
  tvar.py        Burg fit, AR spectra, order selection, Kalman/RLS TVAR
  baseline.py    sliding-window FFT
  preprocess.py  band-pass, standardize
  metrics.py     MNF, MDF, SpectralTrack
  evaluate.py    rise time, trend slope/SNR/roughness
  io.py          public dataset and MyoWare CSV loaders
  synthetic.py   synthetic AR signals, step trials, rep trials with known answers
tests/           pytest suite
tools/           capture.py, check_recording.py
firmware/        ESP32 sketch
tvar.ipynb       TVAR vs sliding-FFT comparison on the public dataset
```

## Setup

```
pip install -r requirements.txt
python -m pytest
```

Tests that need the public dataset skip unless `sEMG_data/subject_<N>/` is present
(`emg.io.extract_subject` unzips it).

## Open items

- ESP32 sampling loop (firmware TODO); no MyoWare recordings yet.
- Tune the per-rep defaults on real curl recordings.
- Compare Burg against a windowed FFT per rep.
- No 50/60 Hz mains notch filter.
- No fatigue classifier yet; the output is per-rep MNF/MDF/RMS and their trends.
