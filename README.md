# Junction2026

Live Rokoko glove capture and gesture-learning starter for the Junction 2026
hackathon. The project can visualize a Rokoko Device SDK stream, record
timestamped/labeled demonstrations, import them into Tiger Data (TimescaleDB),
and train a small gesture classifier locally or on Verda GPU compute.

## Requirements

- Windows 10 version 1903 or later (x64)
- Python 3.12 or later
- Rokoko Device SDK installed with an active SDK license
- A Rokoko glove connected to the computer or reachable over Wi-Fi

The visualizer uses the SDK's `examples/python/rgmp_stream.py` protocol helper.

## Start the live visualizer and capture data

1. Install the [Rokoko Device SDK](https://sdk.rokoko.com/guides/install/windows/)
   and restart the terminal.
2. Verify `ROKOKO_SDK_HOME` points to the SDK folder containing
   `examples/python/rgmp_stream.py`. If needed, set it for this PowerShell
   session (adjust the folder if you chose a custom install location):

   ```powershell
   $env:ROKOKO_SDK_HOME = "$env:LOCALAPPDATA\Rokoko Device SDK"
   ```

3. In one terminal start the SDK driver:

   ```powershell
   rokoko-sdk
   ```

   For a Smartgloves II connected over USB, use `rokoko-sdk --auto-stream-usb`.
4. In a second terminal, from the repository root, start the capture UI:

   ```powershell
   py .\glove_viz\server.py
   ```

   Open <http://127.0.0.1:8000>. When the device is streaming, enter a short
   action label (for example `open hand`) and select **Start recording**.
   Perform that action, then select **Stop recording**. Capture each action
   multiple times as separate labeled sessions; the training script requires
   at least two sessions per label so that evaluation can hold out entire
   recordings.

Files are written as JSON Lines under `data/captures/` by default (or the
directory passed with `--data-dir`). Each file starts with a session record and
then stores timestamped RGMP groups, their streams, device ID, and action
label. Treat raw motion captures as potentially sensitive; the data folder is
git-ignored. Orientation values in the live view are not calibrated anatomical
finger-bend measurements.

## Import captures to Tiger Data

Create a Tiger Data TimescaleDB service and set its PostgreSQL connection URL
as `DATABASE_URL` in the shell. Don’t commit the URL or credentials.

```powershell
py -m pip install -r requirements-data.txt
$env:DATABASE_URL = "postgresql://..."
py .\tools\import_tigerdata.py
```

The importer creates `glove_motion_frames`, a hypertable indexed by session,
group, and device timestamp. It accepts a capture directory or a single JSONL
file. Re-importing a capture is safe: existing session/group/capture-time keys
are skipped.

## Train on local or Tiger Data captures

Install `requirements-verda.txt` in the training environment. It includes
PyTorch and the optional Tiger Data PostgreSQL driver. Set `DATABASE_URL` if
training from Tiger Data; otherwise train directly from local capture files.

```powershell
py -m pip install -r requirements-verda.txt
py .\training\train_gesture.py --captures .\data\captures --group default
```

To load the imported frames instead:

```powershell
py .\training\train_gesture.py --tigerdata --group default
```

The starter model classifies labeled poses from each group's sensor
`TRANSFORM@hub` values. It splits by recording session (not by adjacent frame)
to avoid placing frames from the same take in both train and test sets. The
per-label split requires at least two distinct recording sessions for each
action. The saved model bundle defaults to `artifacts/gesture_model.pt`. This
is a baseline gesture classifier, not yet a robot-control policy: robot
retargeting, simulation, and physical robot evaluation remain follow-up work.

For Verda, run the training command in a Verda GPU instance after making the
captures available to that instance (for example by importing/reading them
from Tiger Data). The script automatically uses CUDA when available and prints
held-out accuracy.

## Tests

```powershell
py -m unittest discover -s .\glove_viz -p "test_*.py" -v
py -m unittest discover -s .\tools -p "test_*.py" -v
py -m unittest discover -s .\training -p "test_*.py" -v
```
