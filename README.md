# rheocam-py

A small Python tool that watches a USB camera during an experiment, saves periodic
screenshots, and logs the average color (RGB) inside one or more regions you define on
the frame. It's built for long, unattended runs — point a camera at your sample, tell it
where to look and for how long, and let it run.

No software engineering background needed to use this — the sections below walk through
setup and day-to-day use. The only file you'll ever need to touch is [main.py](main.py).

## What it actually does

1. Opens a USB camera and lets it warm up for a few seconds (auto-exposure and
   auto-white-balance need a moment to settle, otherwise your first readings look
   different from the rest of the run).
2. On a fixed interval (e.g. every 6 seconds), grabs a frame and computes the average
   and standard deviation of R, G, B pixel values inside each **AOI** ("area of
   interest") you've defined — a small box at a pixel location you choose.
3. Appends one row per capture to a CSV file, with a timestamp and the RGB stats for
   every AOI.
4. Every so often (and always on the very first and very last frame), saves an
   annotated screenshot: the full frame with pixel-coordinate axes and colored boxes
   drawn around each AOI, so you can go back later and check the AOIs were actually
   sitting where you intended.

Everything lands in a `Results/` folder, organized per sample name.

## Requirements

- A Linux machine with a USB camera. The camera is currently opened at the fixed device
  path `/dev/video0` — see [Known limitations](#known-limitations) below.
- Python 3.13 (pinned in [.python-version](.python-version)).
- [`uv`](https://docs.astral.sh/uv/) for managing the Python environment (recommended),
  or plain `pip` if you prefer.

## Setup

With `uv` (recommended — it reads [pyproject.toml](pyproject.toml) and
[uv.lock](uv.lock) and sets everything up for you):

```bash
uv sync
```

Without `uv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python h5py
```

### Camera permissions (Linux)

If you get a "Could not open camera" error, your user probably doesn't have permission
to access the camera device yet:

```bash
sudo usermod -aG video $USER
```

Then log out and back in (group membership changes don't apply to already-open
sessions).

## Running an experiment

Everything you'd want to change for a given run lives in the `if __name__ ==
"__main__":` block at the bottom of [main.py](main.py). Open it and edit the values
described below, then run:

```bash
uv run main.py
```

(or `python main.py` if you're using a plain virtualenv).

### 1. Name your sample

```python
NAME = "CR-C-153-4"  # the sample name -- CHANGE THIS EVERY EXPERIMENT
```

This becomes the prefix for every output file, so give each run a name you'll recognize
later. If two runs share a name, the script won't overwrite your old data — it appends
`_1`, `_2`, etc. to the new CSV.

### 2. Define your AOIs (areas of interest)

An AOI is just a small rectangle on the camera frame that the script averages pixel
colors over. Define as many as you need:

```python
AOIS = [
    {
        "label": "up_1",       # short name, becomes a column prefix in the CSV
        "box_w": 24,           # box width in pixels
        "box_h": 24,           # box height in pixels
        "center_x": 675,       # box center, in pixel coordinates
        "center_y": 300,
        "color": (0, 0, 255),  # box color drawn on screenshots (B, G, R) — red here
    },
    ...
]
```

To figure out where to put an AOI: run a short test capture first (see step 4), open
one of the saved screenshots, and read the pixel coordinates off the axes drawn around
the edge of the image. Adjust `center_x`/`center_y` and re-run until the boxes line up
with what you want to measure.

If you omit `center_x`/`center_y` for an AOI, it defaults to the center of the frame.

### 3. Set the schedule

```python
SCHEDULE = {"hours": 6, "minutes": 0, "seconds": 0}   # total run time
CAPTURE_INTERVAL = 6                                  # seconds between RGB readings
SCREENSHOT_INTERVAL_MINUTES = 1                       # minutes between saved screenshots
```

A screenshot is always saved on the first and last captured frame regardless of this
setting, so you always have a "before" and "after" image.

### 4. Do a short test run first

Before committing to a multi-hour run, it's worth doing a quick sanity check with a
short `SCHEDULE` (e.g. `{"seconds": 30}`) and `NAME = "test"` to confirm the camera
opens, the AOIs are positioned correctly, and the CSV looks right. Unlike other names,
re-running with `NAME = "test"` overwrites the previous test file each time, so it won't
clutter your results folder.

## Output

Each run creates, under `Results/`:

```
Results/
├── <name>-avg-rgb.csv
└── <name>-screencaptures/
    ├── frame_0001_<timestamp>.png
    ├── frame_0002_<timestamp>.png
    ├── ...
    └── latest.png          # always overwritten with the most recent screenshot
```

**`<name>-avg-rgb.csv`** — one row per capture, columns:

| Column | Meaning |
|---|---|
| `frame` | Frame counter (increments once per capture) |
| `timestamp` | Seconds since 2019-01-01 00:00:00 UTC (see below) |
| `<label> avg r/g/b` | Mean R, G, B value inside that AOI |
| `<label> std r/g/b` | Standard deviation of R, G, B inside that AOI (a rough measure of how noisy/textured that region looks) |

There's one `avg r/g/b`/`std r/g/b` triplet per AOI, in the order you listed them in
`AOIS`.

**Timestamps**: rather than a standard Unix timestamp or a human date string, this
project uses seconds elapsed since a fixed reference point (Jan 1, 2019 UTC) — this is
the convention used by the "PolarSpec" instrument software so its logs can be lined up
directly with this camera log on the same time axis. If you need a normal date/time,
convert with:

```python
from datetime import datetime, timedelta
datetime(2019, 1, 1) + timedelta(seconds=<timestamp value>)
```

**Screenshots** — each PNG shows the full camera frame with a pixel ruler drawn around
the border and a colored box + label over every AOI, so you can visually confirm exactly
what region of the sample was being measured at that point in the run.

## Known limitations

- **Camera device is hardcoded.** The script currently always opens `/dev/video0`
  regardless of the `CAMERA_INDEX` value you set — if you have more than one camera
  attached, or your camera shows up at a different path, you'll need to edit the
  `cv2.VideoCapture(...)` line directly (`main.py`, inside `capture_frames`).
- **Linux only.** It depends on the V4L2 camera backend and `/dev/video0`, so it won't
  run as-is on macOS or Windows.
- **Frame resolution is fixed** at 1280x720 in the code. If your camera doesn't support
  that resolution, OpenCV will silently fall back to its default.

## Project layout

```
main.py          # everything: capture loop, AOI averaging, CSV + screenshot output
pyproject.toml   # dependencies (opencv-python, h5py) and Python version requirement
uv.lock          # locked dependency versions, for reproducible installs via `uv sync`
```

It's intentionally a single script rather than a package — there's no installed CLI,
no entry point beyond `main.py`, and no test suite yet.
