# Opti O2

Exploratory analysis of dissolved-oxygen dynamics in a flood-plain monitoring
well, using a 5-minute-cadence environmental record from **Beaver Creek, WA**
(2019-06-26 → 2024-09-30, ~554k rows): dissolved oxygen, salinity, water level,
and co-located weather (solar flux, air temperature, precipitation, barometric
pressure, humidity, …).

The analysis lives in a [marimo](https://marimo.io) reactive notebook,
[`exploratory.py`](exploratory.py), which builds graphs showing the dataset's
**multimodality** (zero-inflated DO, multi-regime salinity) and its **multiple
time-scales** (diurnal, synoptic, and seasonal cycles, plus an FFT periodogram).

## Prerequisites

- [**uv**](https://docs.astral.sh/uv/) — the package/environment manager this
  project uses. Install it with:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  (or see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)
  for Homebrew/Windows options).

- **Python 3.14** — you do *not* need to install this yourself; `uv` reads the
  pinned version from `.python-version` and fetches it automatically.

## Getting started

1. **Clone the repository:**

   ```bash
   git clone <repository-url> opti-o2
   cd opti-o2
   ```

2. **Install dependencies** into a local virtual environment (`.venv/`):

   ```bash
   uv sync
   ```

   This resolves the exact versions in `uv.lock` (marimo, polars, altair, numpy).

3. **Launch the notebook** in edit mode:

   ```bash
   uv run marimo edit exploratory.py
   ```

   marimo opens in your browser. Cells are reactive — they re-run automatically
   when their inputs change. To view it read-only instead, use
   `uv run marimo run exploratory.py`.

## The dataset

The notebook reads:

```
datasets/BeaverCreekWA_EssDive_26Jun2019-30Sep2024/data/
    2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv
```

a [BagIt](https://en.wikipedia.org/wiki/BagIt)-packaged dataset published on
[ESS-DIVE](https://ess-dive.lbl.gov/). The `datasets/` folder also contains the
original `.zip` archives.

If `datasets/` is missing after cloning (large data files may be distributed
separately), recreate the path above by unzipping the corresponding archive, or
point the `data_dir_path` in the notebook's `setup` cell at your local copy. The
loader already handles the file's quirks — spreadsheet `#REF!` artefacts, blank
rows, a trailing spacer column, and physically impossible sentinel values.

## What's in the notebook

- **Section 1 — Multimodality:** histograms of the cleaned variables. Dissolved
  oxygen / O₂% are ~92% exact zeros (anoxic well) with a separated oxic mode (a
  zero-inflated distribution, shown on a symlog count axis); salinity shows
  several regime peaks.
- **Section 2 — Multiple time-scales:** daily means over the full record
  (long-term + seasonal), the average diurnal cycle (mean ± IQR), the seasonal
  cycle by day of year, and an FFT periodogram of air temperature with annotated
  diurnal / semi-diurnal / weekly / monthly / annual peaks.

## Project layout

```
opti-o2/
├── exploratory.py    # the marimo analysis notebook
├── datasets/         # ESS-DIVE source data
├── pyproject.toml    # project metadata and dependencies
├── uv.lock           # pinned, reproducible dependency versions
└── .python-version   # pinned Python (3.14)
```
