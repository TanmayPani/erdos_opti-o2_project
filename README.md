# Opti O2
[proposal](PROPOSAL.md)

An end-to-end, reproducible pipeline that **detects and classifies
dissolved-oxygen "events"** — *hot moments* (abrupt, tidally-driven) vs *oxic
pulses* (symmetric, precipitation-driven) — in a 5-minute-cadence environmental
record from **Beaver Creek, WA** (2019-06-26 → 2024-09-30, ~554k rows):
dissolved oxygen, salinity, water level, and co-located weather (solar flux, air
temperature, precipitation, barometric pressure, humidity, …).

## Contingency note — labels

The project's supervised target is Opti O2's **expert event labels**, which are
**under NDA and not yet delivered**. So that the work stands on its own, the
pipeline is built and validated entirely on the public ESS-DIVE stand-in, with
the classifier trained on a **physics-grounded weak-supervision label**
(`ws_label`) that encodes the published hot-moment vs oxic-pulse taxonomy as
labeling functions. Everything is driven by a single `LABEL_COL` switch in
[`modeling.py`](modeling.py): point it at the real expert column on delivery and
the whole pipeline re-runs unchanged.

## The pipeline (three [marimo](https://marimo.io) notebooks)

| Notebook | Role |
|---|---|
| [`exploratory.py`](exploratory.py) | EDA (multimodality, multi-scale FFT), **event detection**, event-shape + antecedent feature engineering, k-means clusters, and **§4b weak-supervision labels**. Hands off `derived/{events,event_samples}.parquet`. |
| [`modeling.py`](modeling.py) | **Dual-track classifier** — Track A gradient boosting (XGBoost / CatBoost / Logistic) on engineered features; Track B deep learning (ROCKET / contrastive SSL / InceptionTime / transformer) on raw 5-min sequences — through one leakage-checked CV harness. |
| [`discovery.py`](discovery.py) | **Unsupervised driver discovery** (label-free): does the taxonomy emerge from the geometry, which drivers carry the class signal, and is the hot-moment fraction drifting over the six-year record. |

On the stand-in labels the boosting models reach **macro-F1 ≈ 0.92–0.95,
Cohen's κ ≈ 0.85–0.90**; shuffling the labels collapses performance to chance
(≈ 0.51 macro-F1), confirming the harness is leakage-free.

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
[ESS-DIVE](https://ess-dive.lbl.gov/).

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
├── exploratory.py    # EDA + event detection + features + weak-supervision labels
├── modeling.py       # dual-track event classifier (Track A boosting, Track B deep learning)
├── discovery.py      # unsupervised driver discovery (label-free)
├── derived/          # parquet hand-off written by exploratory.py, read by the other two
├── datasets/         # ESS-DIVE source data
├── pyproject.toml    # project metadata and dependencies
├── uv.lock           # pinned, reproducible dependency versions
└── .python-version   # pinned Python (3.14)
```

The full pipeline (`exploratory.py` → `derived/` → `modeling.py` + `discovery.py`)
re-runs from a clean kernel in **well under a minute** — so it can be re-run
verbatim on the Opti O2 NDA dataset the moment it is delivered.
