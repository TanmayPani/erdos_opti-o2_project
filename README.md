# Opti O2

An end-to-end, reproducible pipeline that **detects and classifies dissolved-oxygen
excursions** at **Beaver Creek, WA** — *hot moments* (abrupt, tidally / saline-intrusion
driven) vs *oxic pulses* (symmetric, freshwater / precipitation driven) — from a
5-minute-cadence environmental record: dissolved oxygen, salinity, floodplain water level,
temperature, and co-located weather (solar flux, air temperature, precipitation, barometric
pressure, humidity, …). A third *mixed* class is held out. The project also forecasts
excursion **onset** as an early-warning signal.

## Data & labels (NDA)

The supervised target is Opti O2's **expert event labels**, delivered as a confidential
workbook. **These labels and every dissolved-oxygen-derived artifact are under NDA** — they
are never committed, shared, or sent off-machine. **Git holds code only**; `datasets/` and
`derived/` are gitignored and copied onto each host by hand.

For the parts that can stand on their own, the pipeline also reads the **public ESS-DIVE**
release of the same instrument (2019-06-26 → 2024-09-30), which *is* shareable. The public
rows are spliced ahead of the delivered series so the record is continuous, with a per-row
`is_public_augmented` provenance flag.

## The pipeline

Data preparation and the CV sweep are **plain scripts**; the analysis surfaces are
[marimo](https://marimo.io) notebooks. All heavy lifting lives in the **`core`** workspace
package (`core/src/core/`: `io`, `features`, `estimator`, `nn`).

| Stage | Role |
|---|---|
| [`preprocessing.py`](preprocessing.py) | **Plain script.** Parses the raw workbooks + public CSV + expert-event workbook → the `derived/*.parquet` hand-off: `readouts`, `expert_event_list` / `expert_moment_list`, `expert_samples`, and `processed_{auto,expert,moments}_{features,curves}` (24 engineered features + 5-channel raw curves per event). |
| [`training.py`](training.py) | **Plain script.** The hot/pulse CV sweep through one leakage-checked harness — **Track A** gradient boosting / logistic / TabPFN on engineered features, **Track B** deep learning (ROCKET → boosting/logistic/TabPFN, InceptionTime) on raw 5-min sequences. `--mode excursion` (auto-detected units) or `--mode moment` (expert windows). Writes `derived/model_results*.parquet`. |
| [`eda.py`](eda.py) | EDA (multimodality, multi-scale FFT), an interactive **event viewer**, the engineered-feature methodology, Cusum driver–response, and event seasonality. |
| [`modeling.py`](modeling.py) | Reads the CV artifact and renders the model comparison + **SHAP explainability** (one explainer per model family, on both the excursion and moment routes). |
| [`discovery.py`](discovery.py) | **Label-free driver discovery**: does the taxonomy emerge from the geometry, which drivers carry the class signal, and a physics-rule **weak-supervision** baseline scored against the experts (confusion matrix + Cohen's κ). |
| [`forecast.py`](forecast.py) | Hourly DO-**magnitude** forecast (DLinear vs PatchTST/PatchTSMixer) and excursion-**onset** early-warning from hydrology + deterministic tidal harmonics. |
| [`slides.py`](slides.py) | Presentation deck (marimo reveal.js slides) that **reuses** eda's cells. Renders DO-derived data → **keep local, never publish/export off-machine**. |

On grouped cross-validation the tabular models reach **macro-F1 ≈ 0.92, Cohen's κ ≈ 0.85**;
shuffling the labels collapses performance to chance (≈ 0.44 macro-F1), confirming the
harness is leakage-free.

## Prerequisites

- [**uv**](https://docs.astral.sh/uv/) — the package/environment manager this project uses:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  (or see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/) for
  Homebrew/Windows options).

- **Python 3.14** — you do *not* need to install this yourself; `uv` reads the pinned version
  from `.python-version` and fetches it automatically.

## Getting started

1. **Clone and install** dependencies into a local `.venv/` (also installs the `core`
   workspace package):

   ```bash
   git clone <repository-url> opti-o2
   cd opti-o2
   uv sync
   ```

2. **Provide the data.** `datasets/` and `derived/` are gitignored, so copy them onto the
   host by hand (see *Data & labels* above). The public ESS-DIVE CSV alone is enough to run
   the public-standin path.

3. **Build the `derived/` hand-off** (run from the repo root so relative paths resolve):

   ```bash
   uv run python preprocessing.py
   ```

4. **Run the CV sweep** (optional — needed for `modeling.py`'s comparison panels):

   ```bash
   uv run python training.py --mode excursion   # or --mode moment
   ```

5. **Explore the notebooks** — reactive cells re-run automatically when their inputs change:

   ```bash
   uv run marimo edit eda.py        # or modeling.py / discovery.py / forecast.py
   ```

   Use `uv run marimo run <notebook>.py` for a read-only view.

## The dataset

The public loader reads the [ESS-DIVE](https://ess-dive.lbl.gov/) release:

```
datasets/BeaverCreekWA_EssDive_26Jun2019-30Sep2024/data/
    2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv
```

The confidential loaders (`core.io`) additionally read the delivered NDA workbooks and the
expert-event workbook when present. The loaders already handle the sources' quirks —
spreadsheet `#REF!` artefacts, blank rows, a trailing spacer column, banner header rows,
and the expert workbook's inconsistent moment formatting.

## Project layout

```
opti-o2/
├── preprocessing.py   # plain script: raw + public + expert workbook → derived/*.parquet
├── training.py        # plain script: hot/pulse CV sweep (Track A + Track B) → model_results*
├── eda.py             # EDA + interactive event viewer + feature methodology
├── modeling.py        # model comparison + SHAP explainability
├── discovery.py       # label-free driver discovery + weak-supervision baseline
├── forecast.py        # DO-magnitude forecast + excursion-onset early-warning
├── slides.py          # reveal.js deck (reuses eda cells; NDA — keep local)
├── core/              # workspace package: io, features, estimator, nn (ROCKET / InceptionTime …)
├── derived/           # parquet hand-off written by preprocessing.py (gitignored, NDA)
├── datasets/          # source data — public ESS-DIVE CSV + NDA workbooks (gitignored)
├── pyproject.toml     # project metadata and dependencies
├── uv.lock            # pinned, reproducible dependency versions
└── .python-version    # pinned Python (3.14)
```
