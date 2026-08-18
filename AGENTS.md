# AGENTS.md — opti-o2

Erdős Institute **opti-o2** project: classify dissolved-oxygen (DO) excursions at Beaver
Creek WA as **hot moments** (tidal / saline-intrusion driven) vs **oxic pulses** (freshwater
/ precipitation driven), and forecast excursion **onset** as an early-warning signal.

## Environment
- Use `.venv/bin/python` for ALL Python invocations — never `python3` / `python`.
- **Python 3.14** (bleeding edge): some packages have no wheels or force scientific-stack
  downgrades. Probe `.venv/bin/python -c "import <pkg>"` before assuming availability.
- Add deps with `uv add <pkg>`, remove with `uv remove <pkg>`. `uv add` re-locks the whole
  project and can backtrack to an un-buildable old version — if it fails, `uv pip install
  "<pkg>>=<recent>"` installs into the venv without re-locking (but isn't recorded in
  `pyproject.toml`).
- Stack: torch 2.12.1+cu130 (CUDA available), transformers 5.13.0 (native PatchTST /
  PatchTSMixer), xgboost, catboost, scikit-learn 1.9, polars (not pandas), **skorch** (wraps
  the torch heads — logistic mirror + InceptionTime — as sklearn estimators), shap 0.52
  (TreeExplainer / LinearExplainer / GradientExplainer — explainability), TabPFN v3 (runs
  locally on GPU; token lives in the kernel env / `~/.cache/tabpfn` only). **TabPFN's own
  interpretability** = `tabpfn-extensions[interpretability]` + `shapiq` (the imputation
  explainer) — installed via `uv pip install` (**not in the lock**; `modeling.py` guards the
  import so it still loads without it). **No numba** — ROCKET is torch
  (`core.model.RocketTransform`, grouped `conv1d`, GPU).

## Datasets
- **Public (non-NDA)** — `datasets/BeaverCreekWA_EssDive_26Jun2019-30Sep2024/data/*.csv`:
  553,847 rows × 15 cols; **same floodplain-well instrument as the NDA readouts** (well DO /
  salinity / water-level-BGS / temp / precip / weather), byte-identical on the overlap
  (corr 1.0), starting ~2019-06-26. Loader: `preprocessing.load_public_dataset`
  (`skip_rows_after_header=2`, `%-m/%-d/%Y %-H:%M`,
  `null_values=['#REF!','']`, weather renamed to the workbook's `| unit | agg` names).
  Precip complete through 30 Sep 2024.
- **NDA (CONFIDENTIAL)** — `derived/readouts.parquet` (5-min) is now a **combined series**:
  public rows for **2019-06-26 → 2019-09-30** spliced before the delivered NDA xlsx
  (`datasets/BeaverCreek_DO_Events_Labeled/`, 2019-10-01 → 2025-09-30). A per-row
  **`is_public_augmented`** flag marks provenance; it propagates to `proc_features` /
  `proc_curves` / `expert_events` (per unit/event = window begins before the NDA record).
  This recovers **7 pre-NDA expert events (3 hot / 4 pulse → +13 units, pulse 24→32)**. The
  DO channel and every DO-derived `derived/*.parquet` are still NDA (the labels come from the
  confidential workbook): **never commit, share, or send off-machine; no network egress** —
  the public *raw* readouts are shareable, but anything joined to the event labels is not.
- Datetime column is `'Datetime'`. O2 / temperature / salinity columns contain nulls —
  `.drop_nulls()` / `.fill_null()` before any `group_by` aggregation (else
  `MarimoExceptionRaisedError`). Precip is 100% null in WY2025 → prefer **water level**
  (100% complete) as the flood signal.

## Data IO — local filesystem (repo-relative)
All data lives on the local filesystem under the repo root: `datasets/` (raw NDA workbooks +
expert list + public CSV) and `derived/` (the 8 `*.parquet`). Run everything **from the repo root**
so relative paths resolve. `preprocessing.py` is a **plain script** (`.venv/bin/python preprocessing.py`,
not a marimo notebook): its `load_full_dataset`/`load_public_dataset` glob `Path(data_dir).glob(...)`
and read the public CSV; `core.io` still owns `read_one_year`/`read_expert_event_list`. It writes the
8 `derived/*.parquet` via `df.write_parquet(Path("derived")/…)` — **the raw event workbook is parsed
only here**; the four analysis notebooks (`eda`/`modeling`/`discovery`/`forecast`) read **exclusively**
via `pl.read_parquet("derived/…")` (eda's raw list comes from `derived/expert_event_list.parquet`, no
`.xlsx`/`.csv` reads anywhere downstream). **Git holds code only** —
`datasets/`, `derived/`, `checkpoints/` are gitignored (public repo); **copy `datasets/` + `derived/`
onto any new host by hand** (nothing is committed or fetched). Deployment is handled outside the repo.

## Expert-event workbook (badly formatted → cleaned, then parsed)
The delivered `…list of hot & oxic moments….xlsx` is inconsistently filled (single pulses on the
event row; blank moment indices; `#hot/#oxic` counts on the wrong row; sub-rows that repeat the
`<year>-N<type>` label). Cleanup is **decoupled from parsing**: `core.io.reformat_expert_workbook`
reads the messy xlsx and writes a tidy one-row-per-moment **`expert_annotations_reformatted.csv`**
(never mutating the original; each moment gets an explicit `moment_type` hot/oxic + an `is_moment`
flag; xlsxwriter/openpyxl aren't installed so it's CSV via the stdlib `csv` module).
`core.io.read_expert_event_list` then parses **that CSV** (quirk-free) → one row per umbrella with
nested `moments.*` lists (now incl. `moments.type`). `preprocessing.py` runs the reformat then the
parse. The CSV is NDA-derived → gitignored under `datasets/`, regenerated on each run.

**Stray hot-block timings (healed in `read_expert_event_list`).** The workbook's "Time of Peak
DO" (col E) and "Time of DO Inflection Point" (col F) columns sometimes hold something that is
not a timestamp of that moment. The dominant case is a `=end-start` **duration formula** left in
the peak column with the `[h]:mm` elapsed-time number format — every xlsx reader treats that
format as a time and converts the bare day-fraction against the 1900 date system, so 2h20m comes
back as **`1899-12-31 02:20:00`** (13 moments, all oxic/mixed, whose hot block should be empty;
the value equals the umbrella duration exactly). Three more are plain transcription slips (wrong
month or year) in the inflection column. None is repairable from the sheet, so anything outside
its own moment window is **nulled** and the event ids are printed for reporting back to Opti-O2
(`2019-5o3 2019-7o3 2019-9o3 2020-1o1 2020-3o4 2020-5o5 2020-10o3 2020-13m 2021-1o5 2021-10o1
2022-6o5 2023-5o5 2024-2o3`; inflection: `2022-1h 2024-4h 2025-1h`). A stray value is worse than a
missing one: eda's event viewer layers the peak rule on a **shared temporal x scale**, so one
1899 datum stretched the domain across ~120 years and squeezed the event into a sub-pixel sliver
— the chart looked blank. `eda.py`'s `_peaks` now also clamps to the plotted window as a second
guard. No model feature reads these columns (the only consumer is that peak rule), so the
modelling tables are untouched. Separately, `2022-6o5`'s moment row has `end_time` a day before
`start_time` — the stray duration (22 h) corroborates the umbrella's end, so the moment row lost
a day.

## Notebooks & modules (marimo)
- `preprocessing.py` — preprocessing pipeline → `derived/proc_features.parquet`
  (tabular, 24 `FEATURE_COLS` incl. an antecedent-precip lag ladder `precip_24h/72h/168h`
  + labels) and `proc_curves.parquet` (128×5 raw curves). Also **catalogues orphans**
  (`core.features.catalogue_orphans`) — auto excursions with no expert umbrella, given
  proper `YYYY-N{h|o|x}` ids (h hot-like / o pulse-like / x ignored noise; N after that
  year's expert max) + `is_orphan` + `driver_class` → `derived/orphan_events.parquet` +
  `orphan_event_samples.parquet` (event-viewer only; kept out of the modelling / expert
  tables so `modeling.py`/`discovery.py` are unaffected).
- `modeling.py` — hot/pulse classifier. **Track A** (XGBoost / CatBoost / Logistic /
  TabPFN v3 on `FEATURE_COLS`) + **Track B** (torch-ROCKET → Logistic/XGBoost/TabPFN,
  InceptionTime — consolidated into a real-labels table + a shuffled-label control table).
  Grouped CV (StratifiedGroupKFold on `group_id`); ~0.80 macro-F1 ceiling. **Metric reporting
  is grouped bar-charts** (models on x, metric = tabs, ±std whiskers, dashed chance line) —
  `metric_bar_tabs` (one table), `metric_bars_grouped` (evaluation *protocols* — grouped CV /
  LOGO / temporal / label- & time-shuffled — bunched per model), `metric_bars_routes`
  (excursion vs moment bunched); these **replaced** the old `display_table` tables.
  **SHAP explainability, one explainer per model family, on *both* routes:** XGBoost & the
  ROCKET head via `shap.TreeExplainer`; the torch **logistic** head via `shap.LinearExplainer`
  (exact/linear); **TabPFN** via *its own* imputation explainer
  (`tabpfn_extensions.interpretability.shapiq.get_tabpfn_imputation_explainer` → bridged to a
  `shap.Explanation`); **InceptionTime** via `shap.GradientExplainer` (GradientSHAP → channel
  reliance + a per-timestep temporal profile). Beeswarms show *direction* (salinity → hot,
  water-level rise → pulse); Track B aggregates SHAP back to ROCKET channel/dilation (water
  level ≫ precip — that precip *curve channel* is all-null in WY2024-25, missing not proven
  null; the tabular `precip_24h` is a real mid-tier driver ~6/24, a *secondary* freshwater
  signal). A **TabPFN-on-ROCKET** channel explainer was tried and **dropped**: TabPFN can't
  ingest the 20k tagged features, so PCA back-projection / feature-selection wash the channel
  structure out to ~uniform (honest negative — the XGBoost ROCKET panel stays the channel
  readout). One shared `make_xgb()` factory across all XGB heads.
- `forecast.py` — hourly DO onset early-warning from hydrology + deterministic tidal harmonics.
  Two arcs: (A) DO-**magnitude** forecast — DLinear (from scratch) beats/ties PatchTST/PatchTSMixer,
  weather adds no skill (honest negative); (B) excursion-**onset** early-warning (ROC-AUC ≈ 0.80–0.89,
  tidal-harmonic flood-height covariate helps most at 6–12 h lead). Plus a **Challenge-2 section**
  (deck's final slide — sea-level rise + antecedent timescales): a harmonic+secular-trend **timescale
  variance decomposition** of flood height & salinity (**seasonal ≫ synoptic/event ≫ tidal ≫ multi-year**;
  tides < 1 % — the well is tidally muted, matching eda's periodogram) and a **decorrelated per-water-year
  SLR trend test** (null: no resolvable secular creep in 6 yr; the coastal transition is in *event
  composition*, not a rising baseline). `slr_bandfit` / `slr_bands_df` / `slr_trend_df`.
- `discovery.py` — label-free taxonomy on preprocessing.py's augmented `expert_events.parquet`
  (NDA-derived): unsupervised cluster recovery scored against the real `expert_label` (ARI),
  cluster-count diagnostics, PCA + real-var scatter, mutual-info driver ranking, the
  weak-supervision labeling-function method (physics rules vs experts), cluster profiles, and
  hot-fraction drift. (Absorbed the clustering/weak-supervision viz from the retired
  `exploratory.py`.)
- `eda.py` — EDA + a **multi-scale record-context** section (full-record daily overview + air-temp
  FFT periodogram), an **interactive event viewer** (Ghosh multi-axis chart + detector splice; a
  `shade_mode` toggle flips the shaded overlay between our **auto-detected events** and the
  **expert moments** — hot/oxic-coloured; the picker interleaves expert events **and** catalogued
  orphans by occurrence, flagging `is_orphan` rows and reading `derived/orphan_events.parquet`; its
  controls + viewer cells are **named** `ui_elements` / `event_detection` for cross-notebook reuse
  via `Cell.run` — see `slides.py`), the **handcrafted event-shape
  feature methodology** (per-feature tabs via `utils.feature_methodology_tabs` on `proc`/`proc_curves`:
  formulas, distributions, driver associations, example curves, hysteresis loops — absorbed from
  `exploratory.py`), a **Cusum driver–response** cell (Regier-lab method; water-level rise↑→DO↑,
  air-temp↑→DO↓, matching Regier et al. 2023 Fig. 5) and an **event-seasonality** cell
  (tests that paper's H1: oxic pulses skew to the wet cool season).
- `slides.py` — the **published findings page** (default vertical layout — *not* a slide
  deck; no `layout_file`). A thin wrapper that **reuses eda's named cells**
  via marimo's `Cell.run`: it owns `event_picker` / `shade_mode` (so the viewer stays
  interactive) and renders `event_detection.run(...)` — no data pipeline is duplicated (`.run()`
  auto-computes the refs from eda). Title card = Erdős / Opti O2 logos over a glass pane,
  each embedded as a runtime data-URI from **`assets/`** (`opti_o2_logo.jpg`, `erdos_logo.png`;
  `beaver_creek_site.jpg` also available). **NDA:** it renders DO-derived data →
  keep local, never `marimo export`/publish. `marimo export pdf slides.py --as=slides` → reveal PDF.
- **Page theme (glass over the site photo)** — three pieces that must stay in step:
  (1) **`assets/theme.css`**, wired by `marimo.App(css_file="assets/theme.css")` and inlined into
  the export's `<head>`: a fixed `body::before` layer painting `--o2-photo` under a dark scrim,
  `.marimo-cell` / `.output-area` forced transparent, prose text-shadow, and a faint glass panel
  behind `.vega-embed`. **Two non-obvious rules earn their keep:** `#App` carries tailwind's
  `bg-background` (an opaque `#181c1a` panel over the whole viewport) and must be forced
  transparent or the photo never shows; and every content selector is *unscoped* — marimo renders
  cell output inside `<marimo-mime-renderer>` / `<marimo-tabs>` shadow roots and adopts this
  stylesheet into them, but an `#App ` ancestor can never match from inside a shadow tree, so a
  scoped `#App .vega-embed` silently does nothing. **Two more, learned the hard way:**
  translucent layers compound, so exactly one glass layer per chart (the innermost) — a
  background on `[role="tabpanel"]` stacked three deep in the SHAP cell, which nests route tabs
  inside model tabs, and turned it into a near-solid slab; and `modeling.py`'s SHAP panels
  return `mo.hstack([...])`, i.e. they are **rendered to HTML before `@mo.persistent_cache`
  stores them**, so their cached spec predates the Altair theme and vega-embed's dark theme
  paints an opaque `#333` canvas that no Python-side theme can reach (9 of 11 charts were
  transparent, those 2 were not) — hence `.vega-embed svg.marks { background-color: transparent
  !important }`, which outranks the inline style and covers any cached-HTML panel. (2) the **`page_theme`** cell in `slides.py`, which sets `--o2-photo` to
  `assets/title_slide_bg.jpg` as a base64 data-URI — a standalone export can't fetch `assets/` at
  view time, so the CSS file deliberately ships with `--o2-photo: none` and is filled in at run
  time. (3) the **`optio2_glass` Altair theme** registered in `slides.py`'s setup cell (transparent
  canvas, light axis/legend/title ink); it is process-global and every chart is built by a
  `Cell.run()` *after* setup, so all 62 specs pick it up. Marks with an explicit colour still win.
  `[tool.marimo.display] theme = "dark"` in `pyproject.toml` pins the app theme (and hence what
  marimo hands vega-embed) — at the default `"system"` the page would follow each *visitor's* OS
  setting and light-mode readers would get white chart canvases over the photo, plus modeling.py's
  white-on-white bar labels. `pyproject.toml` is therefore in the export hook's `SOURCES`.
- `utils.py` — plotting glue (`hist_chart`/`binned_hist`), `build_forecast_frame`, readout
  globs/constants. The heavy lifting lives in the **`core`** workspace package (`core/src/core/`):
  `core.io` (readout loaders + the expert-event pipeline, `_class3` suffix→class), `core.features`
  (`FEATURE_COLS`, event detection, curve/feature builders), `core.model` (`RocketTransform`,
  `InceptionTimeLite`), `core.checks` (granularity reconciliation). Plain modules — safe to
  edit directly (not notebooks).

## Marimo workflow
- Validate: `.venv/bin/marimo check <file>`. Export:
  `timeout 300 .venv/bin/marimo export html <file> -o <out.html>`.
- **Exit 0 ≠ success** — grep the output for `MarimoExceptionRaisedError` / `Traceback`;
  cell failures surface in stderr even when the HTML is written.
- The autoformatter rewrites `.py` on save — re-read a notebook immediately before any `Edit`.
- **Live sessions:** drive the kernel via `marimo._code_mode` (cm) per the marimo-pair skill;
  do NOT edit the `.py` directly while a kernel is live, and avoid `uv add` / `uv remove`
  mid-session (can disrupt the live kernel — prefer idle, then restart). **Pairing needs the
  kernel started `--no-token`** (`marimo edit <nb> --no-token`); a plain `marimo edit` requires a
  token and the tooling can't connect. If two notebooks are open, confirm which port hosts which
  (`discover-servers.sh` + cell inspection) before editing — ports can swap between sessions.
- **Layouts:** no notebook declares a `layout_file` — all render in the default vertical
  layout, and `layouts/` is deleted and gitignored. marimo recreates it if you save a
  non-default view in the editor (View → Slides/Grid); delete it again or leave it ignored.
  A cell must be **named** (not `_`) to be importable / reusable via `Cell.run`.

## Deck publishing (commit builds, push deploys)
- Managed by **pre-commit** (`.pre-commit-config.yaml`, `pre-commit>=4.6.2` in the venv).
  The `export-slides` hook runs `scripts/export-slides.sh`, which exports `slides.py` →
  `site/index.html` and stages it. Its `files:` regex limits it to commits that stage
  `slides.py` / `eda.py` / `utils.py` / `core/` / `assets/` / `scripts/export-slides.sh`.
- The published page is a **findings page, not a slide deck** — `slides.py` carries no
  `layout_file`. The publish export therefore uses **`--no-include-code`**; marimo's default
  is `--include-code`, which is right for the *validation* command in "Marimo workflow"
  below but wrong for a reader-facing page. Don't copy one command for the other job.
- `scripts/export-slides.sh` is itself in `SOURCES`, so editing the build recipe invalidates
  the manifest and forces a rebuild on the next commit.
- A full CLI export is **~20 s** (measured; `EXPORT_TIMEOUT=300` is the failure ceiling,
  not an expectation) — the fresh kernel hits `persistent_cache` on disk.
- **`site/.build-manifest`** holds the *index* blob hashes of those 21 sources. The script
  exits in ~20 ms when they already match, so pre-commit's "files were modified by this
  hook" re-run doesn't rebuild.
- The export fails the commit on timeout, on `MarimoExceptionRaisedError` / `Traceback` in
  the log, on marimo's 8 MB output-cap placeholder, or on an empty file — **exit 0 is not
  trusted**. Bypass with `git commit --no-verify`.
- Hooks can't self-install (git never runs anything on clone, and `core.hooksPath` /
  `.git/hooks/` aren't cloned). **On a fresh clone run `bash scripts/setup.sh` once** —
  `uv sync` + `pre-commit install`.
- `.github/workflows/pages.yml` deploys on push to `main` when `site/**` changes (plus
  `workflow_dispatch`). So: commit builds the deck, push ships it.

## Domain (why the features are what they are)
DO excursions are **flooding events** (near-anoxic floodplain well; water table rises toward
the surface). Split by salinity: **rise = saline/tidal intrusion → hot** (~75%, tidally
predictable); **fall = freshwater rise → pulse** (~25%), sub-split by antecedent precip.
Decisive classifier features: `sal_step` (salinity change), water level, `rise_rate`.
Solar / wind / air-temp add no skill (no diel DO signal in an anoxic well), but **antecedent
precip is a real secondary driver** (`precip_24h` ~6th of 24 by SHAP, reads toward pulse) —
longer 72/168 h lags add little, so Regier et al. 2023's "24 h lag under-credits rain" caveat
only partly holds at excursion granularity.

## Safety / NDA
The Opti O2 NDA data is confidential — no commits, no sharing, no egress. Secrets (TabPFN token) live only in the kernel env / cache, never in committed files.
