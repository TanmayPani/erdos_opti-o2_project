import numpy as np
import polars as pl
import altair as alt
import marimo as mo

from core.features import DO_COL, WL_COL, SAL_COL, TEMP_COL, PRECIP_COL, FEATURE_COLS

EVENTS_XLSX = "2019-2026 list of hot & oxic moments rev. 04-29-26.xlsx"


# ---------------------------------------------------------------------------
# Forecasting frame: regular-grid resample of the 5-min readouts for forecast.py.
# Target = DO; covariates = the co-located in-situ weather block + hydrology sensors
# already carried in readouts. Kept RAW / un-imputed (the notebook scales and
# imputes per training fold, mirroring build_curves' contract).
# ---------------------------------------------------------------------------

# verbatim readout column -> short, tidy handle used throughout forecast.py
FORECAST_WEATHER = {
    "SlrFD_kW_Avg | kW/m^2 | Avg": "solar_fd",
    "SlrTF_MJ_Tot | MJ/m^2 | Tot": "solar_tot",
    "AirT_C_Avg | Deg C | Avg": "air_temp",
    "BP_hPa | hPa | Smp": "bp",
    "WS_ms_S_WVT | meters/second | WVc": "wind",
    "VP_hPa_Avg | hPa | Avg": "vp",
    "RH | % | Smp": "rh",
}
FORECAST_HYDRO = {SAL_COL: "sal", WL_COL: "wl", TEMP_COL: "sensor_temp"}
FORECAST_DO = "do"  # <- DO_COL (or `target`), bin mean
FORECAST_PRECIP = "precip"  # <- PRECIP_COL, bin SUM (flux -> accumulation)

# feature-group lists the notebook selects on (precip grouped with weather)
WEATHER_FEATURES = list(FORECAST_WEATHER.values()) + [FORECAST_PRECIP]
HYDRO_FEATURES = list(FORECAST_HYDRO.values())
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

WS_HOT, WS_PULSE, WS_ABSTAIN = 1, -1, 0


# ---------------------------------------------------------------------------
# Discovery routes — the two engineered-feature tables discovery.py runs on.
# `moments` first: it is the default route and the one the deck presents.
#   moments -> per-moment table, each moment typed hot / oxic(pulse) individually
#   events  -> augmented per-event table (expert_label hot / pulse / mixed)
# Identical schemas; the moment route's hysteresis pair is degenerate (single pulse).
# Both are DO-derived => NDA. Shared by discovery.py so the route-dependent cells can
# load *both* routes and switch between them client-side in a static HTML export.
# ---------------------------------------------------------------------------
DISCOVERY_ROUTES = {
    "moments": ("derived/processed_moments_features.parquet", "label"),
    "events": ("derived/processed_expert_features.parquet", "expert_label"),
}


def read_discovery_route(route):
    """Load one discovery route and attach the canonical display class (pulse -> oxic)."""
    _path, _label = DISCOVERY_ROUTES[route]
    return pl.read_parquet(_path).with_columns(
        pl.col(_label).replace({"pulse": "oxic"}).alias("class")
    )


def _ws_band(_df, col, lo_p, hi_p, lo_vote, hi_vote, use_abs=False):
    # Vote lo_vote below the lo_p quantile, hi_vote above the hi_p quantile,
    # abstain (and on nulls) in between — a confident but sparse voter.
    x = _df[col].to_numpy().astype(float)
    if use_abs:
        x = np.abs(x)
    lo, hi = np.nanquantile(x, lo_p), np.nanquantile(x, hi_p)
    v = np.full(_df.height, WS_ABSTAIN, dtype=int)
    v[x <= lo] = lo_vote
    v[x >= hi] = hi_vote
    v[np.isnan(x)] = WS_ABSTAIN
    return v


def weak_supervision_labels(events_df):
    """Physics-grounded weak-supervision labels (hot moment vs oxic pulse).

    Encodes the published DO-event taxonomy as labeling functions (LFs) that
    vote WS_HOT / WS_PULSE / WS_ABSTAIN from individual engineered features, then
    aggregates the noisy votes with a transparent rank-1 spectral label model
    that recovers each LF's accuracy from inter-LF agreement (a dependency-free
    stand-in for Snorkel's LabelModel). Returns the
    events frame augmented with `ws_label`, `ws_proba`, `ws_score` and per-LF
    vote columns, plus an LF-analysis frame (coverage / conflict / learned
    weight). A DEFENSIBLE, NON-CIRCULAR STAND-IN for the NDA expert labels: flip
    `LABEL_COL` in modeling.py to the real column on delivery, nothing else.
    """
    n = events_df.height

    # Directions follow the domain taxonomy: front-loaded / abrupt curve +
    # coincident salinity / water-level step => hot moment (tidal incursion);
    # symmetric curve + antecedent precipitation => oxic pulse.
    _lf_specs = {
        "lf_peak_pos": ("peak_frac", 0.33, 0.66, WS_HOT, WS_PULSE, False),
        "lf_centroid": ("centroid_frac", 0.33, 0.66, WS_HOT, WS_PULSE, False),
        "lf_onset": ("max_rise_norm", 0.33, 0.66, WS_PULSE, WS_HOT, False),
        "lf_sal_step": ("sal_step", 0.50, 0.75, WS_ABSTAIN, WS_HOT, False),
        "lf_wl_step": ("wl_step", 0.50, 0.75, WS_ABSTAIN, WS_HOT, True),
        "lf_sal_hyst": ("hyst_sal", 0.50, 0.75, WS_ABSTAIN, WS_HOT, True),
        "lf_precip": ("precip_24h", 0.50, 0.75, WS_ABSTAIN, WS_PULSE, False),
    }
    # Keep only LFs whose feature exists on the active route (the hysteresis LF is
    # dropped on the moment route, which has no hyst_* columns).
    lfs = {
        _n: _ws_band(events_df, _c, _lo, _hi, _lv, _hv, use_abs=_ab)
        for _n, (_c, _lo, _hi, _lv, _hv, _ab) in _lf_specs.items()
        if _c in events_df.columns
    }
    names = list(lfs)
    L = np.column_stack([lfs[k] for k in names])
    m = len(names)

    # Label model: recover each LF's accuracy from PAIRWISE INTER-LF AGREEMENT
    # rather than agreement with a running majority consensus (which self-reinforces
    # the majority lean). Under conditional independence given the true class, the
    # co-voting agreement M_ij = E[lambda_i * lambda_j] ~= mu_i * mu_j, where
    # mu_j = 2*acc_j - 1 is LF j's correlation with the truth. So mu is the leading
    # factor of the off-diagonal agreement matrix -- a dependency-free rank-1
    # (spectral) label model. It removes the consensus bias but NOT correlated-LF
    # double-counting; Snorkel's LabelModel is the correlation-aware version.
    M = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            _both = (L[:, i] != WS_ABSTAIN) & (L[:, j] != WS_ABSTAIN)
            if _both.any():
                M[i, j] = float(
                    np.mean(L[_both, i] * L[_both, j])
                )  # E[lambda_i*lambda_j]

    # Leading eigenvector of the off-diagonal (self excluded) via power iteration.
    mu = np.ones(m) / np.sqrt(m)
    for _ in range(200):
        _nxt = M @ mu
        _nrm = float(np.linalg.norm(_nxt))
        if _nrm < 1e-12:
            break
        _nxt = _nxt / _nrm
        if np.allclose(np.abs(_nxt), np.abs(mu), atol=1e-7):
            mu = _nxt
            break
        mu = _nxt

    # Scale to accuracy units (||mu||^2 = leading eigenvalue), fix the global sign so
    # most LFs read better-than-chance in their authored (+1 = hot) orientation, and
    # fall back to a uniform vote if the agreement structure is degenerate.
    _eig = float(mu @ M @ mu)
    if _eig > 1e-9:
        mu = mu * np.sqrt(_eig)
        if mu.sum() < 0:
            mu = -mu
        w = np.clip(mu, 0.0, 1.0)
    else:
        w = np.ones(m)
    if not np.any(w > 0):
        w = np.ones(m)

    score = L @ w
    p_hot = 1.0 / (1.0 + np.exp(-score))
    ws_label = np.where(score >= 0, "hot", "pulse")
    ws_proba = np.where(score >= 0, p_hot, 1.0 - p_hot)  # confidence in chosen class

    labeled = events_df.with_columns(
        pl.Series("ws_label", ws_label),
        pl.Series("ws_proba", ws_proba),
        pl.Series("ws_score", score),
        *[pl.Series(k, lfs[k]) for k in names],
    )

    rows = []
    for j, name in enumerate(names):
        col = L[:, j]
        voted = col != WS_ABSTAIN
        conflict = float(
            np.mean(
                [
                    voted[i] and np.any((L[i] != WS_ABSTAIN) & (L[i] != col[i]))
                    for i in range(n)
                ]
            )
        )
        rows.append(
            {
                "LF": name,
                "coverage": round(float(voted.mean()), 3),
                "votes_hot": int(np.sum(col == WS_HOT)),
                "votes_pulse": int(np.sum(col == WS_PULSE)),
                "conflict": round(conflict, 3),
                "weight": round(float(w[j]), 3),
            }
        )
    lf_analysis = pl.DataFrame(rows)
    weights = {k: float(v) for k, v in zip(names, w)}
    return labeled, lf_analysis, weights


def binned_hist(df, col, bins=60):
    s = df.select(pl.col(col).alias("v")).drop_nulls()
    lo = s.select(pl.col("v").min()).item()
    hi = s.select(pl.col("v").max()).item()
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / bins
    return (
        s.with_columns(
            ((pl.col("v") - lo) / width).floor().clip(0, bins - 1).alias("bin")
        )
        .group_by("bin")
        .len(name="count")
        .with_columns(
            (lo + pl.col("bin") * width).alias("left"),
            (lo + (pl.col("bin") + 1) * width).alias("right"),
        )
        .sort("bin")
    )


def hist_chart(df, col, symlog=False, bins=60):
    h = binned_hist(df, col, bins=bins)
    y_scale = alt.Scale(type="symlog") if symlog else alt.Scale(type="linear")
    return (
        alt.Chart(h)
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title=col),
            x2="right:Q",
            y=alt.Y("count:Q", title="count", scale=y_scale),
            tooltip=[
                alt.Tooltip("left:Q", format=".2f", title="bin start"),
                alt.Tooltip("count:Q", format=",", title="count"),
            ],
        )
        .properties(width=330, height=200, title=col)
    )


def build_forecast_frame(readouts, freq="1h", target=DO_COL):
    """Resample the 5-min readout series onto a regular, GAP-COMPLETE grid at `freq`
    for DO forecasting. One row per time bin over the full [min, max] span (empty bins
    are emitted as nulls, never dropped). Columns:

      Datetime               regular grid (left bin edge)
      do                     mean of `target` over the bin
      solar_fd .. rh         mean of each weather covariate over the bin
      precip                 SUM of precipitation over the bin (flux -> accumulation)
      sal / wl / sensor_temp mean of each hydrology covariate over the bin
      <feat>_missing         1.0 where the bin had no raw sample for that covariate
      covar_valid            1.0 where every covariate is present (before imputation)
      hour_sin / hour_cos    diurnal calendar features (known-future)
      doy_sin / doy_cos      seasonal calendar features (known-future)

    Values are RAW and nulls are NOT imputed here; forecast.py applies a single causal
    imputation (forward-fill, then residual leading nulls -> train-safe global median;
    precip -> 0.0 accumulation), keeping the *_missing indicators as features. The median
    only touches LEADING nulls (row-0 prefix = earliest timestamps = train in every
    expanding fold), so it is leakage-safe without per-fold refitting (asserted in forecast.py)."""
    ro = readouts.sort("Datetime")
    mean_map = {DO_COL: FORECAST_DO, **FORECAST_WEATHER, **FORECAST_HYDRO}
    if target != DO_COL:
        mean_map = {k: v for k, v in mean_map.items() if k != DO_COL}
        mean_map[target] = FORECAST_DO
    aggs = [pl.col(src).mean().alias(dst) for src, dst in mean_map.items()]
    # precip summed, but keep NULL when the whole bin is empty so `_missing` stays honest
    aggs.append(
        pl.when(pl.col(PRECIP_COL).is_not_null().sum() > 0)
        .then(pl.col(PRECIP_COL).sum())
        .otherwise(None)
        .alias(FORECAST_PRECIP)
    )
    agg = ro.group_by_dynamic("Datetime", every=freq, closed="left", label="left").agg(
        aggs
    )

    # reindex onto a contiguous grid so gaps become explicit null rows, never dropped
    lo, hi = agg["Datetime"].min(), agg["Datetime"].max()
    grid = pl.DataFrame(
        {
            "Datetime": pl.datetime_range(
                lo, hi, interval=freq, closed="both", eager=True
            )
        }
    )
    out = grid.join(agg, on="Datetime", how="left").sort("Datetime")

    covars = (
        list(FORECAST_WEATHER.values())
        + [FORECAST_PRECIP]
        + list(FORECAST_HYDRO.values())
    )
    out = out.with_columns(
        [pl.col(c).is_null().cast(pl.Float64).alias(f"{c}_missing") for c in covars]
    )
    _valid = pl.lit(True)
    for c in covars:
        _valid = _valid & pl.col(c).is_not_null()
    out = out.with_columns(_valid.cast(pl.Float64).alias("covar_valid"))

    # known-future calendar encodings (diurnal + seasonal)
    _hh = pl.col("Datetime").dt.hour().cast(pl.Float64)
    _dd = pl.col("Datetime").dt.ordinal_day().cast(pl.Float64)
    out = out.with_columns(
        [
            (2 * np.pi * _hh / 24.0).sin().alias("hour_sin"),
            (2 * np.pi * _hh / 24.0).cos().alias("hour_cos"),
            (2 * np.pi * _dd / 365.25).sin().alias("doy_sin"),
            (2 * np.pi * _dd / 365.25).cos().alias("doy_cos"),
        ]
    )
    return out


def feature_methodology_tabs(
    proc, proc_curves, feature_cols=FEATURE_COLS, example_panel="curves"
):
    """Per-feature tabbed handcrafted-feature exposition (one tab per engineered feature:
    distribution + defining formula + a secondary panel) for eda.py. Reads the
    per-unit features (`proc` = proc_features) and resampled curves (`proc_curves`)
    that preprocessing.py already computed on OUR detector's events — no recompute
    of the features themselves; only plot-shaping lives here.

    `feature_cols` selects WHICH features to expose. It defaults to the full unit-route
    `FEATURE_COLS`. Note `splice_readouts` computes and stores the *full* FEATURE_COLS for
    EVERY route — including hyst_wl/hyst_sal on the moment features — so we can't infer the
    route's modelled feature set from the columns present. Pass `MOMENT_FEATURE_COLS` on the
    moment route to hide the hysteresis pair the moment CLASSIFIER drops (degenerate on a
    single tight rise-fall pulse), keeping the methodology slide consistent with the model.

    `example_panel` picks the second chart beside each feature's histogram:
      - "curves"        → example low/high DO curves from `proc_curves` (the eda default).
      - "label_scatter" → a jittered strip/scatter of the feature value stratified by moment
                          class (hot vs oxic), for the moment route (needs a label column)."""
    if example_panel not in ("curves", "label_scatter"):
        raise ValueError(f"example_panel must be 'curves'|'label_scatter', got {example_panel!r}")
    # moment class label for the scatter panel (both `label` and `expert_label` carry hot/pulse)
    _label_col = next((_c for _c in ("label", "expert_label") if _c in proc.columns), None)
    if example_panel == "label_scatter" and _label_col is None:
        raise ValueError("example_panel='label_scatter' needs a 'label'/'expert_label' column")
    # Handcrafted event-shape features, defined on OUR detector's events (proc_features /
    # proc_curves from preprocessing.py) — NOT the expert umbrellas, over which symmetry /
    # hysteresis are ill-defined (one umbrella can span several sub-moments). Each tab shows
    # the feature's distribution (with its defining formula), how it relates to each
    # coincident / antecedent driver, and example DO curves at the low vs high extremes.
    # peak_frac = 0 -> abrupt rise / slow decay ("hot moment"); ~0.5 -> symmetric ("oxic pulse").
    # All engineered features (core.features.FEATURE_COLS), in canonical order: event
    # magnitude / duration, curve shape, coincident driver levels + antecedent steps, the
    # curve-morphology scalars, the DO–driver hysteresis pair, then the antecedent-precip
    # lag ladder. `_feat_meta` = short tab label, `_feat_formula` = the defining formula.
    _feat_meta = {
        "dur_min": "Duration",
        "n_samples": "Sample count",
        "peak_do": r"\(\max(DO_2)\)",
        "mean_do": r"\(\langle DO_2\rangle\)",
        "area_mgLh": "DO area",
        "peak_frac": "Rel. time to peak",
        "rise_min": "Rise time",
        "fall_min": "Fall time",
        "rise_rate": r"\(\text{Rate}_{DO_{2}}\)",
        "fall_rate": r"\(\text{Rate}^{\rm fall}_{DO_{2}}\)",
        "sal_in": r"\(\langle\text{Salinity}\rangle\)",
        "wl_in": r"\(\langle\text{Water-level}\rangle\)",
        "temp_in": r"\(\langle T\rangle\)",
        "sal_step": r"\(\Delta \langle\text{Salinity}\rangle\)",
        "wl_step": r"\(\Delta \langle\text{Water-level}\rangle\)",
        "temp_step": r"\(\Delta \langle T\rangle\)",
        "centroid_frac": "DO centroid",
        "max_rise_norm": "Max step",
        "plateau_frac": "Plateau fraction",
        "hyst_wl": "W.L. hysteresis",
        "hyst_sal": "Salinity hysteresis",
        "precip_24h": "Antecedent precip. (24h)",
        "precip_72h": "Antecedent precip. (72h)",
        "precip_168h": "Antecedent precip. (168h)",
    }

    _feat_formula = {
        "dur_min": r"\(t_{\rm end} - t_{\rm start}\) (min)",
        "n_samples": r"\(N_{\rm steps}\) = number of 5-min readouts in \((\rm start, end)\)",
        "peak_do": r"\(\max_i (DO_2)_i\) (mg/L)",
        "mean_do": r"\(\frac{1}{N_{\rm steps}}\sum_{i=1}^{N_{\rm steps}} (DO_2)_i\) (mg/L)",
        "area_mgLh": r"\(\sum_{i=1}^{N_{\rm steps}} (DO_2)_i\,\Delta t,\; \Delta t = 5\,\text{min}\) (mg/L·h)",
        "peak_frac": r"\(\frac{t_{\rm peak} - t_{\rm start}}{t_{\rm end} - t_{\rm start}}\)",
        "rise_min": r"\(t_{\rm peak} - t_{\rm start}\) (min)",
        "fall_min": r"\(t_{\rm end} - t_{\rm peak}\) (min)",
        "rise_rate": r"\(\frac{\partial(DO_2)}{\partial t}\)",
        "fall_rate": r"\(\frac{\max(DO_2)}{t_{\rm end} - t_{\rm peak}}\) (mg/L·h)",
        "sal_in": r"\(\frac{1}{N_{\rm steps}}\sum_{i=1}^{N_{\rm steps}}\text{Sal.}_i\)",
        "wl_in": r"\(\frac{1}{N_{\rm steps}}\sum_{i=1}^{N_{\rm steps}}\text{W.L.}_i\)",
        "temp_in": r"\(\frac{1}{N_{\rm steps}}\sum_{i=1}^{N_{\rm steps}} T_i\)",
        "sal_step": r"\(\langle \text{Sal.(t)} \rangle_{t \in (\rm start, end)} - \langle \text{Sal.(t)} \rangle_{t \in (\rm start - 24h, start)}\) (PSU)",
        "wl_step": r"\(\langle \text{WL(t)} \rangle_{t \in (\rm start, end)} - \langle \text{WL(t)} \rangle_{t \in (\rm start - 24h, start)}\) (BGS, cm)",
        "temp_step": r"\(\langle T(t) \rangle_{t \in (\rm start, end)} - \langle T(t) \rangle_{t \in (\rm start - 24h, start)}\) (\(\degree C\))",
        "centroid_frac": r"\(\frac{\sum_i i\,(DO_2)_i}{(N_{\rm steps}-1)\sum_i (DO_2)_i}\) — DO-weighted time centroid, normalised to \([0,1]\)",
        "max_rise_norm": r"\(\frac{\max_i[(DO_2)_{i+1} - (DO_2)_i]}{\max(DO_2)}\) — largest single 5-min jump, peak-normalised",
        "plateau_frac": r"\(\frac{1}{N_{\rm steps}}\sum_i \mathbb{1}\!\left[(DO_2)_i \geq 0.8\,\max(DO_2)\right]\)",
        "hyst_wl": r"\(\left\langle DO_2^{\rm rise}(\text{W.L.}) - DO_2^{\rm fall}(\text{W.L.})\right\rangle\) — mean gap between rising & falling limbs of the peak-normalised \(DO_2\)–W.L. loop",
        "hyst_sal": r"\(\left\langle DO_2^{\rm rise}(\text{Sal.}) - DO_2^{\rm fall}(\text{Sal.})\right\rangle\) — mean gap between rising & falling limbs of the peak-normalised \(DO_2\)–Sal. loop",
        "precip_24h": r"\(\sum_{t \in (\rm start - 24h, start)}\text{Precip}(t)\) (mm)",
        "precip_72h": r"\(\sum_{t \in (\rm start - 72h, start)}\text{Precip}(t)\) (mm)",
        "precip_168h": r"\(\sum_{t \in (\rm start - 168h, start)}\text{Precip}(t)\) (mm)",
    }

    # Expose only the features the CALLER asked for (`feature_cols`) that are also physically
    # present in the frame. `splice_readouts` stores the full FEATURE_COLS for every route, so
    # the moment features parquet DOES carry computed hyst_wl/hyst_sal — pass MOMENT_FEATURE_COLS
    # to drop them here, matching the moment classifier (which excludes the degenerate loop).
    _keep = [_c for _c in feature_cols if _c in proc.columns]
    _feat_meta = {_k: _v for _k, _v in _feat_meta.items() if _k in _keep}

    def _example_chart(ycol):
        # 2 lowest + 2 highest units on THIS tab's feature; DO curve over normalised time
        # straight from proc_curves (already resampled to 128 steps by preprocessing.py).
        _v = proc.drop_nulls(ycol).sort(ycol)
        _ids = _v.head(2)["unit_id"].to_list() + _v.tail(2)["unit_id"].to_list()
        _pick = proc.filter(pl.col("unit_id").is_in(_ids)).select("unit_id", ycol)
        _curve = (
            proc_curves.filter(pl.col("unit_id").is_in(_ids) & pl.col("valid"))
            .join(_pick, on="unit_id")
            .with_columns(
                (
                    pl.col("step")
                    / (pl.col("step").max().over("unit_id")).clip(lower_bound=1)
                ).alias("t_norm"),
                pl.format(
                    "{} ({}={})", pl.col("unit_id"), pl.lit(ycol), pl.col(ycol).round(2)
                ).alias("event"),
            )
            .select("t_norm", "do", "event")
            .sort("event", "t_norm")
        )
        return (
            alt.Chart(_curve)
            .mark_line()
            .encode(
                alt.X("t_norm:Q", title="normalised time (0 = start, 1 = end)"),
                alt.Y("do:Q", title="DO (mg/L)"),
                alt.Color(
                    "event:N", title=None, scale=alt.Scale(scheme="redyellowblue")
                ),
            )
            .properties(
                width=400,
                height=190,
                title=f"Example shapes: low vs high {ycol}",
            )
        )

    def _label_scatter(ycol):
        # Feature value stratified by moment class (hot vs oxic): a jittered strip/scatter so
        # the hot-vs-oxic separation each feature carries reads directly off the y-axis. Class
        # from `_label_col` with pulse -> oxic (matching eda's label survey); nulls dropped.
        _d = proc.select(
            pl.col(ycol).cast(pl.Float64),
            pl.col(_label_col).replace({"pulse": "oxic"}).alias("class"),
        ).drop_nulls()
        return (
            alt.Chart(_d)
            .mark_circle(size=55, opacity=0.55)
            .encode(
                alt.X("class:N", title="moment label", sort=["hot", "oxic", "mixed"]),
                alt.Y(f"{ycol}:Q", title=ycol),
                alt.XOffset("jitter:Q"),  # spread points within each class band
                alt.Color(
                    "class:N",
                    title="class",
                    sort=["hot", "oxic", "mixed"],
                    scale=alt.Scale(
                        domain=["hot", "oxic", "mixed"],
                        range=["#c1440e", "#3b7dd8", "#9467bd"],
                    ),
                ),
                tooltip=["class:N", alt.Tooltip(f"{ycol}:Q", format=".3f")],
            )
            # Box–Muller gaussian jitter (Vega-Lite strip-plot idiom) on the offset channel
            .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
            .properties(width=400, height=190, title=f"{ycol} by moment label")
        )

    def _feat_tab(ycol):
        _title = _feat_meta[ycol]
        _hist = (
            alt.Chart(proc.select(ycol))
            .mark_bar(color="#3b7dd8")
            .encode(
                alt.X(f"{ycol}:Q", bin=alt.Bin(maxbins=24), title=ycol),
                alt.Y("count():Q", title="number of events"),
            )
            .properties(
                width=460,
                height=190,
                title=f"{ycol}  (n = {proc.height} of our events)",
            )
        )
        _panel = _example_chart(ycol) if example_panel == "curves" else _label_scatter(ycol)
        return mo.vstack(
            [
                mo.md(f"""
                **{_title} ({ycol}) = {_feat_formula[ycol]}**
                """),
                mo.hstack(
                    [
                        _hist,
                        _panel,
                    ],
                    justify="start",
                ),
            ],
        )

    # Group the 24 features into physically-meaningful categories → two-level tabs
    # (outer = category, inner = the features in it) so the label row stays short.
    # Order within each group follows FEATURE_COLS.
    _groups = {
        "Magnitude & duration": [
            "dur_min",
            "n_samples",
            "peak_do",
            "mean_do",
            "area_mgLh",
        ],
        # "Rise / fall shape": ["peak_frac", "rise_min", "fall_min", "rise_rate", "fall_rate"],
        "Drivers (levels & steps)": [
            "sal_in",
            "wl_in",
            "temp_in",
            "sal_step",
            "wl_step",
            "temp_step",
        ],
        "Curve morphology": [
            "peak_frac",
            "rise_min",
            "fall_min",
            "rise_rate",
            "fall_rate",
            "centroid_frac",
            "max_rise_norm",
            "plateau_frac",
        ],
        "Hysteresis": ["hyst_wl", "hyst_sal"],
        "Antecedent precip.": ["precip_24h", "precip_72h", "precip_168h"],
    }

    # Build each category's inner tabs from only its present features, and drop any group the
    # column guard emptied (e.g. Hysteresis on the moment route). A stray feature not assigned
    # to any group would silently disappear, so assert full coverage of the present features.
    _grouped_keys = {_k for _keys in _groups.values() for _k in _keys}
    assert set(_feat_meta) <= _grouped_keys, (
        f"ungrouped features: {set(_feat_meta) - _grouped_keys}"
    )
    _outer = {}
    for _label, _keys in _groups.items():
        _kept = [_k for _k in _keys if _k in _feat_meta]
        if _kept:
            _outer[_label] = mo.ui.tabs({_feat_meta[_k]: _feat_tab(_k) for _k in _kept})
    return mo.ui.tabs(_outer)
