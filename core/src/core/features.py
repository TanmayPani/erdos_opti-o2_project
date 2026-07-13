import re
from functools import partial

import numpy as np
import polars as pl
from scipy.signal import find_peaks

DO_COL = "Dissolved Oxygen (mg/L)"
SAL_COL = "Well Salinity (PPT)"
WL_COL = "Flood plain water level in BGS (cm)"
TEMP_COL = "DO Sensor Temperature (C) "
PRECIP_COL = "Precip (mm) over 5 minutes"
CURVE_CHANNELS = [DO_COL, SAL_COL, WL_COL, TEMP_COL, PRECIP_COL]
CURVE_SHORT = ["do", "sal", "wl", "temp", "precip"]
FEATURE_COLS = [
    "dur_min",
    "n_samples",
    "peak_do",
    "mean_do",
    "area_mgLh",
    "peak_frac",
    "rise_min",
    "fall_min",
    "rise_rate",
    "fall_rate",
    "sal_in",
    "wl_in",
    "temp_in",
    "sal_step",
    "wl_step",
    "temp_step",
    "centroid_frac",
    "max_rise_norm",
    "plateau_frac",
    "hyst_wl",
    "hyst_sal",
    "precip_24h",
    "precip_72h",
    "precip_168h",
]

# The moment route classifies single tight rise-fall pulses instead of whole excursions. The
# DO-vs-driver HYSTERESIS loop (`hyst_wl`/`hyst_sal`) needs the driver to sweep up and back
# across a multi-pulse excursion — on one moment it's degenerate (flat driver / edge peak), so
# it's dropped. Every other engineered feature stays well-defined (verified <2% null, real
# variance on the moments), keeping the moment route otherwise identical to the unit route.
MOMENT_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ("hyst_wl", "hyst_sal")]


def ws_umbrella_label(umbrella_feat, sal_step_thr=0.4):
    """Driver-based *weak* label at the umbrella level: a salinity-intrusion rule
    (`|sal_step| >= sal_step_thr` -> hot, else pulse), the single most interpretable
    driver of the taxonomy (marine/tidal intrusion vs rainfall). Computed on the
    umbrella feature table (`sync_expert_features` on `events_covered`) because the
    step signature fires at the umbrella ONSET and is washed out on the nested
    per-auto-event windows. On the labelled hot/pulse umbrellas this recovers the
    expert only ~0.74 (base rate 0.72) — the drivers are informative but do NOT
    cleanly determine the class, so this is a QC/agreement signal, not a target.
    Returns `group_id, ws_label`."""
    return umbrella_feat.select(
        pl.col("event_id").alias("group_id"),
        pl.when(pl.col("sal_step").abs() >= sal_step_thr)
        .then(pl.lit("hot"))
        .otherwise(pl.lit("pulse"))
        .alias("ws_label"),
    )


def finalize_units(unit_feat, units, umbrella_feat, sal_step_thr=0.4):
    """Assemble the model-ready tabular table from the per-unit feature roundtrip.

    `unit_feat` is `sync_expert_features(readouts, units.rename(unit_id->event_id))[0]`.
    Attaches provenance (`group_id`/`source`/`is_flood` from `units`), renames the
    inherited 3-class umbrella label to `regime`, and derives the supervised columns:
      label        binary training target in {hot, pulse}; null for held-out rows.
      split        'holdout' for regime=='mixed' or subtype 'b' (unknown flood/
                   composite regimes we don't train on), else 'train'.
      label_source 'expert' for umbrella-derived rows, 'inferred' for adopted
                   orphans (self-assigned shape+driver label) — lets you ablate.
      ws_label / ws_expert_agree  the umbrella driver rule (`ws_umbrella_label`)
                   inherited by group_id (own-row `sal_step` rule for orphans, which
                   belong to no umbrella), and whether it matches `regime` (QC).
    Column order: keys/provenance, labels, then FEATURE_COLS."""
    meta = units.select(pl.col("unit_id"), "group_id", "source", "is_flood")
    ws = ws_umbrella_label(umbrella_feat, sal_step_thr)
    holdout = (pl.col("regime") == "mixed") | pl.col("expert_subtype").str.starts_with(
        "b"
    )
    out = (
        unit_feat.rename({"event_id": "unit_id", "expert_label": "regime"})
        .drop("eid")
        .join(meta, on="unit_id", how="left")
        .join(ws, on="group_id", how="left")
        .with_columns(
            # orphans belong to no umbrella -> ws join misses; fall back to their own
            # sal_step (a standalone excursion, unlike a nested auto event)
            pl.coalesce(
                pl.col("ws_label"),
                pl.when(pl.col("sal_step").abs() >= sal_step_thr)
                .then(pl.lit("hot"))
                .otherwise(pl.lit("pulse")),
            ).alias("ws_label"),
            pl.when(pl.col("source") == "orphan")
            .then(pl.lit("inferred"))
            .otherwise(pl.lit("expert"))
            .alias("label_source"),
        )
        .with_columns(
            pl.when(holdout)
            .then(pl.lit("holdout"))
            .otherwise(pl.lit("train"))
            .alias("split"),
            pl.when(holdout).then(None).otherwise(pl.col("regime")).alias("label"),
        )
        .with_columns((pl.col("ws_label") == pl.col("regime")).alias("ws_expert_agree"))
    )
    lead = [
        "unit_id",
        "group_id",
        "source",
        "label_source",
        "expert_subtype",
        "regime",
        "label",
        "split",
        "is_flood",
        "ws_label",
        "ws_expert_agree",
        "start",
        "end",
        "is_public_augmented",
    ]
    return out.select(lead + FEATURE_COLS).sort(["group_id", "start"])


def explode_moments(events):
    """Expert list (nested `moments.*`) -> one row per MOMENT, in the shape
    `sync_expert_features` expects (a moment reuses the umbrella-event columns at moment
    granularity). `event_id` = `<umbrella>#<moment_idx>` (unique, null-free — moment_idx is the
    reformatter's clean 1..N); `group_id` = the umbrella (kept for grouped CV so an umbrella's
    moments never straddle a fold); `expert_label` = the moment's OWN type mapped hot/pulse
    (this is the direct per-moment label — mixed umbrellas dissolve into individual moments).
    Only typed moments with a start are kept; the readout-coverage filter is handled downstream
    by the `sync_expert_features` sample join (a moment with no in-range samples drops out)."""
    mcols = [c for c in events.columns if c.startswith("moments.")]
    ex = events.explode(mcols).filter(
        pl.col("moments.type").is_not_null() & pl.col("moments.start_time").is_not_null()
    )
    return ex.select(
        (pl.col("event_id") + "#" + pl.col("moments.idx").cast(pl.String)).alias("event_id"),
        pl.col("event_id").alias("group_id"),
        pl.col("moments.start_time").alias("group_start"),
        pl.col("moments.end_time").alias("group_end"),
        pl.when(pl.col("moments.type") == "hot")
        .then(pl.lit("hot"))
        .otherwise(pl.lit("pulse"))
        .alias("expert_label"),
        pl.col("moments.type").alias("expert_subtype"),
    )


def finalize_moments(mom_feat, mom_frame):
    """Assemble the moment-level model table — the mirror of `finalize_units`, minus the
    unit-only machinery: every moment is DIRECTLY typed, so there is no umbrella-inherited
    label, no orphan / weak-supervision logic, and no `mixed` holdout (all moments are
    trainable, `split=='train'`). Drops the hysteresis features -> `MOMENT_FEATURE_COLS`.
    Column order mirrors `finalize_units` (keys/provenance, labels, then features)."""
    _grp = mom_frame.select(pl.col("event_id").alias("unit_id"), "group_id")
    out = (
        mom_feat.rename({"event_id": "unit_id", "expert_label": "label"})
        .drop("eid")
        .join(_grp, on="unit_id", how="left")
        .with_columns(
            pl.col("label").alias("regime"),
            pl.lit("expert").alias("label_source"),
            pl.lit("train").alias("split"),
        )
    )
    lead = [
        "unit_id",
        "group_id",
        "label_source",
        "expert_subtype",
        "regime",
        "label",
        "split",
        "start",
        "end",
        "is_public_augmented",
    ]
    return out.select(lead + MOMENT_FEATURE_COLS).sort(["group_id", "start"])


def build_curves(
    readouts,
    proc,
    T=128,
    pad_frac=0.5,
    max_pad_h=12.0,
    pad_left_h=None,
    pad_right_frac=0.25,
    pad_right_cap_h=6.0,
):
    """Fixed-length multichannel curves for the DL track — one (T x 5) block per unit.

    Each unit window [start, end] is padded so the net sees context before/after, then
    the five `CURVE_CHANNELS` are linearly resampled onto `T` evenly spaced time points.
    Two padding modes:
      * SYMMETRIC (default, `pad_left_h=None`): pad by `pad_frac` of duration each side
        (capped at `max_pad_h`) — the original tight window centred on the excursion.
      * ASYMMETRIC (`pad_left_h` set): pad LEFT by a fixed `pad_left_h` hours of antecedent
        context, RIGHT by `min(pad_right_frac*dur, pad_right_cap_h h)`. This exposes the
        pre-onset salinity/water-level trajectory that `sal_step`/`wl_step` encode but the
        tight window omits — VALIDATED 2026-07-07 to lift ROCKET->XGB on the minority pulse
        class (macroF1 0.79->0.84 random CV, 0.75->0.80 leave-one-water-year-out, shuffle
        floors at .48; best at pad_left_h=48, T=256; 72h over-widens). See [[modeling-scaffold]].
    Values are kept RAW (no normalisation) so scaling can be fit on the training fold only —
    do not normalise here. Returns a tidy long frame: unit_id, step (0..T-1), t_min (minutes
    from padded-window start), do/sal/wl/temp/precip, valid (grid point within the raw sample
    time-span), plus group_id/regime/label/split denormalised from `proc`.
    A unit whose window carries no readout samples is skipped (reported by the caller)."""
    ro = readouts.sort("Datetime")
    tvec = ro["Datetime"].to_numpy().astype("datetime64[s]").astype(np.int64)
    arrs = {c: ro[c].to_numpy().astype(float) for c in CURVE_CHANNELS}
    keep = proc.select(
        "unit_id", "start", "end", "group_id", "regime", "label", "split"
    )
    blocks = []
    for row in keep.iter_rows(named=True):
        s = np.datetime64(row["start"]).astype("datetime64[s]").astype(np.int64)
        e = np.datetime64(row["end"]).astype("datetime64[s]").astype(np.int64)
        if pad_left_h is None:  # symmetric (default, unchanged)
            pad = min(pad_frac * (e - s), max_pad_h * 3600)
            a, b = s - pad, e + pad
        else:  # asymmetric: long antecedent left pad
            a = s - pad_left_h * 3600
            b = e + min(pad_right_frac * (e - s), pad_right_cap_h * 3600)
        m = (tvec >= a) & (tvec <= b)
        if not m.any():
            continue
        tt = tvec[m]
        grid = np.linspace(a, b, T)
        blk = {
            "unit_id": pl.Series([row["unit_id"]] * T, dtype=pl.String),
            "step": np.arange(T, dtype=np.int64),
            "t_min": (grid - a) / 60.0,
            "valid": (grid >= tt.min()) & (grid <= tt.max()),
        }
        for c, sc in zip(CURVE_CHANNELS, CURVE_SHORT):
            y = arrs[c][m]
            ok = ~np.isnan(y)
            blk[sc] = (
                np.interp(grid, tt[ok], y[ok]) if ok.sum() >= 2 else np.full(T, np.nan)
            )
        # string passthrough with explicit dtype so None (held-out label) stays a
        # proper null and blocks concat cleanly across train/holdout rows
        for k in ("group_id", "regime", "label", "split"):
            blk[k] = pl.Series([row[k]] * T, dtype=pl.String)
        blocks.append(pl.DataFrame(blk))
    return pl.concat(blocks)


def detect_events(df, do_col=None, merge_gap=12):
    """One row per oxygenation event (contiguous DO > 0), merging baseline gaps
    of <= merge_gap samples (12 = 1 h at the 5-min cadence) into the surrounding
    event. Returns columns: eid, start, end."""

    f = df.select("Datetime", (pl.col(do_col or DO_COL) > 0).alias("_oxic"))
    f = f.with_columns(
        (pl.col("_oxic") != pl.col("_oxic").shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("_rid")
    )
    runs = (
        f.group_by("_rid")
        .agg(
            pl.col("_oxic").first(),
            pl.len().alias("_n"),
            pl.col("Datetime").min().alias("start"),
            pl.col("Datetime").max().alias("end"),
        )
        .sort("start")
    )
    # absorb a short baseline gap into its neighbours, then re-segment
    runs = runs.with_columns(
        pl.when(~pl.col("_oxic") & (pl.col("_n") <= merge_gap))
        .then(True)
        .otherwise(pl.col("_oxic"))
        .alias("_om")
    )
    runs = runs.with_columns(
        (pl.col("_om") != pl.col("_om").shift(1)).fill_null(True).cum_sum().alias("eid")
    )
    return (
        runs.filter(pl.col("_om"))
        .group_by("eid")
        .agg(pl.col("start").min(), pl.col("end").max())
        .sort("start")
    )


def auto_detected_events(readouts, peak_min=0.1):
    """Auto-detected events (existing `detect_events`) with per-event peak DO, kept
    above `peak_min` — the `exploratory.py` selection, applied to the NDA readouts.
    Returns eid/peak_do/start/end; used both for the reconciliation and to delineate,
    on each expert-event plot, where our detector would splice the window."""
    auto = detect_events(readouts)
    s = (
        readouts.sort("Datetime")
        .join_asof(auto, left_on="Datetime", right_on="start", strategy="backward")
        .filter(pl.col("Datetime") <= pl.col("end"))
    )
    return (
        s.group_by("eid")
        .agg(
            pl.col(DO_COL).max().alias("peak_do"),
            pl.col("start").first(),
            pl.col("end").first(),
        )
        .filter(pl.col("peak_do") >= peak_min)
        .sort("start")
    )


def build_units(readouts, events_covered, peak_min=0.1, adopt_orphans=None):
    """Construct the classification UNITS from auto-detected events + the expert
    umbrellas that cover them (the 'auto events + umbrella fallback' scheme).

    Each auto event (`auto_detected_events`, peak >= peak_min) is assigned to the
    expert umbrella it overlaps most; it inherits that umbrella's class and keeps
    its own tight [start, end] window (where the DO excursion actually happens).
    Umbrellas that no auto event claims (either genuinely undetected, or whose only
    overlapping auto event was won by a neighbour) get a *fallback* unit spanning
    their full expert window, so no labelled umbrella is dropped. Auto events that
    overlap no covered umbrella are ORPHANS (returned separately, excluded).

    `adopt_orphans` optionally promotes specific orphans to proper units: a dict
    mapping an orphan's start date (`"YYYY-MM-DD"`) -> assigned class. Adopted units
    get `source="orphan"`, `expert_subtype="inferred"`, and their OWN `unit_id` as
    `group_id` (a singleton CV group — they belong to no umbrella). These carry a
    *self-assigned* (shape+driver) label, not an expert one — `finalize_units` marks
    them `label_source="inferred"` so they stay separable from the expert-gold rows.

    Returns `(units, orphans, missed)`:
      units    one row per unit: `unit_id` (unique), `group_id` (parent umbrella, or
               own id for adopted orphans — the CV grouping key), `source`
               ('auto'|'fallback'|'orphan'), `group_start`/`group_end` (the unit
               window, named to feed `sync_expert_features` directly),
               `expert_label`/`expert_subtype`, `is_flood`.
      orphans  auto events overlapping no covered umbrella AND not adopted (audit /
               possible expert-missed events — inspect before trusting).
      missed   umbrellas represented only by a fallback window (no auto event)."""
    auto = auto_detected_events(readouts, peak_min)
    umb = events_covered.select(
        pl.col("event_id").alias("group_id"),
        pl.col("group_start").alias("u_gs"),
        pl.col("group_end").alias("u_ge"),
        "expert_label",
        "expert_subtype",
        pl.col("flooding").alias("is_flood"),
    )
    # assign each auto event to the umbrella of maximum temporal overlap
    ov = auto.join_where(
        umb, pl.col("start") <= pl.col("u_ge"), pl.col("end") >= pl.col("u_gs")
    ).with_columns(
        (pl.min_horizontal("end", "u_ge") - pl.max_horizontal("start", "u_gs"))
        .dt.total_seconds()
        .alias("_ov")
    )
    best = ov.sort("_ov", descending=True).unique(subset=["eid"], keep="first")
    auto_units = best.select(
        pl.concat_str([pl.lit("a"), pl.col("eid").cast(pl.Utf8)]).alias("unit_id"),
        "group_id",
        pl.lit("auto").alias("source"),
        pl.col("start").alias("group_start"),
        pl.col("end").alias("group_end"),
        "expert_label",
        "expert_subtype",
        "is_flood",
    )
    orphans = auto.filter(~pl.col("eid").is_in(best["eid"].implode()))

    claimed = best["group_id"].implode()
    missed = umb.filter(~pl.col("group_id").is_in(claimed))
    fb_units = missed.select(
        pl.concat_str([pl.lit("f_"), pl.col("group_id")]).alias("unit_id"),
        "group_id",
        pl.lit("fallback").alias("source"),
        pl.col("u_gs").alias("group_start"),
        pl.col("u_ge").alias("group_end"),
        "expert_label",
        "expert_subtype",
        "is_flood",
    )
    units = pl.concat([auto_units, fb_units]).sort("group_start")

    if adopt_orphans:
        _adf = pl.DataFrame(
            {"_d": list(adopt_orphans), "_lab": list(adopt_orphans.values())}
        )
        _ad = orphans.with_columns(
            pl.col("start").dt.strftime("%Y-%m-%d").alias("_d")
        ).join(_adf, on="_d", how="inner")
        _orphan_units = _ad.select(
            pl.concat_str([pl.lit("o"), pl.col("eid").cast(pl.Utf8)]).alias("unit_id"),
            pl.concat_str([pl.lit("o"), pl.col("eid").cast(pl.Utf8)]).alias("group_id"),
            pl.lit("orphan").alias("source"),
            pl.col("start").alias("group_start"),
            pl.col("end").alias("group_end"),
            pl.col("_lab").alias("expert_label"),
            pl.lit("inferred").alias("expert_subtype"),
            pl.lit(False).alias("is_flood"),
        )
        units = pl.concat([units, _orphan_units]).sort("group_start")
        orphans = orphans.filter(~pl.col("eid").is_in(_ad["eid"].implode()))

    return units, orphans, missed.rename({"u_gs": "group_start", "u_ge": "group_end"})


def catalogue_orphans(readouts, events_covered, expert_event_ids, peak_min=0.1):
    """Auto-detected excursions overlapping no covered expert umbrella ('orphans'),
    catalogued as FIRST-CLASS events so they can sit beside the expert ones.

    Each orphan gets a proper `event_id` in the expert scheme — `YYYY-N{h|o|x}` where
    the suffix encodes its driver signature (**h** = hot-like: a salinity step >= 0.4 or
    a water-level step >= 20 cm; **o** = pulse-like / freshwater; **x** = ignored
    threshold noise, peak DO < 0.15) and `N` continues after that year's highest expert
    number so the id never collides. Also sets a derived `expert_label`
    (hot / pulse / ignore), `expert_subtype` = the suffix, a descriptive `driver_class`,
    and **`is_orphan=True`**.

    Returns `(orphan_events, orphan_samples)` carrying `sync_expert_features`' columns
    plus that orphan metadata, so they concatenate directly onto `expert_events` /
    `expert_samples`. `expert_event_ids` is the FULL expert id list (covered +
    out-of-range) used only to pick each year's next free running number."""
    auto = auto_detected_events(readouts, peak_min)
    _cov = events_covered.select(
        pl.col("group_start").alias("_gs"), pl.col("group_end").alias("_ge")
    )
    _ov = auto.join_where(
        _cov, pl.col("start") <= pl.col("_ge"), pl.col("end") >= pl.col("_gs")
    )
    orph = auto.filter(~pl.col("eid").is_in(_ov["eid"].implode())).sort("start")
    _tmp = orph.select(
        pl.concat_str([pl.lit("o"), pl.col("eid").cast(pl.Utf8)]).alias("event_id"),
        pl.col("start").alias("group_start"),
        pl.col("end").alias("group_end"),
        pl.lit("orphan").alias("expert_label"),
        pl.lit("?").alias("expert_subtype"),
    )
    feat, samples = sync_expert_features(readouts, _tmp)

    feat = feat.sort("start").with_columns(
        pl.when(pl.col("peak_do") < 0.15)
        .then(pl.lit("x"))
        .when((pl.col("sal_step") >= 0.4) | (pl.col("wl_step").abs() >= 20))
        .then(pl.lit("h"))
        .otherwise(pl.lit("o"))
        .alias("_suf"),
        pl.col("start").dt.year().cast(pl.Utf8).alias("_y"),
        pl.when(pl.col("peak_do") < 0.15)
        .then(pl.lit("noise (peak<0.15)"))
        .when(pl.col("sal_step") >= 0.4)
        .then(pl.lit("hot-like (salinity intrusion)"))
        .when(pl.col("wl_step").abs() >= 20)
        .then(pl.lit("hot-like (water-level step)"))
        .when(pl.col("precip_24h") >= 0.5)
        .then(pl.lit("pulse-like (rain)"))
        .otherwise(pl.lit("pulse-like (freshwater)"))
        .alias("driver_class"),
    )

    # next free running number per year, after the expert events
    _maxseq = {}
    for _eid in expert_event_ids:
        _m = re.match(r"^(\d{4})-(\d+)", str(_eid))
        if _m:
            _maxseq[_m.group(1)] = max(_maxseq.get(_m.group(1), 0), int(_m.group(2)))
    _maxdf = pl.DataFrame(
        {"_y": list(_maxseq) or [""], "_mx": list(_maxseq.values()) or [0]},
        schema={"_y": pl.Utf8, "_mx": pl.Int64},
    )
    feat = (
        feat.with_columns(pl.int_range(1, pl.len() + 1).over("_y").alias("_k"))
        .join(_maxdf, on="_y", how="left")
        .with_columns(pl.col("_mx").fill_null(0))
        .with_columns(
            pl.format(
                "{}-{}{}", pl.col("_y"), pl.col("_mx") + pl.col("_k"), pl.col("_suf")
            ).alias("_new_id")
        )
    )
    _idmap = dict(zip(feat["event_id"].to_list(), feat["_new_id"].to_list()))
    orphan_events = feat.with_columns(
        pl.col("_new_id").alias("event_id"),
        pl.lit(True).alias("is_orphan"),
        pl.when(pl.col("_suf") == "h")
        .then(pl.lit("hot"))
        .when(pl.col("_suf") == "o")
        .then(pl.lit("pulse"))
        .otherwise(pl.lit("ignore"))
        .alias("expert_label"),
        pl.col("_suf").alias("expert_subtype"),
    ).drop("_suf", "_y", "_k", "_mx", "_new_id")
    orphan_samples = samples.with_columns(pl.col("event_id").replace(_idmap))
    return orphan_events, orphan_samples


def hysteresis(do, drv):
    rng = drv.max() - drv.min()
    ap = int(np.argmax(do))
    if rng < 1e-9 or ap < 1 or ap >= len(do) - 1:
        return None
    xn = (drv - drv.min()) / rng
    yn = do / do.max()
    grid = np.linspace(0.05, 0.95, 10)
    xr, yr = xn[: ap + 1], yn[: ap + 1]
    xf, yf = xn[ap:], yn[ap:]
    ri = np.interp(grid, np.sort(xr), yr[np.argsort(xr)])
    fi = np.interp(grid, np.sort(xf), yf[np.argsort(xf)])
    return float((ri - fi).mean())


def expert_event_samples(readouts, events):
    """Every 5-min readout row that falls inside an expert window, tagged with the
    event's integer `eid`, `event_id`, labels, and `elapsed_min` (mirrors the
    `event_samples` table exploratory.py builds, but keyed on expert windows rather
    than auto-detected ones). Feeds the shape features and Track B."""
    win = (
        events.sort("group_start")
        .with_row_index("eid")
        .with_columns(pl.col("eid").cast(pl.UInt32))
    )
    return (
        readouts.sort("Datetime")
        .join_asof(
            win.select(
                "eid",
                "group_start",
                "group_end",
                "event_id",
                "expert_label",
                "expert_subtype",
            ),
            left_on="Datetime",
            right_on="group_start",
            strategy="backward",
        )
        .filter(pl.col("Datetime") <= pl.col("group_end"))
        .with_columns(
            (pl.col("Datetime") - pl.col("group_start"))
            .dt.total_minutes()
            .alias("elapsed_min")
        )
    )


def event_shape_features(
    samples,
    do_col=None,
    wl_col=None,
    sal_col=None,
):
    """Trajectory-level shape descriptors that need the full per-event DO curve
    (computed in numpy), one row per eid:
      centroid_frac  - mass-weighted centre of the curve in time (robust symmetry;
                       0.5 = symmetric, < 0.5 = front-loaded / abrupt)
      max_rise_norm  - sharpest single 5-min DO jump / peak (onset abruptness)
      plateau_frac   - fraction of samples within 80-100% of peak (flat-topped vs spiky)
      hyst_wl/hyst_sal - DO-vs-driver hysteresis index: mean(rising - falling) DO over
                       the driver range, split at the peak (+ve = clockwise loop,
                       the abrupt-incursion signature).
    """
    out = []
    for key, sub in samples.sort(["eid", "elapsed_min"]).group_by(
        "eid", maintain_order=True
    ):
        d = sub[do_col or DO_COL].to_numpy().astype(float)
        _hyst = partial(hysteresis, d)
        n = len(d)
        peak = d.max() if n else 0.0
        rec = {"eid": key[0]}
        if n > 1 and peak > 0 and d.sum() > 0:
            t = np.arange(n)
            w = d / d.sum()
            rec["centroid_frac"] = float((w * t).sum() / (n - 1))
            rec["max_rise_norm"] = float(np.diff(d).max() / peak)
            rec["plateau_frac"] = float((d >= 0.8 * peak).mean())

            rec["hyst_wl"] = _hyst(sub[wl_col or WL_COL].to_numpy().astype(float))
            rec["hyst_sal"] = _hyst(sub[sal_col or SAL_COL].to_numpy().astype(float))
        else:
            rec.update(
                centroid_frac=None,
                max_rise_norm=None,
                plateau_frac=None,
                hyst_wl=None,
                hyst_sal=None,
            )
        out.append(rec)

    return pl.DataFrame(out)


def sync_expert_features(readouts, events):
    """Compute the event feature table on the EXPERT windows, reusing the exact
    shape + antecedent-driver features from exploratory.py (§ Section 3). Returns
    (events_feat, samples): `events_feat` mirrors derived/events.parquet's feature
    columns but carries real `expert_label`/`expert_subtype` instead of ws_label,
    and `samples` mirrors derived/event_samples.parquet. Only events with >=1
    in-range readout sample survive (coverage handled by the caller)."""
    samples = expert_event_samples(readouts, events)

    _ctx = readouts.sort("Datetime").select(
        "Datetime",
        pl.col(PRECIP_COL)
        .rolling_sum_by("Datetime", window_size="24h")
        .alias("precip_24h"),
        pl.col(PRECIP_COL)
        .rolling_sum_by("Datetime", window_size="72h")
        .alias("precip_72h"),
        pl.col(PRECIP_COL)
        .rolling_sum_by("Datetime", window_size="168h")
        .alias("precip_168h"),
        pl.col(SAL_COL).rolling_mean_by("Datetime", window_size="24h").alias("sal_pre"),
        pl.col(WL_COL).rolling_mean_by("Datetime", window_size="24h").alias("wl_pre"),
        pl.col(TEMP_COL)
        .rolling_mean_by("Datetime", window_size="24h")
        .alias("temp_pre"),
    )

    _shape = (
        samples.rename({"group_start": "start", "group_end": "end"})
        .group_by("eid")
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("start").first(),
            pl.col("end").first(),
            pl.col("event_id").first(),
            pl.col("expert_label").first(),
            pl.col("expert_subtype").first(),
            pl.col("is_public_augmented").first(),
            pl.col(DO_COL).max().alias("peak_do"),
            pl.col(DO_COL).mean().alias("mean_do"),
            pl.col(DO_COL).sum().alias("_sum_do"),
            pl.col(DO_COL).arg_max().cast(pl.Int64).alias("_argpeak"),
            pl.col(SAL_COL).mean().alias("sal_in"),
            pl.col(WL_COL).mean().alias("wl_in"),
            pl.col(TEMP_COL).mean().alias("temp_in"),
        )
        .sort("start")
        .with_columns(
            (pl.col("end") - pl.col("start")).dt.total_minutes().alias("dur_min"),
            (pl.col("_sum_do") * 5 / 60).alias("area_mgLh"),
            (pl.col("_argpeak") / (pl.col("n_samples") - 1).clip(lower_bound=1)).alias(
                "peak_frac"
            ),
            (pl.col("_argpeak") * 5).alias("rise_min"),
            ((pl.col("n_samples") - 1 - pl.col("_argpeak")) * 5).alias("fall_min"),
        )
        .with_columns(
            (
                pl.col("peak_do") / (pl.col("rise_min") / 60).clip(lower_bound=0.0833)
            ).alias("rise_rate"),
            (
                pl.col("peak_do") / (pl.col("fall_min") / 60).clip(lower_bound=0.0833)
            ).alias("fall_rate"),
        )
    )

    _extra = event_shape_features(samples)

    events_feat = (
        _shape.join_asof(
            _ctx, left_on="start", right_on="Datetime", strategy="backward"
        )
        .with_columns(
            (pl.col("sal_in") - pl.col("sal_pre")).alias("sal_step"),
            (pl.col("wl_in") - pl.col("wl_pre")).alias("wl_step"),
            (pl.col("temp_in") - pl.col("temp_pre")).alias("temp_step"),
        )
        .join(_extra, on="eid")
        .select(
            "eid",
            "event_id",
            "expert_label",
            "expert_subtype",
            "start",
            "end",
            "is_public_augmented",
            "dur_min",
            "n_samples",
            "peak_do",
            "mean_do",
            "area_mgLh",
            "peak_frac",
            "rise_min",
            "fall_min",
            "rise_rate",
            "fall_rate",
            "sal_in",
            "wl_in",
            "temp_in",
            "sal_step",
            "wl_step",
            "temp_step",
            "precip_24h",
            "precip_72h",
            "precip_168h",
            "centroid_frac",
            "max_rise_norm",
            "plateau_frac",
            "hyst_wl",
            "hyst_sal",
        )
        .sort("start")
    )
    return events_feat, samples


def agg_expert_moment(events):
    """Roll each umbrella's nested `moments.*` measurements up to one aggregate row per
    event, so the expert's own numbers can sit beside our computed event features for
    cross-checking / opt-in. Consumes the nested table from
    `core.io.read_expert_event_list`; all outputs are `moments_`-prefixed (plus
    `num_moments`) so they're easy to exclude from a model's X. Moment-less umbrellas
    (empty lists) give 0 counts / null stats."""
    return events.select(
        "event_id",
        pl.col("moments.idx").list.len().alias("num_moments"),
        pl.col("moments.peak_do").list.max().alias("moments_peak_do_max"),
        pl.col("moments.peak_do").list.mean().alias("moments_peak_do_mean"),
        pl.col("moments.sal_at_peak").list.mean().alias("moments_sal_at_peak_mean"),
        pl.col("moments.sal_at_peak").list.max().alias("moments_sal_at_peak_max"),
        pl.col("moments.do_rise_pct").list.mean().alias("moments_do_rise_pct_mean"),
        pl.col("moments.consumption_rate")
        .list.mean()
        .alias("moments_consumption_rate_mean"),
        pl.col("moments.min_to_rise").list.mean().alias("moments_rise_min_mean"),
        pl.col("moments.min_to_fall").list.mean().alias("moments_fall_min_mean"),
        pl.col("moments.duration_min").list.sum().alias("moments_duration_min_sum"),
    )


def enrich_expert_events(feature_events, events):
    """Attach the expert-recorded columns to our computed event feature table, keyed on
    `event_id`, from the nested `core.io.read_expert_event_list` table: the umbrella-level
    count/flag columns (`n_hot_moments`/`n_oxic_pulses`/`xf_qc_checked`/`flooding`/
    `concurrent_precip`) and the per-event `moments_` moment aggregates. Everything
    expert-derived is a raw expert flag/count or `moments_`-prefixed — feed a model only
    after excluding them (they are the label-adjacent expert judgment); use them to
    validate the computed features or opt specific ones in deliberately."""
    meta = events.select(
        "event_id",
        "n_hot_moments",
        "n_oxic_pulses",
        "xf_qc_checked",
        "flooding",
        "concurrent_precip",
    )
    return feature_events.join(meta, on="event_id", how="left").join(
        agg_expert_moment(events), on="event_id", how="left"
    )


def subpeak_features(
    samples, do_col="Dissolved Oxygen (mg/L)", prom_frac=0.1, cadence_min=5
):
    """Sub-peak structure of each excursion's DO curve, one row per eid. Captures
    the multi-peak *tidal complexity* of hot moments vs the single-humped oxic
    pulse, WITHOUT switching the datapoint from the excursion to the moment (which
    collapses class balance, hot fans out into many tidal sub-peaks while the pulse
    stays atomic):
      n_subpeaks          - count of prominent local DO maxima (prominence
                            >= prom_frac * peak); >=1 whenever any O2 rise exists
      subpeak_spacing_min - median spacing between consecutive sub-peaks in minutes
                            (0 if < 2 peaks; ~semidiurnal tidal beat when hot)
      sec_prom_frac       - 2nd-largest peak prominence / peak height (0 if < 2;
                            how strong the secondary structure is relative to the top)

    TESTED 2026-07-06 and deliberately NOT in FEATURE_COLS: at the excursion
    granularity these are near-constant across classes (median 1 sub-peak for both
    hot and pulse) because `detect_events` already segments at DO-baseline returns,
    so the tidal multi-peak structure lives *between* excursions (many per umbrella),
    not *within* one. Adding them nudged every model down within noise (Logistic-L2
    .800->.793, CatBoost .799->.787, XGBoost .771->.760). Kept for the EDA/writeup
    of that negative result, not for training.
    """
    out = []
    for key, sub in samples.sort(["eid", "elapsed_min"]).group_by(
        "eid", maintain_order=True
    ):
        d = sub[do_col].to_numpy().astype(float)
        n = len(d)
        peak = d.max() if n else 0.0
        rec = {
            "eid": key[0],
            "n_subpeaks": 0,
            "subpeak_spacing_min": 0.0,
            "sec_prom_frac": 0.0,
        }
        if n >= 3 and peak > 0:
            # light 3-sample smoothing suppresses single-sample sensor noise so
            # prominence counts real sub-peaks, not jitter
            ds = np.convolve(d, np.ones(3) / 3, mode="same")
            idx, props = find_peaks(ds, prominence=max(prom_frac * peak, 1e-6))
            if idx.size == 0:
                # peak sits at a window boundary (monotone-ish rise/fall) => 1 hump
                rec["n_subpeaks"] = 1
            else:
                rec["n_subpeaks"] = int(idx.size)
                if idx.size >= 2:
                    spac = np.diff(np.sort(idx)) * cadence_min
                    rec["subpeak_spacing_min"] = float(np.median(spac))
                    rec["sec_prom_frac"] = float(
                        np.sort(props["prominences"])[-2] / peak
                    )
        out.append(rec)

    return pl.DataFrame(out)
