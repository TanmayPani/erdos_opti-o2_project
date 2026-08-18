from collections import defaultdict
from typing import Callable
from functools import partial

import numpy as np
import polars as pl

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


def centroid_index(do):
    # degenerate curve (single sample or all-zero DO) -> null, not a NaN: a float NaN
    # slips past polars `fill_null` downstream and crashes NaN-intolerant estimators.
    if len(do) < 2 or do.sum() == 0:
        return None
    time_idx = np.arange(len(do))
    wsum = float(np.dot(do, time_idx) / do.sum())
    return wsum / float(len(do) - 1)


def max_delta_scaled(do):
    if len(do) < 2 or do.max() == 0:
        return None
    return float(np.diff(do).max() / do.max())


def plateau_scaled(do):
    if len(do) < 1:
        return None
    peak = do.max()
    return float((do >= (0.8 * peak)).mean())


def finalize(features, units):
    _grp = units.select("unit_id", "event_id")
    out = (
        features.drop("eid")
        .join(_grp, on="unit_id", how="left")
        .with_columns(
            pl.when(pl.col("source") == "orphan")
            .then(pl.lit("inferred"))
            .otherwise(pl.lit("expert"))
            .alias("label_source"),
            pl.lit("train").alias("split"),
        )
        .with_columns(
            # `mixed` is a FIRST-CLASS target as of the rev 07-31-26 workbook, which promoted
            # it from the ambiguous `hx`/`e` codes to an explicit class with its own column
            # block (39 moments / 13 events — see core.io._MOMENT_BLOCKS). It used to be
            # nulled and held out; now only ORPHANS (auto-detected excursions with no expert
            # umbrella, hence no true label at all) stay out of training.
            pl.when(pl.col("source") == "orphan")
            .then(pl.lit("holdout"))
            .otherwise(pl.lit("train"))
            .alias("split"),
            pl.col("expert_label").alias("label"),
        )
    )
    lead = [
        "unit_id",
        "event_id",
        "label_source",
        "expert_subtype",
        "expert_label",
        "label",
        "split",
        "start_time",
        "end_time",
        "is_public_augmented",
    ]
    return out.select(lead + FEATURE_COLS).sort(["event_id", "start_time"])


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
    ro = readouts.sort("Datetime")
    tvec = ro["Datetime"].to_numpy().astype("datetime64[s]").astype(np.int64)
    arrs = {c: ro[c].to_numpy().astype(float) for c in CURVE_CHANNELS}
    keep = proc.select(
        "unit_id",
        "start_time",
        "end_time",
        "event_id",
        "expert_label",
        "label",
        "split",
    )
    blocks = []
    for row in keep.iter_rows(named=True):
        s = np.datetime64(row["start_time"]).astype("datetime64[s]").astype(np.int64)
        e = np.datetime64(row["end_time"]).astype("datetime64[s]").astype(np.int64)
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
        for k in ("event_id", "expert_label", "label", "split"):
            blk[k] = pl.Series([row[k]] * T, dtype=pl.String)
        blocks.append(pl.DataFrame(blk))
    return pl.concat(blocks).join(
        proc.select("unit_id", "is_public_augmented"),
        on="unit_id",
        how="left",
    )


def auto_detected_events(readouts, peak_min=0.1, merge_gap=12):
    """Auto-detected events (existing `detect_events`) with per-event peak DO, kept
    above `peak_min` — the `exploratory.py` selection, applied to the NDA readouts.
    Returns eid/peak_do/start/end; used both for the reconciliation and to delineate,
    on each expert-event plot, where our detector would splice the window."""

    f = readouts.select("Datetime", (pl.col(DO_COL) > 0).alias("_oxic"))
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
    auto = (
        runs.filter(pl.col("_om"))
        .group_by("eid")
        .agg(pl.col("start").min(), pl.col("end").max())
        .sort("start")
    )

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


def build_auto_units(readouts, umbrella, **kwargs):
    auto = auto_detected_events(readouts, **kwargs)
    umb = umbrella.with_columns(
        pl.col("start_time").alias("u_gs"),
        pl.col("end_time").alias("u_ge"),
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
    auto_units = best.with_columns(
        pl.concat_str([pl.lit("a"), pl.col("eid").cast(pl.Utf8)]).alias("unit_id"),
        pl.lit("auto").alias("source"),
        pl.col("start").alias("start_time"),
        pl.col("end").alias("end_time"),
    )

    claimed = best["event_id"].implode()
    missed = umb.filter(~pl.col("event_id").is_in(claimed))
    fb_units = missed.with_columns(
        pl.concat_str([pl.lit("f_"), pl.col("event_id")]).alias("unit_id"),
        pl.lit("fallback").alias("source"),
        pl.col("u_gs").alias("start_time"),
        pl.col("u_ge").alias("end_time"),
    )
    # Project both onto the canonical event schema before concat: `auto_units` still carries
    # the detector's internal columns (eid/peak_do/start/end/_ov/u_gs/u_ge) that `fb_units`
    # lacks, so a raw concat is width-mismatched; the stray `eid` would also collide with
    # `event_samples`'s own `with_row_index("eid")` downstream.
    _cols = [
        "unit_id",
        "event_id",
        "source",
        "start_time",
        "end_time",
        "expert_label",
        "expert_subtype",
    ]
    units = pl.concat([auto_units.select(_cols), fb_units.select(_cols)]).sort(
        "start_time"
    )

    return units


def event_samples(readouts, events):
    win = (
        events.sort("start_time")
        .with_row_index("eid")
        .with_columns(pl.col("eid").cast(pl.UInt32))
    )
    return (
        readouts.sort("Datetime")
        .join_asof(
            win.select(
                "eid",
                "start_time",
                "end_time",
                "event_id",
                "unit_id",
                "source",
                "expert_label",
                "expert_subtype",
            ),
            left_on="Datetime",
            right_on="start_time",
            strategy="backward",
        )
        .filter(pl.col("Datetime") <= pl.col("end_time"))
        .with_columns(
            (pl.col("Datetime") - pl.col("start_time"))
            .dt.total_minutes()
            .alias("elapsed_min")
        )
    )


def splice_readouts(readouts, events=None, **kwargs):
    events = events if events is not None else build_auto_units(readouts, **kwargs)
    samples = event_samples(readouts, events)

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
        samples.group_by("eid")
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("start_time").first(),
            pl.col("end_time").first(),
            pl.col("unit_id").first(),
            pl.col("event_id").first(),
            pl.col("expert_label").first(),
            pl.col("expert_subtype").first(),
            pl.col("source").first(),
            pl.col("is_public_augmented").first(),
            pl.col(DO_COL).max().alias("peak_do"),
            pl.col(DO_COL).mean().alias("mean_do"),
            pl.col(DO_COL).sum().alias("_sum_do"),
            pl.col(DO_COL).arg_max().cast(pl.Int64).alias("_argpeak"),
            pl.col(SAL_COL).mean().alias("sal_in"),
            pl.col(WL_COL).mean().alias("wl_in"),
            pl.col(TEMP_COL).mean().alias("temp_in"),
        )
        .sort("start_time")
        .with_columns(
            (pl.col("end_time") - pl.col("start_time"))
            .dt.total_minutes()
            .alias("dur_min"),
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

    _extra = defaultdict(list)
    _fn_dict: dict[str, Callable[[np.ndarray], float]] = dict(
        centroid_frac=centroid_index,
        max_rise_norm=max_delta_scaled,
        plateau_frac=plateau_scaled,
    )

    for key, sub in samples.sort(["eid", "elapsed_min"]).group_by(
        "eid", maintain_order=True
    ):
        d = sub[DO_COL].to_numpy().astype(float)

        _fn_dict.update(
            hyst_wl=partial(hysteresis, drv=sub[WL_COL].to_numpy().astype(float)),
            hyst_sal=partial(hysteresis, drv=sub[SAL_COL].to_numpy().astype(float)),
        )
        _extra["eid"].append(key[0])
        for _k, _fn in _fn_dict.items():
            _extra[_k].append(_fn(d))

    _columns = [
        "eid",
        "event_id",
        "unit_id",
        "expert_label",
        "expert_subtype",
        "start_time",
        "end_time",
        "source",
        "is_public_augmented",
    ] + FEATURE_COLS
    events_feat = (
        _shape.join_asof(
            _ctx, left_on="start_time", right_on="Datetime", strategy="backward"
        )
        .with_columns(
            (pl.col("sal_in") - pl.col("sal_pre")).alias("sal_step"),
            (pl.col("wl_in") - pl.col("wl_pre")).alias("wl_step"),
            (pl.col("temp_in") - pl.col("temp_pre")).alias("temp_step"),
        )
        .join(pl.DataFrame(_extra), on="eid")
        .select(*_columns)
        .sort("start_time")
    )

    finalized = finalize(events_feat, events)
    return finalized, build_curves(readouts, finalized)
