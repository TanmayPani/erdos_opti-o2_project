import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    from datetime import datetime

    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt

    from core.features import (
        auto_detected_events,
        DO_COL,
        SAL_COL,
        WL_COL,
        PRECIP_COL,
    )

    from utils import feature_methodology_tabs

    alt.data_transformers.disable_max_rows()
    # Vega SVG renderer for every chart here — a default <canvas> initialises 0x0 inside a
    # hidden mo.ui.tabs panel / off-screen reveal slide and renders blank. marimo merges this
    # into each chart's usermeta.embedOptions; survives static HTML export. See modeling.py.
    _ = alt.renderers.set_embed_options(renderer="svg")  # `_ =`: suppress the repr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exploratory views — Beaver Creek DO events (NDA)

    The **preprocessing pipeline** lives in `preprocessing.py` and writes the
    model-ready `derived/processed_auto_features.parquet` + `derived/processed_auto_curves.parquet`
    plus the reference tables. This notebook holds the **exploratory visualisations**
    moved out of that pipeline: how the expert classes separate in the physics
    features, an interactive per-event browser with our detector's splice track, an
    **event/onset-level detection scorecard** (precision / recall / F1 / IoU / lead-time),
    and the three-level granularity reconciliation. It reads the reference series from
    `derived/readouts.parquet` and re-derives the light tables from the event workbook.
    """)
    return


@app.cell
def reading():
    events = pl.read_parquet("derived/expert_event_list.parquet")
    readouts = pl.read_parquet("derived/readouts.parquet")
    proc = pl.read_parquet("derived/processed_auto_features.parquet")
    _b = proc.group_by("split", "expert_label").len().sort("split", "expert_label")

    _r0, _r1 = readouts["Datetime"].min(), readouts["Datetime"].max()
    _in_range = (pl.col("end_time") >= _r0) & (pl.col("start_time") <= _r1)
    events_covered = events.filter(_in_range)

    expert_moments = pl.read_parquet("derived/expert_moment_list.parquet")
    expert_events_full = pl.read_parquet("derived/processed_expert_features.parquet")
    expert_samples = pl.read_parquet("derived/expert_samples.parquet")

    auto_events = auto_detected_events(readouts)
    return (
        auto_events,
        events,
        events_covered,
        expert_events_full,
        expert_moments,
        expert_samples,
        proc,
        readouts,
    )


@app.cell
def ui_elements(events, events_covered):
    _axis_opts = [
        "dur_min",
        "n_samples",
        "peak_do",
        "peak_frac",
        "centroid_frac",
        "rise_min",
        "fall_min",
        "rise_rate",
        "fall_rate",
        "sal_step",
        "wl_step",
        "temp_step",
        "max_rise_norm",
        "plateau_frac",
    ]
    x_var = mo.ui.dropdown(options=_axis_opts, value="sal_step", label="x axis")
    y_var = mo.ui.dropdown(options=_axis_opts, value="wl_step", label="y axis")

    _cov_ids = set(events_covered["event_id"])
    # picker: expert umbrellas, ordered by OCCURRENCE (time).
    _rows = [
        (
            r["start_time"],
            r["event_id"],
            f"{r['event_id']}  —  {r['expert_label']} ({r['expert_subtype']})"
            + ("" if r["event_id"] in _cov_ids else "   ⚠ no coverage"),
        )
        for r in events.iter_rows(named=True)
    ]
    _rows.sort(key=lambda t: t[0])
    _opts = {_lbl: _eid for _, _eid, _lbl in _rows}
    _default = next(_lbl for _, _eid, _lbl in _rows if _eid in _cov_ids)
    event_picker = mo.ui.dropdown(options=_opts, value=_default, label="Event")
    shade_mode = mo.ui.radio(
        options=["excursions", "moments"],
        value="excursions",
        inline=True,
        label="Shade",
    )
    return event_picker, shade_mode, x_var, y_var


@app.cell(hide_code=True)
def event_detection(
    auto_events,
    event_picker,
    events,
    expert_events_full,
    expert_moments,
    expert_samples,
    shade_mode,
):
    # Interactive event viewer as a SINGLE Altair/Vega spec: the **event picker** and the
    # **shading toggle** are client-side Vega param bindings (`eidSel` / `shadeSel`), so the
    # viewer stays interactive in a STATIC html export with no Python kernel behind it.
    # Every event's traces, splice segments, moment windows, peak rules and summary card are
    # embedded once and filtered client-side. The marimo widgets only SEED the defaults.
    # Only events with readout coverage are offered (the uncovered umbrellas have no series
    # to plot, so they can't be rendered client-side).
    # Every event's series has to be embedded at once, so it is bin-downsampled to at most
    # _NPTS points per event (chart is 720 px wide; events shorter than that keep full 5-min
    # resolution). Levels are averaged within a bin, precipitation is SUMMED so rainfall
    # totals survive the reduction. Without this the cell output is ~20 MB and marimo drops
    # it wholesale (8 MB per-output cap).
    _NPTS = 300
    _series = (
        expert_samples.sort("event_id", "Datetime")
        .select(
            "event_id",
            "Datetime",
            pl.col("Dissolved Oxygen (mg/L)").alias("do"),
            pl.col("Well Salinity (PPT)").alias("sal"),
            pl.col("Flood plain water level in BGS (cm)").alias("wl"),
            pl.col("Precip (mm) over 5 minutes").alias("precip"),
        )
        .with_columns(
            (
                pl.int_range(pl.len()).over("event_id")
                * _NPTS
                // pl.len().over("event_id")
            ).alias("_bin")
        )
        .group_by("event_id", "_bin")
        .agg(
            pl.col("Datetime").min(),
            pl.col("do").mean().round(3),
            pl.col("sal").mean().round(3),
            pl.col("wl").mean().round(2),
            pl.col("precip").sum().round(2),
        )
        .sort("event_id", "_bin")
        .drop("_bin")
    )
    _win = _series.group_by("event_id").agg(
        pl.col("Datetime").min().alias("w0"), pl.col("Datetime").max().alias("w1")
    )
    _covered = _win.select("event_id")

    # our detector's auto-events overlapping each window, clipped to it (one row per segment)
    _splice = (
        _win.join(auto_events, how="cross")
        .filter((pl.col("start") <= pl.col("w1")) & (pl.col("end") >= pl.col("w0")))
        .with_columns(
            pl.max_horizontal("start", "w0").alias("cstart"),
            pl.min_horizontal("end", "w1").alias("cend"),
        )
        .sort("event_id", "cstart")
        .with_columns(pl.int_range(1, pl.len() + 1).over("event_id").alias("seg_no"))
        .select("event_id", "cstart", "cend", "seg_no")
    )
    # expert moment windows (empty for orphans), clipped to the plotted window
    _mrects = (
        expert_moments.select(
            "event_id",
            pl.col("moment_idx").alias("m_no"),
            pl.col("type").alias("m_type"),
            pl.col("start_time").alias("m_start"),
            pl.col("end_time").alias("m_end"),
        )
        .drop_nulls(["m_start", "m_end"])
        .join(_win, on="event_id")
        .with_columns(
            pl.max_horizontal("m_start", "w0").alias("m_start"),
            pl.min_horizontal("m_end", "w1").alias("m_end"),
        )
        .select("event_id", "m_no", "m_type", "m_start", "m_end")
    )
    # Peak-DO rules, clamped to the plotted window. All seven layers share ONE temporal x
    # scale, so a peak timestamp outside the window would stretch the domain and squeeze the
    # event into an invisible sliver. `core.io` already nulls out-of-window peaks at the
    # source; this keeps the viewer safe against any future stray value.
    _peaks = (
        expert_moments.select("event_id", pl.col("m_peak_do_ts").alias("Datetime"))
        .drop_nulls()
        .join(_win, on="event_id")
        .filter(pl.col("Datetime").is_between(pl.col("w0"), pl.col("w1")))
        .select("event_id", "Datetime")
    )

    def _fmt(v, f="{:.2f}"):
        return f.format(v) if v is not None else "—"

    _event_detection_snip = mo.md(
        """
        #Event Detection

        - **Moments (expert-labeled):** Expert detected events based on hydrology (\\(DO_2\\), flooding, tides, precipitation, etc.) and each event split into distinct hot moments (tidal) and oxic pulses (freshwater).
        - **Excursions (auto-detected):** Sustained \\(DO_2\\) departure from anoxic baseline \\(DO_2 = 0\\) mg/L.
        """
    )

    # workbook LEGEND ("hot moment and oxic pulse classification.xlsx") decoded to the
    # fine suffixes used in the moments list (o1..o5 in the sheet's listed order).
    _SUBTYPE_MEANING = {
        "h": "hot moment — tidal DO incursion (water-level rise + salinity step)",
        "b": "unclassified hot moment (held out)",
        "hx": "mixed — a symmetric pulse coincident with a salinity step",
        "e": "oxic event during a flood",
        "o1": "oxic pulse — no change in salinity",
        "o2": "oxic pulse — adiabatic change in salinity",
        "o3": "oxic pulse — prior to flooding",
        "o4": "oxic pulse — during / following flooding",
        "o5": "oxic pulse — coincident with rainfall",
    }

    # Per-event summary card, as a tidy [event_id, ord, field, value] frame that a pair of
    # text marks renders — markdown can't be param-driven, but text marks can.
    _meta = (
        expert_events_full.join(_covered, on="event_id")
        .join(
            events.select("event_id", "flooding").unique(subset=["event_id"]),
            on="event_id",
            how="left",
        )
        .join(
            _splice.group_by("event_id").len().rename({"len": "n_seg"}),
            on="event_id",
            how="left",
        )
        .sort("start_time")
    )
    _card_rows = []
    for _r in _meta.iter_rows(named=True):
        _code = _r["expert_subtype"]
        _fields = [
            ("event", _r["event_id"]),
            ("class", f"{_r['expert_label']}  ({_code})"),
            (
                "code means",
                _SUBTYPE_MEANING.get(_code, _SUBTYPE_MEANING.get(_code[:1], "—")),
            ),
            (
                "window",
                f"{_r['start_time']:%Y-%m-%d %H:%M} → {_r['end_time']:%Y-%m-%d %H:%M}",
            ),
            ("duration", f"{_r['dur_min'] / 60:.1f} h"),
            ("peak DO", f"{_fmt(_r['peak_do'])} mg/L"),
            ("salinity step", _fmt(_r["sal_step"], "{:+.2f}")),
            ("water-level step", _fmt(_r["wl_step"], "{:+.2f}")),
            ("antecedent precip (24 h)", f"{_fmt(_r['precip_24h'], '{:.1f}')} mm"),
            ("flooding", str(_r["flooding"])),
            ("DO excursions", f"{_r['n_seg'] or 0} segment(s)"),
        ]
        _card_rows += [
            {"event_id": _r["event_id"], "ord": _i, "field": _k, "value": str(_v)}
            for _i, (_k, _v) in enumerate(_fields)
        ]
    _card_df = pl.DataFrame(_card_rows)

    # --- controls as Vega params (kernel-free) -------------------------------------
    _MODES = ["excursions", "moments"]  # must match ui_elements' shade_mode radio
    _eids = _meta["event_id"].to_list()
    _labels = [
        f"{_r['event_id']}  —  {_r['expert_label']} ({_r['expert_subtype']})"
        for _r in _meta.iter_rows(named=True)
    ]
    _eid_default = event_picker.value if event_picker.value in _eids else _eids[0]
    _shade_default = shade_mode.value if shade_mode.value in _MODES else _MODES[0]

    _eidSel = alt.param(
        name="eidSel",
        value=_eid_default,
        bind=alt.binding_select(options=_eids, labels=_labels, name="Event "),
    )
    _shadeSel = alt.param(
        name="shadeSel",
        value=_shade_default,
        bind=alt.binding_radio(options=_MODES, name="Shade "),
    )
    _PICK = "datum.event_id === eidSel"

    # --- Dr. Ghosh's multi-axis event format (deck slides 10-14): one panel, four
    # colour-matched y-axes stacked via Axis(offset=...) on layers with independent
    # y-scales. DO (white, left), precipitation (pink bars, far left), water depth
    # below ground (blue, right, axis REVERSED so a rising water table / flood reads
    # as up), salinity (green, far right).
    _C_DO, _C_PR, _C_WL, _C_SAL = "#f0f4f7", "#e377c2", "#1f77b4", "#2ca02c"

    def _yaxis(_t, _color, _orient, _offset):
        return alt.Axis(
            title=_t,
            orient=_orient,
            offset=_offset,
            titleColor=_color,
            labelColor=_color,
            tickColor=_color,
            domainColor=_color,
        )

    _bx = (
        alt.Chart(_series)
        .transform_filter(_PICK)
        .encode(x=alt.X("Datetime:T", title="local time"))
    )
    _l_pr = _bx.mark_bar(color=_C_PR, size=1.5, opacity=0.6).encode(
        y=alt.Y(
            "precip:Q",
            scale=alt.Scale(zero=True),
            axis=_yaxis("Precip (mm/5min)", _C_PR, "left", 46),
        )
    )
    _l_wl = _bx.mark_line(color=_C_WL, strokeWidth=1.6).encode(
        y=alt.Y(
            "wl:Q",
            scale=alt.Scale(zero=False, reverse=True),
            axis=_yaxis("Water depth below ground surface (cm)", _C_WL, "right", 0),
        )
    )
    _l_sal = _bx.mark_line(color=_C_SAL, strokeWidth=1.6).encode(
        y=alt.Y(
            "sal:Q",
            scale=alt.Scale(zero=False),
            axis=_yaxis("Salinity (PPT)", _C_SAL, "right", 54),
        )
    )
    _l_do = _bx.mark_line(color=_C_DO, strokeWidth=2).encode(
        y=alt.Y(
            "do:Q", scale=alt.Scale(zero=True), axis=_yaxis("DO (mg/L)", _C_DO, "left", 0)
        )
    )
    # SHADING (drawn BEHIND the traces) — BOTH variants are embedded and the `shadeSel`
    # param picks one client-side:
    #   "excursions" → our detector's segments (one rect per segment);
    #   "moments"    → the expert moment windows, coloured hot vs oxic.
    _r_auto = (
        alt.Chart(_splice)
        .transform_filter(f"{_PICK} && shadeSel === '{_MODES[0]}'")
        .mark_rect(opacity=0.14)
        .encode(
            x="cstart:T",
            x2="cend:T",
            color=alt.Color(
                "seg_no:N", legend=None, scale=alt.Scale(scheme="category10")
            ),
            tooltip=[
                alt.Tooltip("seg_no:N", title="our event #"),
                alt.Tooltip("cstart:T", title="start"),
                alt.Tooltip("cend:T", title="end"),
            ],
        )
    )
    _r_moments = (
        alt.Chart(_mrects)
        .transform_filter(f"{_PICK} && shadeSel === '{_MODES[1]}'")
        .mark_rect(opacity=0.16)
        .encode(
            x="m_start:T",
            x2="m_end:T",
            color=alt.Color(
                "m_type:N",
                title="moment",
                scale=alt.Scale(
                    domain=["hot", "oxic", "mixed"],
                    range=["#d62728", "#1f77b4", "#9467bd"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("m_no:N", title="moment #"),
                alt.Tooltip("m_type:N", title="type"),
                alt.Tooltip("m_start:T", title="start"),
                alt.Tooltip("m_end:T", title="end"),
            ],
        )
    )
    _r_pk = (
        alt.Chart(_peaks)
        .transform_filter(_PICK)
        .mark_rule(color="#d62728", strokeDash=[3, 3], opacity=0.55)
        .encode(x="Datetime:T")
    )
    _chart = (
        alt.layer(_r_auto, _r_moments, _l_pr, _l_wl, _l_sal, _l_do, _r_pk)
        .resolve_scale(y="independent", color="independent")
        .properties(width=720, height=440)
    )

    # Summary card: two text marks (label / value) sharing an ordinal row scale.
    _cbase = alt.Chart(_card_df).transform_filter(_PICK)
    _crow = alt.Y("ord:O", axis=None, sort="ascending")
    _card = alt.layer(
        _cbase.mark_text(align="right", dx=-8, fontSize=11, color="#8b959e").encode(
            y=_crow, x=alt.value(160), text="field:N"
        ),
        _cbase.mark_text(align="left", dx=8, fontSize=11).encode(
            y=_crow, x=alt.value(160), text="value:N"
        ),
    ).properties(width=470, height=250, title="Selected event")

    _view = (
        alt.hconcat(_card, _chart)
        .resolve_scale(x="independent", y="independent", color="independent")
        .add_params(_eidSel, _shadeSel)
    )
    mo.vstack([_event_detection_snip, _view])
    return


@app.cell
def label_survey():
    _mom_features = pl.read_parquet(
        "derived/processed_moments_features.parquet"
    ).with_columns(pl.col("label").replace({"pulse": "oxic"}).alias("class"))
    _drift = (
        _mom_features.with_columns(pl.col("start_time").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("n_events"),
            (pl.col("class") == "hot").mean().round(3).alias("hot_fraction"),
        )
        .sort("year")
    )
    _trend_yr_bar = (
        alt.Chart(_drift)
        .mark_bar(color="#c44e52")
        .encode(
            alt.X("year:O", title="year"),
            alt.Y(
                "hot_fraction:Q",
                title="hot-moment fraction",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["year:O", "n_events:Q", "hot_fraction:Q"],
        )
        .properties(
            width=480,
            height=280,
            title="Hot-moment fraction per year",
        )
    )

    _trend_yr = _trend_yr_bar.transform_regression("year", "hot_fraction").mark_line(
        color="#f0f4f7", strokeDash=[4, 3]
    )

    _SEASONS = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]
    _ev = _mom_features.filter(pl.col("class").is_in(["hot", "oxic"])).with_columns(
        ((pl.col("start_time").dt.month() % 12) // 3).alias("_sidx"),
        pl.col("start_time").dt.month().alias("month"),
    )
    _by_season = (
        _ev.group_by("_sidx", "label")
        .len(name="count")
        .with_columns(
            pl.col("_sidx")
            .replace_strict(
                {0: _SEASONS[0], 1: _SEASONS[1], 2: _SEASONS[2], 3: _SEASONS[3]}
            )
            .alias("season")
        )
        .sort("_sidx", "label")
    )

    # H1 read-out: pulse share in the cool half (Oct–Mar) vs the warm half (Apr–Sep)
    _cool = _ev.filter(pl.col("month").is_in([10, 11, 12, 1, 2, 3]))
    _warm = _ev.filter(pl.col("month").is_in([4, 5, 6, 7, 8, 9]))
    _pulse_cool = _cool.filter(pl.col("label") == "pulse").height / max(_cool.height, 1)
    _pulse_warm = _warm.filter(pl.col("label") == "pulse").height / max(_warm.height, 1)

    _trend_szn = (
        alt.Chart(_by_season)
        .mark_bar()
        .encode(
            x=alt.X("season:N", sort=_SEASONS, title=None),
            xOffset="label:N",
            y=alt.Y("count:Q", title="expert events"),
            color=alt.Color("label:N", title="expert label"),
            tooltip=["season:N", "label:N", "count:Q"],
        )
        .properties(width=400, height=300, title="Event type by meteorological season")
    )

    mo.vstack(
        [
            mo.md("# Label Imbalance"),
            mo.hstack(
                [
                    _trend_szn,
                    _trend_yr_bar + _trend_yr,
                    mo.md("""
                    - Label imbalance has seasonal and year-by-year trends
                    - Hot moments dominate in summer (less freshwater incursions/flooding)
                    - Oxic pulses dominate in winter (less saline incursion, more freshwater/flooding)
                    - Hot moments frequency increasing with years (sea-level rise?)
                      """),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def label_survey_1(proc, x_var, y_var):
    _xv, _yv = x_var.value, y_var.value
    _scatter = (
        alt.Chart(proc)
        .mark_circle(size=90, opacity=0.75)
        .encode(
            x=alt.X(f"{_xv}:Q", title=_xv),
            y=alt.Y(f"{_yv}:Q", title=_yv),
            color=alt.Color("regime:N", title="expert label"),
            size=alt.Size(
                "peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[30, 400])
            ),
            tooltip=["regime:N", "event_id:N", "start:T", "end:T", "peak_do:Q"],
        )
        .properties(
            width=500, height=300, title="Expert classes separation in feature space"
        )
    )

    _xy_var = mo.hstack([x_var, y_var], justify="start")

    mo.vstack([_xy_var, _scatter])
    return


@app.cell(hide_code=True)
def record_context_intro():
    mo.md(r"""
    ## Multi-scale record context

    Before the event-level views, the raw 5-minute record (2019 → 2025, the augmented NDA
    series) at two scales: the **full record** of daily means (long-term + seasonal
    structure), and a **periodogram** of air temperature placing the diurnal, semi-diurnal
    (tidal), synoptic and annual scales on one axis. That clean scale separation is what
    lets `forecast.py` model the tidal band with deterministic harmonics.
    """)
    return


@app.cell(hide_code=True)
def record_overview(events, readouts):
    _AIRT = "AirT_C_Avg | Deg C | Avg"
    _r0, _r1 = readouts["Datetime"].min(), readouts["Datetime"].max()
    _daily = (
        readouts.group_by_dynamic("Datetime", every="1d")
        .agg(
            pl.col(DO_COL).mean().alias("do"),
            pl.col(_AIRT).mean().alias("air"),
            pl.col(SAL_COL).mean().alias("sal"),
            pl.col(WL_COL).mean().alias("wl"),
        )
        .sort("Datetime")
    )

    # expert-labelled events falling outside the readout span, at each end
    _before = events.filter(pl.col("end_time") < _r0)
    _after = events.filter(pl.col("start_time") > _r1)
    _gap_rows = []
    if _before.height:
        _lo = _before["start_time"].min()
        _gap_rows.append(
            {
                "x": _lo,
                "x2": _r0,
                "mid": _lo + (_r0 - _lo) / 2,
                "dur": f"{(_r0 - _lo).days} d",
            }
        )
    if _after.height:
        _hi = _after["end_time"].max()
        _gap_rows.append(
            {
                "x": _r1,
                "x2": _hi,
                "mid": _r1 + (_hi - _r1) / 2,
                "dur": f"{(_hi - _r1).days} d",
            }
        )
    _gaps = pl.DataFrame(_gap_rows)
    _dom_lo = min([r["x"] for r in _gap_rows] + [_r0])
    _dom_hi = max([r["x2"] for r in _gap_rows] + [_r1])
    _xs = alt.Scale(domain=[_dom_lo, _dom_hi])

    # water-year demarcations (WY = Oct 1 -> Sep 30, labelled by the ending year)
    _wy_bounds, _wy_bands = [], []
    for _end in range(_dom_lo.year, _dom_hi.year + 2):
        _ws, _we = datetime(_end - 1, 10, 1), datetime(_end, 10, 1)
        if _dom_lo < _we and _ws < _dom_hi:
            _cs, _ce = max(_ws, _dom_lo), min(_we, _dom_hi)
            _wy_bands.append({"mid": _cs + (_ce - _cs) / 2, "label": f"WY{_end}"})
        if _dom_lo <= _ws <= _dom_hi:
            _wy_bounds.append(_ws)
    _wy_rules = (
        alt.Chart(pl.DataFrame({"b": _wy_bounds}))
        .mark_rule(color="#c9c9c9")
        .encode(x=alt.X("b:T", scale=_xs))
    )
    _wy_lab = (
        alt.Chart(pl.DataFrame(_wy_bands))
        .mark_text(color="#8a8a8a", fontSize=11, fontWeight="bold", baseline="top")
        .encode(x=alt.X("mid:T", scale=_xs), y=alt.value(3), text="label:N")
    )

    _C_DO, _C_AIR, _C_SAL, _C_WL = "#f0f4f7", "#e6842a", "#2ca02c", "#1f77b4"

    def _yax(_t, _c, _o, _off):
        return alt.Axis(
            title=_t,
            orient=_o,
            offset=_off,
            titleColor=_c,
            labelColor=_c,
            tickColor=_c,
            domainColor=_c,
        )

    _bx = alt.Chart(_daily).encode(x=alt.X("Datetime:T", title="date", scale=_xs))
    _l_air = _bx.mark_line(color=_C_AIR, strokeWidth=1, opacity=0.85).encode(
        y=alt.Y(
            "air:Q",
            scale=alt.Scale(zero=False),
            axis=_yax("Air temp (°C)", _C_AIR, "left", 48),
        )
    )
    _l_sal = _bx.mark_line(color=_C_SAL, strokeWidth=1, opacity=0.9).encode(
        y=alt.Y(
            "sal:Q",
            scale=alt.Scale(zero=False),
            axis=_yax("Salinity (PPT)", _C_SAL, "right", 0),
        )
    )
    _l_wl = _bx.mark_line(color=_C_WL, strokeWidth=1, opacity=0.9).encode(
        y=alt.Y(
            "wl:Q",
            scale=alt.Scale(zero=False, reverse=True),
            axis=_yax("Water level (cm below ground)", _C_WL, "right", 52),
        )
    )
    _l_do = _bx.mark_line(color=_C_DO, strokeWidth=1.3).encode(
        y=alt.Y(
            "do:Q", scale=alt.Scale(zero=True), axis=_yax("DO (mg/L)", _C_DO, "left", 0)
        )
    )

    _rects = (
        alt.Chart(_gaps)
        .mark_rect(color="#9a9a9a", opacity=0.18)
        .encode(x=alt.X("x:T", scale=_xs), x2="x2:T")
    )
    _bounds = (
        alt.Chart(pl.DataFrame({"b": [_r0, _r1]}))
        .mark_rule(color="#9aa7b0", strokeDash=[3, 3])
        .encode(x=alt.X("b:T", scale=_xs))
    )
    _gtext = (
        alt.Chart(_gaps)
        .mark_text(angle=270, color="#b3c4cf", fontSize=11, fontWeight="bold")
        .encode(x=alt.X("mid:T", scale=_xs), y=alt.value(180), text="dur:N")
    )

    _cap = (
        "**Daily means — all four series on one panel** via the multi-axis trick (real units, "
        "colour-matched axes): DO (white), air temp (orange), salinity (green), water level "
        "below ground (blue, reversed). Dashed rules mark the readout span; the grey "
        "**end-bands are expert-labelled periods with no readout coverage** — "
    )
    if _before.height:
        _cap += f"**{(_r0 - _before['start_time'].min()).days} d / {_before.height} events** before it begins"
    if _before.height and _after.height:
        _cap += " and "
    if _after.height:
        _cap += f"**{(_after['end_time'].max() - _r1).days} d / {_after.height} events** after it ends"
    _cap += "."

    mo.vstack(
        [
            mo.md(_cap),
            alt.layer(
                _wy_rules,
                _rects,
                _bounds,
                _l_air,
                _l_sal,
                _l_wl,
                _l_do,
                _gtext,
                _wy_lab,
            )
            .resolve_scale(y="independent")
            .properties(
                width=1500,
                height=400,
                title="Daily means over the full record — multi-axis",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def record_periodogram(readouts):
    _vars = {
        "Salinity": SAL_COL,
        "Water level": WL_COL,
        "Air temp": "AirT_C_Avg | Deg C | Avg",
        "Dissolved O2": DO_COL,
    }
    _frames = []
    for _name, _col in _vars.items():
        _g = (
            readouts.select("Datetime", _col)
            .upsample("Datetime", every="5m")
            .with_columns(pl.col(_col).interpolate().forward_fill().backward_fill())
        )
        _x = _g[_col].to_numpy().astype(float)
        _x = _x - _x.mean()
        _n = len(_x)
        _freq = np.fft.rfftfreq(_n, d=1.0 / 288.0)  # cycles per day
        _power = np.abs(np.fft.rfft(_x)) ** 2
        _frames.append(
            pl.DataFrame({"freq": _freq[1:], "power": _power[1:]})
            .filter(pl.col("freq").is_between(1.0 / 2000.0, 48.0))
            .with_columns((pl.col("freq").log10() * 120).round().alias("logbin"))
            .group_by("logbin")
            .agg(
                pl.col("freq").median().alias("freq"),
                pl.col("power").max().alias("power"),
            )
            .with_columns(
                (pl.col("power") / pl.col("power").max()).alias("rel_power"),
                pl.lit(_name).alias("variable"),
            )
            .sort("freq")
        )
    _spec = pl.concat(_frames)

    _marks = pl.DataFrame(
        {
            "freq": [1.0 / 365.25, 1.0 / 30.0, 1.0 / 7.0, 1.0, 2.0],
            "label": ["annual", "monthly", "weekly", "diurnal", "semi-diurnal"],
        }
    )
    _log_x = alt.Scale(type="log")
    _order = list(_vars.keys())
    _spectrum = (
        alt.Chart(_spec)
        .mark_line(strokeWidth=1.6, opacity=0.85)
        .encode(
            x=alt.X("freq:Q", scale=_log_x, title="frequency (cycles per day)"),
            y=alt.Y(
                "rel_power:Q",
                scale=alt.Scale(type="log"),
                title="relative spectral power (per variable, peak = 1)",
            ),
            color=alt.Color(
                "variable:N",
                scale=alt.Scale(domain=_order, scheme="tableau10"),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                "variable:N",
                alt.Tooltip("freq:Q", format=".4f"),
                alt.Tooltip("rel_power:Q", format=".3f"),
            ],
        )
    )
    _rules = (
        alt.Chart(_marks)
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(x=alt.X("freq:Q", scale=_log_x))
    )
    _labels = (
        alt.Chart(_marks)
        .mark_text(angle=270, align="left", dx=6, dy=-4, color="#888")
        .encode(x=alt.X("freq:Q", scale=_log_x), y=alt.value(8), text="label:N")
    )
    mo.hstack(
        [
            mo.md(
                """
                - **Periodogram of the main variables** each normalised to its own peak. 
                - **Salinity**, **water level** and **air temperature** are all dominated by the 
                **annual** cycle; water level adds broad **synoptic** (weekly\u2013monthly) power 
                from storms, and air temperature is the only series with a clear **diurnal (solar)** 
                peak. 
                - **Dissolved O2** \u2014 dominated by **synoptic / 
                event scales (~1\u20134 weeks, peak near a fortnight)**
                - DO is driven by discrete flooding *events*, not periodic
                - **No variable shows clean semi-diurnal tidal 
                line** \u2014 tidal saline intrusions into floodplain intermittent
                """
            ),
            (_spectrum + _rules + _labels).properties(width=900, height=320),
        ],
        justify="start",
    )
    return


@app.cell(hide_code=True)
def feature_methodology_intro():
    mo.md(r"""
    ## Handcrafted event-shape features — the classifier's vocabulary

    Each excursion is reduced to a handful of physically-motivated **shape + driver**
    descriptors (the same `FEATURE_COLS` the tabular models train on), computed by
    `preprocessing.py` on **our detector's events** — *not* the expert umbrellas, over which
    symmetry / hysteresis are ill-defined (one umbrella can span several sub-moments). Each
    tab **defines one feature** (formula + gloss), shows its **distribution** over the
    `proc_features` units, how it **co-varies with the coincident / antecedent drivers**
    (salinity / water-level / temperature step, antecedent precipitation), and **example DO
    curves** (from `proc_curves`) at the low vs high extremes. `peak_frac` &rarr; 0 = abrupt
    rise then slow decay (**hot moment**); ~0.5 = symmetric rise & fall (**oxic pulse**). The
    two hysteresis tabs instead plot the **DO-vs-driver loop** (the index *is* the
    rising&minus;falling DO gap over the driver range).
    """)
    return


@app.cell
def feature_methodology(proc):
    proc_curves = pl.read_parquet("derived/processed_auto_curves.parquet")
    feature_methodology_tabs(proc, proc_curves)
    return


@app.cell
def cusum_drivers(readouts):
    _AIRT = "AirT_C_Avg | Deg C | Avg"
    # water level is stored as depth Below Ground Surface (BGS): a LARGER number = water
    # table deeper = drier. Negate it (matching forecast.py's `wl_up`) so the driver reads
    # intuitively "water-level rise, ↑ = flood toward the surface".
    _drivers = {
        "water-level rise (cm, ↑=flood)": "wl_up",
        "well salinity (PPT)": SAL_COL,
        "air temperature (°C)": _AIRT,
        "antecedent precip 24 h (mm)": "precip_24h",
    }
    _base = readouts.sort("Datetime").select(
        pl.col(DO_COL).alias("_do"),
        pl.col(PRECIP_COL).rolling_sum_by("Datetime", "24h").alias("precip_24h"),
        pl.col(WL_COL).mul(-1).alias("wl_up"),
        pl.col(SAL_COL).alias(SAL_COL),
        pl.col(_AIRT).alias(_AIRT),
    )

    _rows, _infl = [], []
    for _name, _col in _drivers.items():
        _d = _base.select(
            pl.col("_do").alias("do"), pl.col(_col).alias("drv")
        ).drop_nulls()
        _do = _d["do"].to_numpy().astype(float)
        _drv = _d["drv"].to_numpy().astype(float)
        if _do.std() == 0 or len(_do) < 10:
            continue
        _order = np.argsort(_drv, kind="stable")
        _z = (_do - _do.mean()) / _do.std()
        _cs = np.cumsum(_z[_order])
        _drv_ord = _drv[_order]
        _n = len(_cs)
        # positive (bowl) if the interior trough is deeper than the interior crest is tall
        _imin, _imax = int(_cs.argmin()), int(_cs.argmax())
        _positive = abs(_cs[_imin]) >= abs(_cs[_imax])
        _ii = _imin if _positive else _imax
        _infl.append(
            {
                "driver": _name,
                "relationship": "positive (bowl)" if _positive else "negative (dome)",
                "DO crosses mean at driver =": round(float(_drv_ord[_ii]), 2),
                "n samples": _n,
            }
        )
        # thin the curve for plotting (~1200 pts) but always keep the inflection + ends
        _step = max(1, _n // 1200)
        _keep = sorted(set(list(range(0, _n, _step)) + [_ii, _n - 1]))
        for _k in _keep:
            _rows.append(
                {
                    "driver": _name,
                    "frac": _k / (_n - 1),
                    "cusum": float(_cs[_k]),
                    "drv": float(_drv_ord[_k]),
                    "inflection": bool(_k == _ii),
                }
            )

    cusum_curve = pl.DataFrame(_rows)
    cusum_inflections = pl.DataFrame(_infl)

    # both layers share one top-level data source (cusum_curve) so the layered chart can
    # be faceted; the inflection point is selected with transform_filter, not a pre-filter.
    _src = alt.Chart(cusum_curve)
    _line = _src.mark_line(color="#1f77b4").encode(
        x=alt.X("frac:Q", title="samples ordered by driver (low → high)"),
        y=alt.Y("cusum:Q", title="Cusum of z-normalized DO"),
        tooltip=[
            "driver:N",
            alt.Tooltip("drv:Q", format=".2f", title="driver value"),
            alt.Tooltip("cusum:Q", format=".1f"),
        ],
    )
    _pt = (
        _src.transform_filter(alt.datum.inflection)
        .mark_point(color="#d62728", size=90, filled=True)
        .encode(x="frac:Q", y="cusum:Q")
    )
    _facet = (
        (_line + _pt)
        .properties(width=300, height=200)
        .facet(alt.Facet("driver:N", title=None), columns=4)
        .resolve_scale(y="independent")
    )

    mo.vstack(
        [
            mo.md(
                """
                ### Cusum driver–response (Regier-lab method)

                Order z-normalized DO by each driver and cumulatively sum: a **bowl**
                (red dot at an interior minimum) is a **positive** driver→DO relationship,
                a **dome** (interior maximum) is **negative**. The red inflection marks the
                driver value at which DO crosses its own mean — a **regime threshold**. This
                is the domain team's own tool (Regier et al. 2019/2023) and the natural
                vehicle for **Challenge 2** (driver attribution / sea-level rise): the
                inflection driver values in the table are candidate onset triggers, read
                straight off the hydrology without a trained model. **Water-level rise** is a
                clear positive driver — a rising water table (flood inundation) pushes DO
                above its mean, the same flood-oxygenation mechanism the SHAP beeswarm shows
                and the analog of creek depth in Regier et al. 2023 Fig. 5B. **Air temperature**
                is negative (solubility, their Fig. 5A). Read each panel's shape and the red
                inflection for the sign and threshold — salinity and precip carry the more
                composite well-scale responses.
                """
            ),
            cusum_inflections,
            _facet,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
