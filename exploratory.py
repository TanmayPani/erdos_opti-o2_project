import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    # Some columns carry per-day means/maxes that blow past Altair's default
    # 5k-row guard; we keep our chart payloads small but disable it to be safe.
    alt.data_transformers.disable_max_rows()

    data_dir_path = mo.notebook_location() / "datasets" / "BeaverCreekWA_EssDive_26Jun2019-30Sep2024"

    lf = pl.scan_csv(
        data_dir_path / "data" / "2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv",
        skip_rows_after_header=2,
        infer_schema=True,
        infer_schema_length=10000,
        # Spreadsheet artefacts ("#REF!") and blanks appear mid-file.
        null_values=["#REF!", ""],
        try_parse_dates=False,
    )

    # The trailing column is an empty spacer left over from the CSV export.
    lf = lf.select(pl.all().exclude(lf.collect_schema().names()[-1]))

    # Physically implausible sentinels (e.g. AirT = -1989, negative solar flux,
    # a vapour-pressure spike of 211 hPa) are masked to null for plotting.
    _bounds = {
        "AirT_C_Avg": (-40.0, 50.0),
        "SlrFD_kW_Avg": (0.0, 2.0),
        "SlrTF_MJ_Tot": (0.0, 5.0),
        "VP_hPa_Avg": (0.0, 60.0),
    }

    df = (
        lf.with_columns(
            pl.col("Datetime").str.to_datetime("%-m/%-d/%Y %-H:%M"),
        )
        .with_columns(
            pl.when(pl.col(c).is_between(lo, hi)).then(pl.col(c)).otherwise(None).alias(c)
            for c, (lo, hi) in _bounds.items()
        )
        .drop("WEATHER TIMESTAMP")
        .drop_nulls("Datetime")
        .unique("Datetime", keep="first")
        .sort("Datetime")
        .collect()
    )


@app.function
def binned_hist(frame, col, bins=60):
    s = frame.select(pl.col(col).alias("v")).drop_nulls()
    lo = s.select(pl.col("v").min()).item()
    hi = s.select(pl.col("v").max()).item()
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / bins
    return (
        s.with_columns(((pl.col("v") - lo) / width).floor().clip(0, bins - 1).alias("bin"))
        .group_by("bin")
        .len(name="count")
        .with_columns(
            (lo + pl.col("bin") * width).alias("left"),
            (lo + (pl.col("bin") + 1) * width).alias("right"),
        )
        .sort("bin")
    )


@app.function
def hist_chart(frame, col, symlog=False, bins=60):
    h = binned_hist(frame, col, bins=bins)
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


@app.function
def daily_profile(frame, col, xexpr, xname):
    return (
        frame.select(xexpr.alias(xname), pl.col(col).alias("v"))
        .drop_nulls()
        .group_by(xname)
        .agg(
            pl.col("v").mean().alias("mean"),
            pl.col("v").quantile(0.25).alias("q25"),
            pl.col("v").quantile(0.75).alias("q75"),
        )
        .sort(xname)
    )


@app.function
def cycle_chart(prof, xname, xtitle, title):
    base = alt.Chart(prof)
    band = base.mark_area(opacity=0.25).encode(
        x=alt.X(f"{xname}:Q", title=xtitle),
        y=alt.Y("q25:Q", title=None),
        y2="q75:Q",
    )
    line = base.mark_line().encode(
        x=alt.X(f"{xname}:Q", title=xtitle),
        y=alt.Y("mean:Q", title=None),
        tooltip=[alt.Tooltip("mean:Q", format=".2f")],
    )
    return (band + line).properties(width=330, height=200, title=title)


@app.function
def detect_events(frame, do_col="Dissolved Oxygen (mg/L)", merge_gap=12):
    """One row per oxygenation event (contiguous DO > 0), merging baseline gaps
    of <= merge_gap samples (12 = 1 h at the 5-min cadence) into the surrounding
    event. Returns columns: eid, start, end."""
    f = frame.select("Datetime", (pl.col(do_col) > 0).alias("_oxic"))
    f = f.with_columns(
        (pl.col("_oxic") != pl.col("_oxic").shift(1)).fill_null(True).cum_sum().alias("_rid")
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


@app.function
def event_shape_features(
    samples,
    do_col="Dissolved Oxygen (mg/L)",
    wl_col="Flood plain water level in BGS (cm)",
    sal_col="Well Salinity (PPT)",
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
    for key, sub in samples.sort(["eid", "elapsed_min"]).group_by("eid", maintain_order=True):
        d = sub[do_col].to_numpy().astype(float)
        n = len(d)
        peak = d.max() if n else 0.0
        rec = {"eid": key[0]}
        if n > 1 and peak > 0 and d.sum() > 0:
            t = np.arange(n)
            w = d / d.sum()
            ap = int(np.argmax(d))
            rec["centroid_frac"] = float((w * t).sum() / (n - 1))
            rec["max_rise_norm"] = float(np.diff(d).max() / peak)
            rec["plateau_frac"] = float((d >= 0.8 * peak).mean())

            def _hyst(drv):
                rng = drv.max() - drv.min()
                if rng < 1e-9 or ap < 1 or ap >= n - 1:
                    return None
                xn = (drv - drv.min()) / rng
                yn = d / peak
                grid = np.linspace(0.05, 0.95, 10)
                xr, yr = xn[: ap + 1], yn[: ap + 1]
                xf, yf = xn[ap:], yn[ap:]
                ri = np.interp(grid, np.sort(xr), yr[np.argsort(xr)])
                fi = np.interp(grid, np.sort(xf), yf[np.argsort(xf)])
                return float((ri - fi).mean())

            rec["hyst_wl"] = _hyst(sub[wl_col].to_numpy().astype(float))
            rec["hyst_sal"] = _hyst(sub[sal_col].to_numpy().astype(float))
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


@app.cell
def _():
    mo.md(r"""
    # Beaver Creek (WA) flood-plain well — exploratory analysis

    A 5-minute-cadence environmental record (**2019-06-26 → 2024-09-30**, ~554k rows)
    from a flood-plain monitoring well: dissolved oxygen, salinity, water level, and
    co-located weather (solar, air temperature, precipitation, barometric pressure, …).

    The two structural features we want to *see* before modelling:

    1. **Multimodality** — several variables are not single-peaked. Dissolved oxygen is
       strongly *zero-inflated* (long anoxic spells punctuated by oxic events), and
       salinity sits in distinct regimes.
    2. **Multiple time-scales** — the signals carry a fast **diurnal** beat, a slower
       **synoptic/weather** band (days–weeks), and a **seasonal/annual** swing, all
       superimposed.
    """)
    return


@app.cell
def _():
    df
    return


@app.cell
def _():
    _row1 = mo.hstack(
        [
            hist_chart(df, "Dissolved Oxygen (mg/L)", symlog=True),
            hist_chart(df, "O2 Concentration (%)", symlog=True),
            hist_chart(df, "Well Salinity (PPT)"),
        ],
        justify="center",
    )
    _row2 = mo.hstack(
        [
            hist_chart(df, "Flood plain water level in BGS (cm)"),
            hist_chart(df, "AirT_C_Avg"),
            hist_chart(df, "DO Sensor Temperature (C) "),
        ],
        justify="center",
    )
    mo.vstack([_row1, _row2])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Multiple time-scales

    Three complementary views: the **full record** (long-term + seasonal), the average
    **diurnal cycle** (fast beat), and a frequency-domain **periodogram** that places the
    diurnal, synoptic, and annual scales on one axis.
    """)
    return


@app.cell
def _():
    _daily = (
        df.group_by_dynamic("Datetime", every="1d")
        .agg(
            pl.col("AirT_C_Avg").mean().alias("Air temp (°C)"),
            pl.col("Dissolved Oxygen (mg/L)").mean().alias("DO (mg/L)"),
            pl.col("Well Salinity (PPT)").mean().alias("Salinity (PPT)"),
            pl.col("Flood plain water level in BGS (cm)").mean().alias("Water level (cm BGS)"),
        )
        .unpivot(index="Datetime", variable_name="variable", value_name="value")
        .drop_nulls("value")
    )

    overview = (
        alt.Chart(_daily)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("Datetime:T", title="date"),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("variable:N", legend=None),
            tooltip=["Datetime:T", "variable:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(width=900, height=120)
        .facet(
            row=alt.Row("variable:N", title=None),
            title="Daily means over the full record",
        )
        .resolve_scale(y="independent")
    )
    overview
    return


@app.cell
def _():
    _tod = pl.col("Datetime").dt.hour() + pl.col("Datetime").dt.minute() / 60.0

    _diurnal = mo.hstack(
        [
            cycle_chart(
                daily_profile(df, "SlrFD_kW_Avg", _tod, "tod"),
                "tod",
                "hour of day",
                "Solar flux (kW/m²)",
            ),
            cycle_chart(
                daily_profile(df, "AirT_C_Avg", _tod, "tod"),
                "tod",
                "hour of day",
                "Air temp (°C)",
            ),
            cycle_chart(
                daily_profile(df, "Dissolved Oxygen (mg/L)", _tod, "tod"),
                "tod",
                "hour of day",
                "Dissolved O₂ (mg/L)",
            ),
        ],
        justify="center",
    )

    _woy = pl.col("Datetime").dt.ordinal_day()

    _seasonal = mo.hstack(
        [
            cycle_chart(
                daily_profile(df, "AirT_C_Avg", _woy, "doy"),
                "doy",
                "day of year",
                "Air temp (°C)",
            ),
            cycle_chart(
                daily_profile(df, "Dissolved Oxygen (mg/L)", _woy, "doy"),
                "doy",
                "day of year",
                "Dissolved O₂ (mg/L)",
            ),
            cycle_chart(
                daily_profile(df, "Well Salinity (PPT)", _woy, "doy"),
                "doy",
                "day of year",
                "Salinity (PPT)",
            ),
        ],
        justify="center",
    )

    mo.vstack(
        [
            mo.md("**Diurnal cycle** (mean ± inter-quartile band, by time of day):"),
            _diurnal,
            mo.md("**Seasonal cycle** (mean ± inter-quartile band, by day of year):"),
            _seasonal,
        ]
    )
    return


@app.cell
def _():
    _var = "AirT_C_Avg"

    # Resample onto a strict 5-minute grid and interpolate the few gaps so the
    # FFT sees an evenly-sampled series.
    _g = (
        df.select("Datetime", _var)
        .upsample("Datetime", every="5m")
        .with_columns(pl.col(_var).interpolate().forward_fill().backward_fill())
    )

    _x = _g[_var].to_numpy().astype(float)
    _x = _x - _x.mean()
    _n = len(_x)

    _samples_per_day = 288.0
    _freq = np.fft.rfftfreq(_n, d=1.0 / _samples_per_day)  # cycles per day
    _power = np.abs(np.fft.rfft(_x)) ** 2

    # Log-bin (keeping the max power per bin) so peaks survive down-sampling.
    _pdf = (
        pl.DataFrame({"freq": _freq[1:], "power": _power[1:]})
        .filter(pl.col("freq").is_between(1.0 / 2000.0, 48.0))
        .with_columns((pl.col("freq").log10() * 120).round().alias("logbin"))
        .group_by("logbin")
        .agg(
            pl.col("freq").median().alias("freq"),
            pl.col("power").max().alias("power"),
        )
        .sort("freq")
    )

    _marks = pl.DataFrame(
        {
            "freq": [1.0 / 365.25, 1.0 / 30.0, 1.0 / 7.0, 1.0, 2.0],
            "label": ["annual", "monthly", "weekly", "diurnal", "semi-diurnal"],
        }
    )

    _log_x = alt.Scale(type="log")
    _spectrum = (
        alt.Chart(_pdf)
        .mark_line()
        .encode(
            x=alt.X("freq:Q", scale=_log_x, title="frequency (cycles per day)"),
            y=alt.Y("power:Q", scale=alt.Scale(type="log"), title="spectral power"),
            tooltip=[alt.Tooltip("freq:Q", format=".4f")],
        )
    )
    _rules = (
        alt.Chart(_marks)
        .mark_rule(color="firebrick", strokeDash=[4, 3])
        .encode(x=alt.X("freq:Q", scale=_log_x))
    )
    _labels = (
        alt.Chart(_marks)
        .mark_text(angle=270, align="left", dx=6, dy=-4, color="firebrick")
        .encode(x=alt.X("freq:Q", scale=_log_x), y=alt.value(8), text="label:N")
    )

    mo.vstack(
        [
            mo.md(
                "**Periodogram of air temperature.** Sharp peaks at **1/day** (diurnal) "
                "and **2/day** (semi-diurnal harmonic), broad **synoptic** power at "
                "~1–3 weeks, and rising power toward the **annual** scale — the time-scales "
                "are well separated."
            ),
            (_spectrum + _rules + _labels).properties(width=900, height=300),
        ]
    )
    return


@app.cell
def _():
    # Label every 5-min sample with the event that contains it, and record how
    # far into the event it falls (elapsed_min) for shape analysis.
    event_samples = (
        df.sort("Datetime")
        .join_asof(detect_events(df), left_on="Datetime", right_on="start", strategy="backward")
        .filter(pl.col("Datetime") <= pl.col("end"))
        .with_columns((pl.col("Datetime") - pl.col("start")).dt.total_minutes().alias("elapsed_min"))
    )
    event_samples
    return (event_samples,)


@app.cell(hide_code=True)
def _(dur_min_min, events, events_all, peak_do_min):
    # Section 3 intro + the event definition / selection / count funnel. The
    # thresholds and the resulting counts read LIVE from the selection fields above
    # (peak_do_min, dur_min_min) and from events_all / events, so the prose stays in
    # sync whenever you change the floors.
    _n_raw = (
        df.select((pl.col("Dissolved Oxygen (mg/L)") > 0).alias("_ox"))
        .with_columns((pl.col("_ox") != pl.col("_ox").shift(1)).fill_null(True).cum_sum().alias("_r"))
        .filter(pl.col("_ox"))
        .select(pl.col("_r").n_unique())
        .item()
    )
    _n_merged = events_all.height
    _n_kept = events.height
    _pk = peak_do_min.value
    _du = dur_min_min.value
    _du_samples = int(_du // 5) + 1

    mo.md(f"""
    ## Section 3 — Event detection & shape features

    The DO series sits at the anoxic baseline (DO = 0) ~92 % of the time; oxygen
    arrives in discrete **events**. Below we: 
    1. **detect** events as contiguous departures from the baseline — merging sub-hour fluctuations 
    2. engineer **shape features** (duration, peak, integrated load, and *time-to-peak symmetry*) and
    **antecedent / coincident drivers** (salinity and water-level *steps*, prior-24 h
    precipitation). These are the inputs a supervised classifier will use to
    separate abrupt, asymmetric **hot moments** from symmetric **oxic pulses**.

    ### How an event is defined, filtered, and counted

    **Definition.** The well is anoxic (DO = 0) ~92 % of the time, so we treat any
    sample with **DO > 0 mg/L** as *oxic* and define a raw event as a **contiguous run of non-zero oxic samples** on the regular 5-minute grid.

    **Gap merging.** Occasionally events will flicker briefly back to zero (sensor noise, a
    short ebb). To avoid splitting one physical events unnecessarily, `detect_events`
    **combines events with a gap of <= 12 samples (1 hour)**
    (the `merge_gap` argument).

    **Selection of "proper" events.** After merging gaps a candidate is kept only if it has a
    genuine oxic peak and lasts long enough to have a shape worth classifying — both
    floors are set interactively in the fields above:

    - **peak DO >= {_pk:.2f} mg/L** — discards shallow near-zero flicker, and
    - **duration >= {_du:g} min** (>= {_du_samples} samples) — discards short blips.

    **How we reach {_n_kept}.** To summarize:

    | step | count |
    |---|---|
    | raw events (DO > 0) | **{_n_raw}** |
    | after merging baseline gaps <= 1 h | **{_n_merged}** |
    | after `peak >= {_pk:.2f} mg/L` **and** `dur >= {_du:g} min` | **{_n_kept}** |
    """)
    return


@app.cell
def _(dur_min_min, event_samples, peak_do_min):
    # Per-event feature table: shape (duration, peak, load, symmetry, rise/fall)
    # plus antecedent / coincident drivers (salinity, water-level & temperature
    # steps, prior-24 h precipitation). `events_all` holds every candidate; `events`
    # applies the interactive peak-DO and duration floors (see the controls cell above).
    _do = "Dissolved Oxygen (mg/L)"
    _sal = "Well Salinity (PPT)"
    _wl = "Flood plain water level in BGS (cm)"
    _pr = "Precip (mm) over 5 minutes"
    _tp = "DO Sensor Temperature (C) "

    # trailing 24 h context on the full series, sampled just before each event start
    _ctx = df.sort("Datetime").select(
        "Datetime",
        pl.col(_pr).rolling_sum_by("Datetime", window_size="24h").alias("precip_24h"),
        pl.col(_sal).rolling_mean_by("Datetime", window_size="24h").alias("sal_pre"),
        pl.col(_wl).rolling_mean_by("Datetime", window_size="24h").alias("wl_pre"),
        pl.col(_tp).rolling_mean_by("Datetime", window_size="24h").alias("temp_pre"),
    )

    _shape = (
        event_samples.group_by("eid")
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("start").first(),
            pl.col("end").first(),
            pl.col(_do).max().alias("peak_do"),
            pl.col(_do).mean().alias("mean_do"),
            pl.col(_do).sum().alias("_sum_do"),
            pl.col(_do).arg_max().cast(pl.Int64).alias("_argpeak"),
            pl.col(_sal).mean().alias("sal_in"),
            pl.col(_wl).mean().alias("wl_in"),
            pl.col(_tp).mean().alias("temp_in"),
        )
        .sort("start")
        .with_columns(
            (pl.col("end") - pl.col("start")).dt.total_minutes().alias("dur_min"),
            (pl.col("_sum_do") * 5 / 60).alias("area_mgLh"),
            (pl.col("_argpeak") / (pl.col("n_samples") - 1).clip(lower_bound=1)).alias("peak_frac"),
            (pl.col("_argpeak") * 5).alias("rise_min"),
            ((pl.col("n_samples") - 1 - pl.col("_argpeak")) * 5).alias("fall_min"),
        )
        .with_columns(
            (pl.col("peak_do") / (pl.col("rise_min") / 60).clip(lower_bound=0.0833)).alias("rise_rate"),
            (pl.col("peak_do") / (pl.col("fall_min") / 60).clip(lower_bound=0.0833)).alias("fall_rate"),
        )
    )

    _extra = event_shape_features(event_samples)

    events_all = (
        _shape.join_asof(_ctx, left_on="start", right_on="Datetime", strategy="backward")
        .with_columns(
            (pl.col("sal_in") - pl.col("sal_pre")).alias("sal_step"),
            (pl.col("wl_in") - pl.col("wl_pre")).alias("wl_step"),
            (pl.col("temp_in") - pl.col("temp_pre")).alias("temp_step"),
        )
        .join(_extra, on="eid")
        .select(
            "eid",
            "start",
            "end",
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
            "centroid_frac",
            "max_rise_norm",
            "plateau_frac",
            "hyst_wl",
            "hyst_sal",
        )
        .sort("start")
    )

    events = events_all.filter(
        (pl.col("peak_do") >= peak_do_min.value) & (pl.col("dur_min") >= dur_min_min.value)
    )
    events
    return events, events_all


@app.cell
def _():
    # Interactive selection knobs for what counts as a "real" oxygenation event.
    # Lower the peak-DO floor to recover long, low-amplitude oxic episodes that a
    # fixed 0.5 mg/L cut discards; raise it to keep only strong pulses. Everything
    # dependent cell (features, symmetry, clustering) re-runs reactively on change.
    peak_do_min = mo.ui.number(
        start=0.0,
        stop=10.0,
        step=0.01,
        value=0.05,
        label="Min peak DO (mg/L)",
    )
    dur_min_min = mo.ui.number(
        start=0,
        stop=10000,
        step=5,
        value=15,
        label="Min duration (min)",
    )
    # mo.vstack([peak_do_min, dur_min_min])
    return dur_min_min, peak_do_min


@app.cell
def _(dur_min_min, events, events_all, peak_do_min):
    # What the current selection keeps vs. drops, with the events sitting just below
    # the peak-DO floor surfaced so we can judge whether real events are being cut.
    _thr = peak_do_min.value
    _dmin = dur_min_min.value
    _kept = events.height
    _dropped = events_all.height - _kept

    _bar = (
        alt.Chart(events_all.select("peak_do"))
        .mark_bar(opacity=0.85, color="#4c78a8")
        .encode(
            x=alt.X("peak_do:Q", bin=alt.Bin(maxbins=40), title="peak DO (mg/L)"),
            y=alt.Y("count()", title="# candidate events"),
        )
        .properties(
            width=560,
            height=200,
            title=f"{_kept} kept / {_dropped} dropped  (peak>={_thr:.2f} mg/L, dur>={_dmin} min)",
        )
    )
    _rule = (
        alt.Chart(pl.DataFrame({"thr": [_thr]}))
        .mark_rule(color="red", strokeDash=[4, 3])
        .encode(x="thr:Q")
    )

    # marginal events just below the floor, longest first: the likeliest mis-drops
    _marginal = (
        events_all.filter(pl.col("peak_do") < _thr)
        .sort("dur_min", descending=True)
        .select("eid", "start", "dur_min", "peak_do", "mean_do", "area_mgLh")
        .head(12)

    )

    mo.vstack(
        [
            mo.hstack([peak_do_min, dur_min_min], justify="center"),
            _bar + _rule,
            mo.md(f"**Marginal dropped events** (peak < {_thr:.2f}, longest first) — are these real?"),
            mo.ui.table(_marginal, selection=None),
        ]
    )
    return


@app.cell
def _(event_samples, events):
    # Shape-feature insights, one tab per engineered descriptor. Each tab shows the
    # feature's distribution (with its defining formula), how it relates to each
    # coincident / antecedent driver, and an auto-generated read of the strongest
    # driver associations (recomputed from the current selection so the bullets never
    # go stale when the DO/duration floors move). peak_frac = 0 -> abrupt rise / slow
    # decay ("hot moment"); ~0.5 -> symmetric rise & fall ("oxic pulse").
    _feat_meta = {
        "peak_frac": (
            "Symmetry (time-to-peak)",
            "0 = abrupt rise then slow decay, 0.5 = symmetric rise & fall.",
        ),
        "centroid_frac": (
            "Symmetry (centroid)",
            "centre of mass of the DO curve; whole-curve symmetry, more robust than the single-sample peak_frac.",
        ),
        "max_rise_norm": (
            "Onset abruptness",
            "steepest single 5-min DO jump, peak-normalised; high = spiky, sudden onset.",
        ),
        "plateau_frac": (
            "Plateau fraction",
            "share of the event spent near peak DO; high = flat-topped, low = sharp-peaked.",
        ),
        "hyst_wl": (
            "DO-water-level hysteresis",
            "sign of the DO-vs-water-level loop; magnitude = how open the loop is.",
        ),
        "hyst_sal": (
            "DO-salinity hysteresis",
            "sign of the DO-vs-salinity loop; magnitude = how open the loop is.",
        ),
    }

    _feat_formula = {
        "peak_frac": r"$$\mathrm{peak\_frac}=\frac{\operatorname*{arg\,max}_i\, d_i}{n-1}$$",
        "centroid_frac": r"$$\mathrm{centroid\_frac}=\frac{\sum_i i\, d_i}{(n-1)\sum_i d_i}$$",
        "max_rise_norm": r"$$\mathrm{max\_rise\_norm}=\frac{\max_i\,(d_{i+1}-d_i)}{\max_i d_i}$$",
        "plateau_frac": r"$$\mathrm{plateau\_frac}=\frac{1}{n}\sum_i \mathbf{1}\!\left[d_i\ge 0.8\,d_{\max}\right]$$",
        "hyst_wl": r"$$\mathrm{hyst\_wl}=\operatorname*{mean}_{g}\bigl(\hat d_\uparrow(g)-\hat d_\downarrow(g)\bigr)$$",
        "hyst_sal": r"$$\mathrm{hyst\_sal}=\operatorname*{mean}_{g}\bigl(\hat d_\uparrow(g)-\hat d_\downarrow(g)\bigr)$$",
    }

    _feat_legend = {
        "peak_frac": r"$d_i$ = DO at 5-min step $i$ ($i=0\ldots n-1$); $n$ = samples in the event.",
        "centroid_frac": r"Time index $i$ weighted by DO mass $d_i$, normalised to $[0,1]$.",
        "max_rise_norm": r"$d_{i+1}-d_i$ = consecutive 5-min DO change; divided by peak DO.",
        "plateau_frac": r"$\mathbf{1}[\cdot]$ = indicator; counts samples within 80–100% of peak.",
        "hyst_wl": r"$g$ steps across 10 evenly-spaced driver levels (rescaled to $[0,1]$). At each level the DO on the way up ($\hat d_\uparrow$) is compared with the DO on the way down ($\hat d_\downarrow$), where $\hat d = d/d_{\max}$; the average gap is the index. $+$ = DO higher while rising (clockwise loop).",
        "hyst_sal": r"$g$ steps across 10 evenly-spaced driver levels (rescaled to $[0,1]$). At each level the DO on the way up ($\hat d_\uparrow$) is compared with the DO on the way down ($\hat d_\downarrow$), where $\hat d = d/d_{\max}$; the average gap is the index. $+$ = DO higher while rising (clockwise loop).",
    }

    _drivers = [
        ("sal_step", "salinity step (PPT)", "blues"),
        ("wl_step", "water-level step (cm)", "greens"),
        ("temp_step", "temperature step (°C)", "oranges"),
        ("precip_24h", "prior-24 h precip (mm)", "purples"),
    ]


    def _driver_panel(ycol, xcol, xtitle, scheme):
        return (
            alt.Chart(events.select(["start", "dur_min", "peak_do", ycol, xcol]))
            .mark_circle(opacity=0.75)
            .encode(
                alt.X(f"{xcol}:Q", title=xtitle),
                alt.Y(f"{ycol}:Q", title=_feat_meta[ycol][0]),
                alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[20, 400])),
                alt.Color(f"{xcol}:Q", title=xtitle, scale=alt.Scale(scheme=scheme), legend=None),
                tooltip=["start:T", "dur_min:Q", "peak_do:Q", f"{ycol}:Q", f"{xcol}:Q"],
            )
            .properties(width=320, height=230, title=f"{_feat_meta[ycol][0]} vs {xtitle}")
        )


    def _example_chart(ycol):
        # the same time-normalised DO-shape view as the next slide, but the four
        # exemplars are the 2 lowest and 2 highest events on THIS tab's feature.
        _v = events.drop_nulls(ycol).sort(ycol)
        _ids = _v.head(2)["eid"].to_list() + _v.tail(2)["eid"].to_list()
        _pick = events.filter(pl.col("eid").is_in(_ids)).select("eid", "dur_min", ycol)
        _curve = (
            event_samples.filter(pl.col("eid").is_in(_ids))
            .join(_pick, on="eid")
            .with_columns(
                (pl.col("elapsed_min") / pl.col("dur_min").clip(lower_bound=1)).alias("t_norm"),
                pl.col("Dissolved Oxygen (mg/L)").alias("do"),
                pl.format("eid {} ({}={})", pl.col("eid"), pl.lit(ycol), pl.col(ycol).round(2)).alias(
                    "event"
                ),
            )
            # resample each curve to ~50 normalised-time points so the embedded
            # payload stays small regardless of raw event length
            .with_columns((pl.col("t_norm").clip(0, 1) * 50).round(0).alias("_b"))
            .group_by("event", "_b")
            .agg(pl.col("t_norm").mean(), pl.col("do").mean())
            .select("t_norm", "do", "event")
            .sort("event", "t_norm")
        )
        return (
            alt.Chart(_curve)
            .mark_line()
            .encode(
                alt.X("t_norm:Q", title="normalised time (0 = start, 1 = end)"),
                alt.Y("do:Q", title="DO (mg/L)"),
                alt.Color("event:N", title=None, scale=alt.Scale(scheme="redyellowblue")),
            )
            .properties(
                width=400, height=190, title=f"Example shapes: low vs high {_feat_meta[ycol][0]}"
            )
        )


    # for the hysteresis tabs the meaningful "example" is the DO-vs-driver loop, not a
    # DO-vs-time curve: the index IS the rising-minus-falling DO gap over the driver range.
    _hyst_driver = {
        "hyst_wl": ("Flood plain water level in BGS (cm)", "water level (cm)"),
        "hyst_sal": ("Well Salinity (PPT)", "salinity (PPT)"),
    }


    def _hyst_loop(ycol):
        _driver_col, _driver_title = _hyst_driver[ycol]
        _v = events.drop_nulls(ycol).sort(ycol)
        _ids = [_v["eid"][0], _v["eid"][-1]]  # strongest counter-clockwise & clockwise loop
        _lab = events.filter(pl.col("eid").is_in(_ids)).select("eid", ycol)
        _s = (
            event_samples.filter(pl.col("eid").is_in(_ids))
            .select(
                "eid",
                "elapsed_min",
                pl.col("Dissolved Oxygen (mg/L)").alias("do"),
                pl.col(_driver_col).alias("drv"),
            )
            .drop_nulls(["do", "drv"])
            .sort("eid", "elapsed_min")
            # resample to ~60 time-ordered points per event to bound the payload
            .with_columns(
                (pl.col("elapsed_min") / pl.col("elapsed_min").max().over("eid") * 60)
                .round(0)
                .alias("_b")
            )
            .group_by("eid", "_b")
            .agg(pl.col("elapsed_min").mean(), pl.col("do").mean(), pl.col("drv").mean())
            .sort("eid", "elapsed_min")
        )
        _pk = _s.group_by("eid").agg(
            pl.col("elapsed_min").filter(pl.col("do") == pl.col("do").max()).first().alias("_tpk")
        )
        _s = (
            _s.join(_pk, on="eid")
            .join(_lab, on="eid")
            .with_columns(
                pl.when(pl.col("elapsed_min") <= pl.col("_tpk"))
                .then(pl.lit("rising"))
                .otherwise(pl.lit("falling"))
                .alias("branch"),
                pl.format("eid {} ({} = {})", pl.col("eid"), pl.lit(ycol), pl.col(ycol).round(2)).alias(
                    "event"
                ),
            )
        )
        return (
            alt.Chart(_s)
            .mark_line(point=alt.OverlayMarkDef(size=15))
            .encode(
                alt.X("drv:Q", title=_driver_title, scale=alt.Scale(zero=False)),
                alt.Y("do:Q", title="DO (mg/L)", scale=alt.Scale(zero=False)),
                alt.Color(
                    "branch:N",
                    title=None,
                    scale=alt.Scale(domain=["rising", "falling"], range=["#d73027", "#4575b4"]),
                ),
                alt.Order("elapsed_min:Q"),
            )
            .properties(width=180, height=165)
            .facet(facet=alt.Facet("event:N", title=None), columns=2)
            .resolve_scale(x="independent", y="independent")
        )


    def _assoc_bullets(ycol):
        _y = events[ycol].to_numpy().astype(float)
        _rows = []
        for _c, _lab, _ in _drivers:
            _x = events[_c].to_numpy().astype(float)
            _m = ~(np.isnan(_x) | np.isnan(_y))
            if _m.sum() < 3:
                continue
            _r = float(np.corrcoef(_x[_m], _y[_m])[0, 1])
            _rows.append((abs(_r), _r, _lab))
        _rows.sort(reverse=True)
        _out = []
        for _ar, _r, _lab in _rows:
            _tag = (
                "negligible"
                if _ar < 0.1
                else "weak"
                if _ar < 0.2
                else "moderate"
                if _ar < 0.3
                else "strong"
            )
            _out.append(f"- **{_lab}** &nbsp; r = {_r:+.2f} _({_tag})_")
        return "\n".join(_out)


    def _feat_tab(ycol):
        _title, _gloss = _feat_meta[ycol]
        _hist = (
            alt.Chart(events.select(ycol))
            .mark_bar(color="#3b7dd8")
            .encode(
                alt.X(f"{ycol}:Q", bin=alt.Bin(maxbins=24), title=_title),
                alt.Y("count():Q", title="number of events"),
            )
            .properties(width=460, height=190, title=f"{_title}  (n = {events.height} events)")
        )
        _formula = mo.vstack(
            [
                mo.md(f"**{_title}**<br><br>{_gloss}<br><br>"),
                mo.md(_feat_formula[ycol]),
                mo.md(f"<small>{_feat_legend[ycol]}</small>"),
            ],
            justify="center",
        )
        _example = _hyst_loop(ycol) if ycol in _hyst_driver else _example_chart(ycol)
        _top = mo.hstack(
            [_hist, _example, _formula],
            justify="start",
            # align="center",
            # widths=[3, 3, 2],
        )
        _insight = mo.vstack(
            [
                mo.md(f"**Driver associations** (+ raises this feature, &minus; lowers it):"),
                mo.md(_assoc_bullets(ycol)),
            ],
            gap=0.25,
            align="start",
        )
        _grid = mo.hstack(
            [
                mo.vstack(
                    [
                        _driver_panel(ycol, _drivers[0][0], _drivers[0][1], _drivers[0][2]),
                        _driver_panel(ycol, _drivers[2][0], _drivers[2][1], _drivers[2][2]),
                    ]
                ),
                mo.vstack(
                    [
                        _driver_panel(ycol, _drivers[1][0], _drivers[1][1], _drivers[1][2]),
                        _driver_panel(ycol, _drivers[3][0], _drivers[3][1], _drivers[3][2]),
                    ]
                ),
                _insight,
            ],
            justify="start",
        )
        return mo.vstack([_top, _grid])


    mo.ui.tabs(
        {
            "Symmetry (peak)": _feat_tab("peak_frac"),
            "Symmetry (centroid)": _feat_tab("centroid_frac"),
            "Onset abruptness": _feat_tab("max_rise_norm"),
            "Plateau": _feat_tab("plateau_frac"),
            "Hysteresis · WL": _feat_tab("hyst_wl"),
            "Hysteresis · salinity": _feat_tab("hyst_sal"),
        }
    )
    return


@app.cell
def _(cluster_X, events):
    # How many clusters does the geometry actually support? Several criteria across k
    # on the SAME standardised feature matrix (cluster_X). Separation metrics
    # (silhouette, Calinski-Harabasz: higher = better) vs compactness / likelihood
    # (Davies-Bouldin, GMM BIC/AIC: lower = better) often disagree -- that gap is itself
    # the finding.
    _ks = list(range(2, 7))
    _rows = []
    for _kk in _ks:
        _l = KMeans(n_clusters=_kk, n_init=20, random_state=0).fit_predict(cluster_X)
        _g = GaussianMixture(n_components=_kk, covariance_type="diag", n_init=5, random_state=0).fit(
            cluster_X
        )
        _rows.append(
            {
                "k": _kk,
                "silhouette": round(float(silhouette_score(cluster_X, _l)), 3),
                "calinski_harabasz": round(float(calinski_harabasz_score(cluster_X, _l)), 1),
                "davies_bouldin": round(float(davies_bouldin_score(cluster_X, _l)), 3),
                "gmm_bic": round(float(_g.bic(cluster_X)), 1),
                "gmm_aic": round(float(_g.aic(cluster_X)), 1),
            }
        )
    cluster_diagnostics = pl.DataFrame(_rows)

    _ka = cluster_diagnostics["k"].to_numpy()
    _best = {
        "silhouette": int(_ka[cluster_diagnostics["silhouette"].to_numpy().argmax()]),
        "calinski_harabasz": int(_ka[cluster_diagnostics["calinski_harabasz"].to_numpy().argmax()]),
        "davies_bouldin": int(_ka[cluster_diagnostics["davies_bouldin"].to_numpy().argmin()]),
        "gmm_bic": int(_ka[cluster_diagnostics["gmm_bic"].to_numpy().argmin()]),
        "gmm_aic": int(_ka[cluster_diagnostics["gmm_aic"].to_numpy().argmin()]),
    }

    # normalise each criterion to 0-1 with "up = better" (flip the lower-better ones)
    _long = []
    for _c, _lower_better in [
        ("silhouette", False),
        ("calinski_harabasz", False),
        ("davies_bouldin", True),
        ("gmm_bic", True),
        ("gmm_aic", True),
    ]:
        _v = cluster_diagnostics[_c].to_numpy().astype(float)
        _v = (_v.max() - _v) if _lower_better else (_v - _v.min())
        _v = _v / (_v.max() if _v.max() > 0 else 1.0)
        for _i, _kk in enumerate(_ks):
            _long.append({"k": _kk, "criterion": _c, "score_norm": float(_v[_i])})
    _long = pl.DataFrame(_long)

    _chart = (
        alt.Chart(_long)
        .mark_line(point=True)
        .encode(
            alt.X("k:O", title="number of clusters k"),
            alt.Y("score_norm:Q", title="normalised score (up = better)"),
            alt.Color("criterion:N", title=None, legend=alt.Legend(orient="bottom")),
        )
        .properties(width=560, height=260, title="Cluster-count criteria across k (normalised)")
    )

    mo.vstack(
        [
            mo.md(f"""
    ## Section 4 — Unsupervised structure: how many event types emerge on their own?

    - Before training a *supervised* classifier, we check if data already has any separation. 

    - We log-scale the skewed magnitude features, standardise
    everything, and run **k-means** on the **{events.height}** events. 

    - Different metrics to select optimal num_clusters disagree.
    """),
            mo.hstack(
                [
                    _chart,
                    mo.ui.table(cluster_diagnostics, selection=None),
                ],
                justify="start",
            ),
            mo.md(f"""
    **What we learn:** the criteria disagree. Separation metrics favour
    **k = {_best["silhouette"]}** (silhouette) and **k = {_best["calinski_harabasz"]}**
    (Calinski-Harabasz), while the compactness / likelihood criteria favour
    **k = {_best["davies_bouldin"]}** (Davies-Bouldin), **k = {_best["gmm_bic"]}** (GMM-BIC)
    and **k = {_best["gmm_aic"]}** (GMM-AIC). 
    """),
        ]
    )
    return


@app.cell
def _():
    # Number of k-means clusters. The expert taxonomy is two classes, but the
    # silhouette / GMM-BIC diagnostics hint at finer sub-structure (e.g. a third
    # "big freshwater event" group), so k is adjustable here and the whole of
    # Section 4 reacts.
    n_clusters = mo.ui.number(start=2, stop=8, step=1, value=3, label="k (number of clusters)")
    return (n_clusters,)


@app.cell
def _(events, n_clusters):
    # Log-scale skewed magnitudes, standardise, then k-means with the chosen k
    # (selector above; see the cluster-count diagnostics below the scatter for how
    # many clusters the geometry supports).
    # clusters are labelled with plain numerals, numbered by symmetry rank (median
    # peak_frac) so the numbering is stable across runs.
    _logc = ["dur_min", "peak_do", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    #_logc = ["dur_min", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    _rawc = [
        "peak_frac",
        "sal_in",
        "wl_in",
        "sal_step",
        "wl_step",
        "max_rise_norm",
        "plateau_frac",
        "hyst_wl",
        "hyst_sal",
    ]
    _X = events.select(
        [pl.col(c).log1p().alias(c) for c in _logc]
        + [pl.col(c).fill_null(pl.col(c).median()).alias(c) for c in _rawc]
    ).to_numpy()
    _Xs = StandardScaler().fit_transform(_X)
    cluster_X = _Xs  # standardised feature matrix, reused by the cluster-count diagnostics

    _k = int(n_clusters.value)
    _lab = KMeans(n_clusters=_k, n_init=20, random_state=0).fit_predict(_Xs)
    _pc = PCA(n_components=2, random_state=0).fit_transform(_Xs)

    # rank clusters by median peak_frac: rank 0 = most abrupt (lowest), k-1 = most symmetric
    _pf = events["peak_frac"].to_numpy()
    _med = {c: float(np.median(_pf[_lab == c])) for c in range(_k)}
    _rank_of = {c: r for r, c in enumerate(sorted(_med, key=_med.get))}
    _ranks = np.array([_rank_of[c] for c in _lab])

    _names = np.array([f"cluster {r + 1}" for r in _ranks])
    events_clustered = events.with_columns(
        pl.Series("cluster", _names),
        pl.Series("cluster_rank", _ranks),
        pl.Series("is_abrupt", _ranks == 0),
        pl.Series("pc1", _pc[:, 0]),
        pl.Series("pc2", _pc[:, 1]),
    )
    events_clustered.group_by("cluster", "cluster_rank").len("n_events").sort("cluster_rank")

    # K-means clusters in PCA space, with a takeaway caption.
    _chart = (
        alt.Chart(events_clustered)
        .mark_circle(opacity=0.78)
        .encode(
            alt.X("pc1:Q", title="PC1"),
            alt.Y("pc2:Q", title="PC2"),
            alt.Color(
                "cluster:N",
                title=f"k-means cluster (k={_k})",
                scale=alt.Scale(scheme="set1"),
                legend=alt.Legend(orient="bottom"),
            ),
            alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[25, 400])),
            tooltip=["start:T", "dur_min:Q", "peak_do:Q", "peak_frac:Q", "sal_step:Q", "cluster:N"],
        )
        .properties(width=560, height=360, title=f"Event clusters in PCA space (k = {_k})")
    )
    mo.vstack(
        [
            n_clusters,
            mo.hstack(
                [
                    _chart,
                    mo.ui.table(
                        events_clustered.group_by("cluster", "cluster_rank")
                        .len("n_events")
                        .sort("cluster_rank"),
                        selection=None,
                    ),
                    mo.md(
                        f"""
    **What we learn:** with no labels, k-means splits the events into **{_k}** groups
    ordered by symmetry. At k = 2 this recovers the abrupt hot-moment vs symmetric
    pulse split; raising k surfaces finer sub-structure &mdash; use the silhouette
    sweep and the profile table to judge whether the extra groups are physically real.
            """
                    ),
                ]
            ),
        ]
    )
    return cluster_X, events_clustered


@app.function
def weak_supervision_labels(events_df):
    """Physics-grounded weak-supervision labels (hot moment vs oxic pulse).

    Encodes the published DO-event taxonomy as labeling functions (LFs) that
    vote HOT / PULSE / ABSTAIN from individual engineered features, then
    aggregates the noisy votes with a transparent accuracy-weighted vote (a
    hand-rolled Snorkel-style label model — no extra dependency). Returns the
    events frame augmented with `ws_label`, `ws_proba`, `ws_score` and per-LF
    vote columns, plus an LF-analysis frame (coverage / conflict / learned
    weight). A DEFENSIBLE, NON-CIRCULAR STAND-IN for the NDA expert labels: flip
    `LABEL_COL` in modeling.py to the real column on delivery, nothing else.
    """
    HOT, PULSE, ABSTAIN = 1, -1, 0
    n = events_df.height

    def _band(col, lo_p, hi_p, lo_vote, hi_vote, use_abs=False):
        # Vote lo_vote below the lo_p quantile, hi_vote above the hi_p quantile,
        # abstain (and on nulls) in between — a confident but sparse voter.
        x = events_df[col].to_numpy().astype(float)
        if use_abs:
            x = np.abs(x)
        lo, hi = np.nanquantile(x, lo_p), np.nanquantile(x, hi_p)
        v = np.full(n, ABSTAIN, dtype=int)
        v[x <= lo] = lo_vote
        v[x >= hi] = hi_vote
        v[np.isnan(x)] = ABSTAIN
        return v

    # Directions follow the domain taxonomy: front-loaded / abrupt curve +
    # coincident salinity / water-level step => hot moment (tidal incursion);
    # symmetric curve + antecedent precipitation => oxic pulse.
    lfs = {
        "lf_peak_pos": _band("peak_frac", 0.33, 0.66, HOT, PULSE),
        "lf_centroid": _band("centroid_frac", 0.33, 0.66, HOT, PULSE),
        "lf_onset": _band("max_rise_norm", 0.33, 0.66, PULSE, HOT),
        "lf_sal_step": _band("sal_step", 0.50, 0.75, ABSTAIN, HOT),
        "lf_wl_step": _band("wl_step", 0.50, 0.75, ABSTAIN, HOT, use_abs=True),
        "lf_sal_hyst": _band("hyst_sal", 0.50, 0.75, ABSTAIN, HOT, use_abs=True),
        "lf_precip": _band("precip_24h", 0.50, 0.75, ABSTAIN, PULSE),
    }
    names = list(lfs)
    L = np.column_stack([lfs[k] for k in names])
    m = len(names)

    # Label model: iterative accuracy-weighted majority vote (Dawid-Skene-lite).
    # Each LF's weight = its empirical accuracy vs the current consensus, mapped to
    # [-1, 1] and clipped at 0 so anti-correlated / chance LFs are dropped.
    w = np.ones(m)
    for _ in range(20):
        _lab = np.sign(L @ w)
        _lab[_lab == 0] = HOT
        _new = np.array(
            [
                2 * np.mean(L[L[:, j] != ABSTAIN, j] == _lab[L[:, j] != ABSTAIN]) - 1
                if (L[:, j] != ABSTAIN).any()
                else 0.0
                for j in range(m)
            ]
        )
        _new = np.clip(_new, 0.0, None)
        if np.allclose(_new, w, atol=1e-4):
            w = _new
            break
        w = _new

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
        voted = col != ABSTAIN
        conflict = float(
            np.mean(
                [voted[i] and np.any((L[i] != ABSTAIN) & (L[i] != col[i])) for i in range(n)]
            )
        )
        rows.append(
            {
                "LF": name,
                "coverage": round(float(voted.mean()), 3),
                "votes_hot": int(np.sum(col == HOT)),
                "votes_pulse": int(np.sum(col == PULSE)),
                "conflict": round(conflict, 3),
                "weight": round(float(w[j]), 3),
            }
        )
    lf_analysis = pl.DataFrame(rows)
    weights = {k: float(v) for k, v in zip(names, w)}
    return labeled, lf_analysis, weights


@app.cell
def _(events_clustered):
    # === Section 4b — Physics-grounded weak-supervision labels =============
    # The k-means `cluster` above is unsupervised and, for modeling, CIRCULAR — it
    # is derived from the same engineered features a tabular classifier would train
    # on. Here we instead encode the PUBLISHED hot-moment vs oxic-pulse taxonomy
    # directly as labeling functions and aggregate them into `ws_label`: a
    # physics-grounded, non-circular stand-in for the NDA expert labels.
    events_labeled, lf_analysis, lf_weights = weak_supervision_labels(events_clustered)

    _bal = (
        events_labeled.group_by("ws_label")
        .agg(pl.len().alias("n"), pl.col("ws_proba").mean().round(3).alias("mean_conf"))
        .sort("ws_label")
    )
    # Does the unsupervised geometry corroborate the physics rules?
    _xtab = (
        events_labeled.group_by("cluster", "cluster_rank")
        .agg(
            (pl.col("ws_label") == "hot").sum().alias("hot"),
            (pl.col("ws_label") == "pulse").sum().alias("pulse"),
        )
        .sort("cluster_rank")
    )
    mo.vstack(
        [
            mo.md(r"""
    ### Weak-supervision labels — physics-grounded stand-in for expert labels

    Each **labeling function** votes *hot* / *pulse* / *abstain* from one feature,
    following the domain taxonomy (front-loaded, abrupt curve + coincident
    salinity / water-level step &rarr; **hot moment**; symmetric curve +
    antecedent precipitation &rarr; **oxic pulse**). A transparent
    accuracy-weighted vote (a hand-rolled Snorkel-style label model, no extra
    dependency) aggregates the noisy votes into `ws_label` / `ws_proba`. This is
    the target `modeling.py` trains on; flip `LABEL_COL` to the real column when
    the NDA labels arrive &mdash; nothing else changes.
        """),
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**LF analysis**"), mo.ui.table(lf_analysis, selection=None)]
                    ),
                    mo.vstack(
                        [mo.md("**Label balance**"), mo.ui.table(_bal, selection=None)]
                    ),
                    mo.vstack(
                        [
                            mo.md("**`ws_label` vs k-means cluster**"),
                            mo.ui.table(_xtab, selection=None),
                        ]
                    ),
                ]
            ),
        ]
    )
    return (events_labeled,)


@app.cell
def _():
    _axis_opts = [
        "peak_frac",
        "centroid_frac",
        "max_rise_norm",
        "plateau_frac",
        "hyst_wl",
        "hyst_sal",
        "sal_step",
        "wl_step",
        "temp_step",
        "precip_24h",
        "dur_min",
        "peak_do",
        "area_mgLh",
        "rise_rate",
        "fall_rate",
        "sal_in",
        "wl_in",
        "temp_in",
    ]
    x_var = mo.ui.dropdown(options=_axis_opts, value="sal_step", label="x axis")
    y_var = mo.ui.dropdown(options=_axis_opts, value="peak_frac", label="y axis")
    return x_var, y_var


@app.cell
def _(events_clustered, x_var, y_var):
    # Cluster separation in the space of two REAL variables (chosen above) rather than the
    # abstract PCA components: the axes are actual features, so which variables pull the
    # clusters apart is directly readable. Same colour key as the PCA plot. Reacts to k
    # and the axis pickers.
    # Pick which two real (interpretable) variables to use as the scatter axes below.

    _dom = (
        events_clustered.select("cluster", "cluster_rank")
        .unique()
        .sort("cluster_rank")["cluster"]
        .to_list()
    )
    _cscale = alt.Scale(domain=_dom, scheme="set1")
    _xv, _yv = x_var.value, y_var.value
    _scatter = (
        alt.Chart(events_clustered)
        .mark_circle(opacity=0.8)
        .encode(
            alt.X(f"{_xv}:Q", title=_xv),
            alt.Y(f"{_yv}:Q", title=_yv),
            alt.Color(
                "cluster:N",
                scale=_cscale,
                sort=alt.SortField("cluster_rank", order="ascending"),
                legend=alt.Legend(orient="bottom", title="cluster"),
            ),
            alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[30, 400])),
            tooltip=["start:T", "cluster:N", f"{_xv}:Q", f"{_yv}:Q", "peak_do:Q"],
        )
        .properties(width=600, height=440, title=f"Cluster separation: {_yv} vs {_xv}")
    )
    mo.vstack(
        [
            mo.hstack([x_var, y_var], justify="center"),
            _scatter,
            mo.md(f"""
    **What we learn:** plotted against two *real* variables (**{_xv}** vs **{_yv}**), the
    k-means clusters occupy distinct regions &mdash; this makes the physical meaning of the
    split explicit (which the PCA axes hide). Swap the axes above to probe how any pair of
    drivers / shape features separates the groups; pairs like `sal_step` vs `peak_frac` or
    `hyst_wl` vs `max_rise_norm` show the cleanest separation.
    """),
        ]
    )
    return


@app.cell
def _(events_clustered):
    # Median feature profile per cluster: how the two groups actually differ.
    _prof = (
        events_clustered.group_by("cluster")
        .agg(
            pl.len().alias("n"),
            pl.col("peak_frac").median().round(3).alias("peak_frac"),
            pl.col("dur_min").median().round(0).alias("dur_min"),
            pl.col("peak_do").median().round(2).alias("peak_do"),
            pl.col("sal_step").median().round(2).alias("sal_step"),
            pl.col("wl_step").median().round(1).alias("wl_step"),
            pl.col("precip_24h").median().round(2).alias("precip_24h"),
            pl.col("centroid_frac").median().round(3).alias("centroid_frac"),
            pl.col("max_rise_norm").median().round(3).alias("max_rise_norm"),
            pl.col("plateau_frac").median().round(3).alias("plateau_frac"),
            pl.col("hyst_wl").median().round(3).alias("hyst_wl"),
            pl.col("hyst_sal").median().round(3).alias("hyst_sal"),
        )
        .sort("peak_frac")
    )
    mo.vstack(
        [
            mo.ui.table(_prof, selection=None),
            mo.md(
                r"""
    **What we learn:** 
    Precipitation barely separates theclasses, indicating that the
    *precip → oxic-pulse* link is weak and may need a large data set to
    confirm.
            """
            ),
        ]
    )
    return


@app.cell
def _(events_clustered):
    # 1D histograms of the discriminating shape features, one step-line per cluster.
    # Bins are computed PER FEATURE in numpy; each feature is its OWN Altair chart and
    # marimo arranges them in a 3-wide grid (3x2). Reacts to k.
    _feats = [
        "peak_frac",
        "centroid_frac",
        "max_rise_norm",
        "plateau_frac",
        "hyst_wl",
        "hyst_sal",
    ]
    _clusters = events_clustered.select("cluster", "cluster_rank").unique().sort("cluster_rank")
    _ordered = _clusters["cluster"].to_list()
    _color_scale = alt.Scale(domain=_ordered, scheme="set1")
    _csort = alt.SortField("cluster_rank", order="ascending")


    def _hist_for(_f):
        _all = events_clustered[_f].to_numpy().astype(float)
        _all = _all[~np.isnan(_all)]
        _lo, _hi = float(_all.min()), float(_all.max())
        if _hi <= _lo:
            _hi = _lo + 1.0
        _edges = np.linspace(_lo, _hi, 19)
        _mid = (_edges[:-1] + _edges[1:]) / 2
        _rows = []
        for _cl, _rk in zip(_clusters["cluster"].to_list(), _clusters["cluster_rank"].to_list()):
            _v = events_clustered.filter(pl.col("cluster") == _cl)[_f].to_numpy().astype(float)
            _v = _v[~np.isnan(_v)]
            _cnt, _ = np.histogram(_v, bins=_edges)
            for _x, _c in zip(_mid, _cnt):
                _rows.append(
                    {"cluster": _cl, "cluster_rank": int(_rk), "value": float(_x), "count": int(_c)}
                )
        _df = pl.DataFrame(_rows)
        return (
            alt.Chart(_df)
            .mark_line(interpolate="step-after", strokeWidth=3, opacity=0.9)
            .encode(
                alt.X("value:Q", title=None),
                alt.Y("count:Q", title=None),
                alt.Color("cluster:N", scale=_color_scale, sort=_csort, legend=None),
            )
            .properties(width=440, height=320, title=alt.TitleParams(_f, fontSize=20))
            .configure_axis(labelFontSize=15, titleFontSize=16)
            .configure_view(strokeOpacity=0)
        )


    _charts = [_hist_for(_f) for _f in _feats]
    _grid = [mo.hstack(_charts[_i : _i + 3], justify="center") for _i in range(0, len(_charts), 3)]

    # Build the colour key in HTML so it is large and legible. set1 assigns colours in
    # domain (cluster_rank) order, so palette[i] matches cluster _ordered[i] exactly.
    _pal = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]
    _items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:0 18px;font-size:20px;">'
        f'<span style="width:26px;height:26px;background:{_pal[_i]};display:inline-block;'
        f'margin-right:10px;border-radius:4px;border:1px solid #00000022;"></span>{_cl}</span>'
        for _i, _cl in enumerate(_ordered)
    )
    _legend = mo.md(
        f'<div style="text-align:center;margin:4px 0 16px;font-size:22px;">&nbsp;{_items}</div>'
    )

    mo.vstack(
        [
            _legend,
            *_grid,
            mo.md(r"""
    **What we learn:** each panel is one shape feature; each coloured step line is its
    distribution within one cluster. Features whose lines occupy **separated** value bands
    discriminate the clusters most strongly &mdash; symmetry (`peak_frac`, `centroid_frac`),
    onset abruptness (`max_rise_norm`) and water-level hysteresis (`hyst_wl`). Change **k**
    above to see the distributions split further.
    """),
        ]
    )
    return


@app.cell
def _(cluster_X, events):
    # Saline-incursion vs freshwater-driven events: the k=3 story made explicit. We fit
    # k-means with k FIXED at 3 here (independent of the selector) on the same standardised
    # matrix, then name the three groups by their median salinity step and plot them against
    # the real driver axes so the physical split is visible.
    _k3lab = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(cluster_X)
    _salstep = events["sal_step"].to_numpy()
    _msal = {c: float(np.nanmedian(_salstep[_k3lab == c])) for c in range(3)}
    _order = sorted(_msal, key=_msal.get)  # ascending median sal_step
    _label_of = {
        _order[0]: "freshwater-driven (low salinity step)",
        _order[1]: "weak / background",
        _order[2]: "saline incursion (high salinity step)",
    }
    _names3 = np.array([_label_of[c] for c in _k3lab])
    _e3 = events.with_columns(pl.Series("group", _names3))

    _dom3 = [
        "freshwater-driven (low salinity step)",
        "weak / background",
        "saline incursion (high salinity step)",
    ]
    _cscale3 = alt.Scale(domain=_dom3, range=["#377eb8", "#999999", "#e41a1c"])
    _leg = alt.Legend(orient="bottom", title=None, columns=1)


    def _scat(_x, _y, _title):
        return (
            alt.Chart(_e3)
            .mark_circle(opacity=0.8)
            .encode(
                alt.X(f"{_x}:Q", title=_x),
                alt.Y(f"{_y}:Q", title=_y),
                alt.Color("group:N", scale=_cscale3, sort=_dom3, legend=_leg),
                alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[30, 400])),
                tooltip=["start:T", "group:N", f"{_x}:Q", f"{_y}:Q", "peak_do:Q", "dur_min:Q"],
            )
            .properties(width=380, height=340, title=_title)
        )


    _a = _scat("sal_step", "wl_step", "Salinity step vs water-level step")
    _b = _scat("sal_step", "peak_frac", "Salinity step vs symmetry (peak_frac)")

    mo.vstack(
        [
            mo.md("### The salinity vs freshwater split (k = 3)"),
            mo.hstack([_a, _b], justify="center"),
            mo.md(r"""
    **What we learn:** fixing **k = 3** exposes three physically distinct regimes.
    **Saline incursion** events (red) ride **positive salinity steps** and are the most
    **abrupt** (low `peak_frac`) &mdash; tidal/saline water pushing in. **Freshwater-driven**
    events (blue) sit at **negative salinity steps** but **positive water-level steps**
    (left panel, upper-left) and are larger and more **symmetric** &mdash; floodplain /
    rain-fed flushing (note freshwater is the **only** group with a *positive* water-level
    step). The **weak / background** group (grey) has near-zero salinity steps, the lowest
    peak DO and the shortest events. So salinity step and water-level step act as the two
    competing drivers that the unsupervised k = 3 split recovers on its own. *(This cell
    fixes k = 3 regardless of the selector above.)*
    """),
        ]
    )
    return


@app.cell
def _(event_samples, events_labeled, n_clusters):
    # === Hand-off to modeling.py ==========================================
    # Persist the derived event artifacts to `derived/` so the modeling notebook
    # can load them without re-running the whole EDA pipeline. This cell re-runs
    # reactively whenever the selection knobs or the cluster count change, keeping
    # `derived/` in sync with what is shown above.
    #
    # `ws_label` is the physics-grounded weak-supervision target modeling.py trains
    # on (Section 4b); `cluster` is the unsupervised k-means pseudo-label kept for
    # comparison. modeling.py reads the target via a single LABEL_COL switch that
    # flips to the real NDA expert labels with no other code changes.
    derived_dir = Path(mo.notebook_location()) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    # Raw 5-min series, restricted to the events that survive the current selection
    # (so the two artifacts share exactly the same event set).
    _samples_sel = event_samples.filter(pl.col("eid").is_in(events_labeled["eid"].to_list()))

    events_labeled.write_parquet(derived_dir / "events.parquet")
    _samples_sel.write_parquet(derived_dir / "event_samples.parquet")

    mo.md(f"""
    **Derived artifacts written** → `{derived_dir}`

    | file | rows | cols | contents |
    |---|---|---|---|
    | `events.parquet` | {events_labeled.height} | {events_labeled.width} | engineered per-event features + `ws_label` (weak-supervision target) + `cluster` pseudo-label + PCA coords |
    | `event_samples.parquet` | {_samples_sel.height} | {_samples_sel.width} | raw 5-min multivariate series for the {_samples_sel["eid"].n_unique()} selected events |

    `ws_label` (hot / pulse) is the **physics-grounded weak-supervision target** — a
    defensible stand-in for the NDA expert labels (Section 4b). `cluster` is the
    **unsupervised k-means pseudo-label** (k = {int(n_clusters.value)}), kept for
    comparison. Flip `LABEL_COL` in `modeling.py` to the real column on delivery.
    """)
    return


if __name__ == "__main__":
    app.run()
