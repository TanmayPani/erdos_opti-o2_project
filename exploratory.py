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
    from sklearn.metrics import silhouette_score

    # Some columns carry per-day means/maxes that blow past Altair's default
    # 5k-row guard; we keep our chart payloads small but disable it to be safe.
    alt.data_transformers.disable_max_rows()

    data_dir_path = (
        mo.notebook_location()
        / "datasets"
        / "BeaverCreekWA_EssDive_26Jun2019-30Sep2024"
    )

    lf = pl.scan_csv(
        data_dir_path
        / "data"
        / "2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv",
        skip_rows_after_header=2,
        infer_schema=True,
        infer_schema_length=10000,
        # Spreadsheet artefacts ("#REF!") and blanks appear mid-file.
        null_values=["#REF!", ""],
        try_parse_dates=False,
    )

    # The trailing column is an empty spacer left over from the CSV export.
    lf = lf.select(pl.all().exclude(lf.collect_schema().names()[-1]))


@app.function
def binned_hist(frame, col, bins=60):
    s = frame.select(pl.col(col).alias("v")).drop_nulls()
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
            pl.when(pl.col(c).is_between(lo, hi))
            .then(pl.col(c))
            .otherwise(None)
            .alias(c)
            for c, (lo, hi) in _bounds.items()
        )
        .drop("WEATHER TIMESTAMP")
        .drop_nulls("Datetime")
        .unique("Datetime", keep="first")
        .sort("Datetime")
        .collect()
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Multimodality

    Histograms of the cleaned variables. Dissolved oxygen and O₂% use a **symlog**
    count axis so the secondary (oxic) mode is visible next to the dominant zero spike.
    """)
    return


@app.cell
def _(df):
    _row1 = mo.hstack(
        [
            hist_chart(df, "Dissolved Oxygen (mg/L)", symlog=True),
            hist_chart(df, "O2 Concentration (%)", symlog=True),
            hist_chart(df, "Well Salinity (PPT)"),
        ],
        widths="equal",
    )
    _row2 = mo.hstack(
        [
            hist_chart(df, "Flood plain water level in BGS (cm)"),
            hist_chart(df, "AirT_C_Avg"),
            hist_chart(df, "DO Sensor Temperature (C) "),
        ],
        widths="equal",
    )
    mo.vstack([_row1, _row2])
    return


@app.cell
def _(df):
    mo.vstack(
        [
            mo.md(
                r"""
        **Reading the modes.** Dissolved oxygen / O₂% are ~92% exact zeros (anoxic well)
        with a clearly separated oxic mode — a textbook *zero-inflated* distribution.
        Below, restricting to the oxic samples (DO > 0) reveals the shape of that second
        mode on its own; salinity meanwhile shows several regime peaks.
        """
            ),
            mo.hstack(
                [
                    hist_chart(
                        df.filter(pl.col("Dissolved Oxygen (mg/L)") > 0),
                        "Dissolved Oxygen (mg/L)",
                        bins=50,
                    ),
                    hist_chart(df, "Well Salinity (PPT)", bins=80),
                ],
                widths="equal",
            ),
        ]
    )
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
def _(df):
    _daily = (
        df.group_by_dynamic("Datetime", every="1d")
        .agg(
            pl.col("AirT_C_Avg").mean().alias("Air temp (°C)"),
            pl.col("Dissolved Oxygen (mg/L)").mean().alias("DO (mg/L)"),
            pl.col("Well Salinity (PPT)").mean().alias("Salinity (PPT)"),
            pl.col("Flood plain water level in BGS (cm)")
            .mean()
            .alias("Water level (cm BGS)"),
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
def _(df):
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
        widths="equal",
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
        widths="equal",
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
def _(df):
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
    mo.md(r"""
    ## Section 3 — Event detection & shape features

    The DO series sits at the anoxic baseline (DO = 0) ~92 % of the time; oxygen
    arrives in discrete **events**. Below we (i) **detect** events as contiguous
    departures from the baseline — merging across sub-hour dropouts so a briefly
    interrupted incursion stays one event — then (ii) engineer **shape features**
    (duration, peak, integrated load, and *time-to-peak symmetry*) and
    **antecedent / coincident drivers** (salinity and water-level *steps*, prior-24 h
    precipitation). These are the inputs a supervised classifier will use to
    separate abrupt, asymmetric **hot moments** from symmetric **oxic pulses**.
    """)
    return


@app.function
def detect_events(frame, do_col="Dissolved Oxygen (mg/L)", merge_gap=12):
    """One row per oxygenation event (contiguous DO > 0), merging baseline gaps
    of <= merge_gap samples (12 = 1 h at the 5-min cadence) into the surrounding
    event. Returns columns: eid, start, end."""
    f = frame.select("Datetime", (pl.col(do_col) > 0).alias("_oxic"))
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


@app.cell
def _():
    mo.md(r"""
    ### How an event is defined, filtered, and counted

    **Definition.** The well is anoxic (DO = 0) ~92 % of the time, so we treat any
    sample with **DO > 0 mg/L** as *oxic* and define a raw event as a **contiguous run of non-zero oxic samples** on the regular 5-minute grid.

    **Gap merging.** Occasionally events will flicker briefly back to zero (sensor noise, a
    short ebb). To avoid splitting one physical events unnecessarily, `detect_events`
    **combines events with a gap of <= 12 samples (1 hour)**
    (the `merge_gap` argument).

    **Selection of "proper" events.** After merging gaps a candidate is kept only if it has a
    genuine oxic peak and lasts long enough to have a shape worth classifying:

    - **peak DO >= 0.5 mg/L** — discards shallow near-zero flicker, and
    - **duration >= 15 min** (>= 3 samples) — discards short blips.

    **How we reach 73.** To summarize:

    | step | count |
    |---|---|
    | raw events (DO > 0) | **157** |
    | after merging baseline gaps <= 1 h | **119** |
    | after `peak >= 0.5 mg/L` **and** `dur >= 15 min` | **73** |
    """)
    return


@app.cell
def _(df):
    # Label every 5-min sample with the event that contains it, and record how
    # far into the event it falls (elapsed_min) for shape analysis.
    event_samples = (
        df.sort("Datetime")
        .join_asof(detect_events(df), left_on="Datetime", right_on="start", strategy="backward")
        .filter(pl.col("Datetime") <= pl.col("end"))
        .with_columns(
            (pl.col("Datetime") - pl.col("start")).dt.total_minutes().alias("elapsed_min")
        )
    )
    event_samples
    return (event_samples,)


@app.cell
def _(df, event_samples):
    # Per-event feature table: shape (duration, peak, load, symmetry, rise/fall)
    # plus antecedent / coincident drivers (salinity & water-level steps, prior-24 h
    # precipitation). Kept events have a real oxic peak (>= 0.5 mg/L) lasting >= 15 min.
    _do = "Dissolved Oxygen (mg/L)"
    _sal = "Well Salinity (PPT)"
    _wl = "Flood plain water level in BGS (cm)"
    _pr = "Precip (mm) over 5 minutes"

    # trailing 24 h context on the full series, sampled just before each event start
    _ctx = df.sort("Datetime").select(
        "Datetime",
        pl.col(_pr).rolling_sum_by("Datetime", window_size="24h").alias("precip_24h"),
        pl.col(_sal).rolling_mean_by("Datetime", window_size="24h").alias("sal_pre"),
        pl.col(_wl).rolling_mean_by("Datetime", window_size="24h").alias("wl_pre"),
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

    events = (
        _shape.join_asof(_ctx, left_on="start", right_on="Datetime", strategy="backward")
        .with_columns(
            (pl.col("sal_in") - pl.col("sal_pre")).alias("sal_step"),
            (pl.col("wl_in") - pl.col("wl_pre")).alias("wl_step"),
        )
        .filter((pl.col("peak_do") >= 0.5) & (pl.col("dur_min") >= 15))
        .select(
            "eid", "start", "end", "dur_min", "n_samples",
            "peak_do", "mean_do", "area_mgLh",
            "peak_frac", "rise_min", "fall_min", "rise_rate", "fall_rate",
            "sal_in", "wl_in", "sal_step", "wl_step", "precip_24h",
        )
        .sort("start")
    )
    events
    return (events,)


@app.cell
def _(events):
    # Symmetry of detected events. peak_frac = 0 -> oxygen jumps up then bleeds off
    # slowly (abrupt, asymmetric -> "hot moment"); peak_frac ~ 0.5 -> rises and falls
    # symmetrically ("oxic pulse").
    _chart = alt.Chart(events).mark_bar(color="#3b7dd8").encode(
        alt.X("peak_frac:Q", bin=alt.Bin(maxbins=24),
              title="time-to-peak fraction  (0 = abrupt rise / slow decay,  0.5 = symmetric)"),
        alt.Y("count():Q", title="number of events"),
    ).properties(width=560, height=200, title=f"Event symmetry  (n = {events.height} events)")
    mo.vstack([
        _chart,
        mo.md(
            r"""
    **What we learn:** the distribution is heavily skewed toward **fast rise long decay events** —
    most events rise much faster than they decay. Symmetry is therefore a good signal for classifying events. 
            """
        ),
    ])
    return


@app.cell
def _(events):
    # Coincident salinity step vs symmetry, sized by peak DO, coloured by prior precip.
    _chart = alt.Chart(events).mark_circle(opacity=0.7).encode(
        alt.X("sal_step:Q", title="coincident salinity step  (event mean - prior 24 h, PPT)"),
        alt.Y("peak_frac:Q", title="symmetry (time-to-peak fraction)"),
        alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[20, 400])),
        alt.Color("precip_24h:Q", title="prior-24 h precip (mm)", scale=alt.Scale(scheme="viridis")),
        tooltip=["start:T", "dur_min:Q", "peak_do:Q", "peak_frac:Q", "sal_step:Q", "precip_24h:Q"],
    ).properties(width=560, height=320, title="Event drivers: salinity step vs symmetry")
    mo.vstack([
        _chart,
        mo.md(
            r"""
    **What we learn:** there seems to be a trend toward abrupt (fast rise long decay) 
    events sitting at **higher salinity** — consistent with tidal-water incursion 
    driving abrupt events. 
            """
        ),
    ])
    return


@app.cell
def _(event_samples, events):
    # Representative event shapes, time-normalised: the two most abrupt/asymmetric
    # events (lowest peak_frac) vs the two most symmetric (peak_frac nearest 0.5).
    _lo = events.sort("peak_frac").head(2)["eid"].to_list()
    _sym = (
        events.with_columns((pl.col("peak_frac") - 0.5).abs().alias("_d"))
        .sort("_d").head(2)["eid"].to_list()
    )
    _pick = events.filter(pl.col("eid").is_in(_lo + _sym)).select("eid", "dur_min", "peak_frac")
    _curve = (
        event_samples.filter(pl.col("eid").is_in(_lo + _sym))
        .join(_pick, on="eid")
        .with_columns(
            (pl.col("elapsed_min") / pl.col("dur_min").clip(lower_bound=1)).alias("t_norm"),
            pl.col("Dissolved Oxygen (mg/L)").alias("do"),
            pl.format("event {} (pf={})", pl.col("eid"), pl.col("peak_frac").round(2)).alias("event"),
        )
    )
    _chart = alt.Chart(_curve).mark_line().encode(
        alt.X("t_norm:Q", title="normalised time within event (0 = start, 1 = end)"),
        alt.Y("do:Q", title="dissolved oxygen (mg/L)"),
        alt.Color("event:N", title=None),
    ).properties(width=560, height=260, title="Example event shapes: abrupt/asymmetric vs symmetric")
    mo.vstack([
        _chart,
        mo.md(
            r"""
    **What we learn:** The
    asymmetric events spike almost immediately then taper over the rest of the
    window (the classic asymmetric hot-moment signature), whereas the `peak_frac ~ 0.5`
    events rise and fall roughly evenly.
            """
        ),
    ])
    return


@app.cell
def _():
    mo.md(r"""
    ## Section 4 — Unsupervised structure: do the two classes emerge on their own?

    Before training a *supervised* classifier we ask whether the taxonomy is
    already latent in the data. We log-scale the skewed magnitude features, standardise
    everything, and run **k-means** on the 73 events. If the hot-moment / oxic-pulse
    split is real, an unsupervised method given **k = 2** should rediscover it.
    """)
    return


@app.cell
def _(events):
    # Log-scale skewed magnitudes, standardise, then k-means. Silhouette sweep picks
    # how many clusters the geometry actually supports; we fix k = 2 to test the
    # expert two-class hypothesis and relabel clusters by symmetry for a clear legend.
    _logc = ["dur_min", "peak_do", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    _rawc = ["peak_frac", "sal_in", "wl_in", "sal_step", "wl_step"]
    _X = events.select(
        [pl.col(c).log1p().alias(c) for c in _logc] + [pl.col(c) for c in _rawc]
    ).to_numpy()
    _Xs = StandardScaler().fit_transform(_X)

    cluster_silhouette = pl.DataFrame(
        {
            "k": list(range(2, 6)),
            "silhouette": [
                silhouette_score(_Xs, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(_Xs))
                for k in range(2, 6)
            ],
        }
    )

    _lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(_Xs)
    _pc = PCA(n_components=2, random_state=0).fit_transform(_Xs)
    # name clusters by symmetry: lower median peak_frac = abrupt/asymmetric
    _pf = events["peak_frac"].to_numpy()
    _asym = 0 if _pf[_lab == 0].mean() < _pf[_lab == 1].mean() else 1
    _names = np.where(_lab == _asym, "asymmetric (hot-moment-like)", "symmetric (pulse-like)")
    events_clustered = events.with_columns(
        pl.Series("cluster", _names),
        pl.Series("pc1", _pc[:, 0]),
        pl.Series("pc2", _pc[:, 1]),
    )
    cluster_silhouette
    return (events_clustered,)


@app.cell
def _(events_clustered):
    # K-means clusters in PCA space, with a takeaway caption.
    _chart = (
        alt.Chart(events_clustered)
        .mark_circle(opacity=0.78)
        .encode(
            alt.X("pc1:Q", title="PC1"),
            alt.Y("pc2:Q", title="PC2"),
            alt.Color("cluster:N", title="k-means cluster (k=2)",
                      scale=alt.Scale(scheme="set1"),
                      legend=alt.Legend(orient="bottom")),
            alt.Size("peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[25, 400])),
            tooltip=["start:T", "dur_min:Q", "peak_do:Q", "peak_frac:Q", "sal_step:Q"],
        )
        .properties(width=560, height=360, title="Event clusters in PCA space (k = 2)")
    )
    mo.vstack([
        _chart,
        mo.md(
            r"""
    **What we learn:** with no labels at all, k-means cleanly separates the events into
    two groups that map onto the classes.
            """
        ),
    ])
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
        )
        .sort("peak_frac")
    )
    mo.vstack([
        mo.ui.table(_prof, selection=None),
        mo.md(
            r"""
    **What we learn:** 
    Precipitation barely separates theclasses, indicating that the
    *precip → oxic-pulse* link is weak and may need a large data set to
    confirm.
            """
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
