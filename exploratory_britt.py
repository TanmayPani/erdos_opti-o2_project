import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np

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
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv("datasets/BeaverCreekWA_EssDive_26Jun2019-30Sep2024/data/2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv")
    data = data[data["Flood plain water level in BGS (cm)"] != "#REF!"]
    data = data[data.notna()]
    data = data[data.notnull()]
    data = data[2:]
    data = data.drop(["Unnamed: 15"], axis=1)
    print(data.columns)
    columns = data.columns
    columns = columns[columns != 'Datetime']
    columns = columns[columns != 'WEATHER TIMESTAMP']
    for _c in columns:
        data[_c] = pd.to_numeric(data[_c])
    missing = 497387 -2
    train = data[:missing]
    test = data[missing:]
    plt.plot(train.index.values, train["Dissolved Oxygen (mg/L)"])
    plt.plot(test.index.values, test["Dissolved Oxygen (mg/L)"])
    return plt, train


@app.cell
def _(plt, train):
    train_cut = train[train["Dissolved Oxygen (mg/L)"] != 0.0]

    t = train_cut.index.values*5/(60*60*24*365)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('years')
    ax1.set_ylabel('Flood plain water level in BGS (cm)', color=color)
    ax1.scatter(t, train_cut['Flood plain water level in BGS (cm)'], color=color, s=1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Precip (mm) over 5 minutes', color=color)  # we already handled the x-label with ax1
    ax2.scatter(t, train_cut['Precip (mm) over 5 minutes'], color=color, s=1)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('years')
    ax1.set_ylabel('Dissolved Oxygen (mg/L)', color=color)
    ax1.scatter(t, train_cut['Dissolved Oxygen (mg/L)'], color=color, s=1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Precip (mm) over 5 minutes', color=color)  # we already handled the x-label with ax1
    ax2.scatter(t, train_cut['Precip (mm) over 5 minutes'], color=color, s=1)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
    return (train_cut,)


@app.cell
def _(train_cut):
    _index = train_cut.index.values
    _start = 0
    _last = 0
    k = 0
    snapshots = []
    for _i in _index:
        if _i - _last > 1:

            temp = train_cut[_start:k]
            snapshots.append(temp)
            _start = k+1
        _last = _i
        k += 1
    return (snapshots,)


@app.cell
def _(snapshots):
    print(len(snapshots))
    print(snapshots[43].values)
    return


@app.cell
def _(plt):
    def graph_snapshot(snaps, N):
        s = snaps[N]
        l = len(s.index.values)
        t = np.arange(0, l, 1)*5/(60*60*24*365)
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('years')
        ax1.set_ylabel('Flood plain water level in BGS (cm)', color=color)
        ax1.scatter(t, s['Flood plain water level in BGS (cm)'], color=color, s=1)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Precip (mm) over 5 minutes', color=color)  # we already handled the x-label with ax1
        ax2.scatter(t, s['Precip (mm) over 5 minutes'], color=color, s=1)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax2.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:green'
        ax3.set_xlabel('years')
        ax3.set_ylabel('Dissolved Oxygen (mg/L)', color=color)
        ax3.scatter(t, s['Dissolved Oxygen (mg/L)'], color=color, s=1)
        ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fname = "snapshot_" + str(N)
        plt.savefig(fname)

    return (graph_snapshot,)


@app.cell
def _(graph_snapshot, snapshots):
    n = 0
    for s in snapshots:
        graph_snapshot(snapshots, n)
        n += 1
    return


if __name__ == "__main__":
    app.run()
