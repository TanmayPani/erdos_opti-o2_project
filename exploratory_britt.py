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
    return data, plt, train


@app.cell
def _(plt, train):
    train_cut = train[train["Dissolved Oxygen (mg/L)"] != 0.0]

    t = train_cut.index.values*5/(60*24*365)
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
    return


@app.cell
def _(data):
    data_cut = data[data["Dissolved Oxygen (mg/L)"] != 0.0]
    _index = data_cut.index.values
    _start = 0
    _last = 0
    _k = 0
    snapshots = []
    dt = []
    times = []
    time_since_last = []
    window = 1
    conv = 5.0/60.0
    time_since_last_cut = 12
    dt_cut = 6
    for _i in _index:
        if (_i - _last > time_since_last_cut) and (_k - _start > dt_cut):

            temp = data_cut[_start:_k]
            snapshots.append(temp)
            dt.append((_k - _start)*conv)
            time_since_last.append((_i - _last)*conv)
            times.append(_i*conv)
            _start = _k
        _last = _i
        _k += 1
    return dt, snapshots, time_since_last


@app.cell
def _(dt, plt, snapshots, time_since_last):
    print(len(snapshots))
    print(snapshots[43].values)
    plt.scatter(dt, time_since_last)
    plt.ylabel("Time Since Last Pulse (hr)")
    plt.xlabel('Length of Pulse (hr)')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.show()
    return


@app.cell
def _(plt):
    def graph_snapshot(snaps, N):
        s = snaps[N]
        l = len(s.index.values)
        t = np.arange(0, l, 1)*5
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('minutes')
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
