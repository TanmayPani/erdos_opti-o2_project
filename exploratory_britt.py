import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import scipy.stats as st
    import sklearn.cluster as cl
    import sklearn.decomposition as dec


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
        fname = "Pulse_final/snapshot_" + str(N)
        plt.savefig(fname)

    return


@app.cell
def _():
    '''
    n = 0
    for _s in snapshots:
        graph_snapshot(snapshots, n)
        n += 1
    '''
    return


@app.cell
def _(snapshots):
    DO_max = []
    Prec_max = []
    Flood_max = []
    Sal_max = []

    DO_aver = []
    Prec_aver = []
    Flood_aver = []
    Sal_aver = []

    DO_med = []
    Prec_med = []
    Flood_med = []
    Sal_med = []

    for s in snapshots:
        DO_max.append(np.max(s['Dissolved Oxygen (mg/L)']))
        Prec_max.append(np.max(s['Precip (mm) over 5 minutes']))
        Flood_max.append(np.max(s['Flood plain water level in BGS (cm)']))
        Sal_max.append(np.max(s['Well Salinity (PPT)']))
        DO_aver.append(np.average(s['Dissolved Oxygen (mg/L)']))
        Prec_aver.append(np.average(s['Precip (mm) over 5 minutes']))
        Flood_aver.append(np.average(s['Flood plain water level in BGS (cm)']))
        Sal_aver.append(np.average(s['Well Salinity (PPT)']))
        DO_med.append(np.median(s['Dissolved Oxygen (mg/L)']))
        Prec_med.append(np.median(s['Precip (mm) over 5 minutes']))
        Flood_med.append(np.median(s['Flood plain water level in BGS (cm)']))
        Sal_med.append(np.median(s['Well Salinity (PPT)']))
    return (
        DO_aver,
        DO_max,
        DO_med,
        Flood_aver,
        Flood_max,
        Flood_med,
        Prec_aver,
        Prec_max,
        Prec_med,
        Sal_aver,
        Sal_max,
        Sal_med,
    )


@app.cell
def _(
    DO_aver,
    DO_max,
    DO_med,
    Flood_aver,
    Flood_max,
    Flood_med,
    Prec_aver,
    Prec_max,
    Prec_med,
    Sal_aver,
    Sal_max,
    Sal_med,
    plt,
):
    def graph_3var_color(title, var1, var2, var3, name1, name2, name3):
        fig, ax1 = plt.subplots()

        plt.title(title)
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.scatter(var1, var2, c=var3, cmap='coolwarm', s=8)
        plt.colorbar(label=name3)
        plt.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        #fname = "Pulse_final/snapshot_" + str(N)
        #plt.savefig(fname)

    graph_3var_color("Average", Flood_aver, DO_aver, Prec_aver, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Precip (mm) over 5 minutes')
    graph_3var_color("Max", Flood_max, DO_max, Prec_max, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Precip (mm) over 5 minutes')
    graph_3var_color("Median", Flood_med, DO_med, Prec_med, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Precip (mm) over 5 minutes')

    graph_3var_color("Average", Flood_aver, DO_aver, Sal_aver, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Well Salinity (PPT)')
    graph_3var_color("Max", Flood_max, DO_max, Sal_max, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Well Salinity (PPT)')
    graph_3var_color("Median", Flood_med, DO_med, Sal_med, 'Flood plain water level in BGS (cm)', 'Dissolved Oxygen (mg/L)', 'Well Salinity (PPT)')

    graph_3var_color("Average", Flood_aver, Sal_aver, DO_aver, 'Flood plain water level in BGS (cm)', 'Well Salinity (PPT)', 'Dissolved Oxygen (mg/L)')
    graph_3var_color("Max", Flood_max, Sal_max, DO_max, 'Flood plain water level in BGS (cm)', 'Well Salinity (PPT)', 'Dissolved Oxygen (mg/L)')
    graph_3var_color("Median", Flood_med, Sal_med, DO_med, 'Flood plain water level in BGS (cm)', 'Well Salinity (PPT)', 'Dissolved Oxygen (mg/L)')
    return (graph_3var_color,)


@app.cell
def _(DO_aver, Flood_aver, Sal_aver, graph_3var_color, plt, snapshots):
    #Sal
    rf_ratio = []
    skewness = []


    _col = ['DO Sensor Temperature (C) ', 'Well Salinity (PPT)', 'Flood plain water level in BGS (cm)', \
               'SlrFD_kW_Avg', 'Precip (mm) over 5 minutes', 'BP_hPa', 'WS_ms_S_WVT', 'RH']
    _col_wO = ['Dissolved Oxygen (mg/L)', 'DO Sensor Temperature (C) ', 'Well Salinity (PPT)', 'Flood plain water level in BGS (cm)', \
               'SlrFD_kW_Avg', 'Precip (mm) over 5 minutes', 'BP_hPa', 'WS_ms_S_WVT', 'RH']
    _col_3 = ['Well Salinity (PPT)', 'Flood plain water level in BGS (cm)', 'Precip (mm) over 5 minutes']

    metrics = []
    metrics_rf = []
    metrics_skew = []

    metrics_wO = []
    metrics_rf_wO = []
    metrics_skew_wO = []

    metrics_3 = []
    metrics_rf_3 = []
    metrics_skew_3 = []

    metrics_only = []
    n = 0
    for _snap in snapshots:
        _O2 = _snap['Dissolved Oxygen (mg/L)'].values

        _O2_m = np.max(_O2)
        _t = _snap.index.values
        _peak = np.average(_t[_O2 == _O2_m])
        _dt_rise = _peak - _t[0]
        _dt_fall = _t[-1] - _peak


        print(n, _dt_fall, _dt_rise, _dt_fall/_dt_rise)
        n += 1
        _rf_rat = _dt_fall/_dt_rise
        rf_ratio.append(_rf_rat)
        skewness.append(_snap['Dissolved Oxygen (mg/L)'].skew())


        metrics_only.append([_snap['Dissolved Oxygen (mg/L)'].skew(), _snap['Dissolved Oxygen (mg/L)'].mean()])
        _temp = []
        _temp_rf = []
        _temp_skew = []
        _temp_skew.append(_snap['Dissolved Oxygen (mg/L)'].skew())
        _temp_rf.append(_rf_rat)
        for _c in _col:
            _temp.append(_snap[_c].mean())
            _temp_skew.append(_snap[_c].mean())
            _temp_rf.append(_snap[_c].mean())
        metrics.append(_temp)
        metrics_rf.append(_temp_rf)
        metrics_skew.append(_temp_skew)

        _temp = []
        _temp_rf = []
        _temp_skew = []
        _temp_skew.append(_snap['Dissolved Oxygen (mg/L)'].skew())
        _temp_rf.append(_rf_rat)
        for _c in _col_wO:
            _temp.append(_snap[_c].mean())
            _temp_skew.append(_snap[_c].mean())
            _temp_rf.append(_snap[_c].mean())
        metrics_wO.append(_temp)
        metrics_rf_wO.append(_temp_rf)
        metrics_skew_wO.append(_temp_skew)


        _temp = []
        _temp_rf = []
        _temp_skew = []
        _temp_skew.append(_snap['Dissolved Oxygen (mg/L)'].skew())
        _temp_rf.append(_rf_rat)
        for _c in _col_3:
            _temp.append(_snap[_c].mean())
            _temp_skew.append(_snap[_c].mean())
            _temp_rf.append(_snap[_c].mean())
        metrics_3.append(_temp)
        metrics_rf_3.append(_temp_rf)
        metrics_skew_3.append(_temp_skew)

    metrics_only = np.array(metrics_only, dtype = 'double')

    metrics = np.array(metrics, dtype = 'double')
    metrics_rf = np.array(metrics_rf, dtype = 'double')
    metrics_skew = np.array(metrics_skew, dtype = 'double')

    metrics_wO = np.array(metrics_wO, dtype = 'double')
    metrics_rf_wO = np.array(metrics_rf_wO, dtype = 'double')
    metrics_skew_wO = np.array(metrics_skew_wO, dtype = 'double')

    metrics_3 = np.array(metrics_3, dtype = 'double')
    metrics_rf_3 = np.array(metrics_rf_3, dtype = 'double')
    metrics_skew_3 = np.array(metrics_skew_3, dtype = 'double')


    plt.scatter(Flood_aver, rf_ratio)
    plt.yscale('log')
    plt.show()
    plt.scatter(Flood_aver, np.fabs(skewness))
    plt.show()
    #graph_3var_color("Average", Flood_aver, Prec_aver, rf_ratio, 'Flood plain water level in BGS (cm)', 'Precip (mm) over 5 minutes', 'Rise Fall Ratio')
    graph_3var_color("", Flood_aver, np.log10(rf_ratio), DO_aver, 'Flood plain water level in BGS (cm)', 'log(Fall/Rise)', 'Mean Dissolved Oxygen (mg/L)')
    graph_3var_color("", Flood_aver, skewness, DO_aver, 'Flood plain water level in BGS (cm)', '|Skewness|', 'Mean Dissolved Oxygen (mg/L)')

    graph_3var_color("", Sal_aver, np.log10(rf_ratio), DO_aver, 'Well Salinity (PPT)', 'log(Fall/Rise)', 'Mean Dissolved Oxygen (mg/L)')
    graph_3var_color("", Sal_aver, skewness, DO_aver, 'Well Salinity (PPT)', '|Skewness|', 'Mean Dissolved Oxygen (mg/L)')
    return metrics_only, metrics_skew_wO


@app.cell
def _(plt):
    def Kmeans_dataset(datafork, kname, xn, yn, xname, yname):
        kmeans = cl.KMeans(init="k-means++", n_clusters=3, n_init=4)
        kmeans.fit(datafork)
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = datafork[:, xn].min() - 1, datafork[:, xn].max() + 1
        y_min, y_max = datafork[:, yn].min() - 1, datafork[:, yn].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
            )

        plt.plot(datafork[:, xn], datafork[:, yn], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, xn],
            centroids[:, yn],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
            )
        plt.title(
            "K-means clustering\n"
            "Centroids are marked with white cross"
            )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.show()

    return (Kmeans_dataset,)


@app.cell
def _(Kmeans_dataset, metrics_only):
    Kmeans_dataset(metrics_only, "Metrics from O2 Time Series Only", 0, 1, "Skew", "Mean O2")
    return


@app.cell
def _(metrics_skew_wO, plt):
    #Kmeans_dataset(metrics_skew_wO, "Everything", 0, 1, "Skew", "Mean O2")
    kmeans = cl.KMeans(init="k-means++", n_clusters=3, n_init=4)
    kmeans.fit(metrics_skew_wO)

    datafork = metrics_skew_wO
    Z = kmeans.predict(np.c_[metrics_skew_wO])
    xn = 0
    yn = 1
    plt.scatter(datafork[Z == 0][:, xn], datafork[Z == 0][:, yn], c = 'b', s=2)
    plt.scatter(datafork[Z == 1][:, xn], datafork[Z == 1][:, yn], c = 'k', s=2)
    plt.scatter(datafork[Z == 2][:, xn], datafork[Z == 2][:, yn], c = 'r', s=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, xn],
        centroids[:, yn],
        marker="x",
        s=169,
        linewidths=3,
        color="b",
        zorder=10,
        )
    plt.title(
        "K-means clustering\n"
        "Centroids are marked with white cross"
        )

    x_min, x_max = datafork[:, xn].min() - 1, datafork[:, xn].max() + 1
    y_min, y_max = datafork[:, yn].min() - 1, datafork[:, yn].max() + 1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("Skew")
    plt.ylabel("Mean O2")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
