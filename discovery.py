import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt

    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import (
        cohen_kappa_score,
        confusion_matrix,
    )

    from utils import (
        weak_supervision_labels,
        DISCOVERY_ROUTES,
        read_discovery_route,
    )

    alt.data_transformers.disable_max_rows()
    # Vega SVG renderer for every chart here — a default <canvas> initialises 0x0 inside a
    # hidden mo.ui.tabs panel / off-screen reveal slide and renders blank. marimo merges this
    # into each chart's usermeta.embedOptions; survives static HTML export. See modeling.py.
    _ = alt.renderers.set_embed_options(renderer="svg")  # `_ =`: suppress the repr


@app.cell
def disc_ui():
    _axis_opts = [
        "peak_frac",
        "centroid_frac",
        "max_rise_norm",
        "plateau_frac",
        # "hyst_wl",
        # "hyst_sal",
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
    # _axis_opts = [c for c in _axis_opts if c in events.columns]
    x_var = mo.ui.dropdown(options=_axis_opts, value="sal_step", label="x axis")
    y_var = mo.ui.dropdown(options=_axis_opts, value="peak_frac", label="y axis")
    num_kmeans_clusters = mo.ui.number(
        start=2, stop=8, step=1, value=3, label="k (number of clusters)"
    )
    route = mo.ui.radio(
        options=["events", "moments"],
        value="moments",
        label="Feature route",
        inline=True,
    )
    return route, num_kmeans_clusters, x_var, y_var


@app.cell(hide_code=True)
def _(route):
    mo.vstack(
        [
            route,
            mo.md(r"""
    # Driver discovery — unsupervised analysis (co-headline)

    A label-independent companion to `modeling.py`. The supervised pipeline asks
    *"can we reproduce the hot-moment vs oxic-pulse taxonomy?"*; here we ask the
    complementary questions that need **no labels at all** — and that feed Opti
    O2's longer-term sea-level-rise roadmap:

    1. **Do the two classes emerge on their own?** Cluster the events with no
       labels and ask whether the geometry recovers the physics-grounded
       expert **hot/pulse** split — and whether finer **sub-classes** appear.
    2. **Which drivers carry the class signal?** Rank the engineered features by
       mutual information with the event class (corroborates the known physical
       drivers: salinity / water-level *step* &rarr; hot moment; *antecedent
       precipitation* &rarr; oxic pulse).
    3. **Is the balance drifting over time?** Track the hot-moment fraction across
       the six-year record — the salinization signal behind the transition.

    > Toggle **route** above to run on preprocessing.py's augmented **event** table
    > (`derived/processed_expert_features.parquet`, hot/pulse/mixed) or the **moment** table
    > (`derived/processed_moments_features.parquet`, each moment typed hot/oxic). **DO-derived —
    > NDA-confidential; keep the rendered outputs on-machine.** The clustering is still
    > label-free; the expert labels are used only as a held-out reference to *score* the
    > unsupervised partitions.
    """),
        ]
    )
    return


@app.cell
def disc_input(route):
    # The route selected by the marimo radio — drives the *kernel-side* cells below
    # (the drift monitor). The chart cells that ship in the deck instead consume
    # `events_by_route` (both routes at once) so their route switch survives a static
    # HTML export. Same engineered features either way (the moment route's hysteresis
    # pair is degenerate). 2019 public-ESS-DIVE rows spliced in (is_public_augmented).
    # DO-derived => NDA.
    events = read_discovery_route(route.value)
    return events


@app.cell
def disc_routes():
    # BOTH routes, loaded once and keyed by name (moments first = the default). Cells that
    # render into slides.py take this instead of a single `events` frame: they embed both
    # routes and let a client-side Vega param / mo.ui.tabs do the switching, so the route
    # control keeps working with no Python kernel behind it.
    events_by_route = {_r: read_discovery_route(_r) for _r in DISCOVERY_ROUTES}
    return (events_by_route,)


@app.cell
def mutual_information(events_by_route):
    # Mutual information between each engineered feature and the event class —
    # a model-free importance that should rank the physically-expected drivers
    # (salinity / water-level step => hot; antecedent precip => pulse) at the top.
    # Engineered features: drop identifiers, labels, provenance flags and the expert-
    # annotated moment aggregates (leaky vs the label). Covers either schema.
    # Rendered for BOTH routes as mo.ui.tabs — tab switching is client-side, so the
    # route control still works in a kernel-less static HTML export.
    _drop = {
        "eid",
        "event_id",
        "unit_id",
        "event_id",
        "label_source",
        "regime",
        "split",
        "expert_label",
        "expert_subtype",
        "label",
        "class",
        "start",
        "end",
        # canonical schema renamed start/end -> start_time/end_time; without these the
        # raw timestamps get cast to float and rank as top "drivers" (pure leakage via
        # the multi-year drift in class balance).
        "start_time",
        "end_time",
        "n_samples",
        "rise_min",
        "fall_min",
        "is_public_augmented",
        "xf_qc_checked",
        "flooding",
        "concurrent_precip",
        "num_moments",
        "n_hot_moments",
        "n_oxic_pulses",
    }
    def _mi_chart(_events):
        _FEATURES = [
            c for c in _events.columns if c not in _drop and not c.startswith("moments_")
        ]
        _raw = _events.select(_FEATURES).to_numpy().astype(float)
        _Xs = StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(_raw)
        )
        # Real reference (hot = 1; 'mixed' counts as non-hot) — used ONLY to score the
        # label-free partitions, never fed to them.
        _y_hot = (_events["class"] == "hot").cast(pl.Int8).to_numpy()
        _mi = mutual_info_classif(_Xs, _y_hot, discrete_features=False, random_state=0)
        _rank = pl.DataFrame(
            {"feature": _FEATURES, "mutual_info": [round(float(v), 4) for v in _mi]}
        ).sort("mutual_info", descending=True)

        return (
            alt.Chart(_rank.head(14))
            .mark_bar(color="#3b6fb5")
            .encode(
                alt.X("mutual_info:Q", title="mutual information with event class"),
                alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature:N", "mutual_info:Q"],
            )
            .properties(
                width=520,
                height=360,
                title="Drivers of the hot-moment vs oxic-pulse distinction",
            )
        )

    mo.ui.tabs({_r: _mi_chart(_df) for _r, _df in events_by_route.items()})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 — Is the class balance drifting over time?
    """)
    return


@app.cell
def _(events):
    # Hot-moment fraction by year over the record — the salinization narrative.
    drift = (
        events.with_columns(pl.col("start_time").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("n_events"),
            (pl.col("class") == "hot").mean().round(3).alias("hot_fraction"),
        )
        .sort("year")
    )
    _bars = (
        alt.Chart(drift)
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
            title="Hot-moment fraction of detected events per year",
        )
    )
    _trend = _bars.transform_regression("year", "hot_fraction").mark_line(
        color="black", strokeDash=[4, 3]
    )
    mo.vstack(
        [
            mo.hstack([mo.ui.table(drift, selection=None), (_bars + _trend)]),
            mo.md(
                "**What we learn:** tracking the hot-moment fraction across the "
                "six-year record turns the static classifier into a **transition "
                "monitor** — a rising tidal-incursion share is the salinization "
                "signal Opti O2 ultimately wants to model against sea-level rise. "
                "These are the real expert labels over the augmented record, so the "
                "hot-moment share IS the salinization trend Opti O2 wants to model."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4 — Handcrafted-feature clustering & weak-supervision labels

    A deeper look at the same unsupervised structure probed in section 1, plus the
    **labeling-function method** that produced the `ws_label` target used throughout.
    We (a) sweep the cluster count against five model-selection criteria, (b) show the
    k-means partition in PCA space and against any two *real* features, (c) expose the
    physics-grounded **weak-supervision labeling functions** (coverage / conflict /
    learned weight), and (d) profile how the clusters differ feature-by-feature. All of
    it runs on `derived/processed_expert_features.parquet` (preprocessing.py) — **NDA-derived**.
    """)
    return


@app.cell
def kmeans_clustering(events_by_route, num_kmeans_clusters, x_var, y_var):
    # Interactive clustering as a SINGLE Altair/Vega spec whose controls (route, k, x-axis,
    # y-axis) are client-side Vega param bindings — so it stays interactive in a STATIC html
    # export (no kernel needed). k-means labels for every k in 2..8 are precomputed as columns
    # (k2..k8); a `binding_select` on k picks the colour field client-side via datum['k'+kSel].
    # Both routes are clustered SEPARATELY (KMeans/PCA must be fit within a route) and stacked
    # into one frame with a `route` column that a param-driven transform_filter selects.
    # The marimo widgets only SEED the initial selections (their .value = default).
    _ks = list(range(2, 9))
    _logc = ["dur_min", "peak_do", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    _rawc = [
        "peak_frac",
        "sal_in",
        "wl_in",
        "sal_step",
        "wl_step",
        "max_rise_norm",
        "plateau_frac",
    ]

    def _cluster_one(_events, _route):
        _X = _events.select(
            [pl.col(c).log1p().alias(c) for c in _logc]
            + [pl.col(c).fill_null(pl.col(c).median()).alias(c) for c in _rawc]
        ).to_numpy()
        _cluster_X = StandardScaler().fit_transform(_X)
        # PCA depends only on the (fixed) feature set, not on k → compute once per route.
        _pc = PCA(n_components=2, random_state=0).fit_transform(_cluster_X)

        # One ranked cluster-label column per candidate k (rank 0 = most abrupt = lowest
        # median peak_frac, so the numbering is stable across k and runs) -> k2..k8.
        _pf = _events["peak_frac"].to_numpy()
        _label_cols = {}
        for _kk in _ks:
            _lab = KMeans(n_clusters=_kk, n_init=20, random_state=0).fit_predict(
                _cluster_X
            )
            _med = {c: float(np.median(_pf[_lab == c])) for c in range(_kk)}
            _rank_of = {c: r for r, c in enumerate(sorted(_med, key=_med.get))}
            _label_cols[f"k{_kk}"] = [f"cluster {_rank_of[c] + 1}" for c in _lab]

        return _events.with_columns(
            pl.lit(_route).alias("route"),
            pl.Series("pc1", _pc[:, 0]),
            pl.Series("pc2", _pc[:, 1]),
            *[pl.Series(_n, _v) for _n, _v in _label_cols.items()],
        )

    _clustered = pl.concat(
        [_cluster_one(_df, _r) for _r, _df in events_by_route.items()],
        how="vertical_relaxed",
    )

    # Axis options (same list as disc_ui), restricted to columns actually present.
    _axis_opts = [
        c
        for c in [
            "peak_frac",
            "centroid_frac",
            "max_rise_norm",
            "plateau_frac",
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
        if c in _clustered.columns
    ]
    # Only the columns the spec actually reads get embedded in the exported HTML.
    events_clustered = _clustered.select(
        ["route", "start_time", "class", "pc1", "pc2"]
        + [f"k{_kk}" for _kk in _ks]
        + _axis_opts
    )

    _routes = list(events_by_route)
    _k_default = min(max(int(num_kmeans_clusters.value), _ks[0]), _ks[-1])
    _xf_default = x_var.value if x_var.value in _axis_opts else _axis_opts[0]
    _yf_default = y_var.value if y_var.value in _axis_opts else _axis_opts[1]

    _rSel = alt.param(
        name="routeSel",
        value=_routes[0],
        bind=alt.binding_select(options=_routes, name="route "),
    )
    _kSel = alt.param(
        name="kSel",
        value=_k_default,
        bind=alt.binding_select(options=_ks, name="k (clusters) "),
    )
    _xfSel = alt.param(
        name="xf",
        value=_xf_default,
        bind=alt.binding_select(options=_axis_opts, name="x axis "),
    )
    _yfSel = alt.param(
        name="yf",
        value=_yf_default,
        bind=alt.binding_select(options=_axis_opts, name="y axis "),
    )

    # Route filter first (both routes are stacked in one frame), then `cluster` = the label
    # column selected by k — both resolved client-side from the params.
    _base = (
        alt.Chart(events_clustered)
        .transform_filter("datum.route === routeSel")
        .transform_calculate(cluster="datum['k' + kSel]")
    )
    _pca = (
        _base.mark_circle(opacity=0.78)
        .encode(
            alt.X("pc1:Q", title="PC1"),
            alt.Y("pc2:Q", title="PC2"),
            alt.Color(
                "cluster:N",
                scale=alt.Scale(scheme="set1"),
                legend=alt.Legend(orient="bottom", title="k-means cluster"),
            ),
            alt.Size(
                "peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[25, 400])
            ),
            tooltip=[
                "start_time:T",
                "peak_do:Q",
                "peak_frac:Q",
                "sal_step:Q",
                "cluster:N",
            ],
        )
        .properties(width=360, height=280, title="Event clusters in PCA space")
    )
    _scatter = (
        _base.transform_calculate(xval="datum[xf]", yval="datum[yf]")
        .mark_circle(opacity=0.8)
        .encode(
            alt.X("xval:Q", title="x axis (selector →)"),
            alt.Y("yval:Q", title="y axis (selector →)"),
            alt.Color(
                "cluster:N",
                scale=alt.Scale(scheme="set1"),
                legend=alt.Legend(orient="bottom", title="cluster"),
            ),
            alt.Size(
                "peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[30, 400])
            ),
            tooltip=["start_time:T", "cluster:N", "xval:Q", "yval:Q", "peak_do:Q"],
        )
        .properties(width=360, height=280, title="Cluster separation (real features)")
    )
    # Purity view: per-cluster composition by expert class, normalised — reacts to k as well.
    _bar = (
        _base.mark_bar()
        .encode(
            alt.X("cluster:N", title="cluster", sort="ascending"),
            alt.Y("count():Q", title="fraction of cluster", stack="normalize"),
            alt.Color(
                "class:N",
                scale=alt.Scale(
                        domain=["hot", "oxic", "mixed"],
                        range=["#c1440e", "#3b7dd8", "#9467bd"],
                    ),
                legend=alt.Legend(orient="bottom", title="expert class"),
            ),
            tooltip=["cluster:N", "class:N", "count():Q"],
        )
        .properties(width=360, height=150, title="Cluster composition (purity)")
    )

    # resolve_scale independent: the concat otherwise SHARES one colour/size scale across all
    # three views, so the purity bar's hot/oxic colours get overridden by the scatters' set1
    # cluster scale. SVG renderer (via usermeta embedOptions): Vega defaults to a <canvas>, which
    # initialises at 0x0 inside an initially-hidden reveal slide (display:none) and stays blank;
    # SVG marks lay out correctly when the slide is later shown.
    _view = (
        alt.vconcat(
            alt.hconcat(_pca, _scatter).resolve_scale(
                color="independent", size="independent"
            ),
            _bar,
        )
        .resolve_scale(color="independent", size="independent")
        .add_params(_rSel, _kSel, _xfSel, _yfSel)
    )
    _view.usermeta = {"embedOptions": {"renderer": "svg"}}
    mo.vstack(
        [
            _view,
            mo.md(
                "*The **route / k / x axis / y axis** dropdowns are in-chart Vega controls, "
                "so they stay live even in a static HTML export (no Python kernel needed).*"
            ),
        ]
    )
    return


@app.cell
def weak_supervision(events_by_route):
    # === Physics-rule labels, cross-checked against the experts ============
    # Scored for BOTH routes and shown as mo.ui.tabs — tab switching is client-side, so the
    # route control keeps working in a kernel-less static HTML export.
    _CLS = ["hot", "oxic"]

    def _ws_confusion_chart(_events):
        _labeled, _lf_analysis, _lf_weights = weak_supervision_labels(_events)

        # Confusion matrix + Cohen's kappa: physics `ws_label` (pulse->oxic) vs the expert
        # class. The rules only distinguish hot vs oxic, so any 'mixed' events (event route
        # only) are held out of this 2-class scoring.
        _eval = _labeled.with_columns(
            pl.col("ws_label").replace({"pulse": "oxic"}).alias("ws_class")
        ).filter(pl.col("class").is_in(_CLS))
        _yt = _eval["class"].to_numpy()
        _yp = _eval["ws_class"].to_numpy()
        _kappa = float(cohen_kappa_score(_yt, _yp, labels=_CLS))
        _acc = float((_yt == _yp).mean())
        _cm = confusion_matrix(_yt, _yp, labels=_CLS)
        _conf = pl.DataFrame(
            [
                {"expert": _CLS[_i], "physics": _CLS[_j], "n": int(_cm[_i, _j])}
                for _i in range(len(_CLS))
                for _j in range(len(_CLS))
            ]
        )
        _cmax = int(_cm.max()) or 1
        _heat = (
            alt.Chart(_conf)
            .mark_rect()
            .encode(
                alt.X("physics:N", title="physics vote (ws_label)", sort=_CLS),
                alt.Y("expert:N", title="expert label", sort=_CLS),
                alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
            )
        )
        _txt = (
            alt.Chart(_conf)
            .mark_text(fontSize=26, fontWeight="bold")
            .encode(
                alt.X("physics:N", sort=_CLS),
                alt.Y("expert:N", sort=_CLS),
                text="n:Q",
                color=alt.condition(
                    alt.datum.n > _cmax / 2, alt.value("white"), alt.value("#111")
                ),
            )
        )
        return (_heat + _txt).properties(
            width=260,
            height=260,
            title=alt.TitleParams(
                f"Physics rules vs experts  (n={_eval.height})",
                subtitle=f"Cohen's kappa = {_kappa:.2f}   ·   accuracy = {_acc:.0%}",
                subtitleFontSize=13,
            ),
        )

    mo.vstack(
        [
            mo.md(r"""
    # Weak-supervision labels — physics-only vs expert


        """),
            mo.hstack(
                [
                    mo.md(
                        """
    - **Labeling functions** vote *hot* / *pulse* / *abstain* using single feature,
    - Accuracy-weighted vote aggregation &rarr; *weak-supervised labels*
    - Taxonomy (Approx.):
        1. Abrupt rise + Slow fall + coincident salinity / water-level step &rarr; **hot moment**
        2. Symmetric curve + Antecedent precipitation &rarr; **oxic pulse**
                        """
                    ),
                    mo.ui.tabs(
                        {
                            _r: _ws_confusion_chart(_df)
                            for _r, _df in events_by_route.items()
                        }
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
