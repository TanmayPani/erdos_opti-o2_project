import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", layout_file="layouts/discovery.slides.json")

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

    from utils import weak_supervision_labels

    alt.data_transformers.disable_max_rows()


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
    # Per-event OR per-moment engineered features (toggle `route` in the setup cell).
    #   events  -> preprocessing.py's augmented per-event table (expert_label hot/pulse/mixed)
    #   moments -> per-moment table, each moment typed hot / oxic(pulse) individually.
    # Same engineered features either way (moments have NO hysteresis columns). The 2019
    # public-ESS-DIVE rows are spliced in (is_public_augmented). DO-derived => NDA.
    if route.value == "moments":
        events = pl.read_parquet("derived/processed_moments_features.parquet")
        _LABEL = "label"  # hot / pulse
    else:
        events = pl.read_parquet("derived/processed_expert_features.parquet")
        _LABEL = "expert_label"  # hot / pulse / mixed

    # Canonical display class: pulse -> oxic  (hot / oxic [/ mixed]).
    events = events.with_columns(
        pl.col(_LABEL).replace({"pulse": "oxic"}).alias("class")
    )

    return events


@app.cell
def mutual_information(events):
    # Mutual information between each engineered feature and the event class —
    # a model-free importance that should rank the physically-expected drivers
    # (salinity / water-level step => hot; antecedent precip => pulse) at the top.
    # Engineered features: drop identifiers, labels, provenance flags and the expert-
    # annotated moment aggregates (leaky vs the label). Covers either schema.
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
    _FEATURES = [
        c for c in events.columns if c not in _drop and not c.startswith("moments_")
    ]

    def _std(cols):
        _raw = events.select(cols).to_numpy().astype(float)
        return StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(_raw)
        )

    _Xs = _std(_FEATURES)

    # Real reference (hot = 1; 'mixed' counts as non-hot) — used ONLY to score the label-free
    # partitions, never fed to them.
    _y_hot = (events["class"] == "hot").cast(pl.Int8).to_numpy()
    _mi = mutual_info_classif(_Xs, _y_hot, discrete_features=False, random_state=0)
    mi_rank = pl.DataFrame(
        {"feature": _FEATURES, "mutual_info": [round(float(v), 4) for v in _mi]}
    ).sort("mutual_info", descending=True)

    alt.Chart(mi_rank.head(14)).mark_bar(color="#3b6fb5").encode(
        alt.X("mutual_info:Q", title="mutual information with event class"),
        alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature:N", "mutual_info:Q"],
    ).properties(
        width=520,
        height=360,
        title="Drivers of the hot-moment vs oxic-pulse distinction",
    )
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
def kmeans_clustering(events, num_kmeans_clusters, x_var, y_var):
    # Log-scale skewed magnitudes, standardise, then k-means with the chosen k
    # (selector above; see the cluster-count diagnostics below the scatter for how
    # many clusters the geometry supports).
    # clusters are labelled with plain numerals, numbered by symmetry rank (median
    # peak_frac) so the numbering is stable across runs.
    _logc = ["dur_min", "peak_do", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    # _logc = ["dur_min", "area_mgLh", "rise_rate", "fall_rate", "precip_24h"]
    _rawc = [
        "peak_frac",
        "sal_in",
        "wl_in",
        "sal_step",
        "wl_step",
        "max_rise_norm",
        "plateau_frac",
        # "hyst_wl",
        # "hyst_sal",
    ]
    _X = events.select(
        [pl.col(c).log1p().alias(c) for c in _logc]
        + [pl.col(c).fill_null(pl.col(c).median()).alias(c) for c in _rawc]
    ).to_numpy()
    _cluster_X = StandardScaler().fit_transform(_X)

    _k = int(num_kmeans_clusters.value)
    _lab = KMeans(n_clusters=_k, n_init=20, random_state=0).fit_predict(_cluster_X)
    _pc = PCA(n_components=2, random_state=0).fit_transform(_cluster_X)

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
    # Contingency of the label-free k-means cluster (rows) against the expert hot / oxic
    # typing (cols) — how each cluster splits by true class, plus per-cluster purity.
    _ct = (
        events_clustered.group_by(["cluster_rank", "cluster", "class"])
        .len()
        .pivot(on="class", index=["cluster_rank", "cluster"], values="len")
        .fill_null(0)
        .sort("cluster_rank")
    )
    for _cl in ["hot", "oxic"]:
        if _cl not in _ct.columns:
            _ct = _ct.with_columns(pl.lit(0, dtype=pl.Int64).alias(_cl))
    _class_cols = [c for c in _ct.columns if c not in ("cluster_rank", "cluster")]
    _ordered_class = [c for c in ["hot", "oxic"] if c in _class_cols] + [
        c for c in _class_cols if c not in ("hot", "oxic")
    ]
    cluster_crosstab = (
        _ct.with_columns(pl.sum_horizontal(_ordered_class).alias("total"))
        .with_columns(
            (pl.max_horizontal(_ordered_class) / pl.col("total"))
            .round(3)
            .alias("purity")
        )
        .select(["cluster", *_ordered_class, "total", "purity"])
    )

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
            alt.Size(
                "peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[25, 400])
            ),
            tooltip=[
                "start:T",
                "dur_min:Q",
                "peak_do:Q",
                "peak_frac:Q",
                "sal_step:Q",
                "cluster:N",
            ],
        )
        .properties(
            width=400, height=300, title=f"Event clusters in PCA space (k = {_k})"
        )
    )

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
            alt.Size(
                "peak_do:Q", title="peak DO (mg/L)", scale=alt.Scale(range=[30, 400])
            ),
            tooltip=["start:T", "cluster:N", f"{_xv}:Q", f"{_yv}:Q", "peak_do:Q"],
        )
        .properties(width=400, height=300, title=f"Cluster separation: {_yv} vs {_xv}")
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            num_kmeans_clusters,
                            _chart,
                        ]
                    ),
                    mo.vstack([mo.hstack([x_var, y_var], justify="start"), _scatter]),
                ],
                justify="start",
            ),
            mo.ui.table(cluster_crosstab, selection=None),
        ],
        align="center",
    )
    return


@app.cell
def weak_supervision(events):
    # === Physics-rule labels, cross-checked against the experts ============
    events_labeled, lf_analysis, lf_weights = weak_supervision_labels(events)

    # Confusion matrix + Cohen's kappa: physics `ws_label` (pulse->oxic) vs the expert
    # class. The rules only distinguish hot vs oxic, so any 'mixed' events (event route
    # only) are held out of this 2-class scoring.
    _CLS = ["hot", "oxic"]
    _eval = events_labeled.with_columns(
        pl.col("ws_label").replace({"pulse": "oxic"}).alias("ws_class")
    ).filter(pl.col("class").is_in(_CLS))
    _yt = _eval["class"].to_numpy()
    _yp = _eval["ws_class"].to_numpy()
    ws_kappa = float(cohen_kappa_score(_yt, _yp, labels=_CLS))
    ws_acc = float((_yt == _yp).mean())
    _cm = confusion_matrix(_yt, _yp, labels=_CLS)
    ws_confusion = pl.DataFrame(
        [
            {"expert": _CLS[_i], "physics": _CLS[_j], "n": int(_cm[_i, _j])}
            for _i in range(len(_CLS))
            for _j in range(len(_CLS))
        ]
    )
    _cmax = int(_cm.max()) or 1
    _heat = (
        alt.Chart(ws_confusion)
        .mark_rect()
        .encode(
            alt.X("physics:N", title="physics vote (ws_label)", sort=_CLS),
            alt.Y("expert:N", title="expert label", sort=_CLS),
            alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
        )
    )
    _txt = (
        alt.Chart(ws_confusion)
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
    _conf_chart = (_heat + _txt).properties(
        width=260,
        height=260,
        title=alt.TitleParams(
            f"Physics rules vs experts  (n={_eval.height})",
            subtitle=f"Cohen's kappa = {ws_kappa:.2f}   ·   accuracy = {ws_acc:.0%}",
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
                    _conf_chart,
                    # mo.ui.table(lf_analysis, selection=None),
                ],
                justify="start",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
