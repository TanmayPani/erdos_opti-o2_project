import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", layout_file="layouts/modeling.slides.json")

with app.setup:
    import marimo as mo
    import altair as alt

    import warnings
    from pathlib import Path

    import numpy as np
    import polars as pl
    import torch

    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils.class_weight import compute_sample_weight

    import shap

    from sklearn.impute import SimpleImputer
    from tabpfn import TabPFNClassifier

    # TabPFN's own interpretability (shapiq imputation explainer) — optional dep
    # (`uv pip install "tabpfn-extensions[interpretability]"`); guarded so the notebook
    # still loads without it (the TabPFN-SHAP cell degrades to a hint).
    try:
        from tabpfn_extensions.interpretability.shapiq import (
            get_tabpfn_imputation_explainer,
        )
        from tabpfn_extensions.interpretability.shap import shapiq_to_shap_explanation
    except ImportError:
        get_tabpfn_imputation_explainer = None
        shapiq_to_shap_explanation = None

    from core.features import FEATURE_COLS, MOMENT_FEATURE_COLS
    from core.estimator import RocketTransform

    # All model factories, CV harnesses, data readers, and the results formatter
    # live in `training.py` — the plain script that runs the full grouped-CV sweep
    # and writes `derived/model_results.parquet`. This notebook only *loads* those
    # precomputed tables and adds the SHAP / boundary explainability on top.
    from training import (
        xgb_fn,
        logreg_fn,
        inception_fn,
        read_tabular_features,
        read_time_series_curves,
        CV_REPEATS,
    )

    warnings.filterwarnings("ignore")
    alt.data_transformers.disable_max_rows()

    # Force the Vega SVG renderer for EVERY chart in this notebook. Vega defaults to a
    # <canvas>, which initialises 0x0 when its container is hidden (an inactive mo.ui.tabs
    # panel or an off-screen reveal.js slide) and then renders blank. marimo merges
    # `alt.renderers.options["embed_options"]` into each chart's `usermeta.embedOptions`
    # (per-chart `usermeta` still wins), so this one line covers the whole notebook and
    # survives a static HTML export.
    _ = alt.renderers.set_embed_options(renderer="svg")  # `_ =`: suppress the repr

    LABEL_COL = "label"


@app.cell
def _():
    # MCC / Cohen's kappa are still computed by training.py and live in the results parquet,
    # but are intentionally omitted from _METRIC_ORDER so they are not reported as tabs.
    _METRIC_ORDER = ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]
    _METRIC_LABEL = {
        "macro_f1": "F1-score",
        "bal_acc": "Balanced Acc.",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    def _metric_chance(n_classes=2):
        """Dashed chance reference per metric, for an `n_classes`-way problem. Balanced
        accuracy and macro-F1 under uniform guessing are both 1/n (0.5 binary, 0.333 for the
        3-class hot/pulse/mixed target); one-vs-rest macro ROC-AUC is 0.5 for any n. PR-AUC's
        chance level is the class prevalence, which differs per class, so it gets no line."""
        return {
            "macro_f1": 1 / n_classes,
            "bal_acc": 1 / n_classes,
            "roc_auc": 0.5,
        }

    def metric_bar_tabs(frame, table, sort="macro_f1", height=300, n_classes=2):
        """Per-table metric reporting as bar charts (models on x, one tab per metric),
        reading the tidy `model_results` frame [table, model, metric, mean, std]. Bars carry
        +/-std whiskers where present and a dashed chance reference (0.5 for F1 / bal-acc /
        ROC-AUC, 0 for MCC / kappa). Model order is shared across tabs (ranked by `sort`)."""
        _CHANCE = _metric_chance(n_classes)
        sub = frame.filter(pl.col("table") == table)
        if sub.height == 0:
            return mo.md(f"> no rows for table `{table}`.")
        metrics = [m for m in _METRIC_ORDER if m in set(sub["metric"].to_list())]
        _sm = sort if sort in metrics else metrics[0]
        order = (
            sub.filter(pl.col("metric") == _sm)
            .sort("mean", descending=True)["model"]
            .to_list()
        )
        width = max(340, 78 * len(order))
        tabs = {}
        for m in metrics:
            d = sub.filter(pl.col("metric") == m).with_columns(
                (pl.col("mean") - pl.col("std")).alias("lo"),
                (pl.col("mean") + pl.col("std")).alias("hi"),
            )
            unit = m in ("macro_f1", "bal_acc", "roc_auc", "pr_auc")
            _lo = float((d["mean"] - d["std"].fill_null(0.0)).min())
            dom = [0, 1] if unit else [min(-0.05, _lo), 1.0]
            base = alt.Chart(d).encode(
                x=alt.X(
                    "model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)
                )
            )
            bars = base.mark_bar().encode(
                y=alt.Y("mean:Q", title=_METRIC_LABEL[m], scale=alt.Scale(domain=dom)),
                color=alt.Color(
                    "model:N",
                    sort=order,
                    legend=None,
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=[
                    "model:N",
                    alt.Tooltip("mean:Q", format=".3f"),
                    alt.Tooltip("std:Q", format=".3f"),
                ],
            )
            whisker = base.mark_rule(strokeWidth=1.5, color="white").encode(
                y="lo:Q", y2="hi:Q"
            )
            labels = base.mark_text(
                angle=315,
                align="left",
                baseline="bottom",
                dx=5,
                dy=-4,
                fontSize=11,
                color="white",
            ).encode(y="mean:Q", text=alt.Text("mean:Q", format=".2f"))
            layers = [bars, whisker, labels]
            if m in _CHANCE:
                layers.append(
                    alt.Chart(pl.DataFrame({"y": [_CHANCE[m]]}))
                    .mark_rule(color="#c1440e", strokeDash=[4, 3])
                    .encode(y="y:Q")
                )
            tabs[_METRIC_LABEL[m]] = alt.layer(*layers).properties(
                width=width, height=height
            )
        return mo.ui.tabs(tabs)

    def metric_bars_grouped(frame, modes, sort="macro_f1", height=340, n_classes=2):
        """Grouped metric bars across evaluation `modes` — models on the x-axis, one bunched
        bar per mode within each model, one tab per metric. `modes` maps a display label ->
        table name in the tidy `model_results` frame; absent tables are skipped. +/-std
        whiskers (white) where present; dashed chance reference. Mode order follows `modes`;
        model order ranks by `sort` on the first mode."""
        _CHANCE = _metric_chance(n_classes)
        present = {l: t for l, t in modes.items() if t in set(frame["table"].to_list())}
        if not present:
            return mo.md("> no matching tables in `model_results`.")
        sub = frame.filter(pl.col("table").is_in(list(present.values()))).with_columns(
            pl.col("table").replace({t: l for l, t in present.items()}).alias("mode")
        )
        mode_order = list(present.keys())
        metrics = [m for m in _METRIC_ORDER if m in set(sub["metric"].to_list())]
        _ref = list(present.values())[0]
        _sm = sort if sort in metrics else metrics[0]
        order = (
            sub.filter((pl.col("table") == _ref) & (pl.col("metric") == _sm))
            .sort("mean", descending=True)["model"]
            .to_list()
        )
        for _m in sub["model"].unique(maintain_order=True).to_list():
            if _m not in order:
                order.append(_m)
        width = max(440, 116 * len(order))
        tabs = {}
        for metric in metrics:
            d = sub.filter(pl.col("metric") == metric).with_columns(
                (pl.col("mean") - pl.col("std")).alias("lo"),
                (pl.col("mean") + pl.col("std")).alias("hi"),
            )
            unit = metric in ("macro_f1", "bal_acc", "roc_auc", "pr_auc")
            _lo = float((d["mean"] - d["std"].fill_null(0.0)).min())
            dom = [0, 1] if unit else [min(-0.05, _lo), 1.0]
            base = alt.Chart(d).encode(
                x=alt.X(
                    "model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)
                ),
                xOffset=alt.XOffset("mode:N", sort=mode_order),
            )
            bars = base.mark_bar().encode(
                y=alt.Y(
                    "mean:Q", title=_METRIC_LABEL[metric], scale=alt.Scale(domain=dom)
                ),
                color=alt.Color(
                    "mode:N",
                    sort=mode_order,
                    title="mode",
                    scale=alt.Scale(scheme="tableau10"),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    "model:N",
                    "mode:N",
                    alt.Tooltip("mean:Q", format=".3f"),
                    alt.Tooltip("std:Q", format=".3f"),
                ],
            )
            whisker = base.mark_rule(strokeWidth=1.2, color="white").encode(
                y="lo:Q", y2="hi:Q"
            )
            layers = [bars, whisker]
            if metric in _CHANCE:
                layers.append(
                    alt.Chart(pl.DataFrame({"y": [_CHANCE[metric]]}))
                    .mark_rule(color="#c1440e", strokeDash=[4, 3])
                    .encode(y="y:Q")
                )
            tabs[_METRIC_LABEL[metric]] = alt.layer(*layers).properties(
                width=width, height=height
            )
        return mo.ui.tabs(tabs)

    def _confusion_tab(conf_frames, table):
        """Per-model PLAIN confusion matrices for one CV `table`, faceted column=model,
        row=route. `conf_frames` maps a route label -> the tidy confusion frame
        [table, model, true, pred, count] written by training.py. TRUE label on x, PREDICTED
        label on y, no normalisation — but the stored counts sum every out-of-fold prediction
        over all `CV_REPEATS` repeats, so they are divided by `CV_REPEATS` to read on the scale
        of the dataset (one CV pass = each event held out exactly once; the cells then total
        138 moments / 86 excursion units). Degrades to a note if the confusion artifact is
        absent (training.py not yet re-run)."""
        if not conf_frames:
            return mo.md(
                "> **Confusion matrix not computed** — run `.venv/bin/python training.py` "
                "(and `--mode moment`) to write `derived/model_confusion*.parquet`."
            )
        parts = []
        for route, fr in conf_frames.items():
            s = fr.filter(pl.col("table") == table)
            if s.height:
                parts.append(s.with_columns(pl.lit(route).alias("route")))
        if not parts:
            return mo.md(f"> no confusion rows for table `{table}` — re-run `training.py`.")
        # per-CV-pass counts: undo the summation over repeats (non-integer where a sample
        # sits near a decision boundary and flips between repeats — that spread is real).
        d = pl.concat(parts).with_columns(
            (pl.col("count") / CV_REPEATS).alias("count")
        )
        _classes = sorted(d["true"].unique().to_list())
        _models = d["model"].unique(maintain_order=True).to_list()
        # Row order: moment route on TOP, excursion on BOTTOM (then any others), so both routes
        # are always shown when their confusion artifacts exist.
        _present = set(d["route"].to_list())
        _routes = [r for r in ("moment", "excursion") if r in _present] + [
            r for r in conf_frames if r in _present and r not in ("moment", "excursion")
        ]
        _cmax = float(d["count"].max() or 1)  # text flips to white on the dark (high-count) cells
        _rect = alt.Chart().mark_rect().encode(
            x=alt.X("true:N", title="true label", sort=_classes),
            y=alt.Y("pred:N", title="predicted label", sort=_classes),
            color=alt.Color(
                "count:Q",
                title="events per CV pass",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(orient="bottom"),
            ),
        )
        _txt = alt.Chart().mark_text(baseline="middle", fontSize=11).encode(
            x=alt.X("true:N", sort=_classes),
            y=alt.Y("pred:N", sort=_classes),
            text=alt.Text("count:Q", format=".1f"),
            color=alt.condition(
                f"datum.count > {_cmax * 0.5}", alt.value("white"), alt.value("black")
            ),
        )
        _chart = (
            alt.layer(_rect, _txt, data=d)
            .properties(width=96, height=96)
            .facet(
                column=alt.Column("model:N", sort=_models, title=None),
                row=alt.Row("route:N", sort=_routes, title=None),
            )
        )
        # SVG renderer: this tab sits inside a hidden mo.ui.tabs panel, so a default canvas can
        # initialise 0x0 and stay blank until re-rendered — SVG lays out when the tab is shown.
        _chart.usermeta = {"embedOptions": {"renderer": "svg"}}
        return _chart

    def metric_bars_routes(
        frames, table, sort="macro_f1", conf_frames=None, height=340, n_classes=2
    ):
        """Grouped metric bars comparing classification routes for one CV `table` — models on
        the x-axis, one bunched bar per route within each model, one tab per metric. `frames`
        maps a route label -> its tidy results frame. +/-std whiskers (white); dashed chance
        reference; route order follows `frames`."""
        _CHANCE = _metric_chance(n_classes)
        parts, route_order = [], []
        for route, fr in frames.items():
            s = fr.filter(pl.col("table") == table)
            if s.height:
                parts.append(s.with_columns(pl.lit(route).alias("route")))
                route_order.append(route)
        if not parts:
            return mo.md(f"> no rows for table `{table}`.")
        sub = pl.concat(parts)
        metrics = [m for m in _METRIC_ORDER if m in set(sub["metric"].to_list())]
        _sm = sort if sort in metrics else metrics[0]
        order = (
            sub.filter((pl.col("route") == route_order[0]) & (pl.col("metric") == _sm))
            .sort("mean", descending=True)["model"]
            .to_list()
        )
        for _m in sub["model"].unique(maintain_order=True).to_list():
            if _m not in order:
                order.append(_m)
        width = max(440, 116 * len(order))
        tabs = {}
        for metric in metrics:
            d = sub.filter(pl.col("metric") == metric).with_columns(
                (pl.col("mean") - pl.col("std")).alias("lo"),
                (pl.col("mean") + pl.col("std")).alias("hi"),
            )
            unit = metric in ("macro_f1", "bal_acc", "roc_auc", "pr_auc")
            _lo = float((d["mean"] - d["std"].fill_null(0.0)).min())
            dom = [0, 1] if unit else [min(-0.05, _lo), 1.0]
            base = alt.Chart(d).encode(
                x=alt.X(
                    "model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)
                ),
                xOffset=alt.XOffset("route:N", sort=route_order),
            )
            bars = base.mark_bar().encode(
                y=alt.Y(
                    "mean:Q", title=_METRIC_LABEL[metric], scale=alt.Scale(domain=dom)
                ),
                color=alt.Color(
                    "route:N",
                    sort=route_order,
                    title="route",
                    scale=alt.Scale(scheme="set2"),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    "model:N",
                    "route:N",
                    alt.Tooltip("mean:Q", format=".3f"),
                    alt.Tooltip("std:Q", format=".3f"),
                ],
            )
            whisker = base.mark_rule(strokeWidth=1.2, color="white").encode(
                y="lo:Q", y2="hi:Q"
            )
            layers = [bars, whisker]
            if metric in _CHANCE:
                layers.append(
                    alt.Chart(pl.DataFrame({"y": [_CHANCE[metric]]}))
                    .mark_rule(color="#c1440e", strokeDash=[4, 3])
                    .encode(y="y:Q")
                )
            tabs[_METRIC_LABEL[metric]] = alt.layer(*layers).properties(
                width=width, height=height
            )
        if conf_frames is not None:
            tabs["Confusion Matrix"] = _confusion_tab(conf_frames, table)
        return mo.ui.tabs(tabs)

    return metric_bars_grouped, metric_bars_routes


@app.function
@mo.persistent_cache
def rocket_shap_panel(X_rocket, specs, class_names, y):
    """SHAP over the ROCKET features (XGBoost head), summed back to each kernel's
    (channel · dilation · pool) tag → an hstack of (channel reliance, time-scale
    reliance by dilation, top-kernel direction beeswarm). Shared by the excursion and
    moment routes — pass that route's kernel `specs` (rocket.module_.kernel_specs(),
    deterministic given the seed) so the cache key stays stable across kernels."""
    clf = xgb_fn(n_classes=len(class_names))
    clf.fit(X_rocket, y, sample_weight=compute_sample_weight("balanced", y))
    sv = shap.TreeExplainer(clf)(X_rocket).values  # (n_rows, 2*n_kernels[, n_classes])
    # 3-class target -> TreeExplainer returns a per-class axis. Channel/dilation reliance
    # and the direction beeswarm are all class-specific, so render the WHOLE panel once
    # per class and tab it, rather than averaging away the thing the panel is showing.
    if np.asarray(sv).ndim == 3:
        return mo.ui.tabs(
            {
                f"→ {c}": rocket_shap_one(sv[:, :, k], specs, c)
                for k, c in enumerate(class_names)
            }
        )
    return rocket_shap_one(sv, specs, class_names)


@app.function
def rocket_shap_one(sv, specs, class_names):
    """One ROCKET SHAP panel for a 2-D `(n_rows, 2*n_kernels)` attribution. `class_names`
    is either the binary pair (signed axis `class_names[0]` vs `[1]`) or a single class
    name string (one-vs-rest push toward it)."""
    _pos = class_names if isinstance(class_names, str) else class_names[1]
    _neg = f"not {_pos}" if isinstance(class_names, str) else class_names[0]

    # map every feature back to its kernel spec (feature 2k -> max, 2k+1 -> ppv of k).
    channels = ["do", "sal", "wl", "temp", "precip"]
    k = np.arange(sv.shape[1]) // 2
    feat_ch = np.asarray(specs["channel"])[k]
    feat_dil = np.asarray(specs["dilation"])[k]
    feat_type = np.where(np.arange(sv.shape[1]) % 2 == 0, "max", "ppv")
    mean_abs = np.abs(sv).mean(axis=0)

    feat_df = pl.DataFrame(
        {
            "mean_abs_shap": mean_abs.astype(float),
            "channel": [channels[c] for c in feat_ch],
            "dilation": feat_dil.astype(int),
            "type": feat_type,
        }
    )

    # 1) Channel reliance — Σ mean|SHAP| over all kernels on each channel.
    by_ch = (
        feat_df.group_by("channel")
        .agg(pl.col("mean_abs_shap").sum().alias("shap"))
        .sort("shap", descending=True)
    )
    chart_ch = (
        alt.Chart(by_ch)
        .mark_bar()
        .encode(
            alt.X("channel:N", sort="-y", title="Channel"),
            alt.Y("shap:Q", title="Σ mean|SHAP|"),
            color=alt.Color("channel:N", legend=None),
            tooltip=["channel", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(width=300, height=250, title="Channel reliance (Σ mean|SHAP|)")
    )

    # 2) Time-scale reliance — Σ mean|SHAP| by dilation, split by pooling type.
    by_dil = (
        feat_df.group_by(["dilation", "type"])
        .agg(pl.col("mean_abs_shap").sum().alias("shap"))
        .sort("dilation")
    )
    chart_dil = (
        alt.Chart(by_dil)
        .mark_bar()
        .encode(
            alt.X("dilation:O", title="Dilation (time scale)"),
            alt.Y("shap:Q", title="Σ mean|SHAP|"),
            color=alt.Color("type:N", title="Pool"),
            tooltip=["dilation", "type", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(width=400, height=250, title="Time-scale reliance by dilation")
    )

    # 3) Direction — beeswarm of the top-12 kernels, per-row SHAP coloured by
    #    (normalized) kernel activation, so red-right / red-left reads the push per class.
    top = np.argsort(mean_abs)[::-1][:12]
    rng = np.random.RandomState(0)
    bee_rows = []
    for f in top:
        label = f"{channels[feat_ch[f]]}·d{int(feat_dil[f])}·{feat_type[f]}#{f // 2}"
        col = X_rocket[:, f].astype(float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        for i in range(len(col)):
            bee_rows.append(
                {
                    "kernel": label,
                    "shap": float(sv[i, f]),
                    "act_norm": None if np.isnan(norm[i]) else float(norm[i]),
                    "jitter": float(rng.uniform(-0.35, 0.35)),
                }
            )
    bee_df = pl.DataFrame(bee_rows)
    order = list(dict.fromkeys(r["kernel"] for r in bee_rows))
    bee = (
        alt.Chart(bee_df)
        .mark_circle(size=26, opacity=0.65)
        .encode(
            x=alt.X(
                "shap:Q", title=f"SHAP value  (◀ {_neg} · {_pos} ▶)"
            ),
            y=alt.Y("kernel:N", sort=order, title=None),
            yOffset="jitter:Q",
            color=alt.Color(
                "act_norm:Q",
                scale=alt.Scale(scheme="redblue", reverse=True),
                title="kernel activation (low → high)",
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["kernel", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(width=460, height=320, title="Top kernels — direction & magnitude")
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(x="z:Q")
    )
    return mo.vstack([mo.hstack([chart_ch, bee + zero], justify="start"), chart_dil])


@app.function
@mo.persistent_cache
def shap_feature_panel(X, y, class_names, feature_cols, topk=12):
    """XGBoost + TreeExplainer SHAP for one tabular route → an hstack of (mean-|SHAP|
    bar, per-row beeswarm coloured by feature value). Shared by the excursion and moment
    routes — pass that route's `FEATURE_COLS` / `MOMENT_FEATURE_COLS`. Balanced-weighted
    full-data fit; read directions, not precise magnitudes, at this small n.

    On the 3-class hot/pulse/mixed target TreeExplainer returns `(n, f, K)`; the signed
    beeswarm has no single axis then, so this hands off to `tabular_shap_charts`, which
    renders one tab per class. (The binary path keeps its own charts because it also shows
    the XGBoost gain column alongside mean |SHAP|.)"""
    clf = xgb_fn(n_classes=len(class_names))
    clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    sv = shap.TreeExplainer(clf)(X).values  # (n_rows, n_features[, n_classes])
    if np.asarray(sv).ndim == 3:
        return tabular_shap_charts(sv, X, feature_cols, class_names, topk)

    imp = pl.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(sv).mean(axis=0).astype(float),
            "xgb_gain": clf.feature_importances_.astype(float),
        }
    ).sort("mean_abs_shap", descending=True)
    top = imp["feature"].to_list()[:topk]

    # one dot per (row, top-feature), coloured by the per-feature min-max normalized
    # value, with vertical jitter to reduce overplotting.
    rng = np.random.RandomState(0)
    bee_rows = []
    for f in top:
        j = feature_cols.index(f)
        col = X[:, j].astype(float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        for i in range(len(col)):
            bee_rows.append(
                {
                    "feature": f,
                    "shap": float(sv[i, j]),
                    "feat_norm": None if np.isnan(norm[i]) else float(norm[i]),
                    "value": None if np.isnan(col[i]) else float(col[i]),
                    "jitter": float(rng.uniform(-0.35, 0.35)),
                }
            )
    bee_df = pl.DataFrame(bee_rows)

    bar = (
        alt.Chart(imp.head(topk))
        .mark_bar(color="#4575b4")
        .encode(
            alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"),
            alt.Y("feature:N", sort=top, title=None),
            tooltip=[
                "feature:N",
                alt.Tooltip("mean_abs_shap:Q", format=".3f"),
                alt.Tooltip("xgb_gain:Q", format=".3f"),
            ],
        )
        .properties(width=300, height=430, title="Global importance — mean |SHAP|")
    )
    bee = (
        alt.Chart(bee_df)
        .mark_circle(size=26, opacity=0.65)
        .encode(
            x=alt.X(
                "shap:Q",
                title=f"SHAP value  (◀ {class_names[0]} · {class_names[1]} ▶)",
            ),
            y=alt.Y("feature:N", sort=top, title=None),
            yOffset="jitter:Q",
            color=alt.Color(
                "feat_norm:Q",
                scale=alt.Scale(scheme="redblue", reverse=True),
                title="feature value (low → high)",
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                "feature:N",
                alt.Tooltip("value:Q", format=".3g"),
                alt.Tooltip("shap:Q", format=".3f"),
            ],
        )
        .properties(width=440, height=430, title="Per-row SHAP — direction & magnitude")
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(x="z:Q")
    )
    return mo.hstack([bar, bee + zero], justify="start")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Event classification

    Classifying detected dissolved-oxygen excursions as **hot moment** vs **oxic
    pulse**, on the real **NDA expert labels** (via `preprocessing.py` →
    `derived/processed_auto_features.parquet`). *Mixed* (`hx`) events and the undefined `b`
    subtype are held out (`split=="holdout"`) — the next section asks where the
    binary model puts them.

    - **Unit** = an auto-detected DO excursion, nested in the expert umbrella that
      supplies its class (`event_id`); a few audited orphans were adopted with
      `label_source="inferred"`.
    - **Target** = binary `label` on `split=="train"`; only the 24 `FEATURE_COLS`
      enter X (metadata columns never do). Live counts are printed by the load cell
      below.
    - **CV** = **StratifiedGroupKFold on `event_id`** so an umbrella's nested units
      never straddle a fold (the leakage the old plain-stratified harness allowed),
      plus **leave-one-group-out** as a small-*n* robustness estimate. Metrics lead
      with **macro-F1** and **balanced accuracy** (the class imbalance).

    Two tracks: **A** — gradient-boosting / linear on the engineered features;
    **B** — deep learning on the raw `proc_curves.parquet` 128×5 sequences.
    """)
    return


@app.cell
def _():
    X, y, class_names, train = read_tabular_features(
        "derived/processed_auto_features.parquet"
    )
    class_labels = list(range(len(class_names)))
    groups = train["event_id"].to_numpy()

    proc = pl.read_parquet("derived/processed_auto_features.parquet")
    # Precomputed grouped-CV sweeps (all tables, mean ± std) — this notebook loads them
    # rather than re-running TabPFN / ROCKET / InceptionTime folds. `training.py` writes
    # one per route: `model_results.parquet` (excursion, `--mode excursion`) and
    # `model_results_moment.parquet` (`--mode moment`). The moment file may be absent
    # (route not yet run) → the route-comparison cell degrades to excursion-only.
    model_results = pl.read_parquet("derived/model_results.parquet")
    _mom_path = Path("derived/model_results_moment.parquet")
    model_results_moment = pl.read_parquet(_mom_path) if _mom_path.exists() else None

    _mom_note = (
        f"{model_results_moment['table'].n_unique()} tables"
        if model_results_moment is not None
        else "**absent** — run `training.py --mode moment`"
    )
    mo.md(
        f"""
        - **Loaded** `proc_features` — {proc.height} units; **{train.height} trainable**
        (split=="train"), {proc.filter(pl.col("split") == "holdout").height} held out
        (mixed + `b`). Target = `{LABEL_COL}` (hot vs pulse).

        - **Design matrix:** {X.shape[0]} units × {X.shape[1]} features over
        {len(set(groups))} CV groups. **Classes:**
        {", ".join(f"`{c}`" for c in class_names)}; counts = {np.bincount(y).tolist()}
        (base rate {np.bincount(y).max() / len(y):.0%}).

        - **Results:** excursion route `derived/model_results.parquet`
        ({model_results["table"].n_unique()} CV tables); moment route {_mom_note}
        (both precomputed by `training.py`).
        """
    )
    return (
        X,
        class_names,
        groups,
        model_results,
        model_results_moment,
        train,
        y,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Two routes to Challenge 1 — excursion units vs expert moments

    The same **hot-moment vs oxic-pulse** target is learned two ways, differing only in
    *what an event is* (both share the CV harness, feature builders, and model heads in
    `training.py`; pick with `--mode`):

    | | **excursion** (`--mode excursion`) | **moment** (`--mode moment`) |
    |---|---|---|
    | **event** | auto-detected DO excursion ("unit") | expert-annotated moment window |
    | **label** | inherited from the enclosing expert umbrella | the moment's own hot/oxic tag |
    | **features** | 24 `FEATURE_COLS` | 22 `MOMENT_FEATURE_COLS` (hysteresis dropped — degenerate on one tight pulse) |
    | **`mixed` holdout** | yes (`hx`/`b` held out) | none — every moment is directly typed |
    | **artifact** | `model_results.parquet` | `model_results_moment.parquet` |

    The excursion route sidesteps segmentation (detect-then-inherit); the moment route
    is the literal Challenge-1 ask on the expert windows. Below, the two routes are
    compared head-to-head on the headline grouped-CV tables; the **detailed
    explainability that follows** (SHAP, mixed-event holdout, Track-B ROCKET readout)
    is shown for the **excursion** route — the moment route mirrors the identical
    pipeline.
    """)
    return


@app.cell
def metrics_routes_plot(
    metric_bars_routes,
    model_results,
    model_results_moment,
    train,
):
    _frames = {"excursion": model_results}
    if model_results_moment is not None:
        _frames["moment"] = model_results_moment

    # Per-route confusion matrices (grouped-CV, from training.py) for the Confusion Matrix tab;
    # absent files are skipped and the tab shows a "run training.py" note instead.
    _conf_frames = {}
    for _route, _cp in [
        ("excursion", "derived/model_confusion.parquet"),
        ("moment", "derived/model_confusion_moment.parquet"),
    ]:
        if Path(_cp).exists():
            _conf_frames[_route] = pl.read_parquet(_cp)

    # class count drives the dashed chance line (1/n for macro-F1 / bal-acc); take it from
    # the confusion frames' distinct true labels, else from the loaded class_names.
    _n_cls = max(
        [len(class_names)] + [f["true"].n_unique() for f in _conf_frames.values()]
    )
    _n_exc = f"{train.height} units / {train['event_id'].n_unique()} umbrellas"
    _mp = pl.read_parquet("derived/processed_moments_features.parquet").filter(
        pl.col("split") == "train"
    )
    _n_mom = f"{_mp.height} moments / {_mp['event_id'].n_unique()} umbrellas"

    _note = (
        "> ⚠️ **Moment route not yet computed** — run `.venv/bin/python training.py "
        "--mode moment` to write `derived/model_results_moment.parquet`; only the "
        "excursion route shows until then.\n\n"
        if model_results_moment is None
        else ""
    )
    mo.vstack(
        [
            mo.md(
                f"""
                - **Trainable events:** excursion = {_n_exc}; moment = {_n_mom}. 
                -  **Over 0.9 F1-score** and accuracy across models and feature modes
                  """
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(
                                "**Engineered features, grouped 5×5 (excursion vs moment):**"
                            ),
                            metric_bars_routes(
                                _frames,
                                "base_grouped",
                                conf_frames=_conf_frames,
                                n_classes=_n_cls,
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(
                                "**ROCKET / InceptionTime on raw curves (excursion vs moment):**"
                            ),
                            metric_bars_routes(
                                _frames,
                                "dl_grouped",
                                conf_frames=_conf_frames,
                                n_classes=_n_cls,
                            ),
                        ],
                        # justify="space-around",
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Track A — Gradient-boosting + linear baselines (engineered features)

    **[XGBoost](https://arxiv.org/abs/1603.02754)** and
    **[CatBoost](https://arxiv.org/abs/1706.09516)** (both strong on small tabular
    data), with an **[L2 logistic-regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**
    floor. Balanced class weighting throughout; shallow trees against the tiny *n*.
    ([TabPFN](https://www.nature.com/articles/s41586-024-08328-6) slots in if its
    one-time license is accepted / `TABPFN_TOKEN` is set.)
    """)
    return


@app.cell
def metrics_grouped_plot(metric_bars_grouped, model_results):
    mo.vstack(
        [
            mo.md(
                "**Track A & B Validation Protocols.** Each model's bunched bars compare the main 5x5 grouped CV against "
                "temporal splits, LOGO, and rigorous shuffling controls. "
                "Metric = tabs; whiskers = ±std across the 25 grouped-CV folds."
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(
                                "**Track A (Engineered Features)**: Grouped vs Temporal/LOGO/Label-shuffled"
                            ),
                            metric_bars_grouped(
                                model_results,
                                {
                                    "Grouped CV": "base_grouped",
                                    "LOGO": "base_logo",
                                    "Temporal": "base_temporal",
                                    "Label-shuffled": "base_label_shuffled",
                                },
                                n_classes=len(class_names),
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(
                                "**Track B (Raw Curves)**: Grouped vs Wide/Label-shuffled/Time-shuffled"
                            ),
                            metric_bars_grouped(
                                model_results,
                                {
                                    "Grouped CV": "dl_grouped",
                                    "Wide 48h": "dl_wide_grouped",
                                    "Label-shuffled": "dl_label_shuffled",
                                    "Time-shuffled": "dl_time_shuffled",
                                },
                                n_classes=len(class_names),
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
    return


@app.function
def tabular_shap_charts(sv, X, feature_cols, class_names, topk=12, axis_title=None):
    """Shared bar + beeswarm renderer for a tabular SHAP matrix. Mirrors
    `shap_feature_panel`'s charts so the logistic / TabPFN panels read side-by-side with the
    XGBoost one.

    `sv` is either 2-D `(n_rows, n_features)` — the binary class-1 direction — or 3-D
    `(n_rows, n_features, n_classes)` for the 3-class hot/pulse/mixed target, in which case
    one TAB PER CLASS is returned, each showing that class's one-vs-rest push. SHAP for a
    K-way model is K separate attributions with no single signed axis to collapse them onto,
    so tabs rather than an aggregate."""
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return mo.ui.tabs(
            {
                f"→ {c}": tabular_shap_charts(
                    sv[:, :, k],
                    X,
                    feature_cols,
                    class_names,
                    topk,
                    axis_title=f"SHAP value  (◀ not {c} · {c} ▶)",
                )
                for k, c in enumerate(class_names)
            }
        )
    Xa = np.asarray(X, dtype=float)
    cols = list(feature_cols)
    imp = pl.DataFrame(
        {
            "feature": cols,
            "mean_abs_shap": np.abs(sv).mean(axis=0).astype(float),
        }
    ).sort("mean_abs_shap", descending=True)
    top = imp["feature"].to_list()[:topk]

    rng = np.random.RandomState(0)
    bee_rows = []
    for f in top:
        j = cols.index(f)
        col = Xa[:, j].astype(float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        for i in range(len(col)):
            bee_rows.append(
                {
                    "feature": f,
                    "shap": float(sv[i, j]),
                    "feat_norm": None if np.isnan(norm[i]) else float(norm[i]),
                    "value": None if np.isnan(col[i]) else float(col[i]),
                    "jitter": float(rng.uniform(-0.35, 0.35)),
                }
            )
    bee_df = pl.DataFrame(bee_rows)

    bar = (
        alt.Chart(imp.head(topk))
        .mark_bar(color="#4575b4")
        .encode(
            alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
            alt.Y("feature:N", sort=top, title=None),
            tooltip=["feature:N", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
        )
        .properties(width=300, height=430, title="Global importance — mean |SHAP|")
    )
    bee = (
        alt.Chart(bee_df)
        .mark_circle(size=26, opacity=0.65)
        .encode(
            x=alt.X(
                "shap:Q",
                title=axis_title
                or f"SHAP value  (◀ {class_names[0]} · {class_names[1]} ▶)",
            ),
            y=alt.Y("feature:N", sort=top, title=None),
            yOffset="jitter:Q",
            color=alt.Color(
                "feat_norm:Q",
                scale=alt.Scale(scheme="redblue", reverse=True),
                title="feature value (low → high)",
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                "feature:N",
                alt.Tooltip("value:Q", format=".3g"),
                alt.Tooltip("shap:Q", format=".3f"),
            ],
        )
        .properties(width=440, height=430, title="Per-row SHAP — direction & magnitude")
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(x="z:Q")
    )
    return mo.hstack([bar, bee + zero], justify="start")


@app.function
@mo.persistent_cache
def logistic_shap_panel(X, y, class_names, feature_cols, topk=12):
    """Exact SHAP for the torch logistic head via `shap.LinearExplainer`. The skorch net
    is linear, so binary class-1 log-odds is (w₁−w₀)·x+(b₁−b₀) and LinearExplainer
    attributes it per feature. For the 3-class target the same contrast is taken
    ONE-VS-REST per class — wₖ minus the mean of the other classes' weights — giving a
    signed axis per class, stacked into `(n, f, K)` for `tabular_shap_charts` to tab.
    Balanced loss is internal to the net (no sample_weight); read against the XGBoost
    panel for the additive-only view."""
    _nc = len(class_names)
    pipe = logreg_fn(n_classes=_nc)
    pipe.fit(X, np.asarray(y))
    Xs = pipe[:-1].transform(X)  # median-impute → z-score → float32 (what the net sees)
    _lin = pipe[-1].module_.linear
    _W = _lin.weight.detach().cpu().numpy()
    _b = _lin.bias.detach().cpu().numpy()

    def _ovr(k):
        _rest = [j for j in range(_nc) if j != k]
        coef = (_W[k] - _W[_rest].mean(axis=0)).astype(float)
        intercept = float(_b[k] - _b[_rest].mean())
        return np.asarray(shap.LinearExplainer((coef, intercept), Xs).shap_values(Xs))

    if _nc == 2:  # keep the exact binary contrast (w₁−w₀), not the OVR mean
        sv = _ovr(1)
    else:
        sv = np.stack([_ovr(k) for k in range(_nc)], axis=-1)
    return tabular_shap_charts(sv, X, feature_cols, class_names, topk)


@app.function
@mo.persistent_cache
def tabpfn_shap_panel(X, y, class_names, feature_cols, topk=12, budget=128):
    """SHAP for TabPFN v3 via its **own imputation explainer**
    (`tabpfn_extensions.interpretability.shapiq.get_tabpfn_imputation_explainer` —
    shapiq under the hood, imputation-based feature removal) bridged to a
    `shap.Explanation`. TabPFN is a transformer, not a tree, so TreeExplainer can't
    touch it; `fit_with_cache` + `balance_probabilities` mirror the modelled config. The
    explainer is per-class (`class_index`): binary explains the push toward
    `class_names[1]`; the 3-class target runs it once per class and stacks the results into
    `(n, f, K)` so `tabular_shap_charts` tabs them. Slowest explainer cell — and K× slower
    still on 3 classes."""
    _nc = len(class_names)
    Xi = SimpleImputer(strategy="median").fit_transform(np.asarray(X, dtype=float))
    clf = TabPFNClassifier(
        fit_mode="fit_with_cache",
        ignore_pretraining_limits=True,
        balance_probabilities=True,
        random_state=0,
    )
    clf.fit(Xi, np.asarray(y))

    def _for_class(k):
        expl = get_tabpfn_imputation_explainer(
            model=clf, data=Xi, index="SV", max_order=1, class_index=k
        )
        return np.asarray(
            shapiq_to_shap_explanation(
                expl, Xi, budget=budget, feature_names=list(feature_cols)
            ).values
        )

    sv = (
        _for_class(1)
        if _nc == 2
        else np.stack([_for_class(k) for k in range(_nc)], axis=-1)
    )
    return tabular_shap_charts(sv, Xi, feature_cols, class_names, topk)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Synthesis & honest caveats

    - **Grouped CV is the point.** Nested units from one expert umbrella share a
      `event_id` and are kept on one side of every split, so scores are not inflated
      by near-duplicate excursions leaking across folds — the flaw in a plain
      stratified split at this granularity.
    - **Tiny, imbalanced *n*.** Read **macro-F1 / balanced accuracy / MCC**, not
      accuracy. LOGO agrees with the 5×5 grouped estimate as a variance check.
    - **No SMOTE.** With so few minority units, balanced class weights are used, not
      synthetic oversampling.
    - **Track B.** Deep learning on the raw `proc_curves.parquet` (128×5) reuses this
      grouped-CV harness (it indexes axis 0) — the honest test of whether O2 *shape* alone
      recovers the class.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## M-mixed — *mixed* is now a modelled class

    The challenge defines **three** event types (hot / oxic pulse / **mixed**), and as of
    the **rev 07-31-26** expert workbook we model all three. That revision retired the
    ambiguous `hx` ("unknown hot moment") and `e` ("oxygen event during flood") codes in
    favour of an explicit `m`, and gave mixed events their own column block — so a moment's
    class is now read directly off the block its DO window sits in. Mixed is no longer a
    handful of umbrellas: it is **39 of 139 moments (24%)** and **13 of 58 events**.

    Earlier revisions had too few mixed umbrellas to learn from, so the deck instead held
    them out and asked where the *binary* classifier put them — the expectation being that a
    pulse-shaped DO rise carrying a hot-like **salinity step** should land near the decision
    boundary rather than confidently in either camp. That argument is **superseded**: the
    3-class confusion matrix now answers the same question directly, and the only remaining
    holdout is the orphan excursions (auto-detected, never catalogued by the expert).
    """)
    return


@app.cell
def _(X, class_names, groups, y):
    # Wrapped in a local function so the superseded-case early-return is a normal
    # Python return, not a marimo cell return (a cell's return value is its defined
    # names, and the cell OUTPUT is its last expression).
    def _boundary_panel():
        _hold = (
            pl.read_parquet("derived/processed_auto_features.parquet")
            .filter(pl.col("split") == "holdout")
            .sort("event_id", "start_time")
        )
        # OBSOLETE AS OF THE rev 07-31-26 WORKBOOK. This analysis existed because `mixed` had no
        # trainable label and was held out; it asked whether the binary boundary "recognised"
        # those events by leaving them at high entropy. `mixed` is now a first-class target
        # (core.features.finalize), so the excursion holdout is EMPTY and there is nothing to
        # score — the 3-class confusion matrix answers the same question directly. Degrades to a
        # note rather than erroring; the slide narrative above needs rewriting to match.
        if _hold.height == 0 or len(class_names) != 2:
            return mo.md(
                "> **Superseded.** `mixed` is now a trained class rather than a holdout, so the "
                "held-out boundary analysis no longer applies — see the 3-class confusion "
                "matrix for how the models actually separate hot / pulse / mixed."
            )
        _Xh = _hold.select(FEATURE_COLS).to_numpy().astype(float)

        _clf = xgb_fn()
        _clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        _ph = _clf.predict_proba(_Xh)[:, 1].astype(
            float
        )  # P(pulse) on the unseen held-out events

        # Fair train baseline: out-of-fold P(pulse) from the same grouped CV (never in-sample).
        _oof = np.full(len(y), np.nan)
        _sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for _tr, _te in _sgkf.split(np.zeros(len(y)), y, groups):
            _m = xgb_fn()
            _m.fit(X[_tr], y[_tr], sample_weight=compute_sample_weight("balanced", y[_tr]))
            _oof[_te] = _m.predict_proba(X[_te])[:, 1]

        def _entropy(p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        _Hh, _Ho = _entropy(_ph), _entropy(_oof)

        # Per-event table for the held-out events (why they are ambiguous: pulse-like rise
        # + a hot-like salinity step). sal_step / wl_step / rise_rate are the decisive features.
        mixed_boundary = (
            _hold.select(["event_id", "expert_subtype", "sal_step", "wl_step", "rise_rate"])
            .with_columns(
                p_pulse=pl.Series(_ph).round(3),
                entropy_bits=pl.Series(_Hh).round(3),
                prediction=pl.Series(np.where(_ph >= 0.5, class_names[1], class_names[0])),
            )
            .sort("entropy_bits", descending=True)
        )

        # Train (OOF) vs held-out entropy distributions
        _dist = pl.concat(
            [
                pl.DataFrame({"entropy_bits": _Ho}).with_columns(
                    set=pl.lit("train (hot/pulse, OOF)")
                ),
                pl.DataFrame({"entropy_bits": _Hh}).with_columns(
                    set=pl.lit("held-out (mixed/boundary)")
                ),
            ]
        )
        _chart = (
            alt.Chart(_dist)
            .mark_tick(thickness=2, size=18, opacity=0.7)
            .encode(
                x=alt.X(
                    "entropy_bits:Q",
                    title="prediction entropy (bits) — 1.0 = on the boundary",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("set:N", title=None),
                color=alt.Color("set:N", legend=None),
            )
            .properties(
                width=460,
                height=110,
                title="Mixed events cluster toward the hot/pulse boundary",
            )
        )

        _frac_hi = float((_Hh > np.nanmedian(_Ho)).mean())
        return mo.vstack(
            [
                mo.md(
                    f"**Held-out mixed/boundary events vs the binary boundary.** Median "
                    f"entropy: **{np.nanmedian(_Ho):.2f} bits** (train, OOF) vs "
                    f"**{np.median(_Hh):.2f} bits** (held-out); **{_frac_hi:.0%}** of the "
                    f"{len(_Hh)} held-out events sit above the train median. With only "
                    f"{len(_Hh)} events this is illustrative, not a statistical test — but it "
                    f"shows mixed events land exactly where hot vs pulse is ambiguous, "
                    f"consistent with their definition (pulse-shaped rise + a hot-like "
                    f"salinity step)."
                ),
                _chart,
                mo.ui.table(mixed_boundary, selection=None),
            ]
        )

    _boundary_panel()
    return


@app.cell
def _():
    mo.md(r"""
    ## Track B — Deep learning on raw sequences

    The honest test: can the raw 5-channel waveform (DO, salinity, water level,
    temperature, precipitation — 128 steps × 5 channels) recover the hot-moment
    vs oxic-pulse classification **without hand-crafted features**?

    All heads run through the same grouped-CV harness (`training.py`, which indexes axis
    0 so it handles 3-D tensors natively); the tables below are **loaded** from its
    precomputed `derived/model_results.parquet`:

    - **Real labels** — **ROCKET** (10,000 random convolutional kernels, Dempster et al.
      2020) → XGBoost / CatBoost / Logistic / TabPFN, on the tight ±≤12 h window and a
      wide 48 h-context variant, plus end-to-end **InceptionTime-lite** (focal loss).
    - **Shuffled labels** and **shuffled time** — the identical pipelines with labels
      permuted, or the time axis permuted; both leakage / order controls.
    """)
    return


@app.cell
def _(train):
    _m2i = {uid: i for i, uid in enumerate(train["unit_id"].to_list())}
    X_seq = read_time_series_curves("derived/processed_auto_curves.parquet", _m2i)
    rocket = RocketTransform(n_kernels=10000, seed=42).fit(X_seq)
    X_rocket = rocket.transform(X_seq)
    mo.md(
        f"**ROCKET transform (torch/GPU · core.model):** 10,000 random kernels → "
        f"{X_rocket.shape[1]:,} features from the train sequences "
        f"(loaded here only to explain the precomputed Track-B result)."
    )
    return X_rocket, X_seq, rocket


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Track B synthesis

    - **ROCKET** is a strong baseline: random convolutional kernels extract discriminative
      features from the raw waveform without training, then a simple linear or tree-based
      classifier separates the classes. This is the **fairer test** compared to Track A — no
      hand-crafted features are used, and no gradient descent on the tiny training set is needed.
    - **Wide 48 h context** adds ~+0.04 macro-F1 over the tight window, on the minority
      **pulse** class — the pre-onset salinity/water-level trajectory that `sal_step`/`wl_step`
      encode is genuinely informative.
    - **InceptionTime-lite** trains end-to-end with focal loss to handle the class imbalance.
      With only a few dozen training examples, deep CNNs struggle to converge and tend to
      collapse to the majority class (~0.44 macro-F1).
    - **ROCKET → TabPFN v3** is a *ceiling probe*: it chains the two strongest per-track
      models — ROCKET features (compressed by fold-internal PCA) feeding TabPFN's in-context
      tabular reasoner. It lands at the same ~0.80 wall, confirming the ceiling is a property
      of the label signal at this small n, not of any one estimator.
    - **Shuffled-label controls** collapse to chance, confirming no leakage in the sequence
      pipeline (normalization, CV splits, model training).
    - **Takeaway:** At this small sample size, hand-crafted features + gradient boosting (Track A)
      or ROCKET (Track B) vastly outperform end-to-end deep learning — and no combination
      of the best components breaks the ~0.80 macro-F1 ceiling.
    """)
    return


@app.function
@mo.persistent_cache
def inception_shap_panel(X_seq, y, class_names, max_epochs=120):
    """GradientSHAP (`shap.GradientExplainer`) on the end-to-end InceptionTime-lite net —
    the DL-native explainer. Fits the pipeline, reaches the raw torch module, and
    attributes the class-1 logit over the (channel × time) input. Attributions are
    aggregated to per-channel reliance and a per-timestep temporal profile (the sequence
    analogue of the ROCKET channel / dilation view), plus a per-row channel beeswarm.
    Focal loss handles imbalance internally (no sample_weight)."""
    _nc = len(class_names)
    channels = ["do", "sal", "wl", "temp", "precip"][: X_seq.shape[1]]
    pipe = inception_fn(
        n_channels=X_seq.shape[1], max_epochs=max_epochs, n_classes=_nc
    )
    pipe.fit(X_seq, np.asarray(y))
    Xsq = pipe[0].transform(X_seq).astype(np.float32)  # channel z-norm, NaN-scrubbed
    module = pipe[-1].module_.eval()
    dev = next(module.parameters()).device
    rng = np.random.RandomState(0)
    _bg = torch.tensor(
        Xsq[rng.choice(len(Xsq), size=min(50, len(Xsq)), replace=False)], device=dev
    )
    _sv = shap.GradientExplainer(module, _bg).shap_values(
        torch.tensor(Xsq, device=dev), nsamples=100
    )

    def _logit(k):  # attributions for class k's logit -> (N, C, T)
        return np.asarray(_sv[k] if isinstance(_sv, list) else np.asarray(_sv)[..., k])

    # 3-class target: every view here (channel reliance, temporal profile, signed push) is
    # class-specific, so render the whole panel once per class and tab it.
    if _nc > 2:
        return mo.ui.tabs(
            {
                f"→ {c}": inception_shap_one(_logit(k), Xsq, channels, c)
                for k, c in enumerate(class_names)
            }
        )
    return inception_shap_one(_logit(1), Xsq, channels, class_names)


@app.function
def inception_shap_one(sv, Xsq, channels, class_names):
    """One InceptionTime SHAP panel for a `(N, C, T)` attribution `sv`, with `Xsq` the
    channel-z-normed input it was computed on (used to colour the beeswarm by level).
    `class_names` is the binary pair (signed axis `[0]` vs `[1]`) or a single class name
    string (one-vs-rest push toward it)."""
    _pos = class_names if isinstance(class_names, str) else class_names[1]
    _neg = f"not {_pos}" if isinstance(class_names, str) else class_names[0]
    sv = np.asarray(sv)

    # 1) channel reliance — Σ over time of mean|SHAP| per channel.
    by_ch = pl.DataFrame(
        {
            "channel": channels,
            "shap": np.abs(sv).mean(axis=0).sum(axis=1).astype(float),
        }
    ).sort("shap", descending=True)
    chart_ch = (
        alt.Chart(by_ch)
        .mark_bar()
        .encode(
            alt.X("channel:N", sort="-y", title="Channel"),
            alt.Y("shap:Q", title="Σ mean|SHAP|"),
            color=alt.Color("channel:N", legend=None),
            tooltip=["channel", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(width=300, height=250, title="Channel reliance (Σ mean|SHAP|)")
    )

    # 2) temporal profile — mean|SHAP| per timestep, per channel.
    _prof = np.abs(sv).mean(axis=0)  # (C, T)
    _prof_rows = [
        {"step": int(t), "channel": channels[c], "shap": float(_prof[c, t])}
        for c in range(_prof.shape[0])
        for t in range(_prof.shape[1])
    ]
    chart_time = (
        alt.Chart(pl.DataFrame(_prof_rows))
        .mark_line()
        .encode(
            alt.X("step:Q", title="time step (onset-aligned)"),
            alt.Y("shap:Q", title="mean |SHAP|"),
            color=alt.Color("channel:N", title="channel"),
            tooltip=["channel", "step", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(
            width=400, height=250, title="Temporal reliance (mean|SHAP| over time)"
        )
    )

    # 3) beeswarm — per-row net channel push (Σ signed SHAP over time), coloured by
    #    that channel's (normalized) mean level for the row.
    _push = sv.sum(axis=2)  # (N, C)
    _level = Xsq.mean(axis=2)  # (N, C) scaled channel level
    _order = by_ch["channel"].to_list()
    rng2 = np.random.RandomState(0)
    _bee = []
    for c, ch in enumerate(channels):
        col = _level[:, c].astype(float)
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        norm = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        for i in range(sv.shape[0]):
            _bee.append(
                {
                    "channel": ch,
                    "shap": float(_push[i, c]),
                    "level": float(norm[i]),
                    "jitter": float(rng2.uniform(-0.35, 0.35)),
                }
            )
    bee = (
        alt.Chart(pl.DataFrame(_bee))
        .mark_circle(size=26, opacity=0.65)
        .encode(
            x=alt.X(
                "shap:Q",
                title=f"Σ SHAP over time  (◀ {_neg} · {_pos} ▶)",
            ),
            y=alt.Y("channel:N", sort=_order, title=None),
            yOffset="jitter:Q",
            color=alt.Color(
                "level:Q",
                scale=alt.Scale(scheme="redblue", reverse=True),
                title="channel level (low → high)",
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["channel", alt.Tooltip("shap:Q", format=".3f")],
        )
        .properties(width=440, height=250, title="Per-row channel push — direction")
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(x="z:Q")
    )
    # return mo.hstack([chart_ch, chart_time, bee + zero])
    return mo.vstack(
        [mo.hstack([chart_ch, bee + zero], justify="start"), chart_time],
        align="stretch",
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    - **XGBoost** — Chen & Guestrin (2016), KDD. <https://arxiv.org/abs/1603.02754>
    - **CatBoost** — Prokhorenkova et al. (2018), NeurIPS. <https://arxiv.org/abs/1706.09516>
    - **Logistic regression (L2)** — scikit-learn `LogisticRegression`.
    - **SHAP** — Lundberg & Lee (2017), NeurIPS. <https://arxiv.org/abs/1705.07874>
    - **shapiq / TabPFN imputation explainer** — Muschalik et al. (2024), NeurIPS; `tabpfn-extensions`. <https://arxiv.org/abs/2410.01649>
    - **TabPFN v3** — Hollmann et al. (2025), Nature. <https://www.nature.com/articles/s41586-024-08328-6>
    - **StratifiedGroupKFold / LeaveOneGroupOut** — scikit-learn model selection.
    - **ROCKET** (Track B) — Dempster, Petitjean & Webb (2020), DMKD. <https://arxiv.org/abs/1910.13051>
    - **InceptionTime** (Track B) — Ismail Fawaz et al. (2020), DMKD. <https://arxiv.org/abs/1909.04939>
    """)
    return


@app.cell
def moment_shap_data():
    mf_path = Path("derived/processed_moments_features.parquet")
    mc_path = Path("derived/processed_moments_curves.parquet")
    if mf_path.exists() and mc_path.exists():
        Xm, ym, cnm, tm = read_tabular_features(mf_path, MOMENT_FEATURE_COLS)
        m2i_m = {uid: i for i, uid in enumerate(tm["unit_id"].to_list())}
        Xm_seq = read_time_series_curves(str(mc_path), m2i_m)
        rk_m = RocketTransform(n_kernels=10000, seed=42).fit(Xm_seq)
        Xm_rocket = rk_m.transform(Xm_seq)
    return Xm, Xm_rocket, Xm_seq, cnm, rk_m, ym


@app.cell
def excursion_shap_tabs(excursion_shap_panels):
    mo.ui.tabs(excursion_shap_panels)
    return


@app.cell
def moment_shap_tabs(moment_shap_panels):
    mo.ui.tabs(moment_shap_panels)
    return


@app.cell
def excursion_shap_panel_dict(X, X_rocket, X_seq, class_names, rocket, y):
    # SHAP charts for the excursion route, one panel per model family. The panel
    # functions are decorated `@mo.persistent_cache`, so this cell's first run caches
    # each fit+explainer to `__marimo__/cache/<fn>/` (gitignored, NDA-safe) keyed by
    # its args; later runs — including slides.py reaching them through `Cell.run` —
    # restore from disk (a bare-decorator cache survives `.run()`; a `with`-block one
    # does not). Kept separate from the `mo.ui.tabs` cell (compute vs. display).
    excursion_shap_panels = {
        "XGBoost (Track A)": shap_feature_panel(X, y, class_names, FEATURE_COLS),
        "Logistic (Track A)": logistic_shap_panel(X, y, class_names, FEATURE_COLS),
        "TabPFN (Track A)": tabpfn_shap_panel(X, y, class_names, FEATURE_COLS)
        if get_tabpfn_imputation_explainer is not None
        else mo.md("> ⚠️ `tabpfn-extensions[interpretability]` not installed."),
        "ROCKET (Track B)": rocket_shap_panel(
            X_rocket, rocket.module_.kernel_specs(), class_names, y
        ),
        "InceptionTime (Track B)": inception_shap_panel(X_seq, y, class_names),
    }
    return (excursion_shap_panels,)


@app.cell
def moment_shap_panel_dict(Xm, Xm_rocket, Xm_seq, cnm, rk_m, ym):
    # Moment-route SHAP charts (cached via the decorated panel fns; see `excursion_shap_panels`).
    moment_shap_panels = {
        "XGBoost": shap_feature_panel(Xm, ym, cnm, MOMENT_FEATURE_COLS),
        "Logistic": logistic_shap_panel(Xm, ym, cnm, MOMENT_FEATURE_COLS),
        "TabPFN": tabpfn_shap_panel(Xm, ym, cnm, MOMENT_FEATURE_COLS)
        if get_tabpfn_imputation_explainer is not None
        else mo.md("> ⚠️ `tabpfn-extensions[interpretability]` not installed."),
        "ROCKET + XGBoost": rocket_shap_panel(
            Xm_rocket, rk_m.module_.kernel_specs(), cnm, ym
        ),
        "InceptionTime": inception_shap_panel(Xm_seq, ym, cnm),
    }
    return (moment_shap_panels,)


if __name__ == "__main__":
    app.run()
