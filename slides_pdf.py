import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    import polars as pl
    import marimo as mo

    # Single source of truth: the live deck. Every slide identical to slides.py is
    # imported and re-rendered via Cell.run(). Text/markdown slides run with no args;
    # slides that consume UI elements take them as kwargs from the UI-def cells below
    # (a UIElement's .value can't be read in the same cell that created it, so the
    # create/consume split from slides.py must be preserved). Only the four slides the
    # PDF rasterizer can't handle (mo.ui.tabs) are rebuilt locally.
    from slides import (
        title,
        intro,
        controls,
        disc_controls,
        viewer,
        feature_examples,
        modelling_intro,
        label_survey_slide,
        disc_events_cell,
        clustering_slide,
        weak_supervision_slide,
        conclusions,
    )

    # Needed only by the locally-rebuilt metrics / SHAP / read-out slides.
    from modeling import moment_shap_panel_dict
    from discovery import mutual_information


@app.cell(hide_code=True)
def _():
    _o, _ = title.run()
    _o
    return


@app.cell(hide_code=True)
def _():
    _o, _ = intro.run()
    _o
    return


@app.cell(hide_code=True)
def _():
    # UI-def cell (no output → no slide): creates the event-viewer widgets so the
    # viewer slide can consume their .value in a different cell.
    _, _defs = controls.run()
    event_picker = _defs["event_picker"]
    shade_mode = _defs["shade_mode"]
    return event_picker, shade_mode


@app.cell(hide_code=True)
def _(event_picker, shade_mode):
    _o, _ = viewer.run(event_picker=event_picker, shade_mode=shade_mode)
    _o
    return


@app.cell(hide_code=True)
def _():
    _o, _ = feature_examples.run()
    _o
    return


@app.cell(hide_code=True)
def _():
    _o, _ = modelling_intro.run()
    _o
    return


@app.cell(hide_code=True)
def _():
    _o, _ = label_survey_slide.run()
    _o
    return


@app.cell(hide_code=True)
def slides_metrics_routes():
    # PDF variant: render the Macro-F1 (F1-score) grouped bars directly (no mo.ui.tabs,
    # which the PDF rasterizer can't flatten). Same grouped-bar form as metrics_routes_plot:
    # models on x, one bunched bar per route (excursion vs moment), ±std whiskers, dashed
    # chance line at 0.5. Reads the on-disk CV artifacts, like modeling.py does.
    import altair as _alt

    def _macro_bars(_table, _height=340):
        _frames = {"excursion": "derived/model_results.parquet"}
        _mm = "derived/model_results_moment.parquet"
        if Path(_mm).exists():
            _frames["moment"] = _mm
        _parts, _routes = [], []
        for _route, _path in _frames.items():
            _s = pl.read_parquet(_path).filter(
                (pl.col("table") == _table) & (pl.col("metric") == "macro_f1")
            )
            if _s.height:
                _parts.append(_s.with_columns(pl.lit(_route).alias("route")))
                _routes.append(_route)
        if not _parts:
            return mo.md(f"> no rows for `{_table}`")
        _d = pl.concat(_parts).with_columns(
            (pl.col("mean") - pl.col("std").fill_null(0.0)).alias("lo"),
            (pl.col("mean") + pl.col("std").fill_null(0.0)).alias("hi"),
        )
        _order = (
            _d.filter(pl.col("route") == _routes[0])
            .sort("mean", descending=True)["model"]
            .to_list()
        )
        for _m in _d["model"].unique(maintain_order=True).to_list():
            if _m not in _order:
                _order.append(_m)
        _base = _alt.Chart(_d).encode(
            x=_alt.X("model:N", sort=_order, title=None, axis=_alt.Axis(labelAngle=-30)),
            xOffset=_alt.XOffset("route:N", sort=_routes),
        )
        _bars = _base.mark_bar().encode(
            y=_alt.Y("mean:Q", title="F1-score", scale=_alt.Scale(domain=[0, 1])),
            color=_alt.Color(
                "route:N",
                sort=_routes,
                title="route",
                scale=_alt.Scale(scheme="set2"),
                legend=_alt.Legend(orient="bottom"),
            ),
            tooltip=["model:N", "route:N", _alt.Tooltip("mean:Q", format=".3f")],
        )
        _wh = _base.mark_rule(strokeWidth=1.2, color="white").encode(y="lo:Q", y2="hi:Q")
        _chance = (
            _alt.Chart(pl.DataFrame({"y": [0.5]}))
            .mark_rule(color="#c1440e", strokeDash=[4, 3])
            .encode(y="y:Q")
        )
        return _alt.layer(_bars, _wh, _chance).properties(
            width=max(440, 116 * len(_order)), height=_height
        )

    mo.vstack(
        [
            mo.md("# Model comparison — Macro-F1 (grouped 5×5 CV)"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Engineered features (excursion vs moment):**"),
                            _macro_bars("base_grouped"),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**ROCKET / InceptionTime on raw curves:**"),
                            _macro_bars("dl_grouped"),
                        ]
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    # PDF variant: each supervised SHAP panel is two charts wide, so one per slide (no
    # mo.ui.tabs, which the rasterizer can't flatten); the unsupervised mutual-info panel
    # moves to the read-out slide below (which it validates).
    _, _outs = moment_shap_panel_dict.run()
    shap_panels = _outs["moment_shap_panels"]
    return (shap_panels,)


@app.cell(hide_code=True)
def _(shap_panels):
    mo.vstack(
        [
            mo.md("# Which drivers carry hot-vs-oxic signal? — XGBoost (engineered)"),
            shap_panels["XGBoost"],
        ]
    )
    return


@app.cell(hide_code=True)
def _(shap_panels):
    mo.vstack(
        [
            mo.md(
                "# Which drivers carry hot-vs-oxic signal? — ROCKET + XGBoost (raw curves)"
            ),
            shap_panels["ROCKET + XGBoost"],
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    # UI-def cell (no output → no slide): creates the discovery widgets.
    _, _defs = disc_controls.run()
    disc_clusters = _defs["disc_clusters"]
    disc_x_var = _defs["disc_x_var"]
    disc_y_var = _defs["disc_y_var"]
    return disc_clusters, disc_x_var, disc_y_var


@app.cell(hide_code=True)
def _():
    # Both discovery routes for the read-out + clustering + weak-supervision slides.
    # In a PDF the route switch is frozen at the default (moments) — the tab / Vega
    # selectors only come alive in the html export.
    _, _defs = disc_events_cell.run()
    disc_events_by_route = _defs["disc_events_by_route"]
    return (disc_events_by_route,)


@app.cell(hide_code=True)
def _(disc_events_by_route):
    _mi, _ = mutual_information.run(events_by_route=disc_events_by_route)
    mo.vstack(
        [
            mo.md("# Which drivers carry hot-vs-oxic signal? — read-out"),
            mo.hstack(
                [
                    mo.md(r"""
    - **SHAP, Engineered Features (XGBoost, Logistic, TabPFN):**
        - Best &rarr; **Hydrology-based** e.g., \(\frac{d(DO_2)}{dt}\) (*rise_rate*), \(\frac{t_{\rm peak-DO_2} - t_{\rm start}}{t_{\rm end} - t_{\rm start}}\) (*peak_frac*), \(\Delta \text{Salinity}\) (*sal_step*), \(\max{(DO_2)}\) (*peak_do*)
        - Worst &rarr; **Weather-based:** Air Temperature (temp_in), Antecedent Precipitation (precip_168h)
    - **SHAP, Time Series (ROCKET+XGBoost, InceptionTime):**
        - \(DO_2\) and Temperature most important, Precipitation least
        - ROCKET kernels with bigger dilation more important (~40-80 min correlations)
    - Mutual information (unsupervised, right) validates conclusions from SHAP &rarr;
                      """),
                    mo.vstack([mo.md("**Unsupervised (mutual info)**"), _mi]),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(disc_clusters, disc_events_by_route, disc_x_var, disc_y_var):
    _o, _ = clustering_slide.run(
        disc_clusters=disc_clusters,
        disc_events_by_route=disc_events_by_route,
        disc_x_var=disc_x_var,
        disc_y_var=disc_y_var,
    )
    _o
    return


@app.cell(hide_code=True)
def _(disc_events_by_route):
    _o, _ = weak_supervision_slide.run(disc_events_by_route=disc_events_by_route)
    _o
    return


@app.cell(hide_code=True)
def _():
    _o, _ = conclusions.run()
    _o
    return


if __name__ == "__main__":
    app.run()
