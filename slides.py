import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", layout_file="layouts/slides.slides.json")

with app.setup:
    import base64
    from pathlib import Path
    import polars as pl
    import marimo as mo

    from utils import feature_methodology_tabs

    from eda import ui_elements, event_detection, label_survey, feature_methodology
    from modeling import (
        metrics_routes_plot,
        metrics_grouped_plot,
        excursion_shap_panel_dict,
        moment_shap_panel_dict,
    )

    from discovery import (
        mutual_information,
        disc_ui,
        kmeans_clustering,
        weak_supervision,
        disc_input,
    )


@app.cell(hide_code=True)
def title():
    _photo = Path("assets/title_slide_bg.jpg")
    _mime = {"jpg": "jpeg", "png": "png", "webp": "webp"}.get(
        _photo.suffix.lower().lstrip("."), "jpeg"
    )
    _layer = (
        f", url(data:image/{_mime};base64,{base64.b64encode(_photo.read_bytes()).decode()})"
        if _photo.exists()
        else ""
    )
    _title1 = "Tidal Hiccups and Oxic Sighs : Machine Learning Meets Marshland Respiration"
    _title2 = (
        "Formalizing Classification Methodologies for Environmental Drivers of Dissolved Oxygen"
    )
    _authors = ("Tanmay Pani", "Eric Britt")

    # Logos pinned bottom-right (Erdős Institute + Opti O2). Each embedded as a data-URI;
    # a missing file is simply skipped. Add/reorder/resize by editing the list below.
    def _img_tag(path, height=56):
        _p = Path(path)
        if not _p.exists():
            return ""
        _t = {"jpg": "jpeg", "png": "png", "webp": "webp"}.get(
            _p.suffix.lower().lstrip("."), "jpeg"
        )
        return (
            f'<img src="data:image/{_t};base64,{base64.b64encode(_p.read_bytes()).decode()}"'
            f' style="height: {height}px;" />'
        )

    # Logos row, with a data-courtesy footnote pinned just beneath them (right-aligned).
    _footnote = (
        '<div style="margin-top: 0.4rem; font-size: 0.72rem; color: #cfd8dd;'
        ' text-align: right; font-style: italic;">'
        "The data for this study is the courtesy of Dr. Ruby Ghosh and Opti-O2"
        "</div>"
    )
    _logos = (
        '<div style="position: absolute; bottom: 1.1rem; right: 2rem; display: flex;'
        ' flex-direction: column; align-items: flex-end;">'
        '<div style="display: flex; align-items: center; gap: 1.25rem;">'
        + _img_tag("assets/erdos_logo.png")
        + _img_tag("assets/opti_o2_logo.jpg")
        + "</div>"
        + _footnote
        + "</div>"
    )

    _authors_joined = " · ".join(_authors)
    mo.md(
        f"""
    <div style="
        position: relative;
        background-image: linear-gradient(rgba(12,20,28,0.62), rgba(12,20,28,0.62)){_layer};
        background-size: cover; background-position: center; border-radius: 14px;
        padding: 4.5rem 3.5rem; min-height: 460px; display: flex;
        flex-direction: column; justify-content: center; color: #fff;">
        <h1 style="margin: 0; color: #fff; font-size: 2.8rem; line-height: 1.1;">
            {_title1}
        </h1>
        <h3 style="margin: 0.9rem 0 0; color: #e8eef2; font-size: 1.25;">
            {_title2}
        </h3>
        <p style="margin: 2rem 0 0; color: #cfd8dd; font-size:1;">{_authors_joined}</p>
        {_logos}
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md("""
    # Opti O2: Coastal Wetland \( DO_2 \) Dynamics
    """),
            mo.hstack(
                [
                    mo.md("""
    * **The Site**: Beaver Creek, WA — a former freshwater floodplain transitioning to a coastal wetland.
    * **The Data**: 6-year, 5-minute resolution time-series of subsurface **Dissolved Oxygen (\(DO_2\))**, **Hydrology** (*Water level*, *Salinity*, etc.), and **Weather** (*Temperature*, *Precipitation*, etc.) timeseries data
    * **The Phenomena**:
      - **\(DO_2 \) ~ 0** for 92% of timestamps (Anoxic baseline)
      - **\(DO_2\) event:** Sufficient departure from the baseline (\(\Delta DO_2 > 0.1\)mg/L)

    * **The Challenge**: Classifying \(DO_2\) events into:
      - **Hot Moments** (abrupt asymmetric and tidal/salinity-driven)
      - **Oxic Pulses** (symmetric and freshwater/precipitation-driven)
    """),
                    mo.image("assets/intro_beaver_creek.png", height=500),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def controls():
    _, _defs = ui_elements.run()
    event_picker = _defs["event_picker"]
    shade_mode = _defs["shade_mode"]
    return event_picker, shade_mode


@app.cell(hide_code=True)
def _():
    _, _defs = disc_ui.run()
    disc_clusters = _defs["num_kmeans_clusters"]
    disc_x_var = _defs["x_var"]
    disc_y_var = _defs["y_var"]
    disc_route = _defs["route"]
    return disc_clusters, disc_route, disc_x_var, disc_y_var


@app.cell(hide_code=True)
def viewer(event_picker, shade_mode):
    _view, _ = event_detection.run(event_picker=event_picker, shade_mode=shade_mode)
    _view
    return


@app.cell(hide_code=True)
def _():
    _moment_features = pl.read_parquet("derived/processed_moments_features.parquet")
    _moment_curves = pl.read_parquet("derived/processed_moments_curves.parquet")

    # _moment_features.head()
    _tabs = feature_methodology_tabs(_moment_features, _moment_curves)
    mo.vstack([mo.md("# Engineered Features: Examples"), _tabs])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Modelling

    *   **Engineered Features**:
      - Evaluates tabular models on 24 domain-engineered features
      - Calculated based on shape of oxic event (abrupt vs slow), hydrology and antecedent precipitation
      - **Models**: Linear (*Logistic Regression*), Tree-based (*XGBoost, CatBoost*),  and Deep/Foundation (*TabPFN*).
    *   **Raw Sequences / Time Series**:
      - Evaluates deep learning models directly on the raw 5-minute multivariate time-series waveforms.
      - **Models**: Sequences transformed by ROCKET, passed on to XGB/CatBoost/Logistic/TabPFN, InceptionTime-Lite (CNN).
    *   **Cross Validation**: Evaluated via 5x5 StratifiedGroupKFold and Leave-One-Group-Out (LOGO) cross-validation, holding out expert events from training
    *   **Temporal Generalization**: Includes a 70/30 chronological split to test model robustness against multi-year ecosystem salinization and drift.
    """)
    return


@app.cell(hide_code=True)
def _():
    _view, _ = label_survey.run()
    _view
    return


@app.cell(hide_code=True)
def slides_metrics_routes():
    _out, _ = metrics_routes_plot.run()
    _out
    return


@app.cell(hide_code=True)
def _(disc_events):
    _chart, _ = mutual_information.run(events=disc_events)
    _, _outs = moment_shap_panel_dict.run()
    _mom_shap = _outs["moment_shap_panels"]
    _mom_shap["Unsupervised"] = _chart
    mo.vstack(
        [
            mo.md("""
        # Which drivers carry hot-vs-oxic signal?
        """),
            mo.hstack(
                [
                    mo.ui.tabs(_mom_shap),
                    mo.md(r"""
            - **SHAP scores, Engineered Features (XGBoost, Logistic, TabPFN):**
                - Best scores &rarr; **Hydrology-based**
                e.g., \(\frac{d(DO_2)}{dt}\) (*rise_rate*), \(\frac{t_{\rm peak-DO_2} - t_{\rm start}}{t_{\rm end} - t_{\rm start}}\) (*peak_frac*), \(\Delta \text{Salinity}\) (*sal_step*), \(\max{(DO_2)}\) (*peak_do*)
                - Worst scores &rarr;**Weather-based:** e.g., Air Temperature (temp_in), Cumulative Antecedent Precipitation (precip_168h)  
            - **SHAP scores, Time Series (ROCKET+XGBoost, InceptionTime):**
                - \(DO_2\) and Temperature most important, Precipitaiton least
                - ROCKET kernels with bigger dilation (time-scales) more important (correlations over ~40-80 minutes)

            - Mutual information (unsupervised) validates conclusions from SHAP
                      """),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(disc_route):
    _, _defs = disc_input.run(route=disc_route)
    disc_events = _defs["events"]
    return (disc_events,)


@app.cell(hide_code=True)
def _(disc_clusters, disc_events, disc_x_var, disc_y_var):
    _view, _ = kmeans_clustering.run(
        events=disc_events, num_kmeans_clusters=disc_clusters, x_var=disc_x_var, y_var=disc_y_var
    )

    mo.vstack([mo.md("# Unsupervised: Clustering"), _view])
    return


@app.cell(hide_code=True)
def _(disc_events):
    _view, _ = weak_supervision.run(events=disc_events)
    _view
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conclusions and Future work

    - Formalized classification of Hot moments and Oxic pulses using timeseries data from Beaver Creek, WA
    - Developed various Deep-Learning (DL) classifiers trained engineered features, encoded features and raw time series
    - All model validated using various cross validation methods and non-DL models (XGBoost, Logistic)
    - Validated the expert labels against our own labelling algorithm
    - Feature importances attributed using SHAP (supervised) and validated against mutual information (unsupervised)
    - Groundwork laid for unsupervised classification (KMeans, Weak-Supervision Label)
    """)
    return


if __name__ == "__main__":
    app.run()
