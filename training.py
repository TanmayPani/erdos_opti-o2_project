import argparse
import inspect
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import polars as pl
import torch

import xgboost as xgb
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.model_loading import ModelVersion

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, FunctionTransformer, label_binarize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.features import FEATURE_COLS, MOMENT_FEATURE_COLS
from core.model import RocketTransform, InceptionTimeLite, FocalLoss, ChannelScaler
from core.model import LogisticRegression, LogitsNetClassifier

warnings.filterwarnings("ignore")

# Two classification routes, selected with `--mode` (see __main__): "excursion" (auto-detected
# DO excursions, the default) and "moment" (expert-window moments). The moment route drops the
# hysteresis features (MOMENT_FEATURE_COLS), has no wide-context curves, and writes a separate
# results file so it never clobbers the excursion-route artifact.
MODES = {
    "excursion": dict(
        features="derived/proc_features.parquet",
        curves="derived/proc_curves.parquet",
        wide="derived/proc_curves_wide.parquet",
        results="derived/model_results.parquet",
        feature_cols=FEATURE_COLS,
    ),
    "moment": dict(
        features="derived/moment_features.parquet",
        curves="derived/moment_curves.parquet",
        wide=None,
        results="derived/model_results_moment.parquet",
        feature_cols=MOMENT_FEATURE_COLS,
    ),
}


def fmt_stat(stat):
    """Format a (mean, std) tuple, or a bare float, as a string."""
    if isinstance(stat, tuple):
        m, s = stat
        return f"{m:.3f} ± {s:.3f}"
    return f"{stat:.3f}"


def _device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def xgb_fn():
    """Shared XGBoost config — shallow trees + L1/L2 against the tiny n. One
    definition for Track A, the ROCKET→XGB heads, and the importance probes."""
    _dev = _device()
    _dev = "cpu" if _dev not in {"cuda", "cpu"} else _dev
    # print("XGBoost running on:", _dev)
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        # n_jobs=2,
        verbosity=0,
        device=_dev,
    )


def catboost_fn():
    # GPU (~5 s/fit) only pays off on the wide ROCKET heads (20k features); on the 24-dim
    # tabular track CPU is 0.15 s and GPU would be overhead-bound, so default is CPU.
    _dev_str = _device()
    _dev = (
        dict(task_type="GPU", devices="0")
        if _dev_str == "cuda"
        else dict(thread_count=10)
    )
    # print("Catboost on:", _dev)
    return CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        **_dev,
    )


def tabpfn_pipeline_fn(rocket=False):
    # TabPFN v3 (latest) with balance_probabilities to correct the skewed class
    # prior — the one lever that helps the minority class (bal_acc/MCC) at equal
    # macro-F1. Runs LOCALLY on GPU; weights are authenticated by TABPFN_TOKEN in
    # the kernel env and never stored in this notebook (no data leaves the machine).
    # SimpleImputer guards residual nulls (e.g. hysteresis on degenerate curves).

    _model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device=_device(),
        ignore_pretraining_limits=True,
        balance_probabilities=True,
        random_state=0,
    )
    if rocket:
        return make_pipeline(
            StandardScaler(), PCA(n_components=0.95, random_state=0), _model
        )

    return make_pipeline(SimpleImputer(strategy="median"), _model)


def inception_fn(n_channels=5, max_epochs=120, lr=1e-3, weight_decay=1e-3):
    """InceptionTimeLite as an sklearn Pipeline: per-channel train-fold z-norm
    (`ChannelScaler`, the 3-D analogue of `StandardScaler`) → skorch-wrapped net. Consumes
    the raw `(N, C, T)` sequence tensors, so it slots straight into a `model_fn_dict`."""
    return make_pipeline(
        ChannelScaler(),
        LogitsNetClassifier(
            module=InceptionTimeLite,
            criterion=FocalLoss,
            optimizer=torch.optim.AdamW,
            lr=lr,
            optimizer__weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=-1,  # full-batch, like the original loop (tiny n)
            train_split=None,  # use every training row (no held-out valid split)
            callbacks=[
                ("cosine", LRScheduler(policy=CosineAnnealingLR, T_max=max_epochs)),
                ("clip", GradientNormClipping(gradient_clip_value=1.0)),
            ],
            device=_device(),
            verbose=0,
            module__n_channels=n_channels,
            module__n_classes=2,
        ),
    )


def logreg_fn(C=0.5, impute=True):
    """Torch logistic head as an sklearn Pipeline that MIRRORS
    `LogisticRegression(penalty='l2', C, class_weight='balanced').
    Preprocessing: median-impute the engineered features (ROCKET features are already dense →
    `StandardScaler` only) → float32 cast → the skorch net. `n_inputs` is the feature count;
    wire into a `model_fn_dict` with `partial(logreg_fn, n_inputs=X.shape[1])` (as with
    `partial(catboost_fn, ...)`). `C` mirrors sklearn's inverse-regularization (default 0.5;
    the ROCKET head uses 1.0)."""
    _net = LogisticRegression(
        # module__n_inputs=n_inputs,
        module__n_outputs=2,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__reduction="sum",  # Σᵢ, so C·Σ + ½‖W‖² matches sklearn's objective
        optimizer=torch.optim.LBFGS,
        lr=1.0,
        optimizer__max_iter=500,
        optimizer__line_search_fn="strong_wolfe",
        max_epochs=1,  # one LBFGS.step (max_iter internal iters) = solve to convergence
        batch_size=-1,  # full-batch (LBFGS is a batch solver)
        train_split=None,
        device=_device(),
        verbose=0,
        C=C,
    )
    _steps = []
    if impute:
        _steps.append(SimpleImputer(strategy="median"))

    return make_pipeline(
        *_steps,
        StandardScaler(),
        FunctionTransformer(partial(np.asarray, dtype=np.float32)),
        _net,
    )


def rocket_head_pipeline(clf_fn, *args, num_kernels=10000, seed=42, **kwargs):
    """Factory: `RocketTransform` on raw `(N, C, T)` sequences → the given 2-D classifier
    factory's estimator, as one Pipeline. Lets every ROCKET head live in the SAME
    sequence-input `model_fn_dict` as InceptionTime (RocketTransform is an sklearn
    transformer, so it composes). ROCKET's random kernels depend only on shape + seed, so
    the per-fold transform reproduces the old precomputed `X_rock` features exactly."""
    return make_pipeline(
        RocketTransform(n_kernels=num_kernels, seed=seed), clf_fn(*args, **kwargs)
    )


def evaluate(y_true, y_pred, y_proba=None, labels=None):
    """Metric bundle. ROC/PR-AUC (binary, positive class index 1) only when the
    test set actually has both classes — small grouped folds can be single-class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and labels is not None and len(np.unique(y_true)) > 1:
        Yb = label_binarize(y_true, classes=labels)
        if Yb.shape[1] == 1:  # binary
            out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            out["pr_auc"] = average_precision_score(y_true, y_proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(
                Yb, y_proba, average="macro", multi_class="ovr"
            )
            out["pr_auc"] = average_precision_score(Yb, y_proba, average="macro")
    return out


def _accepts_sample_weight(estimator):
    """True iff `estimator.fit` takes an explicit `sample_weight` param — XGBoost and
    CatBoost do; skorch nets (`**fit_params`) and TabPFN (`fit(X, y)`) do not, so they are
    fit unweighted and rely on their own imbalance handling."""
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def fit_predict(make_model, Xtr, ytr, Xte):
    """Fit, passing balanced sample weights only to estimators that accept them — routed to
    a Pipeline's FINAL step, so the ROCKET heads' XGBoost/CatBoost stay balanced through the
    `RocketTransform` pipeline. Skorch nets (balanced/focal loss) and TabPFN
    (`balance_probabilities`) handle imbalance internally and are fit unweighted."""
    m = make_model()
    _final = m.steps[-1][1] if isinstance(m, Pipeline) else m
    if _accepts_sample_weight(_final):
        _key = (
            f"{m.steps[-1][0]}__sample_weight"
            if isinstance(m, Pipeline)
            else "sample_weight"
        )
        m.fit(Xtr, ytr, **{_key: compute_sample_weight("balanced", ytr)})
    else:
        m.fit(Xtr, ytr)

    proba = m.predict_proba(Xte) if hasattr(m, "predict_proba") else None
    pred = proba.argmax(1) if proba is not None else np.asarray(m.predict(Xte)).ravel()
    return m, pred, proba


def grouped_cv(make_model, X, y, labels, groups, n_splits=5, n_repeats=5, seed=0):
    rows = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + r
        )
        for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
            m, pred, proba = fit_predict(make_model, X[tr], y[tr], X[te])
            rows.append(evaluate(y[te], pred, proba, labels))
    keys = rows[0].keys()
    return {
        k: (
            float(np.mean([r[k] for r in rows if k in r])),
            float(np.std([r[k] for r in rows if k in r])),
        )
        for k in keys
    }


def logo_cv(make_model, X, y, labels, groups):
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros((len(y), len(labels)), dtype=float)
    for tr, te in logo.split(X, y, groups):
        m, p, pb = fit_predict(make_model, X[tr], y[tr], X[te])
        pred[te] = p
        if pb is not None:
            proba[te] = pb
    return evaluate(y, pred, proba, labels)


def temporal_split(make_model, X, y, labels, order, train_frac=0.7):
    idx = np.argsort(order)
    cut = int(train_frac * len(idx))
    tr, te = idx[:cut], idx[cut:]
    m, pred, proba = fit_predict(make_model, X[tr], y[tr], X[te])
    return (
        evaluate(y[te], pred, proba, labels),
        np.bincount(y[tr]).tolist(),
        np.bincount(y[te]).tolist(),
    )


def read_tabular_features(path, feature_cols=FEATURE_COLS):
    data = (
        pl.read_parquet(path)
        .filter(pl.col("split") == "train")
        .sort("group_id", "start")
    )

    X = data.select(feature_cols).to_numpy().astype(float)
    class_names = sorted(data["label"].unique().to_list())  # ["hot", "pulse"]
    _c2i = {c: i for i, c in enumerate(class_names)}
    y = np.array([_c2i[v] for v in data["label"].to_list()])

    return X, y, class_names, data


def read_time_series_curves(path, map_uid_to_idx, channels=None, num_steps=128):
    _data = (
        pl.read_parquet(path).filter(pl.col("split") == "train").sort("unit_id", "step")
    )
    _uids = _data.select("unit_id").unique(maintain_order=True)["unit_id"].to_list()
    channels = channels or ["do", "sal", "wl", "temp", "precip"]

    _extra = set(_uids) - set(map_uid_to_idx)
    _missing = set(map_uid_to_idx) - set(_uids)
    if _extra or _missing:
        raise ValueError(
            f"{path}: train-unit mismatch vs feature table — "
            f"{len(_missing)} missing, {len(_extra)} extra "
            f"(e.g. extra {sorted(_extra)[:3]}, missing {sorted(_missing)[:3]}); "
            "regenerate this curve file from the current proc."
        )

    # Build (n, C, T) array aligned with the same unit order as X/y/groups
    X_raw = np.zeros((len(_uids), len(channels), num_steps), dtype=np.float32)
    for _i, _uid in enumerate(_uids):
        _unit_data = _data.filter(pl.col("unit_id") == _uid).sort("step")
        for _j, _ch in enumerate(channels):
            X_raw[map_uid_to_idx[_uid], _j, :] = _unit_data[_ch].to_numpy()

    return X_raw


def _cv(cv_fn, model_fn_dict, X, y, *args, **kwargs):
    """Run `cv_fn` for every model factory → `{model_name: {metric: (mean, std)}}`.
    Returns the RAW results dict (not a formatted table) so callers can merge/splice tables.
    A cv_fn may return either a metrics dict or a `(metrics, *extra)` tuple
    (`temporal_split`); only the metrics dict is kept. Any single model that raises
    (e.g. TabPFN with no token) is skipped with a warning rather than sinking the run."""
    _results = {}
    for name, model_fn in model_fn_dict.items():
        try:
            _res = cv_fn(model_fn, X, y, *args, **kwargs)
            _results[name] = _res[0] if isinstance(_res, tuple) else _res
            print(f"___[done] {name}")
        except Exception as _e:
            print(f"___[skip] {name}: {type(_e).__name__}: {_e}")
    return _results


def results_frame(results, table):
    """Flatten `{model: {metric: (mean, std) | float}}` into a tidy long frame
    `[table, model, metric, mean, std]` (std null for bare-float metrics, e.g. the
    single-shot temporal split). This is the on-disk artifact shape."""
    rows = []
    for model, metrics in results.items():
        for metric, val in metrics.items():
            mean, std = val if isinstance(val, tuple) else (val, None)
            rows.append(
                {
                    "table": table,
                    "model": model,
                    "metric": metric,
                    "mean": float(mean),
                    "std": None if std is None else float(std),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "table": pl.String,
            "model": pl.String,
            "metric": pl.String,
            "mean": pl.Float64,
            "std": pl.Float64,
        },
    )


def display_table(frame, table, sort="model"):
    """Pivot the tidy `results_frame` back to a wide, display-ready table for one
    `table` name: one row per model, one `"mean ± std"` string per metric (via
    `fmt_stat`). `sort` is either `"model"` (alphabetical) or a metric name, which
    ranks models by that metric's mean, descending."""
    sub = frame.filter(pl.col("table") == table)
    metrics = sub["metric"].unique(maintain_order=True).to_list()
    rows = []
    for model in sub["model"].unique(maintain_order=True).to_list():
        _m = sub.filter(pl.col("model") == model)
        _row = {"model": model}
        for metric in metrics:
            _r = _m.filter(pl.col("metric") == metric)
            if _r.height == 0:
                _row[metric] = None
                continue
            _mean, _std = _r["mean"][0], _r["std"][0]
            _row[metric] = fmt_stat(_mean if _std is None else (_mean, _std))
        rows.append(_row)
    out = pl.DataFrame(rows)
    if sort == "model":
        return out.sort("model")
    _order = (
        sub.filter(pl.col("metric") == sort)
        .sort("mean", descending=True)["model"]
        .to_list()
    )
    return (
        out.join(pl.DataFrame({"model": _order, "_r": range(len(_order))}), on="model")
        .sort("_r")
        .drop("_r")
    )


def compare_modes(frames, table, metrics=("macro_f1", "bal_acc", "mcc")):
    """Side-by-side comparison of the classification routes for one CV `table`.
    `frames` maps a route name (e.g. "excursion" / "moment") → its tidy
    `results_frame` (training's on-disk shape). Returns one row per model with a
    "mean ± std" column per (route, metric), blocked by route; a model missing
    from a route is null there, and a route missing this `table` contributes no
    columns. Rows are ranked by the first route's first metric (descending)."""
    modes = list(frames)
    _cell = {}  # (mode, model, metric) -> (mean, std)
    _models = []
    for mode, frame in frames.items():
        for row in frame.filter(pl.col("table") == table).iter_rows(named=True):
            _cell[(mode, row["model"], row["metric"])] = (row["mean"], row["std"])
            if row["model"] not in _models:
                _models.append(row["model"])
    rows = []
    for model in _models:
        _row = {"model": model}
        for mode in modes:
            for metric in metrics:
                v = _cell.get((mode, model, metric))
                _row[f"{mode} · {metric}"] = (
                    None if v is None else fmt_stat(v[0] if v[1] is None else v)
                )
        rows.append(_row)
    if not _models:  # `table` absent from every route → nothing to compare
        return pl.DataFrame({"model": []}, schema={"model": pl.String})
    out = pl.DataFrame(rows)
    _rk = (modes[0], metrics[0])
    _order = sorted(
        _models,
        key=lambda m: (_cell.get((_rk[0], m, _rk[1])) or (float("-inf"),))[0],
        reverse=True,
    )
    return (
        out.join(pl.DataFrame({"model": _order, "_r": range(len(_order))}), on="model")
        .sort("_r")
        .drop("_r")
    )


def _begin(msg):
    print(f"\n▶ Beginning {msg} ...", flush=True)


def _report(name, results):
    """Print a finished CV result dict as a compact table (main-loop progress)."""
    if not results:
        print(f"  (no results for {name})", flush=True)
        return
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=60):
        print(
            display_table(results_frame(results, name), name, sort="macro_f1"),
            flush=True,
        )


def main(mode="excursion"):
    cfg = MODES[mode]
    feature_cols = cfg["feature_cols"]
    TABULAR_FEATURES = cfg["features"]
    TIME_SERIES = cfg["curves"]
    # WIDE_PATH = cfg["wide"]
    RESULTS_PATH = cfg["results"]
    print(f"mode={mode!r} | {len(feature_cols)} features")

    print("Reading tabular features from:", TABULAR_FEATURES)
    X_tab, y, _class_names, data = read_tabular_features(TABULAR_FEATURES, feature_cols)

    model_fn_dict_base = {
        "XGBoost": xgb_fn,
        "CatBoost": catboost_fn,
        "Logistic (L2)": partial(logreg_fn, C=0.5),
        "TabPFN v3": tabpfn_pipeline_fn,
    }

    groups = data["group_id"].to_numpy()
    class_labels = list(range(len(_class_names)))

    print("\n===== engineered features =====")
    _begin("Track A grouped StratifiedGroupKFold (5x5)")
    base_grouped_cv = _cv(
        grouped_cv, model_fn_dict_base, X_tab, y, class_labels, groups
    )
    _report("base_grouped", base_grouped_cv)

    _begin("Track A shuffled-label control (5x5)")
    y_shuf = np.random.RandomState(999).permutation(y)
    base_label_shuffled_group_cv = _cv(
        grouped_cv, model_fn_dict_base, X_tab, y_shuf, class_labels, groups
    )
    _report("base_label_shuffled", base_label_shuffled_group_cv)

    _begin("Track A leave-one-group-out")
    base_logo_cv = _cv(logo_cv, model_fn_dict_base, X_tab, y, class_labels, groups)
    _report("base_logo", base_logo_cv)

    _begin("Track A chronological temporal split (70/30)")
    base_temporal_split = _cv(
        temporal_split,
        model_fn_dict_base,
        X_tab,
        y,
        class_labels,
        data["start"].to_numpy(),
    )
    _report("base_temporal", base_temporal_split)

    map_uid_to_idx = {uid: i for i, uid in enumerate(data["unit_id"].to_list())}

    print("\n===== DL: ROCKET / InceptionTime on raw curves =====")
    print("Reading time-series data from:", TIME_SERIES, flush=True)
    X_seq = read_time_series_curves(TIME_SERIES, map_uid_to_idx)

    # Every head consumes the raw `(N, C, T)` sequences: the ROCKET heads carry a
    # `RocketTransform` step (via `_rocket_head`), InceptionTime a `ChannelScaler` — so
    # InceptionTime is a first-class `model_fn_dict` entry, no special-casing.
    model_fn_dict_dl = {
        "ROCKET + XGBoost": partial(rocket_head_pipeline, xgb_fn),
        "ROCKET + CatBoost": partial(rocket_head_pipeline, catboost_fn),
        "ROCKET + Logistic (L2)": partial(
            rocket_head_pipeline, logreg_fn, C=1.0, impute=False
        ),
        "ROCKET + TabPFN v3": partial(
            rocket_head_pipeline, tabpfn_pipeline_fn, rocket=True
        ),
        "InceptionTime-lite": inception_fn,
    }

    _begin("Track B — ROCKET heads + InceptionTime, tight window (5x5)")
    dl_grouped_cv = _cv(grouped_cv, model_fn_dict_dl, X_seq, y, class_labels, groups)
    _report("dl_grouped", dl_grouped_cv)

    # Wide 48 h-context variant — optional: only the unit route has `proc_curves_wide.parquet`
    # (the moment route sets WIDE_PATH=None), and even there it may be stale/missing, so a skip
    # is graceful rather than crashing the run.
    # dl_wide_grouped_cv = {}
    # if WIDE_PATH:
    #    try:
    #        _begin("Track B — wide 48h window (5x5)")
    #        X_seq_wide = read_time_series_curves(
    #            WIDE_PATH, map_uid_to_idx, num_steps=256
    #        )
    #        dl_wide_grouped_cv = _cv(
    #            grouped_cv, model_fn_dict_dl, X_seq_wide, y, class_labels, groups
    #        )
    #        _report("dl_wide_grouped", dl_wide_grouped_cv)
    #    except (FileNotFoundError, ValueError) as _e:
    #        print(f"[skip] dl_wide_grouped: {type(_e).__name__}: {_e}")
    #        dl_wide_grouped_cv = {}

    _begin("Track B — shuffled-label control (5x5)")
    dl_label_shuf_grouped_cv = _cv(
        grouped_cv, model_fn_dict_dl, X_seq, y_shuf, class_labels, groups
    )
    _report("dl_label_shuffled", dl_label_shuf_grouped_cv)

    _begin("Track B — shuffled-TIME control (5x5)")
    _perm = np.random.RandomState(777).permutation(X_seq.shape[2])
    dl_time_shuf_grouped_cv = _cv(
        grouped_cv, model_fn_dict_dl, X_seq[:, :, _perm], y, class_labels, groups
    )
    _report("dl_time_shuffled", dl_time_shuf_grouped_cv)

    # Persist every CV table as one tidy artifact the modeling notebook loads
    # (instead of recomputing this whole grouped-CV sweep). Keyed by `table`.
    print("\n===== Saving artifact =====")
    results = {
        "base_grouped": base_grouped_cv,
        "base_logo": base_logo_cv,
        "base_temporal": base_temporal_split,
        "base_label_shuffled": base_label_shuffled_group_cv,
        "dl_grouped": dl_grouped_cv,
        # "dl_wide_grouped": dl_wide_grouped_cv,
        "dl_label_shuffled": dl_label_shuf_grouped_cv,
        "dl_time_shuffled": dl_time_shuf_grouped_cv,
    }
    out = pl.concat([results_frame(r, t) for t, r in results.items() if r])
    Path("derived").mkdir(exist_ok=True)
    out.write_parquet(RESULTS_PATH)
    print(
        f"wrote {RESULTS_PATH} — {out.height} rows over "
        f"{out['table'].n_unique()} tables, {out['model'].n_unique()} models"
    )


if __name__ == "__main__":
    _p = argparse.ArgumentParser(
        description="Run the hot/pulse CV sweep and save the results."
    )
    _p.add_argument(
        "--mode",
        choices=list(MODES),
        default="excursion",
        help="classification route: 'excursion' (auto-detected units, default) or 'moment'",
    )
    main(_p.parse_args().mode)
