import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

with app.setup:
    import os
    import warnings
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt

    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
    from sklearn.metrics import (
        f1_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        matthews_corrcoef,
        roc_auc_score,
        average_precision_score,
    )

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import shap

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    warnings.filterwarnings("ignore")
    alt.data_transformers.disable_max_rows()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Event classification — modeling

    Two tracks in tandem on the detected dissolved-oxygen events:

    - **Track A — gradient-boosting baseline** on the engineered per-event
      feature table (`derived/events.parquet`).
    - **Track B — deep learning** on the raw 5-min event sequences
      (`derived/event_samples.parquet`).

    > ⚠️ **Scaffold caveat.** The target is `ws_label`, the **physics-grounded
    > weak-supervision label** from the EDA notebook (§4b) — a defensible
    > stand-in for the NDA expert labels, built by encoding the published
    > hot-moment vs oxic-pulse taxonomy as labeling functions. It is *not*
    > circular the way the k-means `cluster` is, but **Track A is still partly
    > circular**: the labeling functions read some of the same engineered
    > features Track A trains on, so read Track A as an upper bound. **Track B
    > learns from the raw 5-min shape with no hand features — it is the fairer
    > test, so we lead the discussion with it.** Flip `LABEL_COL` to the real
    > expert column (one line) when it lands; set it to `cluster` or a shuffled
    > column to reproduce the unsupervised / leakage-check rows.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## M0 — Foundation (shared by both tracks)
    """)
    return


@app.cell
def _():
    # Derived artifacts written by exploratory.py's hand-off cell.
    _derived = Path(mo.notebook_location()) / "derived"
    events = pl.read_parquet(_derived / "events.parquet")
    event_samples = pl.read_parquet(_derived / "event_samples.parquet")

    # The single scaffold switch: point this at the real expert-label column
    # once it exists in events.parquet; nothing else changes. Default is the
    # physics-grounded weak-supervision target `ws_label` (exploratory.py §4b);
    # set to "cluster" for the unsupervised pseudo-label, or a shuffled column
    # for the leakage check.
    LABEL_COL = "ws_label"
    return LABEL_COL, event_samples, events


@app.cell
def _(LABEL_COL, events):
    # Engineered numeric features. We DROP identifiers (eid/start/end), the label
    # itself, the cluster-DERIVED columns (cluster_rank/is_abrupt/pc1/pc2), and
    # every weak-supervision artifact (ws_* aggregates + lf_* votes) — all of
    # which leak the target directly.
    DROP_COLS = {
        "eid",
        "start",
        "end",
        LABEL_COL,
        "cluster",
        "cluster_rank",
        "is_abrupt",
        "pc1",
        "pc2",
    }
    feature_cols = [
        c
        for c in events.columns
        if c not in DROP_COLS and not c.startswith("ws_") and not c.startswith("lf_")
    ]

    X = events.select(feature_cols).to_numpy().astype(float)
    class_names = sorted(events[LABEL_COL].unique().to_list())
    _c2i = {c: i for i, c in enumerate(class_names)}
    y = np.array([_c2i[v] for v in events[LABEL_COL].to_list()])
    class_labels = list(range(len(class_names)))

    mo.md(
        f"""
        **Design matrix:** {X.shape[0]} events × {X.shape[1]} engineered features.
        **Classes ({len(class_names)}):** {", ".join(f"`{c}`" for c in class_names)};
        counts = {np.bincount(y).tolist()}.
        """
    )
    return X, class_labels, class_names, feature_cols, y


@app.function
def evaluate(y_true, y_pred, y_proba=None, labels=None):
    """Multiclass-safe metric bundle. PR/ROC-AUC use macro one-vs-rest."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and labels is not None:
        Yb = label_binarize(y_true, classes=labels)
        if Yb.shape[1] == 1:  # binary
            out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            out["pr_auc"] = average_precision_score(y_true, y_proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(Yb, y_proba, average="macro", multi_class="ovr")
            out["pr_auc"] = average_precision_score(Yb, y_proba, average="macro")
    return out


@app.cell
def _():
    def _fit(make_model, Xtr, ytr):
        """Fit with balanced sample weights; fall back when unsupported."""
        m = make_model()
        sw = compute_sample_weight("balanced", ytr)
        try:
            m.fit(Xtr, ytr, sample_weight=sw)
        except (TypeError, ValueError):
            m.fit(Xtr, ytr)
        return m


    def repeated_stratified_cv(make_model, X, y, labels, n_splits=5, n_repeats=5, seed=0):
        """Headline estimate: repeated stratified k-fold, mean ± std per metric.

        Indexes only axis 0 of X, so it works for 2-D tabular matrices and 3-D
        (n, channels, time) sequence tensors alike.
        """
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        rows = []
        for tr, te in rkf.split(np.zeros(len(y)), y):
            m = _fit(make_model, X[tr], y[tr])
            proba = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else None
            # Derive the hard prediction from proba (argmax) so we are robust to
            # each backend's predict() shape quirks (CatBoost MultiClass returns
            # (n,1), XGBoost multi:softprob returns (n,k)); falls back to predict().
            pred = proba.argmax(1) if proba is not None else np.asarray(m.predict(X[te])).ravel()
            rows.append(evaluate(y[te], pred, proba, labels))
        return {
            k: (
                float(np.mean([r[k] for r in rows])),
                float(np.std([r[k] for r in rows])),
            )
            for k in rows[0]
        }


    def temporal_split(make_model, X, y, labels, order, train_frac=0.7):
        """Generalization stress test: chronological early→late split.

        Returns (metrics, train_dist, test_dist) so the imbalance is reported,
        not hidden — class balance drifts across years (the salinization story).
        """
        idx = np.argsort(order)
        cut = int(train_frac * len(idx))
        tr, te = idx[:cut], idx[cut:]
        m = _fit(make_model, X[tr], y[tr])
        proba = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else None
        pred = proba.argmax(1) if proba is not None else np.asarray(m.predict(X[te])).ravel()
        metrics = evaluate(y[te], pred, proba, labels)
        return metrics, np.bincount(y[tr]).tolist(), np.bincount(y[te]).tolist()

    return repeated_stratified_cv, temporal_split


@app.function
def fmt_stat(stat):
    """Format a (mean, std) tuple, or a bare float, as a string."""
    if isinstance(stat, tuple):
        m, s = stat
        return f"{m:.3f} ± {s:.3f}"
    return f"{stat:.3f}"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Track A — Gradient-boosting baseline (engineered features)

    Models: **[XGBoost](https://arxiv.org/abs/1603.02754)** (primary),
    **[CatBoost](https://arxiv.org/abs/1706.09516)** (strong on small data), and an
    **[L2 logistic-regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**
    floor. All use balanced class weighting; tree depths are kept shallow against the
    tiny *n*. ([TabPFN](https://www.nature.com/articles/s41586-024-08328-6) slots in
    once `TABPFN_TOKEN` is set.)
    """)
    return


@app.cell
def _(class_names):
    def _xgb():
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.5,
            objective="multi:softprob",
            num_class=len(class_names),
            n_jobs=2,
            verbosity=0,
        )


    def _cat():
        return CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )


    def _logreg():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5),
        )


    models = {"XGBoost": _xgb, "CatBoost": _cat, "Logistic (L2)": _logreg}
    return (models,)


@app.cell
def _(X, class_labels, models, repeated_stratified_cv, y):
    cv_results = {name: repeated_stratified_cv(mk, X, y, class_labels) for name, mk in models.items()}
    _metrics = list(next(iter(cv_results.values())).keys())
    cv_table = pl.DataFrame(
        [{"model": name, **{k: fmt_stat(res[k]) for k in _metrics}} for name, res in cv_results.items()]
    )
    mo.vstack(
        [
            mo.md(
                "**Repeated stratified 5×5 CV** (headline estimate, mean ± std). "
                "High scores reflect the pseudo-label circularity, not real accuracy."
            ),
            mo.ui.table(cv_table, selection=None),
        ]
    )
    return (cv_results,)


@app.cell
def _(X, class_labels, class_names, events, models, temporal_split, y):
    _order = events["start"].to_numpy()
    _rows = []
    _dists = None
    for _name, _mk in models.items():
        _m, _td, _te = temporal_split(_mk, X, y, class_labels, _order)
        _rows.append({"model": _name, **{k: f"{v:.3f}" for k, v in _m.items()}})
        _dists = (_td, _te)
    temporal_table = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(
                f"""
                **Chronological early→late split** (70/30 by event start — stress test).
                Train class counts = {_dists[0]}, test = {_dists[1]} over
                classes {class_names}. The drift in balance is itself a finding:
                class prevalence shifts across years (the salinization story), so a
                pure temporal split can leave a class nearly absent on one side.
                """
            ),
            mo.ui.table(temporal_table, selection=None),
        ]
    )
    return


@app.cell
def _(X, class_names, feature_cols, y):
    # SHAP TreeExplainer importances on a full-data XGBoost fit. We expect the
    # salinity-/water-level-step and precip drivers to rank highly (corroborates
    # the EDA driver-association finding).
    _m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=len(class_names),
        n_jobs=2,
        verbosity=0,
    )
    _m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    _sv = np.array(shap.TreeExplainer(_m).shap_values(X))
    # normalize to mean|SHAP| per feature regardless of (n,feat,k)/(k,n,feat) layout
    if _sv.ndim == 3 and _sv.shape[-1] == len(class_names):
        _imp = np.abs(_sv).mean(axis=(0, 2))
    elif _sv.ndim == 3:
        _imp = np.abs(_sv).mean(axis=(0, 1))
    else:
        _imp = np.abs(_sv).mean(axis=0)

    shap_importance = pl.DataFrame({"feature": feature_cols, "mean_abs_shap": _imp}).sort(
        "mean_abs_shap", descending=True
    )

    _chart = (
        alt.Chart(shap_importance.head(15))
        .mark_bar(color="#4575b4")
        .encode(
            alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
            alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature:N", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
        )
        .properties(width=460, height=380, title="XGBoost feature importance (SHAP)")
    )
    mo.vstack(
        [
            mo.md("**Which features drive the split** (mean |SHAP|, top 15):"),
            _chart,
        ]
    )
    return


@app.cell
def _(X, class_labels, repeated_stratified_cv, y):
    # TabPFN v2 — zero-tuning in-context-learning sanity check (n < 1000). It needs
    # a one-time Prior Labs license: set TABPFN_TOKEN in the environment to enable.
    # Without it we skip gracefully rather than block the notebook.
    if os.environ.get("TABPFN_TOKEN"):
        from tabpfn import TabPFNClassifier

        def _tabpfn():
            return make_pipeline(
                SimpleImputer(strategy="median"),
                TabPFNClassifier(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    ignore_pretraining_limits=True,
                ),
            )

        tabpfn_cv = repeated_stratified_cv(_tabpfn, X, y, class_labels, n_repeats=2)
        _out = mo.vstack(
            [
                mo.md("**TabPFN v2** (repeated stratified CV, 5×2):"),
                mo.ui.table(
                    pl.DataFrame(
                        [{"model": "TabPFN", **{k: fmt_stat(v) for k, v in tabpfn_cv.items()}}]
                    ),
                    selection=None,
                ),
            ]
        )
    else:
        tabpfn_cv = None
        _out = mo.md(
            "ℹ️ **TabPFN skipped** — set `TABPFN_TOKEN` (one-time Prior Labs license "
            "at <https://ux.priorlabs.ai>) to enable the zero-tuning ICL sanity check."
        )
    _out
    return (tabpfn_cv,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Track B — Deep learning (raw event sequences)

    Each event's multivariate 5-min series is resampled to a fixed length on a
    normalized-time grid. Models, in priority order:

    - **[B1 ROCKET](https://arxiv.org/abs/1910.13051) → logistic** *(primary)* —
      random convolutional kernels, the small-data sweet spot; no GPU needed.
    - **B2 Self-supervised contrastive pretrain + linear probe** — a conv encoder
      trained label-free ([NT-Xent](https://arxiv.org/abs/2002.05709),
      [TS2Vec](https://arxiv.org/abs/2106.10466)) on the event windows, then a linear head.
    - **[B3 InceptionTime](https://arxiv.org/abs/1909.04939) + [focal loss](https://arxiv.org/abs/1708.02002)**
      *(comparison)* — expected to hit the small-*n* overfitting ceiling.
    - **B4 [Patch-transformer](https://arxiv.org/abs/2211.14730)** *(stretch)* — the
      most data-hungry option ([attention](https://arxiv.org/abs/1706.03762)).
    """)
    return


@app.cell
def _(event_samples, events):
    # Resample every event's channels onto a fixed-length normalized-time grid →
    # (n_events, n_channels, SEQ_LEN) tensor aligned to events row order.
    CHANNELS = [
        "Dissolved Oxygen (mg/L)",
        "DO Sensor Temperature (C) ",
        "Well Salinity (PPT)",
        "Flood plain water level in BGS (cm)",
        "Precip (mm) over 5 minutes",
        "AirT_C_Avg",
        "SlrFD_kW_Avg",
    ]
    SEQ_LEN = 128


    def build_event_tensor(events, samples, channels, seq_len):
        grid = np.linspace(0.0, 1.0, seq_len)
        by = {k[0]: v for k, v in samples.partition_by("eid", as_dict=True).items()}
        out = np.zeros((events.height, len(channels), seq_len), dtype=np.float32)
        for i, e in enumerate(events["eid"].to_list()):
            d = by[e].sort("elapsed_min")
            t = d["elapsed_min"].to_numpy().astype(float)
            tn = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else np.zeros_like(t)
            for c, ch in enumerate(channels):
                v = d[ch].to_numpy().astype(float)
                mask = ~np.isnan(v)
                if mask.sum():
                    out[i, c] = np.interp(grid, tn[mask], v[mask])
        return out


    X3d = build_event_tensor(events, event_samples, CHANNELS, SEQ_LEN)
    mo.md(
        f"**Sequence tensor:** {X3d.shape[0]} events × {X3d.shape[1]} channels × "
        f"{X3d.shape[2]} timesteps (resampled on normalized time)."
    )
    return CHANNELS, SEQ_LEN, X3d


@app.cell
def _(SEQ_LEN):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    def set_torch_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


    def focal_loss(logits, target, weight, gamma=2.0):
        logp = F.log_softmax(logits, 1)
        ce = F.nll_loss(logp, target, weight=weight, reduction="none")
        pt = logp.exp().gather(1, target[:, None]).squeeze(1)
        return ((1 - pt) ** gamma * ce).mean()


    class InceptionBlock(nn.Module):
        def __init__(self, cin, nf=24):
            super().__init__()
            self.bottle = nn.Conv1d(cin, nf, 1, padding="same", bias=False)
            self.convs = nn.ModuleList(
                [nn.Conv1d(nf, nf, k, padding="same", bias=False) for k in (9, 19, 39)]
            )
            self.mp = nn.MaxPool1d(3, 1, padding=1)
            self.cmp = nn.Conv1d(cin, nf, 1, padding="same", bias=False)
            self.bn = nn.BatchNorm1d(nf * 4)
            self.act = nn.ReLU()

        def forward(self, x):
            b = self.bottle(x)
            outs = [c(b) for c in self.convs] + [self.cmp(self.mp(x))]
            return self.act(self.bn(torch.cat(outs, 1)))


    class InceptionTime(nn.Module):
        def __init__(self, cin, nclass, nf=24, depth=3):
            super().__init__()
            self.blocks = nn.ModuleList()
            ch = cin
            for _ in range(depth):
                self.blocks.append(InceptionBlock(ch, nf))
                ch = nf * 4
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(ch, nclass)

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.fc(self.gap(x).squeeze(-1))


    class PatchTransformer(nn.Module):
        def __init__(self, cin, nclass, d_model=64, nhead=4, depth=2, patch=8):
            super().__init__()
            self.embed = nn.Conv1d(cin, d_model, patch, stride=patch)
            n_tok = SEQ_LEN // patch
            self.pos = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.tr = nn.TransformerEncoder(enc, depth)
            self.fc = nn.Linear(d_model, nclass)

        def forward(self, x):
            z = self.embed(x).transpose(1, 2) + self.pos
            return self.fc(self.tr(z).mean(1))


    class ConvEncoder(nn.Module):
        """Dilated-conv encoder for the self-supervised track."""

        def __init__(self, cin, emb=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, 64, 7, padding=3),
                nn.ReLU(),
                nn.Conv1d(64, 64, 7, padding=6, dilation=2),
                nn.ReLU(),
                nn.Conv1d(64, emb, 7, padding=12, dilation=4),
                nn.ReLU(),
            )
            self.gap = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            return self.gap(self.net(x)).squeeze(-1)


    class TorchSeqClassifier:
        """sklearn-style wrapper so torch sequence models reuse the CV harness.

        Standardizes channels on the training fold, trains with focal loss and
        balanced class weights, exposes predict / predict_proba.
        """

        def __init__(
            self, build_fn, n_classes, epochs=140, lr=1e-3, wd=1e-3, jitter=0.05, gamma=2.0, seed=0
        ):
            self.build_fn = build_fn
            self.n_classes = n_classes
            self.epochs, self.lr, self.wd = epochs, lr, wd
            self.jitter, self.gamma, self.seed = jitter, gamma, seed

        def fit(self, X, y, sample_weight=None):
            set_torch_seed(self.seed)
            self.mu_ = X.mean((0, 2), keepdims=True)
            self.sd_ = X.std((0, 2), keepdims=True) + 1e-6
            Xs = (X - self.mu_) / self.sd_
            cw = compute_class_weight("balanced", classes=np.arange(self.n_classes), y=y)
            cw = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
            self.net_ = self.build_fn(X.shape[1], self.n_classes).to(DEVICE)
            opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
            xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
            self.net_.train()
            for _ in range(self.epochs):
                xb = xt + self.jitter * torch.randn_like(xt)
                opt.zero_grad()
                focal_loss(self.net_(xb), yt, cw, self.gamma).backward()
                opt.step()
            return self

        def predict_proba(self, X):
            Xs = (X - self.mu_) / self.sd_
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(torch.tensor(Xs, dtype=torch.float32, device=DEVICE))
                return F.softmax(logits, 1).cpu().numpy()

        def predict(self, X):
            return self.predict_proba(X).argmax(1)


    def nt_xent(z1, z2, temp=0.2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], 0)
        n = z1.shape[0]
        sim = z @ z.t() / temp
        sim.fill_diagonal_(-1e9)
        targets = torch.arange(n, device=z.device)
        targets = torch.cat([targets + n, targets])
        return F.cross_entropy(sim, targets)

    return (
        ConvEncoder,
        DEVICE,
        InceptionTime,
        PatchTransformer,
        TorchSeqClassifier,
        nt_xent,
        set_torch_seed,
    )


@app.cell
def _(CHANNELS, X3d, class_labels, repeated_stratified_cv, y):
    # B1 — ROCKET (multivariate, hand-rolled): random dilated conv kernels with
    # PPV + max pooling, then a balanced logistic head. Kernels are data- and
    # label-independent, so transforming all events up front is leakage-free; only
    # the scaler + classifier are refit per fold.
    def _rocket_kernels(n_channels, seq_len, n_kernels=1000, seed=0):
        rng = np.random.default_rng(seed)
        ks = []
        for _ in range(n_kernels):
            klen = int(rng.choice([7, 9, 11]))
            w = rng.standard_normal((n_channels, klen)).astype(np.float32)
            w -= w.mean(axis=1, keepdims=True)
            nch = rng.integers(1, n_channels + 1)
            chans = rng.choice(n_channels, size=nch, replace=False)
            a = int(np.log2((seq_len - 1) / (klen - 1))) if seq_len > klen else 0
            dil = int(2 ** rng.uniform(0, max(a, 0)))
            bias = float(rng.uniform(-1, 1))
            pad = ((klen - 1) * dil) // 2 if rng.random() < 0.5 else 0
            ks.append((w, chans, dil, bias, pad))
        return ks


    def _rocket_transform(X3d, kernels):
        n = X3d.shape[0]
        feats = np.zeros((n, 2 * len(kernels)), dtype=np.float32)
        for j, (w, chans, dil, bias, pad) in enumerate(kernels):
            klen = w.shape[1]
            Xp = np.pad(X3d[:, chans, :], ((0, 0), (0, 0), (pad, pad))) if pad else X3d[:, chans, :]
            span = (klen - 1) * dil + 1
            if Xp.shape[2] < span:
                continue
            nout = Xp.shape[2] - span + 1
            acc = np.full((n, nout), bias, dtype=np.float32)
            for ci_, _ch in enumerate(chans):
                wc = w[_ch]
                for k in range(klen):
                    acc += wc[k] * Xp[:, ci_, k * dil : k * dil + nout]
            feats[:, 2 * j] = (acc > 0).mean(axis=1)
            feats[:, 2 * j + 1] = acc.max(axis=1)
        return feats


    rocket_features = _rocket_transform(
        X3d, _rocket_kernels(len(CHANNELS), X3d.shape[2], n_kernels=1000, seed=0)
    )


    def _rocket_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        )


    rocket_cv = repeated_stratified_cv(_rocket_clf, rocket_features, y, class_labels)
    mo.vstack(
        [
            mo.md(
                f"**B1 ROCKET → logistic** ({rocket_features.shape[1]} random features, "
                "5×5 CV). Learns from raw shape — less circular than Track A:"
            ),
            mo.ui.table(
                pl.DataFrame(
                    [{"model": "B1 ROCKET", **{k: fmt_stat(v) for k, v in rocket_cv.items()}}]
                ),
                selection=None,
            ),
        ]
    )
    return (rocket_cv,)


@app.cell
def _(
    ConvEncoder,
    DEVICE,
    X3d,
    class_labels,
    nt_xent,
    repeated_stratified_cv,
    set_torch_seed,
    y,
):
    # B2 — Self-supervised contrastive pretrain (label-free) + linear probe. The
    # encoder is trained once on ALL event windows with two jittered/scaled views
    # per event (NT-Xent); embeddings are then probed per fold. Pretraining never
    # sees labels, so embedding all events up front is leakage-free. (Scaling this
    # to sliding windows over the full 553k-row series is the documented next step.)
    def _train_ssl(X3d, epochs=300, emb=64, seed=0):
        set_torch_seed(seed)
        mu = X3d.mean((0, 2), keepdims=True)
        sd = X3d.std((0, 2), keepdims=True) + 1e-6
        xt = torch.tensor((X3d - mu) / sd, dtype=torch.float32, device=DEVICE)
        enc = ConvEncoder(X3d.shape[1], emb).to(DEVICE)
        opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
        enc.train()
        for _ in range(epochs):
            v1 = xt + 0.1 * torch.randn_like(xt)
            v2 = xt * (1 + 0.1 * torch.randn_like(xt))
            opt.zero_grad()
            nt_xent(enc(v1), enc(v2)).backward()
            opt.step()
        enc.eval()
        with torch.no_grad():
            return enc(xt).cpu().numpy()


    ssl_embeddings = _train_ssl(X3d)


    def _probe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        )


    ssl_cv = repeated_stratified_cv(_probe, ssl_embeddings, y, class_labels)
    mo.vstack(
        [
            mo.md(
                f"**B2 Self-supervised + linear probe** ({ssl_embeddings.shape[1]}-d "
                "contrastive embedding, 5×5 CV):"
            ),
            mo.ui.table(
                pl.DataFrame(
                    [{"model": "B2 SSL+probe", **{k: fmt_stat(v) for k, v in ssl_cv.items()}}]
                ),
                selection=None,
            ),
        ]
    )
    return (ssl_cv,)


@app.cell
def _(
    InceptionTime,
    TorchSeqClassifier,
    X3d,
    class_labels,
    class_names,
    repeated_stratified_cv,
    y,
):
    # B3 — Supervised InceptionTime + focal loss. Expected to overfit at n~100;
    # included to demonstrate the small-data DL ceiling. 5×2 CV to bound runtime.
    def _make_inception():
        return TorchSeqClassifier(
            lambda cin, k: InceptionTime(cin, k),
            n_classes=len(class_names),
            epochs=140,
        )


    inception_cv = repeated_stratified_cv(_make_inception, X3d, y, class_labels, n_repeats=2)
    mo.vstack(
        [
            mo.md("**B3 InceptionTime + focal loss** (5×2 CV):"),
            mo.ui.table(
                pl.DataFrame(
                    [{"model": "B3 InceptionTime", **{k: fmt_stat(v) for k, v in inception_cv.items()}}]
                ),
                selection=None,
            ),
        ]
    )
    return (inception_cv,)


@app.cell
def _(
    PatchTransformer,
    TorchSeqClassifier,
    X3d,
    class_labels,
    class_names,
    repeated_stratified_cv,
    y,
):
    # B4 — Patch-transformer (stretch). Most data-hungry; here mainly to show the
    # ceiling holds for attention models too at this n. 5×2 CV.
    def _make_transformer():
        return TorchSeqClassifier(
            lambda cin, k: PatchTransformer(cin, k),
            n_classes=len(class_names),
            epochs=160,
        )


    transformer_cv = repeated_stratified_cv(_make_transformer, X3d, y, class_labels, n_repeats=2)
    mo.vstack(
        [
            mo.md("**B4 Patch-transformer** (5×2 CV):"),
            mo.ui.table(
                pl.DataFrame(
                    [{"model": "B4 Transformer", **{k: fmt_stat(v) for k, v in transformer_cv.items()}}]
                ),
                selection=None,
            ),
        ]
    )
    return (transformer_cv,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## M-final — Comparison & synthesis
    """)
    return


@app.cell
def _(cv_results, inception_cv, rocket_cv, ssl_cv, tabpfn_cv, transformer_cv):
    # One table, every model on the same repeated-stratified-CV protocol & metrics.
    _entries = []
    for _name, _res in cv_results.items():
        _entries.append(("A · engineered", _name, _res))
    if tabpfn_cv is not None:
        _entries.append(("A · engineered", "TabPFN", tabpfn_cv))
    _entries += [
        ("B · raw seq", "B1 ROCKET", rocket_cv),
        ("B · raw seq", "B2 SSL+probe", ssl_cv),
        ("B · raw seq", "B3 InceptionTime", inception_cv),
        ("B · raw seq", "B4 Transformer", transformer_cv),
    ]
    _metrics = list(rocket_cv.keys())
    results_all = pl.DataFrame(
        [
            {"track": _t, "model": _n, **{k: fmt_stat(_r[k]) for k in _metrics}}
            for _t, _n, _r in _entries
        ]
    )
    # numeric macro-F1 for ranking
    results_all = results_all.with_columns(
        pl.Series("macro_f1_num", [_r["macro_f1"][0] for _, _, _r in _entries])
    ).sort("macro_f1_num", descending=True)
    mo.vstack(
        [
            mo.md("**All models, repeated-stratified CV (mean ± std), ranked by macro-F1:**"),
            mo.ui.table(results_all.drop("macro_f1_num"), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Synthesis & honest caveats

    - **Circularity (Track A).** The `cluster` target was derived *from* the
      engineered features, so boosting on those same features is the most
      circular setup — strong scores here mostly validate the pipeline. Track B
      (raw sequences) is a different representation and a fairer, if still
      pseudo-labelled, test.
    - **Tiny-*n* ceiling (Track B).** With ~100 events the supervised deep nets
      (InceptionTime, transformer) sit below ROCKET and the boosted baselines —
      the expected small-data ceiling. ROCKET (fixed random kernels) and the
      self-supervised probe are the right tools at this scale.
    - **Temporal imbalance.** The early→late split leaves one class nearly
      absent in the late period — class prevalence genuinely drifts across years
      (the salinization story). Treat that as a finding, not just a split issue.
    - **No SMOTE.** At ~20 minority samples, balanced class weights / focal loss
      are used instead of synthetic oversampling.
    - **Drop-in real labels.** When the NDA expert labels arrive, add them to
      `events.parquet`, set `LABEL_COL` to that column, and rerun — every model,
      split, and metric above recomputes unchanged.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    Citations for each model and method used above.

    **Track A — gradient boosting & tabular**

    - **XGBoost** — Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD. <https://arxiv.org/abs/1603.02754>
    - **CatBoost** — Prokhorenkova et al. (2018), *CatBoost: unbiased boosting with categorical features*, NeurIPS. <https://arxiv.org/abs/1706.09516>
    - **Logistic regression (L2)** — scikit-learn `LogisticRegression`. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
    - **TabPFN v2** — Hollmann et al. (2025), *Accurate predictions on small data with a tabular foundation model*, Nature. <https://www.nature.com/articles/s41586-024-08328-6> · orig. ICLR 2023: <https://arxiv.org/abs/2207.01848>

    **Track B — time-series deep learning**

    - **ROCKET** — Dempster, Petitjean & Webb (2020), *ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels*, DMKD. <https://arxiv.org/abs/1910.13051> · **MiniRocket**: <https://arxiv.org/abs/2012.08791>
    - **Self-supervised contrastive (B2)** — NT-Xent / SimCLR: Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations*. <https://arxiv.org/abs/2002.05709> · time-series instantiation **TS2Vec**: Yue et al. (2022), AAAI. <https://arxiv.org/abs/2106.10466>
    - **InceptionTime** — Ismail Fawaz et al. (2020), *InceptionTime: Finding AlexNet for Time Series Classification*, DMKD. <https://arxiv.org/abs/1909.04939>
    - **Patch-transformer** — attention: Vaswani et al. (2017), *Attention Is All You Need*. <https://arxiv.org/abs/1706.03762> · patch-based TS transformer **PatchTST**: Nie et al. (2023), ICLR. <https://arxiv.org/abs/2211.14730>

    **Shared methods**

    - **Focal loss** (Track B class imbalance) — Lin et al. (2017), *Focal Loss for Dense Object Detection*, ICCV. <https://arxiv.org/abs/1708.02002>
    - **SHAP** (Track A interpretability) — Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS. <https://arxiv.org/abs/1705.07874>
    """)
    return


if __name__ == "__main__":
    app.run()
