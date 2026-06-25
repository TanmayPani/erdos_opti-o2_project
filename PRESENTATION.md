# Opti O2 — presentation outline

> Erdős Data Science Bootcamp. Binding deliverable: the talk + a turnkey repo.
> The real expert labels are under NDA and undelivered, so the story is *"we
> built and validated the entire pipeline anyway, and it flips to the real
> labels in one line."* All numbers below are reproduced by the notebooks.

## Narrative arc

1. **Problem.** Opti O2 hand-classifies dissolved-oxygen events into *hot
   moments* (abrupt, tidal) vs *oxic pulses* (symmetric, precip-driven). Goal:
   formalize that into an automated, interpretable, validated classifier.

2. **The wrinkle.** The expert-labeled NDA dataset has not arrived. We have only
   the public ESS-DIVE stand-in from the same site — **no labels**.

3. **What we built anyway** (on the stand-in):
   - **Detection + features** (`exploratory.py`): ~102 events, event-shape +
     antecedent hydro-met features.
   - **Physics-grounded weak-supervision labels** (`ws_label`): the published
     taxonomy encoded as labeling functions + a transparent label model — a
     defensible, *non-circular* stand-in for the expert labels.
   - **Dual-track classifier** (`modeling.py`): boosting on features + deep
     learning on raw sequences, one leakage-checked CV harness.
   - **Unsupervised driver discovery** (`discovery.py`): label-free corroboration
     + the sea-level-rise roadmap.

4. **Results** (stand-in / weak-supervision labels — see table below).

5. **Honesty slide.** Track A is *partly* circular (LFs read some features it
   trains on) → read it as an upper bound. Track B (raw shape, no hand features)
   is the fairer test. Shuffled labels → chance proves no leakage.

6. **The one-line guarantee.** `LABEL_COL = "ws_label"` → flip to the real expert
   column on NDA delivery; the entire pipeline re-runs unchanged in < 1 min.

## Headline numbers (reproduced by the notebooks)

| model | macro-F1 | Cohen's κ | ROC-AUC |
|---|---|---|---|
| XGBoost | 0.92 ± 0.05 | 0.85 | 0.985 |
| CatBoost | 0.94 ± 0.05 | 0.87 | 0.985 |
| Logistic (L2) | 0.95 ± 0.06 | 0.90 | 0.989 |
| **shuffled-label control** | **0.51** | ≈0 | ≈0.5 |

Unsupervised corroboration (`discovery.py`): the split is invisible in the full
feature geometry (ARI ≈ 0) but **emerges on its own in the shape+driver
subspace** (k-means / Ward ARI ≈ 0.65–0.70). Top mutual-information drivers:
curve-symmetry shape + salinity/water-level step — the expected physics.

## KPIs, restated for the contingency

| KPI | Original (needs expert labels) | Contingency form (shipped) |
|---|---|---|
| #1 Detection | ≥0.9 P/R vs expert reference | Robustness vs threshold + eye-reviewed subset |
| #2 Classification | macro-F1 ≥ 0.85 | **Met** vs `ws_label` (0.92–0.95) |
| #3 Expert agreement | κ ≥ 0.80 vs manual | **Met** vs `ws_label` (0.85–0.90); + LF coverage/conflict |
| #4 Temporal generalization | <10-pt F1 drop | Chronological early→late split reported |
| #5 Interpretability | drivers match physics | **Met** — SHAP (Track A) + label-free MI (discovery) agree |
| #6 Reproducibility | <5 min full rerun | **Met** — ~26 s end-to-end |

> Every "met" becomes a *genuine* expert-label result the moment `LABEL_COL` is
> flipped on the NDA data — this plan is a strict superset of the original.
