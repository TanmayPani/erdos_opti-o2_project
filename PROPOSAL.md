# Formalizing Dissolved-Oxygen Event Classification in a Transitioning Coastal Wetland

**Erdős Institute Data Science Bootcamp — Project Proposal**
**Industry partner:** Opti O2, LLC (East Lansing / Okemos, MI)

> One-line summary: Build a reproducible, interpretable machine-learning pipeline that
> detects and classifies dissolved-oxygen (DO) "events" in a six-year, 5-minute
> groundwater time series — turning Opti O2's expert, hand-labeled event taxonomy
> ("hot moments" vs "oxic pulses") into a formal, automated, validated classifier.

---

## Project scope

Opti O2 has continuously measured subsurface dissolved oxygen (DO) at 5-minute
resolution for six years at a former freshwater floodplain on the US Pacific coast
that is transitioning into a coastal wetland after culvert removal exposed it to
tidal seawater intrusion. Against a near-permanently anoxic baseline, oxygen arrives
in discrete *events* that domain experts have manually sorted into two classes —
**"hot moments"** (abrupt, temporally *asymmetric* DO rises driven by incursions of
high-DO tidal water, coincident with step changes in salinity, water level, and
temperature) and **"oxic pulses"** (temporally *symmetric* DO rises and falls,
typically following precipitation or preceding a series of tidal hot moments). This
project will **formalize that manual classification methodology** by building a
supervised pipeline that (i) detects discrete oxygenation events against the anoxic
baseline, (ii) engineers event-shape features (rise/fall symmetry, inflection
structure, duration, magnitude) together with *antecedent* hydro-meteorological
features spanning the diel, tidal, seasonal, and yearly scales known to drive these
events, and (iii) trains an interpretable classifier to reproduce the expert labels
with quantified accuracy and uncertainty. The pipeline will be developed and
validated **now** on a public stand-in dataset from the same research site, then run
on Opti O2's NDA-protected labeled dataset when it arrives — providing Opti O2 a
reproducible, auditable replacement for hand-classification and a first step toward
their longer-term goal of modeling coastal-ecosystem response to sea-level rise
(an unsupervised driver-discovery extension is noted below as a stretch goal).

---

## Data sources

**Primary (real) dataset — Opti O2, under NDA, delivered after signing.**
A six-year ($\approx$ 2019-2025), continuous, 5-minute-resolution time series of subsurface
DO measured with Opti O2's optical sensor technology, co-located with **three
water-quality / hydrological variables** and **five meteorological parameters**, plus
the team's **manual DO-event class labels** (the supervised targets). The actual dataset is not yet in hand.

**Stand-in dataset — public, used for immediate prototyping.**
The same group's public release on **ESS-DIVE** (DOE's Environmental Systems Science
data repository) for the *same study site*:

- *Title:* "Five Years of Dissolved Oxygen, Temperature, Salinity, Depth, Weather
  Data from a Transitioning Wetland at Beaver Creek, Washington, USA"
  (Ghosh, McIntire, Freeman, Ward, Regier, Norwood, Shooltz, Myers-Pigg, et al.).
- *Site:* PNNL-managed floodplain on Beaver Creek, WA ( $\approx (46.9056\degree N, -123.9806\degree W)$ ).
- *Coverage explored:* 2019-06-26 &rarr; 2024-09-30, **~553,845 rows at 5-minute cadence**.
- *Links:*
  - [ESS-DIVE landing page](https://data.ess-dive.lbl.gov/view/doi:10.15485/2530966)
  - [Related peer-reviewed article (Limnology & Oceanography)](https://doi.org/10.1002/lno.12426)
- *Variables:* dissolved oxygen (mg/L) and $O_{2}$ concentration (%), DO-sensor
  temperature ($\degree C$), well salinity (PPT), floodplain water level (cm below ground
  surface); and weather: solar flux density, precipitation, air temperature,
  barometric pressure, total solar flux, wind speed, vapor pressure, relative
  humidity.

**Why the stand-in is fit for purpose (exploratory evidence).** Initial analysis in
[`exploratory.py`](exploratory.py) confirms the dataset has exactly the structure the
modeling relies on:

- **Episodic, zero-inflated DO.** DO / $O_{2}$% are **$\approx 92.5$% exact zeros** (the anoxic
  baseline) with a clearly separated oxic mode (oxygenation is genuinely *event*-like).
- **Multiple regimes in the drivers.** Well salinity is **multimodal** (peaks near
  ~5–6 and ~10 PPT), consistent with the salinization / step-change narrative behind
  hot moments.
- **Multiple time-scales.** An FFT periodogram shows a dominant **diurnal (1/day)**
  peak, a **semi-diurnal (~2/day, tidal-band)** harmonic, broad **synoptic** power
  (days–weeks), and an **annual** component — confirming that antecedent conditions
  must be encoded across scales, as the domain hypothesis requires.

---

## Stakeholders

- **Opti O2, LLC — primary decision-maker / client.**
  - *Ruby N. Ghosh* (PI, Opti O2) — defines the event taxonomy, owns the scientific
    questions, and will decide whether the formalized classifier replaces manual
    labeling and feeds their sea-level-rise modeling roadmap.
  - *Charles McIntire* (Opti O2) — sensor technology and dataset.
- **Coastal-biogeochemistry & sea-level-rise research community** — the broader
  audience (e.g., Ocean Sciences Meeting; the related L&O publication) that benefits
  from a defensible, automated event-classification methodology.
- **Erdős project team** — *Tanmay Pani, Eric Britt*, responsible for delivering the
  pipeline, evaluation, and documentation.

---

## Key Performance Indicators (KPIs)

Because the DO series is dominated by the anoxic baseline, raw accuracy is misleading;
KPIs emphasize class-balanced metrics and uplift over naive baselines.

1. **Event detection quality.** $\geq 0.9$ **precision and recall** for detected
   oxygenation events versus an expert-reviewed reference set (event = contiguous
   departure from the anoxic baseline).
2. **Classification performance.** **Macro-F1** $\geq 0.85$ on held-out events for the
   hot-moment vs oxic-pulse task, and **$\geq 20$ % above** a
   majority-class / simple-threshold baseline (report PR-AUC alongside F1).
3. **Agreement with experts.** **Cohen's $\kappa \geq 0.80$** between model predictions and the
   manual labels — i.e., the classifier substantially reproduces expert judgment.
4. **Temporal generalization.** Train on early years, test on later years (where the
   hot-moment vs oxic-pulse balance shifts with progressive salinization); target a
   **$< 10$-point macro-F1 drop** relative to random-split performance — evidence the
   method is robust to the multi-year ecosystem transition.
5. **Interpretability / scientific validity.** Ranked feature importances must
   corroborate the known physical drivers (salinity & water-level *step changes* &rarr;
   hot moments; *antecedent precipitation* &rarr; oxic pulses), giving Opti O2 a
   mechanistically defensible model rather than a black box.
6. **Reproducibility & efficiency.** End-to-end pipeline (load &rarr; detect &rarr; feature &rarr;
   classify &rarr; evaluate) reruns on the full six-year, 5-minute series in **under ~5
   minutes on a laptop** (polars-based, no manual steps), so it can be re-run
   verbatim on the Opti O2 NDA dataset on delivery.

---

### Stretch goal (Challenge 2)

Once the supervised classifier is validated, apply **unsupervised** methods
(clustering of event-shape + antecedent-condition features; wavelet / information-
theoretic driver analysis) to test whether the two expert classes emerge naturally,
to surface sub-classes, and to begin parsing scale-specific hydro-meteorological
drivers - the foundation for Opti O2's longer-term sea-level-rise modeling objective.
