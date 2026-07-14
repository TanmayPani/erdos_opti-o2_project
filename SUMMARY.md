# Executive Summary: Classification of hot-moment and oxic-pulses in transitioning coastal wetlands.

## Overview
This project aims to understand biogeochemical dynamics of *coastal interfaces* (e.g., estuaries, marshes, mangroves, swamps) through *Dissolved oxygen* (DO) measurements.
DO is an indicator of coastal ecosystem health (carbon cycles, nutrient mobility, vegetation, etc.), coupled to processes that are both terrestrial (e.g., precipitation, flooding) and aquatic (e.g., tides). Thus precise, dynamic DO measurements, followed by characterization of anomalous and periodic DO events, and understanding their environmental drivers will give us important insights into how climate change effects (e.g. rising sea levels, droughts) translate into short/long term consequences for these complex ecosystems. 

DO readings in floodplains usually sit at zero (anoxic baseline). However, due to various environmental conditions which oxygenate the floodplain (e.g., tidal incursions, flooding, precipitation), DO readings temporarily leave this baseline, giving rise to a DO event. These events can last few hours to several days, and can have many sub-events called moments. Based on the way DO rises and the environmental drivers, DO moments can be:
1. **Hot Moments:** Abrupt, asymmetric events driven by tidal and saline intrusions.
2. **Oxic Pulses:** Slower, symmetric events driven by freshwater and precipitation changes.

We use ML techniques to formalize classification of measured DO moments as hot/oxic, and isolate environmental drivers of DO activity.

## Data Foundation
The analysis is developed using proprietary, labelled datasets collected Opti-O2, using their own detector tenchology. It is a robust 6-year, 5-minute resolution time-series dataset comprising:
- Subsurface Dissolved Oxygen ($DO_2$) levels
- Hydrological metrics (Water level, Salinity)
- Weather indicators (Air Temperature, Precipitation)

## Methodology and Results 
The pipeline combines data engineering with advanced machine learning paradigms to robustly identify the mechanisms behind oxygen excursions.

### 1. Supervised Modeling & Classification
- **Engineered Features:** Evaluated tabular models on 24 domain-engineered features capturing event shape, hydrology, and antecedent precipitation.
- **Time-Series / Sequences:** Deployed deep learning models directly on raw multivariate waveforms (e.g., InceptionTime-Lite, ROCKET transforms).
- **Model Ensembles:** Linear models (Logistic Regression), Tree-based algorithms (XGBoost, CatBoost), and foundational Deep Learning models (TabPFN).
- **Validation:** Ensured strict generalizability via 5x5 StratifiedGroupKFold, Leave-One-Group-Out (LOGO) cross-validation, and multi-year temporal splits to account for chronological drift.
- F1-score, Balanced accuracy > 0.9 for the best performing models, showing good generalizability

### 2. Interpretability & Feature Attribution
- **Supervised Attribution (SHAP):** Deployed exact, gradient, and imputation SHAP explainers (via GPU) to rank driver significance. Results decisively indicated that hydrological markers (e.g., salinity steps, water-level rise rates) heavily outweigh weather markers in predicting Hot Moments.
- **Unsupervised Validation:** Utilized Mutual Information to independently corroborate the feature dependencies highlighted by SHAP.

### 3. Unsupervised Discovery
- Explored label-free taxonomy using K-Means clustering and weak-supervision algorithms to align physical domain rules with raw clustering behavior, laying the groundwork for automated, label-free classification.
- Unsupervised models like KMeans and Snorkel labelling detect hot moments with ease, but struggle with oxic pulses. 
- Since all the models are small and unsupervised 

## Conclusions
- The study successfully formalizes the detection and categorization of Hot Moments and Oxic Pulses. 
- By validating the model interpretations against both expert-provided labels and unsupervised metrics, the project delivers a highly interpretable, dual-track machine learning methodology for ongoing coastal respiration analysis.
- Ongoing work will look at more sophisticated models for segementing moments and formalize the oxic pulses better.
