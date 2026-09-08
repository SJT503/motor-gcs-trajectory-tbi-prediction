# Motor GCS trajectories for real-time mortality prediction after traumatic brain injury

Analysis code, cross-database item mappings, random seeds and frozen model artefacts for the manuscript
*"Motor GCS trajectories enable real-time prediction of mortality after traumatic brain injury"*
(submitted to *npj Digital Medicine*).

## What this repository contains

| Directory | Contents |
|---|---|
| `scripts/` | 34 Python and 2 R scripts: cohort construction, cross-database item mapping, latent-class trajectory modelling, feature building, model fitting, ablation, external validation, benchmarks, calibration and recalibration, fairness subgroups, rolling-window prediction |
| `reports/` | 33 JSON result files and 18 aggregate CSVs — every number reported in the manuscript and supplement, as produced by the scripts. **Aggregates only; no individual-level records.** |
| `models/` | The frozen primary model: 5 LightGBM boosters (one per multiple-imputation replicate) as plain text, plus the v2 model card; `08d_model_card.json`, the card of the earlier 79-variable model, is retained for provenance |
| `figures/fig_scripts/` | Scripts that draw main Figures 1–4 and Supplementary Figures S1–S4 |

## What this repository does **not** contain, and cannot

No patient-level data. The three source databases are not owned by the authors and cannot be
redistributed under their data use agreements. They are available from PhysioNet to researchers who
complete the required training and sign the agreement:

- MIMIC-IV v3.1 — https://doi.org/10.13026/kpb9-mt58
- eICU-CRD v2.0 — https://doi.org/10.13026/C2WM1R
- MIMIC-III v1.4 — https://doi.org/10.13026/C2XW26

Consequently the intermediate `.parquet` files the scripts read and write are absent here, and the
scripts cannot be run end-to-end without first obtaining the databases and rebuilding those
intermediates. Of the six figure scripts that draw the submitted figures, five read per-patient
intermediates and will not run without them — `make_fig1_v1.py` (binned motor-GCS series),
`make_fig2_v1.py`, `make_fig3_v1.py` (per-patient predicted risks), `make_supp_shapdep_v1.py`
(per-patient SHAP values) and `make_supp_v1.py`. Only `make_fig4_v1.py` runs from the aggregate JSON
alone. The `reports/` JSON files let you check every reported number against the code that produced
it without any data access.

`reports/` holds results from two model generations. Files with `v2` in the name (`08d_v2_*`,
`11a_v2_*`, `11b_v2_*`, `11c_v2_*`, `11d_v2_*`) are the 25-variable primary model reported in the
manuscript. Earlier files without the `v2` tag (`08d_lgbm_main.json`, `08g_*`, `08j_*`, `08k_*`) come
from a 79-variable full-feature version that was demoted to a reference model during development;
they are retained for provenance. Supplementary Table S8 of the manuscript records that switch and
which downstream analyses were re-run.

## Reproducing the analysis

1. Obtain the three databases from PhysioNet (above).
2. Set up the two environments in `requirements.txt`. Modelling ran under Python 3.12.13; the figure
   and document build under Python 3.14.4. The two are separate because `pandas`/`numpy` majors differ
   between them. Latent-class trajectory modelling and the sample-size calculation ran under R 4.5.3
   (`lcmm` 2.2.2, `pmsampsize` 1.1.3).
3. Scripts are numbered in execution order (`07*` extraction and mapping → `08*` matrices, model
   fitting → `09*` rolling prediction → `10*` Table 1 → `11*` ablation, benchmarks, fairness,
   calibration).
4. Paths are currently absolute and point at the authors' local layout (`E:/TBI subtype/...`); adjust
   the `ROOT` constant at the top of each script.

## Model and validation summary

A 25-variable LightGBM model, selected by dual screening (L1-penalised logistic ∩ random-forest
importance) on the training split only, predicts 28-day mortality at a 24-hour landmark. Motor-GCS
trajectory information enters only through admission-to-checkpoint summaries: truncated posteriors
from a three-class latent-class growth model plus four motor dynamics.

| Cohort | n | AUROC (95% CI) |
|---|---|---|
| MIMIC-IV internal validation (temporal, 2020–2022) | 483 | 0.905 (0.875–0.936) |
| eICU-CRD (external, 130 hospitals) | 4,440 | 0.818 (0.798–0.835) |
| MIMIC-III CareVue (external, 2001–2008) | 839 | 0.884 (0.854–0.910) |

External validation was zero-touch: no refitting, no threshold or feature changes. Discrimination
transported better than calibration — external predictions were inflated, and a pre-specified
recalibration-only analysis showed a two-parameter logistic update restores calibration without
affecting discrimination. Post-landmark neurological deterioration was **not** predictable
(AUROC 0.55–0.59); this null is reported in the manuscript abstract.

The statistical analysis plan was frozen before any predictive model was fitted. Deviations,
including analyses added after results were seen, are logged in the manuscript supplement.

## Citation

Sheng J, Liu X, Lin R, Sun Q, Chen X, Li K, Chen W. Motor GCS trajectories enable real-time
prediction of mortality after traumatic brain injury. Submitted, 2026.

## Contact

Jiangtao Sheng — jtsheng@stu.edu.cn — Department of Microbiology and Immunology, Shantou University
Medical College

## Licence

Code is released under the MIT Licence (`LICENSE`). This covers the code in this repository only; it
does not extend to the source databases, which remain governed by their PhysioNet data use agreements.
