# Post-resuscitation motor phenotyping and conformal routing guide proactive traumatic brain injury triage

Open analytical pipeline, six-cohort external validation scripts, sequential 0-to-72-h Mondrian split-conformal triage, and zero-server 1,500-tree bedside calculator for:

> Sheng J†,*, Huang S†, Liu X, Lin R, Sun Q, Chen X, Li K, Chen W*. **Post-resuscitation motor phenotyping and conformal routing guide proactive traumatic brain injury triage.** *npj Digital Medicine* (submitted to Thematic Collection: *Digital Biomarkers for Enabling Proactive Clinical Decisions*).

## Overview (`v5.3` Six-Cohort Transcontinental Edition)
- **Study scale**: `12,181` primary trauma/TBI admissions (`21,131` expanded acute brain injury [ABI] stays) across `134` unique hospitals in North America, Europe, and Asia (`MIMIC-IV`, `MIMIC-III CareVue`, `eICU-CRD`, Swiss `HiRID`, Dutch `AmsterdamUMCdb`, and Chinese `JinhuaNSICU` [CNCB-NGDC `OMIX011033`]).
- **Sedation-decoupled 24-h bedside motor awakening & dilution-paradox resolution**: Across `1,443` severe TBI stays (`MIMIC-IV` `n = 683`, `HiRID` `n = 376`, `AmsterdamUMCdb` `n = 384`) with patient-level IV sedation and RASS verification in three countries, `60.5%` (`782/1,293`) of actively sedated stays awakened to `mGCS >= 5` by 24 h with `4.0-fold` lower 28-day mortality (`10.0%` vs `40.1%`), while unsedated stays marked older, profoundly obtunded coma requiring no intubation sedation (`134/150` [`89.3%`] from `MIMIC-IV`; median age `68.5` vs `55.0` yr; `60.7%` awakening, `2.6-fold` separation; `56.9%` [`1,076/1,891`] and `4.3-fold` across all five flowsheet cohorts). Severity stratification shows mild-injury ceiling effects dilute whole-cohort dynamic gains (`+0.060` to `+0.130` AUROC jump in severe TBI; zero-laboratory `Bedside-5` retains `92.7%–99.3%`).
- **Sequential 0-to-72-h Mondrian conformal corridor**: From arrival `Static-3` (`43.8%` [`2,837/6,472`] uncertain `{0, 1}`), 24-h phenotyping chronologically resolved `59.6%` (`1,691/2,837`, `selective NNR = 1.68`; `41.6%` [`914/2,198`, `NNR = 2.40`] marginal motor gain over 24-h `Static S0`), while by 72 h pure conformal singleton assignment resolved `79.9%` (`5,173/6,472`; trajectory stopping coverage `88.7%`) alongside `10.4%` (`675/6,472`; `88.3%` 28-day survivors) pre-72-h clinical ICU exits (`90.0%–93.9%` empirical coverage; `0.844–0.970` singleton AUROC; `0.919` out-of-hospital across 130 US ICUs; stable coverage by `150` local stays), isolating `9.6%` (`624/6,472`, or `20.0%` of active 72-h stays) persistent 72-h `{0, 1}` stays for multimodal escalation.
- **Modality-adaptive static & bilingual head-CT routing**: Across unrecorded-flowsheet ICUs (`eICU-CRD` `n = 1,551`; `JinhuaNSICU` `n = 1,941`), dynamic-feature-free static and structured bilingual head-CT routing (`6,551` stays `<= 24 h`) prevented naive MICE degradation (`+0.050` to `+0.110` AUROC) and resolved `42.1%` (`NNR = 2.38`; `25.8%` marginal CT gain) of uncertain stays.
- **Public repository**: https://github.com/SJT503/motor-gcs-trajectory-tbi-prediction
- **Standalone bedside calculator**: `deployment/tbi_conformal_bedside_calculator.html` embeds all `1,500` `Bedside-5` and `1,000` `Static-3` LightGBM trees with Local Site Calibration Mode and a `< 72 h` anti-WLST safeguard.

## Key 24-h Discrimination & Conformal Summary (`Supplementary Table S1 & S5`)
| Cohort (Country) | Primary `n` (Deaths) | Primary 24-h AUROC | 90% Conformal Coverage | Definitive Singleton AUROC |
|---|---:|---:|---:|---:|
| MIMIC-IV Validation (USA) | 483 (81) | 0.911 | 92.3% | 0.970 |
| MIMIC-III CareVue (USA) | 839 (155) | 0.885 | 90.3% | 0.924 |
| HiRID Trauma (Switzerland) | 1,209 (101) | 0.894 | 90.0% | 0.924 |
| AmsterdamUMCdb Trauma (Netherlands) | 1,052 (92) | 0.845 | 90.1% | 0.908 |
| eICU-CRD 130 Hospitals (USA) | 4,440 (520; charted: 2,889) | 0.826 (0.876 charted) | 93.9% | 0.921 (0.919 LHO) |
| JinhuaNSICU Neurotrauma (China) | 1,941 (240) | 0.809 | 93.0% | 0.844 |

## Repository Structure
- `scripts/` — SQL/Python/R cohort harmonization, train-only `lcmm` LCGM projection, multi-architecture benchmark (`LightGBM`, `CatBoost`, `Tabular-MLP`, `CT-DecayNet`, `Bedside-5`), Mondrian conformal triage, sedation-decoupled awakening & sequential 0-to-72-h conformal corridor (`19a`–`19j`).
- `fig_scripts/` — Deterministic `600 DPI` figure scripts (`make_fig1_v5_4.py`–`make_supp_v5_4.py`).
- `deployment/tbi_conformal_bedside_calculator.html` — Audited standalone client-side calculator.
- `reports/` & `models/` — Summary JSON metrics and model artifacts (zero patient-level data).

## License
MIT
