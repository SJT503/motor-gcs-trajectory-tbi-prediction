# -*- coding: utf-8 -*-
# 10a v2 — Table 1 (fixes the MIMIC-IV sex bug of 10a: 100*(1-sex_female) reported MALE % as female)
# Adds: ICU length of stay, motor-GCS assessments in first 24 h, share with >=2 motor assessments, hospitals (eICU).
import pandas as pd, numpy as np, duckdb
from pathlib import Path
ROOT = Path("E:/TBI subtype"); DATA = ROOT / "07_prediction_system/data"; OUT = ROOT / "07_prediction_system/manuscript"
def cont(s, nd=0):
    s = s.dropna().astype(float); f = f"{{:.{nd}f}}"
    return f"{f.format(s.median())} [{f.format(s.quantile(.25))}–{f.format(s.quantile(.75))}]"
rows = {}
m4 = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
sofa4 = pd.read_parquet(ROOT / "results/features/08z_sofa24.parquet")[["stay_id", "sofa"]]
m4 = m4.merge(sofa4, on="stay_id", how="left")
rows["MIMIC-IV (development)"] = {"n": f"{len(m4):,}", "Age, years": cont(m4.admission_age), "Female, %": f"{100*m4.sex_female.mean():.1f}",
    "Charlson index": cont(m4.charlson_comorbidity_score), "GCS total, first": cont(m4.gcs_total_first), "GCS motor, first": cont(m4.gcs_motor_first),
    "SOFA, day 1": cont(m4.sofa), "ICU length of stay, days": cont(m4.los_h / 24, 1),
    "Motor-GCS assessments, first 24 h": cont(m4.n_obs_m.fillna(0)), "≥2 motor-GCS assessments, %": f"{100*(m4.n_obs_m.fillna(0)>=2).mean():.1f}",
    "Hospitals, n": "1", "28-day deaths, n (%)": f"{int(m4.d28.sum())} ({100*m4.d28.mean():.1f})"}
fe = pd.read_parquet(DATA / "features_eicu.parquet").rename(columns={"age": "admission_age"})
post = pd.read_parquet(DATA / "08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
d = fe.merge(post, on="icustay_id_eicu", how="left"); d = d[~(((d.died_hosp == 1) & (d.los_h <= 24)) | (d.los_h < 24))].copy()
se = pd.read_parquet(DATA / "08i_sofa_eicu.parquet"); sk = "icustay_id_eicu"; sc = [c for c in se.columns if "sofa" in c.lower()][0]
d = d.merge(se[[sk, sc]].rename(columns={sc: "sofa"}), on=sk, how="left")
rows["eICU-CRD (external)"] = {"n": f"{len(d):,}", "Age, years": cont(d.admission_age), "Female, %": f"{100*(1-d.male.astype(float)).mean():.1f}",
    "Charlson index": cont(d.charlson_comorbidity_score), "GCS total, first": cont(d.gcs_total_first), "GCS motor, first": cont(d.gcs_motor_first),
    "SOFA, day 1": cont(d.sofa), "ICU length of stay, days": cont(d.los_h / 24, 1),
    "Motor-GCS assessments, first 24 h": cont(d.n_obs_m.fillna(0)), "≥2 motor-GCS assessments, %": f"{100*(d.n_obs_m.fillna(0)>=2).mean():.1f}",
    "Hospitals, n": str(d.hospitalid.nunique()), "28-day deaths, n (%)": f"{int(d.died_hosp_28d.sum())} ({100*d.died_hosp_28d.mean():.1f})"}
fm = pd.read_parquet(DATA / "features_mimic3.parquet").rename(columns={"age": "admission_age"})
cv = duckdb.connect().execute("SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'").df()
fm = fm[fm.icustay_id.isin(cv.icustay_id)].copy()
p3 = pd.read_parquet(DATA / "08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
d3 = fm.merge(p3, on="icustay_id", how="left"); d3 = d3[~((d3.d28 == 1) & (d3.days_to_death < 1.0)) & ~(d3.los_h < 24)].copy()
s3 = pd.read_parquet(DATA / "08h_sofa_mimic3.parquet"); s3c = [c for c in s3.columns if "sofa" in c.lower() and c != "icustay_id"][0]
d3 = d3.merge(s3[["icustay_id", s3c]].rename(columns={s3c: "sofa"}), on="icustay_id", how="left")
rows["MIMIC-III CareVue (external)"] = {"n": f"{len(d3):,}", "Age, years": cont(d3.admission_age), "Female, %": f"{100*(1-d3.male.astype(float)).mean():.1f}",
    "Charlson index": cont(d3.charlson_comorbidity_score), "GCS total, first": cont(d3.gcs_total_first), "GCS motor, first": cont(d3.gcs_motor_first),
    "SOFA, day 1": cont(d3.sofa), "ICU length of stay, days": cont(d3.los_h / 24, 1),
    "Motor-GCS assessments, first 24 h": cont(d3.n_obs_m.fillna(0)), "≥2 motor-GCS assessments, %": f"{100*(d3.n_obs_m.fillna(0)>=2).mean():.1f}",
    "Hospitals, n": "1", "28-day deaths, n (%)": f"{int(d3.d28.sum())} ({100*d3.d28.mean():.1f})"}
T = pd.DataFrame(rows); T.index.name = "Characteristic"
T.to_csv(OUT / "Table1_v2.csv")
md = "**Table 1 | Baseline characteristics, monitoring and outcomes of the three 24-hour landmark risk sets.**\n\n"
md += "| Characteristic | " + " | ".join(T.columns) + " |\n|---|" + "---|" * len(T.columns) + "\n"
for feat in T.index: md += f"| {feat} | " + " | ".join(str(T.loc[feat, c]) for c in T.columns) + " |\n"
md += ("\nContinuous variables are median [IQR]. GCS, Glasgow Coma Scale; SOFA, Sequential Organ Failure Assessment (first ICU day). "
       "MIMIC-IV is the development database (temporal split: training 2008–2017, model selection 2018–2019, internal validation 2020–2022); "
       "eICU-CRD and MIMIC-III CareVue (2001–2008) are zero-touch external validation cohorts. Motor-GCS assessments are counts of charted motor "
       "scores in the first 24 hours after ICU admission (0 where the motor component was never charted). eICU-CRD outcome = in-hospital death within "
       "28 days of ICU admission; other databases = 28-day all-cause mortality.\n")
(OUT / "Table1_v2.md").write_text(md, encoding="utf-8")
print(T.to_string()); print("[saved] Table1_v2.csv/md")
