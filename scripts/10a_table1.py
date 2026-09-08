# -*- coding: utf-8 -*-
# 10a — Table 1: 三库基线特征 (R1, 审稿团 must-fix)
# 行: n / 年龄 / 女性 / Charlson / 入院 GCS 总分 / 入院 motor GCS / 首日 SOFA / 28d 死亡
# 口径: 各库 24h landmark 风险集 (M4=08c 全 2700; eICU=4440 风险集; M3=1423 风险集)
# 产物: manuscript/Table1.csv + Table1.md (独立文件, 不进正文 MD —— 表格职责边界铁律)
import pandas as pd, numpy as np, json
from pathlib import Path

ROOT = Path("E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
OUT = ROOT / "07_prediction_system/manuscript"

def summ_cont(s):
    s = s.dropna().astype(float)
    return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"

rows = {}

# --- MIMIC-IV (08c 矩阵全风险集) ---
m4 = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
sofa4 = pd.read_parquet(ROOT / "results/features/08z_sofa24.parquet")
m4 = m4.merge(sofa4[["stay_id", "sofa"]].rename(columns={"sofa": "sofa24"}), on="stay_id", how="left")
rows["MIMIC-IV (dev)"] = {
    "n": len(m4), "Age, yr": summ_cont(m4.admission_age),
    "Female, %": f"{100*(1-m4.sex_female).mean():.1f}" if "sex_female" in m4 else None,
    "Charlson": summ_cont(m4.charlson_comorbidity_score),
    "GCS total (first)": summ_cont(m4.gcs_total_first),
    "GCS motor (first)": summ_cont(m4.gcs_motor_first),
    "SOFA (day 1)": summ_cont(m4.sofa24),
    "28-d deaths, n (%)": f"{int(m4.d28.sum())} ({100*m4.d28.mean():.1f})",
}

# --- eICU (08e 风险集口径 = features_eicu 排除后) ---
fe = pd.read_parquet(DATA / "features_eicu.parquet")
if "age" in fe.columns: fe = fe.rename(columns={"age": "admission_age"})
post = pd.read_parquet(DATA / "08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
d = fe.merge(post, on="icustay_id_eicu", how="left")
died24 = ((d.died_hosp == 1) & (d.los_h <= 24)); short = d.los_h < 24
d = d[~(died24 | short)].copy()
sofa_e = pd.read_parquet(DATA / "08i_sofa_eicu.parquet")
sk = "icustay_id_eicu" if "icustay_id_eicu" in sofa_e.columns else sofa_e.columns[0]
scol = [c for c in sofa_e.columns if "sofa" in c.lower()][0]
d = d.merge(sofa_e[[sk, scol]].rename(columns={sk: "icustay_id_eicu", scol: "sofa24"}), on="icustay_id_eicu", how="left")
gt = d.gcs_total_first if "gcs_total_first" in d.columns else None
gm = d.gcs_motor_first if "gcs_motor_first" in d.columns else None
rows["eICU-CRD (ext)"] = {
    "n": len(d), "Age, yr": summ_cont(d.admission_age),
    "Female, %": f"{100*(1-d.male).mean():.1f}" if "male" in d.columns else None,
    "Charlson": summ_cont(d.charlson_comorbidity_score) if "charlson_comorbidity_score" in d.columns else "—",
    "GCS total (first)": summ_cont(gt) if gt is not None else "—",
    "GCS motor (first)": summ_cont(gm) if gm is not None else "—",
    "SOFA (day 1)": summ_cont(d.sofa24),
    "28-d deaths, n (%)": f"{int(d.died_hosp_28d.sum())} ({100*d.died_hosp_28d.mean():.1f})",
}

# --- MIMIC-III (CareVue-only) ---
fm = pd.read_parquet(DATA / "features_mimic3.parquet")
if "age" in fm.columns: fm = fm.rename(columns={"age": "admission_age"})
# CareVue 过滤 (2026-09-07)
import duckdb as _dd
_cv3 = _dd.connect().execute("""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true)
WHERE DBSOURCE = 'carevue'""").df()
fm = fm[fm.icustay_id.isin(_cv3.icustay_id)].copy()
post3 = pd.read_parquet(DATA / "08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
d3 = fm.merge(post3, on="icustay_id", how="left")
d3 = d3[~((d3.d28 == 1) & (d3.days_to_death < 1.0)) & ~(d3.los_h < 24)].copy()
sofa3 = pd.read_parquet(DATA / "08h_sofa_mimic3.parquet")
s3k = "icustay_id" if "icustay_id" in sofa3.columns else sofa3.columns[0]
s3c = [c for c in sofa3.columns if "sofa" in c.lower() and c != s3k][0]
d3 = d3.merge(sofa3[[s3k, s3c]].rename(columns={s3k: "icustay_id", s3c: "sofa24"}), on="icustay_id", how="left")
gt3 = d3.gcs_total_first if "gcs_total_first" in d3.columns else None
gm3 = d3.gcs_motor_first if "gcs_motor_first" in d3.columns else None
rows["MIMIC-III (ext)"] = {
    "n": len(d3), "Age, yr": summ_cont(d3.admission_age),
    "Female, %": f"{100*(1-d3.male).mean():.1f}" if "male" in d3.columns else None,
    "Charlson": summ_cont(d3.charlson_comorbidity_score) if "charlson_comorbidity_score" in d3.columns else "—",
    "GCS total (first)": summ_cont(gt3) if gt3 is not None else "—",
    "GCS motor (first)": summ_cont(gm3) if gm3 is not None else "—",
    "SOFA (day 1)": summ_cont(d3.sofa24),
    "28-d deaths, n (%)": f"{int(d3.d28.sum())} ({100*d3.d28.mean():.1f})",
}

# --- 落盘 ---
T = pd.DataFrame(rows).T
T.index.name = "Cohort"
T.to_csv(OUT / "Table1.csv")
md = "**Table 1 | Baseline characteristics and outcomes by database (24-hour landmark risk sets).**\n\n"
md += "| " + " | ".join(["Characteristic"] + list(T.columns)) + " |\n"
md += "|" + "---|" * (len(T.columns) + 1) + "\n"
for feat in T.columns:
    md += f"| {feat} | " + " | ".join(str(T.loc[c, feat]) for c in T.index) + " |\n"
md += "\nContinuous variables: median [IQR]. GCS = Glasgow Coma Scale; SOFA = Sequential Organ Failure Assessment (first ICU day). MIMIC-IV is the development database (temporal split: training 2008–2017, model selection 2018–2019, internal validation 2020–2022); eICU-CRD and MIMIC-III are zero-touch external validation cohorts. eICU outcome = in-hospital death within 28 days (discharge-time approximation); other databases = 28-day all-cause mortality.\n"
(OUT / "Table1.md").write_text(md, encoding="utf-8")
print(T.to_string())
print("[saved]", OUT / "Table1.csv", "and Table1.md")
