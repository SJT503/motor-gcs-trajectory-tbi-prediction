# -*- coding: utf-8 -*-
# ============================================================
# 09a — 三库 GCS 长表延展至 120h (rolling END_post 窗: t<=72 + 48h = 120h)
# 07a_gcs_long_harmonize.py 逐字复刻, 仅窗 72h→120h (机械延展, 见 09_rolling_spec.md §一)
# 产出: data/gcs_long120_{mimic4,eicu,mimic3}.parquet (schema 同 07a) + reports/09a_gcs120_qc.json
# ============================================================
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M4 = ROOT/"data/mimic-iv-3.1"; M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(); con.execute("PRAGMA threads=4")
COH4 = (ROOT/"results/cohort_audit/02_tbi_cohort.parquet").as_posix()  # M4 n=2751 冻结队列(07a 同源)
WIN_H = 120
qc = {}

# ========== A1. MIMIC-IV (07a 逐字, 72→120) ==========
print("[A1] MIMIC-IV GCS 120h 长表...")
m4 = con.execute(f"""
WITH ce AS (
  SELECT c.stay_id, c.charttime, co.intime,
    MAX(CASE WHEN c.itemid=220739 AND c.valuenum BETWEEN 1 AND 4 THEN c.valuenum END) AS eye,
    MAX(CASE WHEN c.itemid=223900 AND c.value='No Response-ETT' THEN 1
             WHEN c.itemid=223900 AND c.valuenum BETWEEN 1 AND 5 THEN c.valuenum END) AS verbal,
    MAX(CASE WHEN c.itemid=223901 AND c.valuenum BETWEEN 1 AND 6 THEN c.valuenum END) AS motor
  FROM read_csv_auto('{M4.as_posix()}/icu/chartevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) c
  JOIN read_parquet('{COH4}') co ON c.stay_id=co.stay_id
  WHERE c.itemid IN (220739,223900,223901)
    AND (c.valuenum IS NOT NULL OR (c.itemid=223900 AND c.value='No Response-ETT'))
    AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL {WIN_H} HOUR
  GROUP BY c.stay_id, c.charttime, co.intime)
SELECT CAST(stay_id AS VARCHAR) id, date_diff('minute',intime,charttime)/60.0 offset_hr,
       eye, verbal, motor,
       CASE WHEN eye BETWEEN 1 AND 4 AND verbal BETWEEN 1 AND 5 AND motor BETWEEN 1 AND 6
            THEN eye+verbal+motor END gcs_total
FROM ce WHERE eye IS NOT NULL OR verbal IS NOT NULL OR motor IS NOT NULL
""").df()
m4_out = pd.DataFrame({"id": m4.id, "source_db": "mimic_iv", "system": "metavision",
    "offset_hr": m4.offset_hr, "gcs_total": m4.gcs_total, "gcs_motor": m4.motor,
    "gcs_eye": m4.eye, "gcs_verbal": m4.verbal})
m4_out.to_parquet(OUT/"gcs_long120_mimic4.parquet", index=False)
t4 = m4_out.dropna(subset=["gcs_total"])
qc["mimic4"] = dict(n_pat=int(t4.id.nunique()), rows=int(len(m4_out)))
print(f"  {len(m4_out)} rows | total覆盖 {t4.id.nunique()}/2751")

# ========== A2. MIMIC-III (07a 逐字: D_ITEMS 动态查证 + 内联队列, 72→120) ==========
print("[A2] MIMIC-III GCS 120h (D_ITEMS 动态查证 CareVue itemid)...")
d_items = con.execute(f"""
SELECT itemid AS itemid, label AS label, dbsource AS dbsource, category AS category
FROM read_csv_auto('{M3.as_posix()}/D_ITEMS.csv.gz')
WHERE (label ILIKE '%glasgow%' OR label ILIKE '%gcs%'
       OR label IN ('Eye Opening','Motor Response','Verbal Response'))
  AND dbsource IN ('carevue','metavision')
  AND (category IS NULL OR category NOT ILIKE '%alarm%')
ORDER BY dbsource, itemid""").df()
def pick(dbs, label_kw):   # 07a pick() 逐字: 无命中即 assert 失败
    hits = d_items[(d_items.dbsource==dbs) & (d_items.label.str.contains(label_kw, case=False, na=False, regex=True))]
    assert len(hits)>=1, f"D_ITEMS 未命中: {dbs} / {label_kw}"
    return int(hits.sort_values("itemid").iloc[0].itemid)
CV_TOT = 198                                        # SAP_v2:64 锚点(CareVue GCS total)
CV_EYE = pick("carevue", r"^Eye Opening$")
CV_VER = pick("carevue", r"^Verbal Response$")
CV_MOT = pick("carevue", r"^Motor Response$")
MV_EYE, MV_VER, MV_MOT = 220739, 223900, 223901     # MIMIC-IV 同号(07a 常量)
print(f"  CareVue itemids: eye={CV_EYE} verbal={CV_VER} motor={CV_MOT} total={CV_TOT}")

# 07a 内联 M3 队列 (ICD-9 850-854, 成人+首ICU+LOS>=24h) — 逐字
m3_coh = con.execute(f"""
WITH tbi AS (SELECT DISTINCT hadm_id FROM read_csv_auto('{M3.as_posix()}/DIAGNOSES_ICD.csv.gz', types={{'ICD9_CODE':'VARCHAR'}})
             WHERE substr(ICD9_CODE,1,3) IN ('850','851','852','853','854')),
icu AS (SELECT i.*, ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) rn,
               date_diff('hour',i.intime,i.outtime) los_h
        FROM read_csv_auto('{M3.as_posix()}/ICUSTAYS.csv.gz') i JOIN tbi t ON i.hadm_id=t.hadm_id)
SELECT ic.icustay_id icustay_id, ic.subject_id subject_id, ic.hadm_id hadm_id,
       CAST(ic.intime AS TIMESTAMP) intime, p.dob dob
FROM icu ic JOIN read_csv_auto('{M3.as_posix()}/PATIENTS.csv.gz') p ON ic.subject_id=p.subject_id
WHERE ic.rn=1 AND ic.los_h>=24""").df()
m3_coh["age"] = ((pd.to_datetime(m3_coh.intime)-pd.to_datetime(m3_coh.dob)).dt.days/365.25).clip(upper=91)
m3_coh = m3_coh[m3_coh.age>18].copy()
print(f"  M3 内联队列 n={len(m3_coh)} (07a 同口径)")
m3_coh[["icustay_id","intime"]].to_parquet(OUT/"_tmp_m3_coh120.parquet", index=False)
TMP = (OUT/"_tmp_m3_coh120.parquet").as_posix()

m3 = con.execute(f"""
WITH raw AS (
  SELECT c.icustay_id, c.charttime, c.itemid, c.valuenum, c.value
  FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
       types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
  JOIN read_parquet('{TMP}') co ON c.icustay_id=co.icustay_id
  WHERE c.itemid IN ({CV_TOT},{CV_EYE},{CV_VER},{CV_MOT},{MV_EYE},{MV_VER},{MV_MOT})
    AND (c.valuenum IS NOT NULL OR (c.itemid IN ({MV_VER},{CV_VER}) AND c.value ILIKE '%ET%'))),
piv AS (
  SELECT icustay_id AS icustay_id, charttime AS charttime,
    MAX(CASE WHEN itemid IN (220739,{CV_EYE}) AND valuenum BETWEEN 1 AND 4 THEN valuenum END) eye,
    MAX(CASE WHEN itemid IN (223900,{CV_VER}) AND value ILIKE '%ET%' THEN 1
             WHEN itemid IN (223900,{CV_VER}) AND valuenum BETWEEN 1 AND 5 THEN valuenum END) verbal,
    MAX(CASE WHEN itemid IN (223901,{CV_MOT}) AND valuenum BETWEEN 1 AND 6 THEN valuenum END) motor,
    MAX(CASE WHEN itemid={CV_TOT} AND valuenum BETWEEN 3 AND 15 THEN valuenum END) cv_total
  FROM raw GROUP BY icustay_id, charttime)
SELECT p.*, c.intime FROM piv p JOIN read_parquet('{TMP}') c ON p.icustay_id=c.icustay_id
""").df()
m3["gcs_total"] = m3.cv_total.fillna(m3[["eye","verbal","motor"]].dropna().sum(axis=1)).where(
    lambda s: s.between(3,15))
m3["offset_hr"] = (pd.to_datetime(m3.charttime)-pd.to_datetime(m3.intime)).dt.total_seconds()/3600
m3 = m3[(m3.offset_hr>=0)&(m3.offset_hr<=WIN_H)]
stay_sys = m3.groupby("icustay_id").cv_total.apply(lambda s: "carevue" if s.notna().any() else "metavision")
m3_out = pd.DataFrame({"id": m3.icustay_id.astype(str), "source_db": "mimic_iii",
    "system": m3.icustay_id.map(stay_sys), "offset_hr": m3.offset_hr,
    "gcs_total": m3.gcs_total, "gcs_motor": m3.motor, "gcs_eye": m3.eye, "gcs_verbal": m3.verbal})
m3_out.to_parquet(OUT/"gcs_long120_mimic3.parquet", index=False)
t3 = m3_out.dropna(subset=["gcs_total"])
qc["mimic3"] = dict(n_pat=int(t3.id.nunique()), rows=int(len(m3_out)),
    carevue_stays=int((stay_sys=="carevue").sum()), metavision_stays=int((stay_sys=="metavision").sum()))
print(f"  {len(m3_out)} rows | total覆盖 {t3.id.nunique()}")

# ========== A3. eICU (07a 逐字: 全库 discovery + 内联队列, 4320→7200) ==========
print("[A3] eICU GCS 120h (nurseCharting valname 全库 discovery)...")
lab = con.execute(f"""
SELECT nursingchartcelltypevallabel, nursingchartcelltypevalname, count(*) n
FROM read_csv_auto('{EI.as_posix()}/nurseCharting.csv.gz')
WHERE nursingchartcelltypevallabel ILIKE '%glasgow%' OR nursingchartcelltypevalname ILIKE '%gcs%'
GROUP BY 1,2 ORDER BY n DESC LIMIT 30""").df()
def eicu_name(kw):   # 07a eicu_name() 逐字
    h = lab[lab.nursingchartcelltypevalname.str.contains(kw, case=False, na=False)]
    assert len(h)>=1, f"eICU valname 未命中: {kw}"
    return h.sort_values("n", ascending=False).iloc[0].nursingchartcelltypevalname
E_TOT, E_MOT, E_EYE, E_VER = eicu_name("total"), eicu_name("motor"), eicu_name("eye"), eicu_name("verbal")
print(f"  valnames: total='{E_TOT}' motor='{E_MOT}'")

# 07a 内联 eICU 队列 — 逐字
ei_coh = con.execute(f"""
WITH tbi AS (
  SELECT DISTINCT patientunitstayid FROM read_csv_auto('{EI.as_posix()}/diagnosis.csv.gz')
  WHERE lower(diagnosisstring) LIKE '%traumatic brain%' OR lower(diagnosisstring) LIKE '%head injury%'
   OR lower(diagnosisstring) LIKE '%intracranial%' OR lower(diagnosisstring) LIKE '%subdural%'
   OR lower(diagnosisstring) LIKE '%epidural%' OR lower(diagnosisstring) LIKE '%cerebral contusion%'
   OR lower(diagnosisstring) LIKE '%concussion%'
   OR regexp_matches(COALESCE(icd9code,''), '^(85[0-4])')),
p AS (SELECT patientunitstayid, patienthealthsystemstayid, age FROM read_csv_auto('{EI.as_posix()}/patient.csv.gz'))
SELECT t.patientunitstayid, p.patienthealthsystemstayid, p.age FROM tbi t JOIN p ON t.patientunitstayid=p.patientunitstayid""").df()
ei_coh["age_num"] = pd.to_numeric(ei_coh.age.replace({"> 89":"90"}), errors="coerce")
ei_coh = ei_coh[ei_coh.age_num>18].sort_values("patientunitstayid").drop_duplicates("patienthealthsystemstayid", keep="first")
print(f"  eICU 内联队列 n={len(ei_coh)} (07a 同口径)")
ei_coh[["patientunitstayid"]].to_parquet(OUT/"_tmp_ei_coh120.parquet", index=False)
TMP2 = (OUT/"_tmp_ei_coh120.parquet").as_posix()

ei = con.execute(f"""
SELECT patientunitstayid, nursingchartoffset/60.0 offset_hr, nursingchartcelltypevalname nm,
       TRY_CAST(nursingchartvalue AS DOUBLE) v
FROM read_csv_auto('{EI.as_posix()}/nurseCharting.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM read_parquet('{TMP2}'))
  AND nursingchartoffset BETWEEN 0 AND {WIN_H*60}
  AND nursingchartcelltypevalname IN ('{E_TOT}','{E_MOT}','{E_EYE}','{E_VER}')
  AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 0 AND 15""").df()
ei_agg = ei.groupby(["patientunitstayid","offset_hr","nm"], as_index=False).v.median()
ei_agg["v"] = ei_agg["v"].round(0)   # 只圆整 GCS 值 (07a 同)
ei_piv = ei_agg.pivot(index=["patientunitstayid","offset_hr"], columns="nm", values="v").reset_index()
for c in [E_TOT, E_MOT, E_EYE, E_VER]:
    assert c in ei_piv.columns, f"pivot 缺列 {c}"
ei_out = pd.DataFrame({"id": ei_piv.patientunitstayid.astype(str), "source_db": "eicu", "system": "eicu",
    "offset_hr": ei_piv.offset_hr, "gcs_total": ei_piv[E_TOT], "gcs_motor": ei_piv[E_MOT],
    "gcs_eye": ei_piv[E_EYE], "gcs_verbal": ei_piv[E_VER]})
ei_out.loc[~ei_out.gcs_total.between(3,15), "gcs_total"] = None
for c, lo, hi in [("gcs_motor",1,6),("gcs_eye",1,4),("gcs_verbal",1,5)]:
    ei_out.loc[~ei_out[c].between(lo,hi), c] = None
ei_out.to_parquet(OUT/"gcs_long120_eicu.parquet", index=False)
te = ei_out.dropna(subset=["gcs_total"])
qc["eicu"] = dict(n_pat=int(te.id.nunique()), rows=int(len(ei_out)))

# 增量核查: 120h 窗相对 72h 冻结表新增的首测>72h患者 (延展是否带来增量信息)
for tag, old in [("mimic4","gcs_long_mimic4.parquet"),("mimic3","gcs_long_mimic3.parquet"),("eicu","gcs_long_eicu.parquet")]:
    try:
        old_ids = set(pd.read_parquet(OUT/old).dropna(subset=["gcs_total"]).id.astype(str))
        new_ids = set(pd.read_parquet(OUT/{"mimic4":"gcs_long120_mimic4.parquet","mimic3":"gcs_long120_mimic3.parquet","eicu":"gcs_long120_eicu.parquet"}[tag]).dropna(subset=["gcs_total"]).id.astype(str))
        qc[tag]["added_patients_vs_72h"] = len(new_ids - old_ids)
    except Exception as e:
        qc[tag]["added_patients_vs_72h"] = f"对照失败: {e}"
(REP/"09a_gcs120_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n=== 09a 汇总 === {json.dumps(qc, ensure_ascii=False)}")
print("DONE 09a")
