# Phase1-①: 三库 gcs_long harmonization (SAP_prediction §2 / SAP_v2 §10)
# 输出: data/gcs_long_{mimic4,eicu,mimic3}.parquet + 合并表 + reports/07a_itemid_map.csv + 质检JSON
# 统一 schema: (id, source_db, system, offset_hr, gcs_total, gcs_motor, gcs_eye, gcs_verbal)
# 原则: itemid 动态查证(D_ITEMS/nurseCharting 实测), 禁凭记忆硬编码; 映射表=交付物
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M4 = ROOT/"data/mimic-iv-3.1"; M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
OUT.mkdir(parents=True, exist_ok=True); REP.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(); con.execute("PRAGMA threads=4")
COH4 = (ROOT/"results/cohort_audit/02_tbi_cohort.parquet").as_posix()  # MIMIC-IV n=2751 冻结队列
map_rows, qc = [], {}   # 映射表行 / 质检
ANCHOR = {"mimic_iv": 27, "eicu": 19, "mimic_iii": 22}   # 中位测量数锚点 (SAP_v2:11)

# ========== A1. MIMIC-IV (discovery, chartevents 220739/223900/223901, 03b/12e 已验证) ==========
print("[A1] MIMIC-IV GCS 72h 长表...")
m4 = con.execute(f"""
WITH ce AS (
  SELECT c.stay_id, c.charttime, co.intime,
    MAX(CASE WHEN c.itemid=220739 AND c.valuenum BETWEEN 1 AND 4 THEN c.valuenum END) AS eye,
    MAX(CASE WHEN c.itemid=223900 AND c.value='No Response-ETT' THEN 1
             WHEN c.itemid=223900 AND c.valuenum BETWEEN 1 AND 5 THEN c.valuenum END) AS verbal,
    MAX(CASE WHEN c.itemid=223901 AND c.valuenum BETWEEN 1 AND 6 THEN c.valuenum END) AS motor,
    MAX(CASE WHEN c.itemid=223900 AND c.value='No Response-ETT' THEN 1 ELSE 0 END) AS ett_verbal
  FROM read_csv_auto('{M4.as_posix()}/icu/chartevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) c
  JOIN read_parquet('{COH4}') co ON c.stay_id=co.stay_id
  WHERE c.itemid IN (220739,223900,223901)
    AND (c.valuenum IS NOT NULL OR (c.itemid=223900 AND c.value='No Response-ETT'))
    AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL 72 HOUR
  GROUP BY c.stay_id, c.charttime, co.intime)
SELECT CAST(stay_id AS VARCHAR) id, date_diff('minute',intime,charttime)/60.0 offset_hr,
       eye, verbal, motor, ett_verbal,
       CASE WHEN eye BETWEEN 1 AND 4 AND verbal BETWEEN 1 AND 5 AND motor BETWEEN 1 AND 6
            THEN eye+verbal+motor END gcs_total
FROM ce WHERE eye IS NOT NULL OR verbal IS NOT NULL OR motor IS NOT NULL
""").df()
m4.insert(1, "source_db", "mimic_iv"); m4.insert(2, "system", "metavision")
for iid, name in [(220739,"GCS eye"),(223900,"GCS verbal (ETT→1 rule)"),(223901,"GCS motor")]:
    map_rows.append(dict(source_db="mimic_iv", system="metavision", concept=name,
                         key_type="itemid", key=str(iid), verified_by="03b/12e 已跑通"))
tot = m4.dropna(subset=["gcs_total"])
qc["mimic_iv"] = dict(n_patients=int(tot.id.nunique()), n_rows=int(len(tot)),
    median_meas_72h=float(tot.groupby("id").size().median()),
    n_patients_motor=int(m4.dropna(subset=["motor"]).id.nunique()),
    ett_verbal_rows=int(m4.ett_verbal.sum()),
    ett_verbal_patients=int(m4[m4.ett_verbal==1].id.nunique()),
    gcs_total_domain=[float(tot.gcs_total.min()), float(tot.gcs_total.max())])
m4.drop(columns=["ett_verbal"]).to_parquet(OUT/"gcs_long_mimic4.parquet", index=False)
print(f"  {len(m4)} rows | total覆盖 {qc['mimic_iv']['n_patients']}/2751 | 中位测 {qc['mimic_iv']['median_meas_72h']:.0f}/72h (锚点27)")

# ========== A2. MIMIC-III (D_ITEMS 动态查证 CareVue/MetaVision 双系统) ==========
print("[A2] MIMIC-III GCS itemid 实测查证 (D_ITEMS)...")
d_items = con.execute(f"""
SELECT itemid AS itemid, label AS label, dbsource AS dbsource, category AS category
FROM read_csv_auto('{M3.as_posix()}/D_ITEMS.csv.gz')
WHERE (label ILIKE '%glasgow%' OR label ILIKE '%gcs%'
       OR label IN ('Eye Opening','Motor Response','Verbal Response'))  -- CareVue 分项实测无 GCS 前缀(184/454/723)
  AND dbsource IN ('carevue','metavision')
  AND (category IS NULL OR category NOT ILIKE '%alarm%')
ORDER BY dbsource, itemid""").df()
print(d_items.to_string(index=False))
def pick(dbs, label_kw):
    """D_ITEMS 按 label 精确/前缀匹配选 itemid; 无命中即 assert 失败(禁止凭记忆兜底)"""
    hits = d_items[(d_items.dbsource==dbs) & (d_items.label.str.contains(label_kw, case=False, na=False, regex=True))]
    assert len(hits)>=1, f"D_ITEMS 未命中: {dbs} / {label_kw}"
    row = hits.sort_values("itemid").iloc[0]
    map_rows.append(dict(source_db="mimic_iii", system=dbs, concept=label_kw,
                         key_type="itemid", key=str(int(row.itemid)),
                         verified_by=f"D_ITEMS 实测 label={row.label!r}"))
    return int(row.itemid)
CV_TOT = 198                                   # SAP_v2:64 锚点(CareVue GCS total)
CV_EYE = pick("carevue", r"^Eye Opening$")     # 实测: CareVue 分项无 GCS 前缀 (D_ITEMS 探针 2026-09-02)
CV_VER = pick("carevue", r"^Verbal Response$")
CV_MOT = pick("carevue", r"^Motor Response$")
map_rows.append(dict(source_db="mimic_iii", system="carevue", concept="GCS total",
                     key_type="itemid", key="198", verified_by="SAP_v2:64 锚点"))
MV_EYE, MV_VER, MV_MOT = 220739, 223900, 223901   # MIMIC-IV 同号(SAP_v2:64)
for iid, name in [(MV_EYE,"GCS eye"),(MV_VER,"GCS verbal"),(MV_MOT,"GCS motor")]:
    map_rows.append(dict(source_db="mimic_iii", system="metavision", concept=name,
                         key_type="itemid", key=str(iid), verified_by="MIMIC-IV 同号+SAP_v2:64"))

print("  建 MIMIC-III TBI 队列(ICD-9 850-854 [README冻结口径, 不含颅骨骨折], 成人+首ICU+LOS≥24h)...")
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
print(f"  MIMIC-III 队列 n={len(m3_coh)} (锚点1950=850-854 raw stays/1783患者, 已逐位闭环; 本行=854-only+成人+首ICU+LOS≥24h 分析口径)")
m3_coh[["icustay_id","intime"]].to_parquet(OUT/"_tmp_m3_coh.parquet", index=False)
TMP = (OUT/"_tmp_m3_coh.parquet").as_posix()

print("  提 MIMIC-III GCS 72h (CHARTEVENTS, strict_mode=false, ~4min)...")
m3 = con.execute(f"""
WITH raw AS (
  SELECT c.icustay_id, c.charttime, c.itemid, c.valuenum, c.value
  FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
       types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
  JOIN read_parquet('{TMP}') co ON c.icustay_id=co.icustay_id
  WHERE c.itemid IN (198,{CV_EYE},{CV_VER},{CV_MOT},{MV_EYE},{MV_VER},{MV_MOT})
    AND (c.valuenum IS NOT NULL OR (c.itemid IN (223900,{CV_VER}) AND c.value ILIKE '%ET%'))),
piv AS (
  SELECT icustay_id AS icustay_id, charttime AS charttime,
    MAX(CASE WHEN itemid IN (220739,{CV_EYE}) AND valuenum BETWEEN 1 AND 4 THEN valuenum END) eye,
    MAX(CASE WHEN itemid IN (223900,{CV_VER}) AND value ILIKE '%ET%' THEN 1
             WHEN itemid IN (223900,{CV_VER}) AND valuenum BETWEEN 1 AND 5 THEN valuenum END) verbal,
    MAX(CASE WHEN itemid IN (223901,{CV_MOT}) AND valuenum BETWEEN 1 AND 6 THEN valuenum END) motor,
    MAX(CASE WHEN itemid=198 AND valuenum BETWEEN 3 AND 15 THEN valuenum END) cv_total,
    MAX(CASE WHEN itemid IN (223900,{CV_VER}) AND value ILIKE '%ET%' THEN 1 ELSE 0 END) ett_verbal
  FROM raw GROUP BY icustay_id, charttime)
SELECT p.*, c.intime FROM piv p JOIN read_parquet('{TMP}') c ON p.icustay_id=c.icustay_id
""").df()
# 合成 gcs_total: CareVue total 优先, 缺则分项求和 (SAP_v2:64 规则)
m3["gcs_total"] = m3.cv_total.fillna(m3[["eye","verbal","motor"]].dropna().sum(axis=1)).where(
    lambda s: s.between(3,15))
m3["offset_hr"] = (pd.to_datetime(m3.charttime)-pd.to_datetime(m3.intime)).dt.total_seconds()/3600
m3 = m3[(m3.offset_hr>=0)&(m3.offset_hr<=72)]
stay_sys = m3.groupby("icustay_id").cv_total.apply(lambda s: "carevue" if s.notna().any() else "metavision")
m3_out = pd.DataFrame({"id": m3.icustay_id.astype(str), "source_db": "mimic_iii",
    "system": m3.icustay_id.map(stay_sys), "offset_hr": m3.offset_hr,
    "gcs_total": m3.gcs_total, "gcs_motor": m3.motor, "gcs_eye": m3.eye, "gcs_verbal": m3.verbal})
tot3 = m3_out.dropna(subset=["gcs_total"])
qc["mimic_iii"] = dict(n_patients=int(tot3.id.nunique()), n_rows=int(len(tot3)),
    median_meas_72h=float(tot3.groupby("id").size().median()),
    n_patients_motor=int(m3_out.dropna(subset=["gcs_motor"]).id.nunique()),
    ett_verbal_rows=int(m3.ett_verbal.sum()),
    ett_verbal_patients=int(m3[m3.ett_verbal==1].icustay_id.nunique()),
    carevue_stays=int((stay_sys=="carevue").sum()), metavision_stays=int((stay_sys=="metavision").sum()),
    gcs_total_domain=[float(tot3.gcs_total.min()), float(tot3.gcs_total.max())])
m3_out.to_parquet(OUT/"gcs_long_mimic3.parquet", index=False)
print(f"  {len(m3_out)} rows | total覆盖 {qc['mimic_iii']['n_patients']} 患者 | 中位测 {qc['mimic_iii']['median_meas_72h']:.0f}/72h (锚点22)")

# ========== A3. eICU (nurseCharting valname 分布实测) ==========
print("[A3] eICU GCS: 先实测 nursingchartcelltypevalname 分布...")
lab = con.execute(f"""
SELECT nursingchartcelltypevallabel, nursingchartcelltypevalname, count(*) n,
       count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{EI.as_posix()}/nurseCharting.csv.gz')
WHERE nursingchartcelltypevallabel ILIKE '%glasgow%' OR nursingchartcelltypevalname ILIKE '%gcs%'
GROUP BY 1,2 ORDER BY n DESC LIMIT 30""").df()
print(lab.to_string(index=False))
lab.to_csv(REP/"07a_eicu_gcs_cellnames_raw.csv", index=False)
def eicu_name(kw):
    h = lab[lab.nursingchartcelltypevalname.str.contains(kw, case=False, na=False)]
    assert len(h)>=1, f"eICU valname 未命中: {kw}"
    name = h.sort_values("n", ascending=False).iloc[0].nursingchartcelltypevalname
    map_rows.append(dict(source_db="eicu", system="eicu", concept=f"GCS {kw}",
                         key_type="nursingchartcelltypevalname", key=name, verified_by="nurseCharting 实测分布"))
    return name
E_TOT, E_MOT, E_EYE, E_VER = eicu_name("total"), eicu_name("motor"), eicu_name("eye"), eicu_name("verbal")

print("  建 eICU 队列(diagnosisstring TBI + 成人 + 首stay; 正式队列由 07b 出)...")
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
print(f"  eICU raw 队列 n={len(ei_coh)} (成人+首stay口径; 锚点6681=诊断命中口径, 07b 已闭环 6677)")
ei_coh[["patientunitstayid"]].to_parquet(OUT/"_tmp_ei_coh.parquet", index=False)
TMP2 = (OUT/"_tmp_ei_coh.parquet").as_posix()

print(f"  提 eICU GCS 72h (nurseCharting: total='{E_TOT}' motor='{E_MOT}')...")
ei = con.execute(f"""
SELECT patientunitstayid, nursingchartoffset/60.0 offset_hr, nursingchartcelltypevalname nm,
       TRY_CAST(nursingchartvalue AS DOUBLE) v
FROM read_csv_auto('{EI.as_posix()}/nurseCharting.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM read_parquet('{TMP2}'))
  AND nursingchartoffset BETWEEN 0 AND 4320
  AND nursingchartcelltypevalname IN ('{E_TOT}','{E_MOT}','{E_EYE}','{E_VER}')
  AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 0 AND 15""").df()
# 同一 (患者,offset) 多来源时取中位 → pivot 前聚合, 避免重复 index
ei_agg = ei.groupby(["patientunitstayid","offset_hr","nm"], as_index=False).v.median()
ei_agg["v"] = ei_agg["v"].round(0)   # 只圆整 GCS 值; round 全表会把分钟级 offset_hr 塌成小时 → pivot 重复索引
ei_piv = ei_agg.pivot(index=["patientunitstayid","offset_hr"], columns="nm", values="v").reset_index()
tot_c, mot_c, eye_c, ver_c = E_TOT, E_MOT, E_EYE, E_VER
for c in [tot_c, mot_c, eye_c, ver_c]:
    assert c in ei_piv.columns, f"pivot 缺列 {c}"
ei_out = pd.DataFrame({"id": ei_piv.patientunitstayid.astype(str), "source_db": "eicu", "system": "eicu",
    "offset_hr": ei_piv.offset_hr, "gcs_total": ei_piv[tot_c], "gcs_motor": ei_piv[mot_c],
    "gcs_eye": ei_piv[eye_c], "gcs_verbal": ei_piv[ver_c]})
n_ver0 = int((ei_out.gcs_verbal==0).sum())   # verbal=0: 插管无言语反应代理(语义未经源文档证实) — 量化供 SAP_v2:64 统一规则决策, 不擅自替 1
ei_out.loc[~ei_out.gcs_total.between(3,15), "gcs_total"] = None
for c, lo, hi in [("gcs_motor",1,6),("gcs_eye",1,4),("gcs_verbal",1,5)]:
    ei_out.loc[~ei_out[c].between(lo,hi), c] = None
totE = ei_out.dropna(subset=["gcs_total"])
qc["eicu"] = dict(n_patients=int(totE.id.nunique()), n_rows=int(len(totE)),
    median_meas_72h=float(totE.groupby("id").size().median()),
    n_patients_motor=int(ei_out.dropna(subset=["gcs_motor"]).id.nunique()),
    verbal_zero_rows=n_ver0,
    motor_row_share_pct=round(100*ei_out.gcs_motor.notna().mean(),1),
    gcs_total_domain=[float(totE.gcs_total.min()), float(totE.gcs_total.max())])
ei_out.to_parquet(OUT/"gcs_long_eicu.parquet", index=False)
print(f"  {len(ei_out)} rows | total覆盖 {qc['eicu']['n_patients']} 患者 中位 {qc['eicu']['median_meas_72h']:.0f}/72h (锚点19) | motor 行占比 {qc['eicu']['motor_row_share_pct']}%")

# ========== 合并 + 产物 ==========
SCHEMA = ["id","source_db","system","offset_hr","gcs_total","gcs_motor","gcs_eye","gcs_verbal"]
m4_out = m4.rename(columns={"motor":"gcs_motor","eye":"gcs_eye","verbal":"gcs_verbal"})[SCHEMA]
merged = pd.concat([m4_out, m3_out[SCHEMA], ei_out[SCHEMA]], ignore_index=True)
merged.to_parquet(OUT/"gcs_long_3db.parquet", index=False)
pd.DataFrame(map_rows).to_csv(REP/"07a_itemid_map.csv", index=False)
qc["anchors"] = dict(median_meas_expect="mimic_iv≈27 / eicu≈19 / mimic_iii≈22 (SAP_v2:11)",
                     mimic4_coverage_expect="2748/2751")
(REP/"07a_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== 07a 汇总 ===")
print(f"merged {len(merged)} rows | 患者数: " + ", ".join(
    f"{k}={v['n_patients']}" for k,v in qc.items() if isinstance(v,dict) and "n_patients" in v))
print(f"映射表 {len(map_rows)} 行 → reports/07a_itemid_map.csv (SAP 附录素材)")
for k in ["mimic_iv","eicu","mimic_iii"]:
    v = qc[k]
    flag = "OK" if abs(v["median_meas_72h"]-ANCHOR[k])<=max(4, 0.2*ANCHOR[k]) else "⚠️ 偏差>20%, 查原因不擅改"
    print(f"  {k:9} 中位测/72h={v['median_meas_72h']:.0f} (锚点{ANCHOR[k]}) [{flag}]")
print("DONE 07a")
