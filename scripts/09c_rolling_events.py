# -*- coding: utf-8 -*-
# ============================================================
# 09c — Block A 事件流缓存 (三库, 滚动特征的一次性慢提取)
# 规则逐字复刻: M4=03b (itemid+域+labs -6h前锚) / eICU=07c (periodic∪aperiodic∪nurseCharting双源温度
# +labname概念映射重发现) / M3=07d (07d_itemid_map.csv 冻结映射+域)
# 窗口延展: vitals/GCS [0,72h]; labs M4/M3 [-6h,72h] eICU [0,72h] (各库与其24h冻结口径同构)
# 产物: data/09c_events_{mimic4,eicu,mimic3}.parquet (id,offset_hr,concept,value)
#       + data/09c_gcs_triples_mimic4.parquet (id,offset_hr,eye,motor,verbal — 03b fd_gcs 口径)
#       + reports/09c_qc.json
# 说明: eICU/M3 的 GCS 不在此提 (09d 直接读 gcs_long120_*, 与 07c/07d 同源)
# 注: value 为 DuckDB 保留字 → 内部列名用 val, 仅最终 COPY 处 AS "value"
# ============================================================
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M4 = ROOT/"data/mimic-iv-3.1"; M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
COH4 = (ROOT/"results/cohort_audit/02_tbi_cohort.parquet").as_posix()
qc = {}

def concept_coverage(tbl_rel):
    """产物级 QC: 每 concept 覆盖患者数"""
    d = con.execute(f"SELECT concept, count(*) n_rows, count(DISTINCT id) n_pat FROM {tbl_rel} GROUP BY 1 ORDER BY 1").df()
    return {r.concept: dict(n_rows=int(r.n_rows), n_pat=int(r.n_pat)) for r in d.itertuples()}

# ========== M4: chartevents vitals + GCS 三分项 + labevents ==========
print("[M4] chartevents vitals+GCS 流 [0,72h] (~5min)...")
W = 72
con.execute(f"""
CREATE TABLE ev4_vital AS
SELECT CAST(c.stay_id AS VARCHAR) id,
       date_diff('minute', CAST(co.intime AS TIMESTAMP), CAST(c.charttime AS TIMESTAMP))/60.0 offset_hr,
       CASE c.itemid
         WHEN 220045 THEN 'hr' WHEN 223761 THEN 'temp' WHEN 223762 THEN 'temp' WHEN 220277 THEN 'spo2'
         WHEN 220179 THEN 'sbp' WHEN 220050 THEN 'sbp' WHEN 225309 THEN 'sbp'
         WHEN 220180 THEN 'dbp' WHEN 220051 THEN 'dbp' WHEN 225310 THEN 'dbp'
         WHEN 220052 THEN 'mbp' WHEN 220181 THEN 'mbp' WHEN 225312 THEN 'mbp'
         WHEN 220210 THEN 'rr'  WHEN 224690 THEN 'rr' END concept,
       CASE WHEN c.itemid=220045 AND c.valuenum>0 AND c.valuenum<300 THEN c.valuenum
            WHEN c.itemid IN(220179,220050,225309) AND c.valuenum>0 AND c.valuenum<400 THEN c.valuenum
            WHEN c.itemid IN(220180,220051,225310) AND c.valuenum>0 AND c.valuenum<300 THEN c.valuenum
            WHEN c.itemid IN(220052,220181,225312) AND c.valuenum>0 AND c.valuenum<300 THEN c.valuenum
            WHEN c.itemid IN(220210,224690) AND c.valuenum>0 AND c.valuenum<70 THEN c.valuenum
            WHEN c.itemid=223761 AND c.valuenum>70 AND c.valuenum<120 THEN (c.valuenum-32)/1.8
            WHEN c.itemid=223762 AND c.valuenum>10 AND c.valuenum<50 THEN c.valuenum
            WHEN c.itemid=220277 AND c.valuenum>0 AND c.valuenum<=100 THEN c.valuenum END val
FROM read_csv_auto('{M4.as_posix()}/icu/chartevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) c
JOIN read_parquet('{COH4}') co ON c.stay_id=co.stay_id
WHERE c.itemid IN (220045,220179,220050,225309,220180,220051,225310,220052,220181,225312,
                   220210,224690,223761,223762,220277)
  AND CAST(c.charttime AS TIMESTAMP) BETWEEN CAST(co.intime AS TIMESTAMP)
      AND CAST(co.intime AS TIMESTAMP) + INTERVAL {W} HOUR
""")
n1 = con.execute("SELECT count(*) FROM ev4_vital WHERE val IS NOT NULL").fetchone()[0]
print(f"  vitals 流 {n1} rows")

print("[M4] GCS 三分项 (03b fd_gcs 口径逐字, 72h)...")
con.execute(f"""
CREATE TABLE gcs4 AS
WITH g AS (
  SELECT c.stay_id, c.charttime, co.intime,
    max(CASE WHEN c.itemid=220739 THEN c.valuenum END) eye,
    max(CASE WHEN c.itemid=223901 THEN c.valuenum END) motor,
    max(CASE WHEN c.itemid=223900 AND c.value='No Response-ETT' THEN 1
             WHEN c.itemid=223900 THEN c.valuenum END) verbal
  FROM read_csv_auto('{M4.as_posix()}/icu/chartevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) c
  JOIN read_parquet('{COH4}') co ON c.stay_id=co.stay_id
  WHERE c.itemid IN (223900,223901,220739)
    AND CAST(c.charttime AS TIMESTAMP) BETWEEN CAST(co.intime AS TIMESTAMP)
        AND CAST(co.intime AS TIMESTAMP) + INTERVAL {W} HOUR
  GROUP BY c.stay_id, c.charttime, co.intime)
SELECT CAST(stay_id AS VARCHAR) id,
       date_diff('minute', CAST(intime AS TIMESTAMP), CAST(charttime AS TIMESTAMP))/60.0 offset_hr,
       eye, motor, verbal
FROM g WHERE eye IS NOT NULL OR motor IS NOT NULL OR verbal IS NOT NULL
""")
n1g = con.execute("SELECT count(*) FROM gcs4").fetchone()[0]
print(f"  GCS triples {n1g} rows")

print("[M4] labevents 流 [-6h,72h] (03b 域逐字, 白名单16项)...")
con.execute(f"""
CREATE TABLE ev4_lab AS
SELECT CAST(co.stay_id AS VARCHAR) id,
       date_diff('minute', CAST(co.intime AS TIMESTAMP), CAST(l.charttime AS TIMESTAMP))/60.0 offset_hr,
       CASE l.itemid WHEN 50868 THEN 'aniongap' WHEN 50882 THEN 'bicarbonate' WHEN 51006 THEN 'bun'
         WHEN 50893 THEN 'calcium' WHEN 50902 THEN 'chloride' WHEN 50912 THEN 'creatinine'
         WHEN 50931 THEN 'glucose_lab' WHEN 50983 THEN 'sodium' WHEN 50971 THEN 'potassium'
         WHEN 51221 THEN 'hematocrit' WHEN 51222 THEN 'hemoglobin' WHEN 51265 THEN 'platelet'
         WHEN 51301 THEN 'wbc' WHEN 51237 THEN 'inr' WHEN 51274 THEN 'pt' WHEN 51275 THEN 'ptt' END concept,
       CASE WHEN l.itemid=50868 AND l.valuenum>0 THEN l.valuenum
            WHEN l.itemid=50882 AND l.valuenum>0 AND l.valuenum<=10000 THEN l.valuenum
            WHEN l.itemid=51006 AND l.valuenum>0 AND l.valuenum<=300 THEN l.valuenum
            WHEN l.itemid=50893 AND l.valuenum>0 AND l.valuenum<=10000 THEN l.valuenum
            WHEN l.itemid=50902 AND l.valuenum>0 AND l.valuenum<=10000 THEN l.valuenum
            WHEN l.itemid=50912 AND l.valuenum>0 AND l.valuenum<=150 THEN l.valuenum
            WHEN l.itemid=50931 AND l.valuenum>0 AND l.valuenum<=10000 THEN l.valuenum
            WHEN l.itemid=50983 AND l.valuenum>0 AND l.valuenum<=200 THEN l.valuenum
            WHEN l.itemid=50971 AND l.valuenum>0 AND l.valuenum<=30 THEN l.valuenum
            WHEN l.itemid IN (51221,51222,51265,51301,51237,51274,51275) AND l.valuenum>0 THEN l.valuenum END val
FROM read_csv_auto('{M4.as_posix()}/hosp/labevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) l
JOIN read_parquet('{COH4}') co ON l.subject_id=co.subject_id
WHERE l.itemid IN (50868,50882,51006,50893,50902,50912,50931,50983,50971,51221,51222,51265,51301,51237,51274,51275)
  AND CAST(l.charttime AS TIMESTAMP) BETWEEN CAST(co.intime AS TIMESTAMP) - INTERVAL 6 HOUR
      AND CAST(co.intime AS TIMESTAMP) + INTERVAL {W} HOUR
""")
n2 = con.execute("SELECT count(*) FROM ev4_lab WHERE val IS NOT NULL").fetchone()[0]
EV4 = "(SELECT id, offset_hr, concept, val FROM ev4_vital WHERE val IS NOT NULL UNION ALL SELECT id, offset_hr, concept, val FROM ev4_lab WHERE val IS NOT NULL)"
con.execute(f"COPY (SELECT id, offset_hr, concept, val AS \"value\" FROM {EV4}) "
            f"TO '{(OUT/'09c_events_mimic4.parquet').as_posix()}' (FORMAT PARQUET)")
con.execute(f"COPY gcs4 TO '{(OUT/'09c_gcs_triples_mimic4.parquet').as_posix()}' (FORMAT PARQUET)")
qc["mimic4"] = dict(vital_rows=n1, lab_rows=n2, gcs_rows=n1g, concepts=concept_coverage(EV4))
print(f"  labs 流 {n2} rows → 09c_events_mimic4.parquet")
con.execute("DROP TABLE ev4_vital"); con.execute("DROP TABLE ev4_lab"); con.execute("DROP TABLE gcs4")

# ========== eICU: periodic ∪ aperiodic ∪ nurseCharting-temp + labs ==========
print("[eICU] vitalPeriodic [0,72h] (07c 域逐字)...")
coh_ei = pd.read_parquet(OUT/"cohort_eicu.parquet")
coh_ei[["icustay_id_eicu"]].rename(columns={"icustay_id_eicu":"pid"}).to_parquet(OUT/"_tmp_ei_ids.parquet", index=False)
IDS = (OUT/"_tmp_ei_ids.parquet").as_posix()
VP = (EI/"vitalPeriodic.csv.gz").as_posix(); VA = (EI/"vitalAperiodic.csv.gz").as_posix()
NC = (EI/"nurseCharting.csv.gz").as_posix()
con.execute(f"""
CREATE TABLE ei_vit AS SELECT pid, offset_hr,
    CASE concept WHEN 'tcel' THEN 'temp' ELSE concept END concept, val FROM (
  SELECT patientunitstayid pid, observationoffset/60.0 offset_hr,
    CASE WHEN heartrate BETWEEN 20 AND 300 THEN heartrate END hr,
    CASE WHEN respiration BETWEEN 4 AND 70 THEN respiration END rr,
    CASE WHEN sao2 BETWEEN 50 AND 100 THEN sao2 END spo2,
    CASE WHEN temperature>70 AND temperature<120 THEN (temperature-32)/1.8
         WHEN temperature>30 AND temperature<45 THEN temperature END tcel,
    CASE WHEN systemicsystolic BETWEEN 30 AND 400 THEN systemicsystolic END sbp,
    CASE WHEN systemicdiastolic BETWEEN 10 AND 300 THEN systemicdiastolic END dbp,
    CASE WHEN systemicmean BETWEEN 20 AND 300 THEN systemicmean END mbp
  FROM read_csv_auto('{VP}')
  WHERE patientunitstayid IN (SELECT pid FROM read_parquet('{IDS}'))
    AND observationoffset BETWEEN 0 AND {W*60})
UNPIVOT (val FOR concept IN (hr, rr, spo2, tcel, sbp, dbp, mbp))
""")
con.execute(f"""
INSERT INTO ei_vit SELECT * FROM (
  SELECT patientunitstayid pid, observationoffset/60.0 offset_hr,
    CASE WHEN noninvasivesystolic BETWEEN 30 AND 400 THEN noninvasivesystolic END sbp,
    CASE WHEN noninvasivediastolic BETWEEN 10 AND 300 THEN noninvasivediastolic END dbp,
    CASE WHEN noninvasivemean BETWEEN 20 AND 300 THEN noninvasivemean END mbp
  FROM read_csv_auto('{VA}')
  WHERE patientunitstayid IN (SELECT pid FROM read_parquet('{IDS}'))
    AND observationoffset BETWEEN 0 AND {W*60})
UNPIVOT (val FOR concept IN (sbp, dbp, mbp))
""")
con.execute(f"""
INSERT INTO ei_vit
SELECT patientunitstayid pid, nursingchartentryoffset/60.0 offset_hr, 'temp' concept,
  CASE WHEN nursingchartcelltypevalname ILIKE '%(F)%' AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 70 AND 120
            THEN (TRY_CAST(nursingchartvalue AS DOUBLE)-32)/1.8
       WHEN nursingchartcelltypevalname ILIKE '%(C)%' AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 30 AND 45
            THEN TRY_CAST(nursingchartvalue AS DOUBLE)
       WHEN TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 70 AND 120
            THEN (TRY_CAST(nursingchartvalue AS DOUBLE)-32)/1.8
       WHEN TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 30 AND 45
            THEN TRY_CAST(nursingchartvalue AS DOUBLE) END val
FROM read_csv_auto('{NC}')
WHERE patientunitstayid IN (SELECT pid FROM read_parquet('{IDS}'))
  AND nursingchartentryoffset BETWEEN 0 AND {W*60}
  AND nursingchartcelltypevalname ILIKE '%temperature%'
""")
ne = con.execute("SELECT count(*) FROM ei_vit WHERE val IS NOT NULL").fetchone()[0]
print(f"  vitals+temp 双源流 {ne} rows")

print("[eICU] labs: labname 重发现 (07c CONCEPTS/排除正则逐字, 窗 0-72h)...")
LAB = (EI/"lab.csv.gz").as_posix()
lab_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{LAB}')").df().column_name)
off_col = "labresultoffset" if "labresultoffset" in lab_cols else (
          "laboffset" if "laboffset" in lab_cols else None)
assert off_col, f"eICU lab 表无时间列, 实测列名: {sorted(lab_cols)}"
names = con.execute(f"""
SELECT labname, count(*) n FROM read_csv_auto('{LAB}')
WHERE patientunitstayid IN (SELECT pid FROM read_parquet('{IDS}'))
  AND {off_col} BETWEEN 0 AND {W*60} AND labresult IS NOT NULL
GROUP BY labname ORDER BY n DESC LIMIT 500""").df()
CONCEPTS = {
 "aniongap": r"anion\s*gap", "bicarbonate": r"bicarb|HCO3", "bun": r"\bBUN\b",
 "calcium": r"calcium", "chloride": r"chloride", "creatinine": r"creatinine",
 "glucose_lab": r"glucose", "sodium": r"\bsodium\b|\bNa\+?\b", "potassium": r"\bpotassium\b|\bK\+?\b",
 "hematocrit": r"hematocrit|\bHCT\b", "hemoglobin": r"hemoglobin|\bHgb?\b", "platelet": r"platelet",
 "wbc": r"\bWBC\b", "inr": r"\bINR\b|international.{0,12}ratio", "pt": r"\bPT\b|prothrombin",
 "ptt": r"\bPTT\b|partial.{0,12}thromboplastin"}
_EXCL_BASE = "ionized|urine|csf|cerebrospinal|body fluid|dialysate|peritoneal|pleural|ascites|synovial"
_EXTRA = {"pt": "inr", "bun": "ratio", "creatinine": "ratio"}
case_sql, labname_sets = [], {}
for concept, pat in CONCEPTS.items():
    excl = _EXCL_BASE + ("|"+_EXTRA[concept] if concept in _EXTRA else "")
    hit = names[names.labname.str.contains(rf"^(?!.*(?:{excl})).*(?:{pat})", case=False, regex=True, na=False)]
    if len(hit)==0: print(f"  [warn] {concept} 无 labname 命中"); continue
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in hit.labname)
    labname_sets[concept] = sorted(hit.labname.tolist())
    case_sql.append(f"WHEN labname IN ({quoted}) THEN '{concept}'")
con.execute(f"""
CREATE TABLE ei_lab AS
SELECT CAST(patientunitstayid AS VARCHAR) id, {off_col}/60.0 offset_hr,
       CASE {' '.join(case_sql)} END concept, labresult val
FROM read_csv_auto('{LAB}')
WHERE patientunitstayid IN (SELECT pid FROM read_parquet('{IDS}'))
  AND {off_col} BETWEEN 0 AND {W*60} AND labresult IS NOT NULL
""")
# 与 07c 冻结映射对照 (top-3 展示名都应在 72h 发现集内)
diff = {}
try:
    m07c = pd.read_csv(REP/"07c_labname_map.csv")
    for r in m07c.itertuples():
        for nm in str(r.labnames).split(" | "):
            c = r.concept
            if c in labname_sets and nm not in labname_sets[c]:
                diff[c] = diff.get(c, []) + [nm]
except Exception as e:
    diff = {"对照失败": [str(e)]}
qc["eicu_labname_vs_07c"] = diff
n2e = con.execute("SELECT count(*) FROM ei_lab").fetchone()[0]
EI_EV = "(SELECT CAST(pid AS VARCHAR) id, offset_hr, concept, val FROM ei_vit WHERE val IS NOT NULL UNION ALL SELECT id, offset_hr, concept, val FROM ei_lab)"
con.execute(f"COPY (SELECT id, offset_hr, concept, val AS \"value\" FROM {EI_EV}) "
            f"TO '{(OUT/'09c_events_eicu.parquet').as_posix()}' (FORMAT PARQUET)")
qc["eicu"] = dict(vital_rows=int(ne), lab_rows=int(n2e), concepts=concept_coverage(EI_EV))
print(f"  labs 流 {n2e} rows (概念 {len(labname_sets)}/16) → 09c_events_eicu.parquet | labname-vs-07c diff: {diff or '无'}")
con.execute("DROP TABLE ei_vit"); con.execute("DROP TABLE ei_lab")

# ========== M3: 07d_itemid_map.csv 冻结映射 + 域 ==========
print("[M3] 冻结 itemid 映射读取 (07d_itemid_map.csv)...")
imap = pd.read_csv(REP/"07d_itemid_map.csv")
def ids_of(block, concept):
    r = imap[(imap.block==block) & (imap.concept==concept)]
    assert len(r)==1, f"映射表缺 {block}/{concept}"
    return r.iloc[0].itemids
def I(s): return "(" + s + ")"
vcase = []
for key, dom in [("hr","BETWEEN 20 AND 300"),("sbp","BETWEEN 30 AND 400"),("dbp","BETWEEN 10 AND 300"),
                 ("map","BETWEEN 20 AND 300"),("rr","BETWEEN 4 AND 70"),("spo2","BETWEEN 50 AND 100")]:
    vcase.append(f"WHEN c.itemid IN {I(ids_of('chart', key))} AND c.valuenum {dom} "
                 f"THEN '{'mbp' if key=='map' else key}'")   # 映射表 concept=map, 特征列名=mbp (07d line82/96 同构)
vcase.append(f"WHEN c.itemid IN {I(ids_of('chart','temp'))} THEN 'temp'")   # concept 不设值域; F/C 单位由 val CASE 自判 (07d 同构)
con.execute(f"""
CREATE TABLE m3_vit AS
WITH ce AS (
  SELECT CAST(c.icustay_id AS VARCHAR) id,
         date_diff('minute', CAST(co.intime AS TIMESTAMP), CAST(c.charttime AS TIMESTAMP))/60.0 offset_hr,
         CASE {' '.join(vcase)} END concept,
         CASE WHEN c.itemid IN {I(ids_of('chart','hr'))} AND c.valuenum BETWEEN 20 AND 300 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','sbp'))} AND c.valuenum BETWEEN 30 AND 400 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','dbp'))} AND c.valuenum BETWEEN 10 AND 300 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','map'))} AND c.valuenum BETWEEN 20 AND 300 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','rr'))} AND c.valuenum BETWEEN 4 AND 70 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','temp'))} AND c.valuenum>70 AND c.valuenum<120 THEN (c.valuenum-32)/1.8
              WHEN c.itemid IN {I(ids_of('chart','temp'))} AND c.valuenum>30 AND c.valuenum<45 THEN c.valuenum
              WHEN c.itemid IN {I(ids_of('chart','spo2'))} AND c.valuenum BETWEEN 50 AND 100 THEN c.valuenum END val
  FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
       types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
  JOIN read_parquet('{(OUT/'cohort_mimic3.parquet').as_posix()}') co ON c.icustay_id=co.icustay_id
  WHERE c.valuenum IS NOT NULL
    AND CAST(c.charttime AS TIMESTAMP) BETWEEN CAST(co.intime AS TIMESTAMP)
        AND CAST(co.intime AS TIMESTAMP) + INTERVAL {W} HOUR)
SELECT * FROM ce WHERE concept IS NOT NULL
""")
n3v = con.execute("SELECT count(*) FROM m3_vit WHERE val IS NOT NULL").fetchone()[0]
print(f"  vitals 流 {n3v} rows")
print("[M3] labs [-6h,72h] (LABEVENTS subject join)...")
lcase, ldom = [], []
for concept in CONCEPTS:   # 与 eICU 同 16 概念名 (07d LAB_CONCEPTS 同集)
    iids = ids_of("lab", concept)
    lcase.append(f"WHEN l.itemid IN {I(iids)} THEN '{concept}'")
    ldom.append(f"WHEN l.itemid IN {I(iids)} THEN l.valuenum")
con.execute(f"""
CREATE TABLE m3_lab AS
SELECT CAST(co.icustay_id AS VARCHAR) id,
       date_diff('minute', CAST(co.intime AS TIMESTAMP), CAST(l.charttime AS TIMESTAMP))/60.0 offset_hr,
       CASE {' '.join(lcase)} END concept,
       CASE {' '.join(ldom)} END val
FROM read_csv_auto('{M3.as_posix()}/LABEVENTS.csv.gz', types={{'valuenum':'DOUBLE'}}) l
JOIN read_parquet('{(OUT/'cohort_mimic3.parquet').as_posix()}') co ON l.subject_id=co.subject_id
WHERE l.valuenum IS NOT NULL
  AND CAST(l.charttime AS TIMESTAMP) BETWEEN CAST(co.intime AS TIMESTAMP) - INTERVAL 6 HOUR
      AND CAST(co.intime AS TIMESTAMP) + INTERVAL {W} HOUR
""")
n3l = con.execute("SELECT count(*) FROM m3_lab WHERE val IS NOT NULL").fetchone()[0]
M3_EV = "(SELECT id, offset_hr, concept, val FROM m3_vit WHERE val IS NOT NULL UNION ALL SELECT id, offset_hr, concept, val FROM m3_lab WHERE val IS NOT NULL)"
con.execute(f"COPY (SELECT id, offset_hr, concept, val AS \"value\" FROM {M3_EV}) "
            f"TO '{(OUT/'09c_events_mimic3.parquet').as_posix()}' (FORMAT PARQUET)")
qc["mimic3"] = dict(vital_rows=int(n3v), lab_rows=int(n3l), concepts=concept_coverage(M3_EV))
print(f"  labs 流 {n3l} rows → 09c_events_mimic3.parquet")

(REP/"09c_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n=== 09c 汇总 === mimic4/eicu/mimic3 流行数: "
      f"{qc['mimic4']['vital_rows']+qc['mimic4']['lab_rows']}/{qc['eicu']['vital_rows']+qc['eicu']['lab_rows']}/"
      f"{qc['mimic3']['vital_rows']+qc['mimic3']['lab_rows']}")
print("DONE 09c")
