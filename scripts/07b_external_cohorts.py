# Phase1-②: eICU + MIMIC-III TBI 队列复算 (SAP_prediction §2 外验队列, 对照 SAP_v2:11 锚点)
# 口径: TBI识别 + 成人>18 + 首次ICU + ICU LOS≥24h (三库统一排除; ICD/文本识别跨库不完全等价→Methods声明)
# 输出: data/cohort_eicu.parquet, data/cohort_mimic3.parquet + reports/07b_cohort_qc.json
# 注意: eICU 无院外死亡随访 → 28d 结局操作化为院内死亡(双列输出, Methods 声明, 不改 SAP §4)
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
OUT.mkdir(parents=True, exist_ok=True); REP.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
qc = {}

# ========== eICU ==========
print("[eICU] TBI 队列复算...")
pat_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{EI.as_posix()}/patient.csv.gz')").df().column_name)
need_cols = {"hospitaldischargestatus","hospitaldischargeoffset","unitdischargeoffset"}
miss = need_cols - pat_cols
assert not miss, f"eICU patient 表缺死亡/出院随访列 {miss} — 实测列: {sorted(pat_cols)}"
print(f"  [核验] patient 表含 hospitaldischargestatus/hospitaldischargeoffset; 无 DOD 类出院后随访列 → 28d 结局只能院内死亡近似(已入 Methods 声明)")
ei = con.execute(f"""
WITH tbi AS (
  SELECT DISTINCT patientunitstayid FROM read_csv_auto('{EI.as_posix()}/diagnosis.csv.gz')
  WHERE lower(diagnosisstring) LIKE '%traumatic brain%' OR lower(diagnosisstring) LIKE '%head injury%'
   OR lower(diagnosisstring) LIKE '%intracranial%' OR lower(diagnosisstring) LIKE '%subdural%'
   OR lower(diagnosisstring) LIKE '%epidural%' OR lower(diagnosisstring) LIKE '%cerebral contusion%'
   OR lower(diagnosisstring) LIKE '%concussion%'
   OR regexp_matches(COALESCE(icd9code,''), '^(85[0-4])')),
p AS (SELECT patientunitstayid, patienthealthsystemstayid, gender, age, ethnicity,
             hospitaldischargestatus, hospitaldischargeoffset, unitdischargeoffset,
             unittype, hospitalid, apacheadmissiondx FROM read_csv_auto('{EI.as_posix()}/patient.csv.gz')),
joined AS (SELECT t.patientunitstayid, p.* FROM tbi t JOIN p ON t.patientunitstayid=p.patientunitstayid),
aged AS (SELECT *, TRY_CAST(replace(age,'> 89','90') AS DOUBLE) age_num FROM joined)
SELECT * FROM aged""").df()
n_raw = len(ei)
ei_adult = ei[ei.age_num>18]
ei_adult = ei_adult.sort_values("patientunitstayid").drop_duplicates("patienthealthsystemstayid", keep="first")
n_nolos = len(ei_adult)                                   # raw 口径(对照 ~6681, 19a 无LOS过滤)
ei = ei_adult.copy(); ei["los_h"] = ei.unitdischargeoffset/60.0
ei_los = ei[ei.los_h>=24].copy()                          # LOS≥24h 口径(三库统一)
ei_los["died_hosp"] = (ei_los.hospitaldischargestatus=="Expired").astype(int)
# 28d 院内死亡近似: Expired 且出院 offset ≤ 28d
ei_los["died_hosp_28d"] = ((ei_los.died_hosp==1) & (ei_los.hospitaldischargeoffset<=28*1440)).astype(int)
ei_los["male"] = (ei_los.gender=="Male").astype(int)
ei_out = ei_los[["patientunitstayid","patienthealthsystemstayid","age_num","male","ethnicity",
    "unittype","hospitalid","los_h","died_hosp","died_hosp_28d"]].rename(columns={"age_num":"age","patientunitstayid":"icustay_id_eicu"})
ei_out.to_parquet(OUT/"cohort_eicu.parquet", index=False)
qc["eicu"] = dict(n_dx_hits=int(n_raw), n_adult_firststay=int(n_nolos), anchor_nolos=6681,
    n_los24=int(len(ei_out)), died_hosp_pct=round(100*ei_out.died_hosp.mean(),1),
    died_hosp_28d_pct=round(100*ei_out.died_hosp_28d.mean(),1),
    age_median=float(ei_out.age.median()), male_pct=round(100*ei_out.male.mean(),1))
print(f"  dx命中 {n_raw} (锚点6681=诊断命中raw口径[19a], 偏差{100*(n_raw-6681)/6681:+.1f}%) → 成人+首stay {n_nolos} → +LOS≥24h {len(ei_out)}")
print(f"  院内死亡 {qc['eicu']['died_hosp_pct']}% | 28d窗院内死亡 {qc['eicu']['died_hosp_28d_pct']}% | 年龄中位 {qc['eicu']['age_median']:.0f}")

# ========== MIMIC-III ==========
print("[MIMIC-III] TBI 队列复算 (LOS≥24h 口径, 对照 ~1950)...")
m3 = con.execute(f"""
WITH tbi AS (SELECT DISTINCT hadm_id FROM read_csv_auto('{M3.as_posix()}/DIAGNOSES_ICD.csv.gz', types={{'ICD9_CODE':'VARCHAR'}})
             WHERE substr(ICD9_CODE,1,3) IN ('850','851','852','853','854')),
icu AS (SELECT i.*, ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) rn,
               date_diff('hour',i.intime,i.outtime) los_h
        FROM read_csv_auto('{M3.as_posix()}/ICUSTAYS.csv.gz') i JOIN tbi t ON i.hadm_id=t.hadm_id)
SELECT ic.icustay_id icustay_id, ic.subject_id subject_id, ic.hadm_id hadm_id,
       CAST(ic.intime AS TIMESTAMP) intime,
       CAST(ic.outtime AS TIMESTAMP) outtime, ic.los_h los_h, ic.first_careunit first_careunit,
       a.hospital_expire_flag hospital_expire_flag, CAST(a.deathtime AS TIMESTAMP) deathtime,
       p.gender gender, CAST(p.dob AS TIMESTAMP) dob, CAST(p.dod AS TIMESTAMP) dod, p.expire_flag expire_flag
FROM icu ic JOIN read_csv_auto('{M3.as_posix()}/ADMISSIONS.csv.gz') a ON ic.hadm_id=a.hadm_id
            JOIN read_csv_auto('{M3.as_posix()}/PATIENTS.csv.gz') p ON ic.subject_id=p.subject_id
WHERE ic.rn=1 AND ic.los_h>=24""").df()
n_los = len(m3)
m3["age"] = ((pd.to_datetime(m3.intime)-pd.to_datetime(m3.dob)).dt.days/365.25).clip(upper=91)
m3 = m3[m3.age>18].copy()
n_final = len(m3)
m3["days_to_death"] = (pd.to_datetime(m3.dod).dt.normalize()-pd.to_datetime(m3.intime).dt.normalize()).dt.days  # normalize: dod 日期精度, 当日死亡=0 而非 -1
m3["d28"] = ((m3.days_to_death>=0)&(m3.days_to_death<=28)).astype(int)   # 28d 全因(含院外, DOD 口径)
m3["died_hosp"] = m3.hospital_expire_flag.astype(int)
m3["male"] = (m3.gender=="M").astype(int)
m3_out = m3[["icustay_id","subject_id","hadm_id","intime","outtime","los_h","first_careunit",
             "age","male","d28","died_hosp","days_to_death"]]
m3_out.to_parquet(OUT/"cohort_mimic3.parquet", index=False)
qc["mimic_iii"] = dict(n_los24_preadult=int(n_los), n_final=int(n_final), anchor=1950,
    d28_pct=round(100*m3_out.d28.mean(),1), died_hosp_pct=round(100*m3_out.died_hosp.mean(),1),
    age_median=float(m3_out.age.median()), male_pct=round(100*m3_out.male.mean(),1))
print(f"  LOS≥24h {n_los} → +成人 {n_final} (锚点1950=850-854 raw口径已闭环; 分析口径=854-only+统一排除 [README冻结])")
print(f"  28d全因死亡 {qc['mimic_iii']['d28_pct']}% | 院内死亡 {qc['mimic_iii']['died_hosp_pct']}% | 年龄中位 {qc['mimic_iii']['age_median']:.0f}")

# 口径敏感性: 仅 850-854 (锚点原口径) 复算 n
m3_854 = con.execute(f"""
SELECT count(*) n FROM (
  SELECT i.icustay_id, ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) rn,
         date_diff('hour',i.intime,i.outtime) los_h
  FROM read_csv_auto('{M3.as_posix()}/ICUSTAYS.csv.gz') i
  WHERE i.hadm_id IN (SELECT DISTINCT hadm_id FROM read_csv_auto('{M3.as_posix()}/DIAGNOSES_ICD.csv.gz', types={{'ICD9_CODE':'VARCHAR'}})
                      WHERE substr(ICD9_CODE,1,3) BETWEEN '850' AND '854')) x
WHERE rn=1 AND los_h>=24""").fetchone()[0]
qc["mimic_iii"]["n_los24_854only_prerank"] = int(m3_854)
print(f"  [口径定位] 仅850-854+LOS≥24h(未剔未成年) = {m3_854} — 与锚点~1950 对照定位差异来源")
# 成人过滤版(锚点精确口径: 850-854 + 首ICU + LOS≥24h + >18)
m3_854a = con.execute(f"""
SELECT i.icustay_id, i.subject_id, CAST(i.intime AS TIMESTAMP) intime, CAST(p.dob AS TIMESTAMP) dob,
       ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) rn,
       date_diff('hour',i.intime,i.outtime) los_h
FROM read_csv_auto('{M3.as_posix()}/ICUSTAYS.csv.gz') i
JOIN read_csv_auto('{M3.as_posix()}/PATIENTS.csv.gz') p ON i.subject_id=p.subject_id
WHERE i.hadm_id IN (SELECT DISTINCT hadm_id FROM read_csv_auto('{M3.as_posix()}/DIAGNOSES_ICD.csv.gz', types={{'ICD9_CODE':'VARCHAR'}})
                    WHERE substr(ICD9_CODE,1,3) BETWEEN '850' AND '854')""").df()
_age = ((pd.to_datetime(m3_854a.intime)-pd.to_datetime(m3_854a.dob)).dt.days/365.25).clip(upper=91)
n_854_adult = int(((m3_854a.rn==1)&(m3_854a.los_h>=24)&(_age>18)).sum())
qc["mimic_iii"]["n_los24_854only_adult"] = n_854_adult
print(f"  [口径定位] 仅850-854+LOS≥24h+成人 = {n_854_adult} — 锚点~1950 精确口径对照")

# ========== 三库总表 ==========
m4_n = con.execute(f"SELECT count(*) FROM read_parquet('{(ROOT/'results/cohort_audit/02_tbi_cohort.parquet').as_posix()}')").fetchone()[0]
qc["three_db_total"] = dict(mimic_iv=int(m4_n), eicu_los24=int(len(ei_out)), mimic3_final=int(n_final),
    total=int(m4_n+len(ei_out)+n_final), expect_note="SAP §7.7 工程量级按 2751+6681+1950≈11382 估")
(REP/"07b_cohort_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=== 07b 汇总 === 三库 n = {m4_n} + {len(ei_out)} + {n_final} = {m4_n+len(ei_out)+n_final}")
print("DONE 07b")
