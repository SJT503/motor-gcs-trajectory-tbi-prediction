# -*- coding: utf-8 -*-
# 08i-probe: eICU SOFA 映射前实测侦察 — 列名/labname/drugname/celllabel 分布落盘
# (eICU 无官方 mimic-code SOFA; 08i 需以下实测字面量后才写定)
import sys, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:/TBI subtype")
EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
coh = pd.read_parquet(OUT/"cohort_eicu.parquet")
ids = tuple(int(x) for x in coh.icustay_id_eicu.tolist())
print(f"eICU 队列 n={len(coh)}")

def desc(tbl):
    cols = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{(EI/tbl).as_posix()}.csv.gz')").df()
    print(f"[{tbl}] 列: {sorted(cols.column_name)}")
    return set(cols.column_name)

# 1) lab 表: SOFA 相关 labname + 单位列
lc = desc("lab")
off = "labresultoffset" if "labresultoffset" in lc else "laboffset"
unit_col = next((c for c in lc if "unit" in c.lower()), None)
print(f"lab off={off} unit={unit_col}")
sel = f"""SELECT labname, {unit_col} unit, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{(EI/'lab.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids} AND {off} BETWEEN -360 AND 1440 AND labresult IS NOT NULL
  AND (labname ILIKE '%pao2%' OR labname ILIKE '%fio2%' OR labname ILIKE '%platelet%'
    OR labname ILIKE '%bilirubin%' OR labname ILIKE '%creatinine%')
GROUP BY 1,2 ORDER BY n DESC"""
lab_dist = con.execute(sel).df()
print(lab_dist.to_string(index=False))
lab_dist.to_csv(REP/"08i_probe_lab.csv", index=False)

# 2) infusionDrug: 升压药 drugname + 速率列 + 单位列
ic = desc("infusionDrug")
print(f"infusionDrug 候选速率/单位列: {[c for c in ic if 'rate' in c.lower() or 'unit' in c.lower() or 'amount' in c.lower()]}")
drug = con.execute(f"""SELECT drugname, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{(EI/'infusionDrug.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids}
  AND (drugname ILIKE '%pinephrine%' OR drugname ILIKE '%opamine%')
GROUP BY 1 ORDER BY n DESC LIMIT 60""").df()
print(drug.to_string(index=False))
drug.to_csv(REP/"08i_probe_drug.csv", index=False)

# 3) intakeOutput: 尿量 celllabel
oc = desc("intakeOutput")
print(f"intakeOutput 值候选列: {[c for c in oc if 'value' in c.lower() or 'val' in c.lower()]}")
uo = con.execute(f"""SELECT celllabel, cellpath AS cellattr, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{(EI/'intakeOutput.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids} AND celllabel ILIKE '%urine%'
GROUP BY 1,2 ORDER BY n DESC LIMIT 30""").df()
print(uo.to_string(index=False))
uo.to_csv(REP/"08i_probe_uo.csv", index=False)

# 4) respiratoryCharting: 通气相关 celllabel (vent 判定)
rc = desc("respiratoryCharting")
rc_lab = con.execute(f"""SELECT respchartvaluelabel AS celllabel, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv('{(EI/'respiratoryCharting.csv.gz').as_posix()}', quote='"')
WHERE patientunitstayid IN {ids}
GROUP BY 1 ORDER BY n DESC LIMIT 40""").df()
print(rc_lab.to_string(index=False))
rc_lab.to_csv(REP/"08i_probe_respchart.csv", index=False)

# 5) treatment: ventilation 相关 treatmentstring
tc = desc("treatment")
trt = con.execute(f"""SELECT treatmentstring, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv('{(EI/'treatment.csv.gz').as_posix()}', quote='"')
WHERE patientunitstayid IN {ids} AND treatmentstring ILIKE '%ventilat%'
GROUP BY 1 ORDER BY n DESC LIMIT 30""").df()
print(trt.head(15).to_string(index=False))
trt.to_csv(REP/"08i_probe_treatment.csv", index=False)

# 6) patient 表: 体重 (升压药 kg 换算不需要 — eICU drugrate 常为 mcgkgmin, 但实测确认)
pc = desc("patient")

# 7) vitalPeriodic/vitalAperiodic 已由 07c 实测 (systemicmean/noninvasivemean), 复核列名
_ = desc("vitalPeriodic"); _ = desc("vitalAperiodic")
print("DONE probe")
