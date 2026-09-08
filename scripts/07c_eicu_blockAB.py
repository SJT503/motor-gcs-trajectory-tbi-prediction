# Phase1-③a: eICU Block A 静态 + Block B 神经动态 (SAP_prediction §3.1/§3.2)
# 列名/labname 全部实测验证(DESRIBE + distinct 分布打印), 禁凭记忆硬编码
# 输出: data/features_eicu.parquet + reports/07c_labname_map.csv + coverage CSV
import sys, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
com = import_module("07_common")

ROOT = Path(r"E:/TBI subtype")
EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
coh = pd.read_parquet(OUT/"cohort_eicu.parquet")
ids = tuple(int(x) for x in coh.icustay_id_eicu.tolist())
print(f"eICU 队列 n={len(coh)} (07b 产物)")

# ========== 列名实测验证 ==========
for tbl in ["vitalPeriodic","lab"]:
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{(EI/tbl).as_posix()}.csv.gz')").df().column_name)
    print(f"[{tbl}] 列: {sorted(cols)}")
    need = {"vitalPeriodic": ["patientunitstayid","observationoffset"],
            "lab": ["patientunitstayid","labname","labresult"]}[tbl]
    miss = [c for c in need if c not in cols and not any(c in x for x in cols)]
    assert not miss, f"{tbl} 缺关键列 {miss} — 按上方实测列名修正脚本后重跑"

# ========== Block A-1: vitals (vitalPeriodic [0,1440]; BP 另并 vitalAperiodic 袖带压) ==========
print("提 eICU vitals (首24h)...")
VP = (EI/"vitalPeriodic.csv.gz").as_posix()
vp_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{VP}')").df().column_name)
# 2026-09-02 修复: BP 若只读 vitalPeriodic systemic* 会漏袖带压 — eICU 离散袖带压记在
# vitalAperiodic noninvasive*, 必须 ∪ 并池 (与 discovery 动脉+NBP 全池口径一致); VITAL_MAP 只留 3 项防列名冲突
# (注: "修复前 periodic-only=50.4%" 的晨版记录系库归因写反、未单独实测, 已废弃; 并池后 4388/4440=98.8% 两轮复现)
VITAL_MAP = {"heartrate":"hr","respiration":"rr","sao2":"spo2"}   # temp 移出: vitalPeriodic.temperature 实测极稀疏, 单独双源并池 (见下)
DOMAIN = {"temperature":"30 AND 45","heartrate":"20 AND 300","respiration":"4 AND 70",
          "sao2":"50 AND 100","systemicsystolic":"30 AND 400","systemicdiastolic":"10 AND 300",
          "systemicmean":"20 AND 300","noninvasivesystolic":"30 AND 400",
          "noninvasivediastolic":"10 AND 300","noninvasivemean":"20 AND 300"}
agg = []
for col, short in VITAL_MAP.items():
    if col not in vp_cols: print(f"  [warn] vitalPeriodic 无 {col}, 跳过"); continue
    v = f"(CASE WHEN {col} BETWEEN {DOMAIN[col]} THEN {col} END)"
    agg.append(f"min({v}) {short}_min")
    agg.append(f"avg({v}) {short}_mean")
    agg.append(f"max({v}) {short}_max")
assert agg, "vitalPeriodic 无任何 VITAL_MAP 列 — 按上方实测列名修正后重跑"   # 空池兜底(2026-09-02 核验建议): 全跳过时防 SQL 语法错
vit = con.execute(f"""
SELECT patientunitstayid, {', '.join(agg)} FROM read_csv_auto('{VP}')
WHERE patientunitstayid IN {ids} AND observationoffset BETWEEN 0 AND 1440
GROUP BY patientunitstayid""").df()
vit = vit.rename(columns={"patientunitstayid":"icustay_id_eicu"})
print(f"  periodic vitals (hr/rr/spo2) 覆盖 {len(vit)} stays")

# temp 双源并池 (2026-09-03 三修): vitalPeriodic.temperature 实测极稀疏 (两轮 12.4%/12.7%, 4440 仅 ~560 有值;
# °F 假说证伪 — 单位自判仅 +0.3pp) → eICU 口腔/腋温由护士记于 nurseCharting (Temperature (C)/(F) 行)
# 并池两源; 单位按 行名优先、值域兜底 自判; nurseCharting 值列为 VARCHAR → TRY_CAST
temp_parts = [f"""SELECT patientunitstayid,
  CASE WHEN temperature>70 AND temperature<120 THEN (temperature-32)/1.8
       WHEN temperature>30 AND temperature<45 THEN temperature END tv
FROM read_csv_auto('{VP}')
WHERE patientunitstayid IN {ids} AND observationoffset BETWEEN 0 AND 1440 AND temperature IS NOT NULL"""]
n_src1 = con.execute(f"SELECT count(DISTINCT patientunitstayid) AS n FROM ({temp_parts[0]}) WHERE tv IS NOT NULL").df().n[0]
print(f"  [temp 源1] vitalPeriodic 温度有值 {int(n_src1)} stays (单位自判后)")
NC = (EI/"nurseCharting.csv.gz")
nc_ok = False
if NC.exists():
    nc_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{NC.as_posix()}')").df().column_name)
    print(f"[nurseCharting] 实测列: {sorted(nc_cols)}")
    # v5 (2026-09-03): 实测值列 = nursingchartvalue (v4 猜 nursingchartcelltypevalvalue 不存在 → 防线退回) → 动态发现
    val_guess = next((c for c in nc_cols if c.endswith("value") and "label" not in c), "nursingchartcelltypevalvalue")
    key = {"id":"patientunitstayid","off":"nursingchartentryoffset",
           "name":"nursingchartcelltypevalname","val":val_guess}
    miss = [k for k,v in key.items() if v not in nc_cols]
    if miss:
        print(f"  [warn] nurseCharting 缺关键列 {miss} → temp 退回 periodic-only (12.7%, 将 DROP)")
    else:
        nc_ok = True
        nc_names = con.execute(f"""SELECT {key['name']} AS nm, count(*) AS n, count(DISTINCT patientunitstayid) AS n_pat
FROM read_csv_auto('{NC.as_posix()}')
WHERE patientunitstayid IN {ids} AND {key['name']} ILIKE '%temperature%'
  AND TRY_CAST({key['val']} AS DOUBLE) IS NOT NULL
GROUP BY 1 ORDER BY n DESC LIMIT 20""").df()
        print(nc_names.to_string(index=False))
        nc_names.to_csv(REP/"07c_nursetemp_names.csv", index=False)
        v = f"TRY_CAST({key['val']} AS DOUBLE)"
        temp_parts.append(f"""SELECT patientunitstayid,
  CASE WHEN {key['name']} ILIKE '%(F)%' AND {v} BETWEEN 70 AND 120 THEN ({v}-32)/1.8
       WHEN {key['name']} ILIKE '%(C)%' AND {v} BETWEEN 30 AND 45 THEN {v}
       WHEN {v} BETWEEN 70 AND 120 THEN ({v}-32)/1.8
       WHEN {v} BETWEEN 30 AND 45 THEN {v} END tv
FROM read_csv_auto('{NC.as_posix()}')
WHERE patientunitstayid IN {ids} AND {key['off']} BETWEEN 0 AND 1440
  AND {key['name']} ILIKE '%temperature%'""")
else:
    print(f"  [warn] 无 {NC} → temp 退回 periodic-only")
tp = con.execute(f"SELECT patientunitstayid, min(tv) temp_min, avg(tv) temp_mean, max(tv) temp_max "
                 f"FROM ({' UNION ALL '.join(temp_parts)}) WHERE tv IS NOT NULL GROUP BY patientunitstayid").df() \
      .rename(columns={"patientunitstayid":"icustay_id_eicu"})
src2 = 0
if nc_ok:
    src2 = con.execute(f"SELECT count(DISTINCT patientunitstayid) AS n FROM ({temp_parts[1]}) WHERE tv IS NOT NULL").df().n[0]
print(f"  temp 并池 (periodic ∪ nurseCharting) 覆盖 {len(tp)}/{len(ids)} stays | 源1 periodic {int(n_src1)} | 源2 nurse-only {int(src2)}")

# BP: vitalPeriodic systemic*(监护仪流) ∪ vitalAperiodic noninvasive*(袖带压) → min/mean/max
VA = (EI/"vitalAperiodic.csv.gz").as_posix()
va_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{VA}')").df().column_name)
print(f"[vitalAperiodic] 实测列: {sorted(va_cols)}")
assert {"patientunitstayid","observationoffset"} <= va_cols, f"vitalAperiodic 缺关键列, 实测列名: {sorted(va_cols)}"
BP_SRC = [("systemicsystolic", VP, vp_cols), ("systemicdiastolic", VP, vp_cols), ("systemicmean", VP, vp_cols),
          ("noninvasivesystolic", VA, va_cols), ("noninvasivediastolic", VA, va_cols), ("noninvasivemean", VA, va_cols)]
parts = []
for col, path, cols_ok in BP_SRC:
    if col not in cols_ok: print(f"  [warn] BP 源无 {col}, 跳过"); continue
    parts.append(f"SELECT patientunitstayid, '{col}' c, {col} v FROM read_csv_auto('{path}') "
                 f"WHERE patientunitstayid IN {ids} AND observationoffset BETWEEN 0 AND 1440 "
                 f"AND {col} BETWEEN {DOMAIN[col]}")
assert parts, "BP 无可用源列 — 按上方 vitalAperiodic 实测列名修正后重跑"
bp = con.execute(f"""
WITH u AS ({" UNION ALL ".join(parts)}),
m AS (SELECT patientunitstayid, CASE WHEN c LIKE '%systolic' THEN 'sbp'
                                    WHEN c LIKE '%diastolic' THEN 'dbp' ELSE 'mbp' END k, v FROM u)
SELECT patientunitstayid,
  min(CASE WHEN k='sbp' THEN v END) sbp_min, avg(CASE WHEN k='sbp' THEN v END) sbp_mean, max(CASE WHEN k='sbp' THEN v END) sbp_max,
  min(CASE WHEN k='dbp' THEN v END) dbp_min, avg(CASE WHEN k='dbp' THEN v END) dbp_mean, max(CASE WHEN k='dbp' THEN v END) dbp_max,
  min(CASE WHEN k='mbp' THEN v END) mbp_min, avg(CASE WHEN k='mbp' THEN v END) mbp_mean, max(CASE WHEN k='mbp' THEN v END) mbp_max
FROM m GROUP BY patientunitstayid""").df().rename(columns={"patientunitstayid":"icustay_id_eicu"})
print(f"  BP (periodic systemic ∪ aperiodic noninvasive) 覆盖 {len(bp)}/{len(ids)} stays")

# ========== Block A-2: labs (labname 实测分布 + 概念子串映射) ==========
print("实测 eICU labname 分布(队列内首24h)...")
LAB = (EI/"lab.csv.gz").as_posix()
lab_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{LAB}')").df().column_name)
off_col = "labresultoffset" if "labresultoffset" in lab_cols else (
          "laboffset" if "laboffset" in lab_cols else None)
assert off_col, f"lab 表无时间列, 实测列名: {sorted(lab_cols)}"
names = con.execute(f"""
SELECT labname, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{LAB}') WHERE patientunitstayid IN {ids}
  AND {off_col} BETWEEN 0 AND 1440 AND labresult IS NOT NULL
GROUP BY labname ORDER BY n DESC LIMIT 200""").df()
print(names.head(40).to_string(index=False))
names.to_csv(REP/"07c_labname_raw_top200.csv", index=False)
CONCEPTS = {  # SAP §3.1 概念 → labname 子串(正则, word-boundary)
 "aniongap": r"anion\s*gap", "bicarbonate": r"bicarb|HCO3", "bun": r"\bBUN\b",
 "calcium": r"calcium", "chloride": r"chloride", "creatinine": r"creatinine",
 "glucose_lab": r"glucose", "sodium": r"\bsodium\b|\bNa\+?\b", "potassium": r"\bpotassium\b|\bK\+?\b",
 "hematocrit": r"hematocrit|\bHCT\b", "hemoglobin": r"hemoglobin|\bHgb?\b", "platelet": r"platelet",
 "wbc": r"\bWBC\b", "inr": r"\bINR\b|international.{0,12}ratio", "pt": r"\bPT\b|prothrombin",
 "ptt": r"\bPTT\b|partial.{0,12}thromboplastin"}
lab_sel, map_rows = [], []
_EXCL_BASE = "ionized|urine|csf|cerebrospinal|body fluid|dialysate|peritoneal|pleural|ascites|synovial"
_EXTRA = {"pt": "inr", "bun": "ratio", "creatinine": "ratio"}   # 同名异义排除: PT-INR 归 inr; BUN/Cr 比值不进 bun/creatinine
for concept, pat in CONCEPTS.items():
    excl = _EXCL_BASE + ("|"+_EXTRA[concept] if concept in _EXTRA else "")
    hit = names[names.labname.str.contains(rf"^(?!.*(?:{excl})).*(?:{pat})", case=False, regex=True, na=False)]
    if len(hit)==0: print(f"  [warn] {concept} 无 labname 命中"); continue
    nms = hit.sort_values("n", ascending=False).head(3).labname.tolist()
    map_rows.append(dict(concept=concept, labnames=" | ".join(nms), n_rows=int(hit.n.sum())))
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in hit.labname)  # labname 含撇号(WBC's…)需 SQL 转义
    lab_sel.append(f"min(CASE WHEN labname IN ({quoted}) THEN labresult END) {concept}_min, "
                   f"max(CASE WHEN labname IN ({quoted}) THEN labresult END) {concept}_max")
labs = con.execute(f"""
SELECT patientunitstayid, {', '.join(lab_sel)} FROM read_csv_auto('{LAB}')
WHERE patientunitstayid IN {ids} AND {off_col} BETWEEN 0 AND 1440 AND labresult IS NOT NULL
GROUP BY patientunitstayid""").df().rename(columns={"patientunitstayid":"icustay_id_eicu"})
pd.DataFrame(map_rows).to_csv(REP/"07c_labname_map.csv", index=False)
print(f"  labs 概念映射 {len(map_rows)}/16 → reports/07c_labname_map.csv | 覆盖 {len(labs)} stays")

# ========== Block A-3: GCS 首24h (从 07a gcs_long) + 人口学 ==========
g = pd.read_parquet(OUT/"gcs_long_eicu.parquet"); g["id"]=g.id.astype(int)
g24 = g[(g.offset_hr>=0)&(g.offset_hr<=24)].sort_values("offset_hr")
ga = g24.groupby("id").agg(gcs_total_min=("gcs_total","min"), gcs_total_first=("gcs_total","first"),
                           gcs_motor_min=("gcs_motor","min"), gcs_motor_first=("gcs_motor","first"),
                           gcs_eye_min=("gcs_eye","min"), gcs_eye_first=("gcs_eye","first"),
                           gcs_verbal_min=("gcs_verbal","min"), gcs_verbal_first=("gcs_verbal","first")).reset_index()
ga = ga.rename(columns={"id":"icustay_id_eicu"})

# ========== Block B: 神经动态 (motor 主, total 敏感性, 0→24h; 统一 dyn_ 前缀与 Block A 区分) ==========
def with_dyn_prefix(bdf):
    return bdf.rename(columns={c: "dyn_"+c for c in bdf.columns if c!="id"})
b_motor = with_dyn_prefix(com.neuro_dynamic_features(g, "gcs_motor", "motor", win_hr=24.0)).rename(columns={"id":"icustay_id_eicu"})
b_total = with_dyn_prefix(com.neuro_dynamic_features(g, "gcs_total", "gcs_total", win_hr=24.0)).rename(columns={"id":"icustay_id_eicu"})
print(f"  Block B: motor {len(b_motor)} | total {len(b_total)} | motor_n≥2 占比 {100*(b_motor.dyn_motor_n>=2).mean():.1f}%")

# ========== 合并 ==========
df = coh.merge(vit, on="icustay_id_eicu", how="left").merge(bp, on="icustay_id_eicu", how="left") \
       .merge(tp, on="icustay_id_eicu", how="left").merge(labs, on="icustay_id_eicu", how="left") \
       .merge(ga, on="icustay_id_eicu", how="left").merge(b_motor, on="icustay_id_eicu", how="left") \
       .merge(b_total, on="icustay_id_eicu", how="left")
df.to_parquet(OUT/"features_eicu.parquet", index=False)
cov = com.coverage_audit(df, id_cols=("icustay_id_eicu","patienthealthsystemstayid","ethnicity","unittype",
                         "hospitalid","los_h","died_hosp","died_hosp_28d"), out_csv=REP/"07c_coverage.csv")
print(f"\n=== 07c 汇总 === features_eicu: {df.shape[0]} × {df.shape[1]}")
print(cov.to_string(index=False))
print("DONE 07c")
