# Phase1-③b: MIMIC-III Block A 静态 + Block B 神经动态 (SAP_prediction §3.1/§3.2)
# itemid 全部 D_ITEMS/D_LABITEMS 实测查证(label 正则匹配, 运行时构造 SQL), 禁凭记忆硬编码
# 输出: data/features_mimic3.parquet + reports/07d_itemid_map.csv + coverage CSV
import sys, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
com = import_module("07_common")

ROOT = Path(r"E:/TBI subtype")
M3 = ROOT/"data/mimic-iii-1.4"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
coh = pd.read_parquet(OUT/"cohort_mimic3.parquet")
coh[["icustay_id","subject_id","intime"]].to_parquet(OUT/"_tmp_m3_feat_coh.parquet", index=False)
TMP = (OUT/"_tmp_m3_feat_coh.parquet").as_posix()
print(f"MIMIC-III 队列 n={len(coh)} (07b 产物)")
map_rows = []

# ========== itemid 实测查证 ==========
d_items = con.execute(f"""SELECT itemid AS itemid, label AS label, dbsource AS dbsource
FROM read_csv_auto('{M3.as_posix()}/D_ITEMS.csv.gz')
WHERE category ILIKE '%hemodynamics%' OR category ILIKE '%respiratory%' OR category ILIKE '%routine%'
   OR dbsource='carevue'""").df()   # carevue 行 category 多为 NULL(实测), 不加此支池子只剩 metavision
def pick_items(label_pat, concept, must=True):
    hits = d_items[d_items.label.str.contains(label_pat, case=False, regex=True, na=False)]
    hits = hits[hits.dbsource.isin(["carevue","metavision"])]
    if len(hits)==0:
        assert not must, f"D_ITEMS 未命中: {concept} / {label_pat}"
        return []
    iids = sorted(int(i) for i in hits.itemid.unique())
    map_rows.append(dict(block="chart", concept=concept, label_pat=label_pat,
                         itemids=",".join(map(str,iids)),
                         labels=" | ".join(sorted(set(hits.label))[:6])))
    return iids
HR   = pick_items(r"^Heart Rate$", "hr")
# 2026-09-02 修复: BP 原正则只匹配动脉压 label (映射表铁证: 池仅 6/51/52/220050 系) → NBP 袖带压全缺
# (动脉-only 实测 n_pat: Art 502+ABP 220≈722/1425≈50.7%; 晨版"25.8%"系库归因写反, 已废弃; M4 锚点 99.8);
# 收全 动脉+无创 两支, 与 eICU systemic∪noninvasive / discovery 动脉+NBP 全池口径一致
SBP  = pick_items(r"Blood Pressure (systolic|Systolic)|^Arterial BP \[Systolic\]$|^ABP \[Systolic\]$|^NBP \[Systolic\]$", "sbp")
DBP  = pick_items(r"Blood Pressure (diastolic|Diastolic)|^Arterial BP \[Diastolic\]$|^ABP \[Diastolic\]$|^NBP \[Diastolic\]$", "dbp")
MAP_ = pick_items(r"Blood Pressure mean|^Arterial BP Mean$|^NBP Mean$|^MAP$", "map")
RR   = pick_items(r"^Respiratory Rate$|^Resp Rate$|^RR$", "rr")
# 2026-09-02 修复: 原 ^Temperature F$ 精确锚拒掉 "Temperature F,…" 类变体 → 覆盖塌 12.4% (M4 锚点 99.6%);
# 改单池宽匹配 ^Temperature* (负前瞻排 Core/Skin 站点), F/C 单位由值域自判 (见下方 CASE)
TEMP = pick_items(r"^(?!.*core)(?!.*skin)Temperature", "temp")
SPO2 = pick_items(r"O2 saturation pulseoxymetry|^SpO2$", "spo2")   # carevue 标签实测补齐 2026-09-02
def I(lst): return "("+",".join(map(str,lst))+")" if lst else "(NULL)"

d_labitems = con.execute(f"""SELECT itemid AS itemid, label AS label FROM read_csv_auto('{M3.as_posix()}/D_LABITEMS.csv.gz')""").df()
LAB_CONCEPTS = {"aniongap": r"anion gap", "bicarbonate": r"bicarbonate", "bun": r"urea nitrogen|BUN",
 "calcium": r"calcium", "chloride": r"chloride", "creatinine": r"creatinine", "glucose_lab": r"glucose",
 "sodium": r"sodium", "potassium": r"potassium", "hematocrit": r"hematocrit", "hemoglobin": r"hemoglobin",
 "platelet": r"platelet count|platelets", "wbc": r"\bwbc\b|white blood cell", "inr": r"inr|international normalized",
 "pt": r"prothrombin|\bpt\b|protime", "ptt": r"ptt|partial thromboplastin"}
_EXCL_BASE = "ionized|urine|csf|dialysate|peritoneal|pleural|ascites|synovial"
_EXTRA = {"pt": "inr", "bun": "ratio", "creatinine": "ratio",
          "wbc": "cast|clump|other fluid|joint"}   # WBC, Joint/Other Fluid 与 Casts/Clumps 不混池; 主label=White Blood Cells(51301)
lab_sel = []
for concept, pat in LAB_CONCEPTS.items():
    excl = _EXCL_BASE + ("|"+_EXTRA[concept] if concept in _EXTRA else "")
    hits = d_labitems[d_labitems.label.str.contains(rf"^(?!.*(?:{excl})).*(?:{pat})", case=False, regex=True, na=False)]
    if len(hits)==0: print(f"  [warn] {concept} D_LABITEMS 未命中"); continue
    iids = sorted(int(i) for i in hits.itemid.unique())
    map_rows.append(dict(block="lab", concept=concept, label_pat=pat,
                         itemids=",".join(map(str,iids)), labels=" | ".join(sorted(set(hits.label))[:6])))
    inlist = ",".join(map(str,iids))
    lab_sel.append(f"min(CASE WHEN itemid IN ({inlist}) THEN valuenum END) {concept}_min, "
                   f"max(CASE WHEN itemid IN ({inlist}) THEN valuenum END) {concept}_max")
pd.DataFrame(map_rows).to_csv(REP/"07d_itemid_map.csv", index=False)
print(f"itemid 映射 {len(map_rows)} 行 → reports/07d_itemid_map.csv")

# ========== Block A-1: vitals (CHARTEVENTS [intime,+24h]) ==========
print("提 MIMIC-III vitals (CHARTEVENTS, strict_mode=false, ~4min)...")
vit = con.execute(f"""
WITH ce AS (
  SELECT c.icustay_id,
    CASE WHEN c.itemid IN {I(HR)} AND c.valuenum BETWEEN 20 AND 300 THEN c.valuenum END hr,
    CASE WHEN c.itemid IN {I(SBP)} AND c.valuenum BETWEEN 30 AND 400 THEN c.valuenum END sbp,
    CASE WHEN c.itemid IN {I(DBP)} AND c.valuenum BETWEEN 10 AND 300 THEN c.valuenum END dbp,
    CASE WHEN c.itemid IN {I(MAP_)} AND c.valuenum BETWEEN 20 AND 300 THEN c.valuenum END mbp,
    CASE WHEN c.itemid IN {I(RR)} AND c.valuenum BETWEEN 4 AND 70 THEN c.valuenum END rr,
    CASE WHEN c.itemid IN {I(TEMP)} AND c.valuenum>70 AND c.valuenum<120 THEN (c.valuenum-32)/1.8
         WHEN c.itemid IN {I(TEMP)} AND c.valuenum>30 AND c.valuenum<45 THEN c.valuenum END "temp",
    CASE WHEN c.itemid IN {I(SPO2)} AND c.valuenum BETWEEN 50 AND 100 THEN c.valuenum END spo2
  FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
       types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
  JOIN read_parquet('{TMP}') co ON c.icustay_id=co.icustay_id
  WHERE c.valuenum IS NOT NULL
    AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL 24 HOUR)
SELECT icustay_id AS icustay_id,
  min(hr) hr_min, avg(hr) hr_mean, max(hr) hr_max,
  min(sbp) sbp_min, avg(sbp) sbp_mean, max(sbp) sbp_max,
  min(dbp) dbp_min, avg(dbp) dbp_mean, max(dbp) dbp_max,
  min(mbp) mbp_min, avg(mbp) mbp_mean, max(mbp) mbp_max,
  min(rr) rr_min, avg(rr) rr_mean, max(rr) rr_max,
  min("temp") temp_min, avg("temp") temp_mean, max("temp") temp_max,
  min(spo2) spo2_min, avg(spo2) spo2_mean, max(spo2) spo2_max
FROM ce GROUP BY icustay_id""").df()
print(f"  vitals 覆盖 {len(vit)} stays")

# 覆盖缺口诊断探针 (2026-09-02 二轮): temp/BP 池逐 itemid 审计 → 落盘自证, 免凭猜归因
probe_ids = sorted(set(TEMP + SBP + DBP + MAP_))
probe = con.execute(f"""
SELECT c.itemid AS itemid, any_value(d.label) AS label, count(*) AS n_rows, count(DISTINCT c.icustay_id) AS n_pat,
       round(avg(c.valuenum),1) v_mean
FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
     types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
JOIN read_parquet('{TMP}') co ON c.icustay_id=co.icustay_id
JOIN read_csv_auto('{M3.as_posix()}/D_ITEMS.csv.gz') d ON c.itemid=d.itemid
WHERE c.itemid IN {I(probe_ids)} AND c.valuenum IS NOT NULL
  AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL 24 HOUR
GROUP BY c.itemid ORDER BY n_pat DESC""").df()
probe.to_csv(REP/"07d_vitals_itemid_probe.csv", index=False)
print(probe.to_string(index=False))

# ========== Block A-2: labs (LABEVENTS [intime-6h,+24h], subject join) ==========
print("提 MIMIC-III labs (LABEVENTS)...")
labs = con.execute(f"""
WITH le AS (
  SELECT co.icustay_id, l.itemid, l.valuenum
  FROM read_csv_auto('{M3.as_posix()}/LABEVENTS.csv.gz', types={{'valuenum':'DOUBLE'}}) l
  JOIN read_parquet('{TMP}') co ON l.subject_id=co.subject_id
  WHERE l.valuenum IS NOT NULL
    AND l.charttime BETWEEN co.intime - INTERVAL 6 HOUR AND co.intime + INTERVAL 24 HOUR)
SELECT icustay_id AS icustay_id, {', '.join(lab_sel)} FROM le GROUP BY icustay_id""").df()
print(f"  labs 覆盖 {len(labs)} stays")

# 凝血探针 (2026-09-03 勘误: 2026-09-02 二轮注释"M3 pt 45.3/inr 47.3/ptt 30.5"系库归因写反——
# 那三数是 eICU 的; M3 v3 轮实测 pt/inr 97.1 ptt 96.8 全 keep, 凝血池=INR(PT) 51237 n_pat 1383)
# → 候选 D_LABITEMS label 全打印 + 逐 itemid 队列内覆盖, 映射表证据落盘
coag_lab = d_labitems[d_labitems.label.str.contains(r"prothrombin|protime|thromboplastin|inr|coag|partial", case=False, na=False)]
print("  [coag 候选 D_LABITEMS]"); print(coag_lab.to_string(index=False))
if len(coag_lab):
    coag_cov = con.execute(f"""
    SELECT l.itemid AS itemid, count(DISTINCT co.icustay_id) AS n_pat
    FROM read_csv_auto('{M3.as_posix()}/LABEVENTS.csv.gz', types={{'valuenum':'DOUBLE'}}) l
    JOIN read_parquet('{TMP}') co ON l.subject_id=co.subject_id
    WHERE l.itemid IN {I(sorted(set(coag_lab.itemid.astype(int))))} AND l.valuenum IS NOT NULL
      AND l.charttime BETWEEN co.intime - INTERVAL 6 HOUR AND co.intime + INTERVAL 24 HOUR
    GROUP BY l.itemid ORDER BY n_pat DESC""").df()
    print("  [coag 候选 itemid 队列内覆盖(n/1425)]"); print(coag_cov.to_string(index=False))

# ========== Block A-3: GCS 首24h + Block B (07a gcs_long) ==========
g = pd.read_parquet(OUT/"gcs_long_mimic3.parquet")
g24 = g[(g.offset_hr>=0)&(g.offset_hr<=24)]
ga = g24.sort_values("offset_hr").groupby("id").agg(
    gcs_total_min=("gcs_total","min"), gcs_total_first=("gcs_total","first"),
    gcs_motor_min=("gcs_motor","min"), gcs_motor_first=("gcs_motor","first"),
    gcs_eye_min=("gcs_eye","min"), gcs_eye_first=("gcs_eye","first"),
    gcs_verbal_min=("gcs_verbal","min"), gcs_verbal_first=("gcs_verbal","first")).reset_index()
ga = ga.rename(columns={"id":"icustay_id"}); g["id"]=g.id.astype(int); ga["icustay_id"]=ga.icustay_id.astype(int)
b_motor = com.neuro_dynamic_features(g, "gcs_motor", "motor", win_hr=24.0)
b_total = com.neuro_dynamic_features(g, "gcs_total", "gcs_total", win_hr=24.0)
for _b in (b_motor, b_total):   # dyn_ 前缀防与 Block A 静态 gcs_total_min 等列冲突 (与 07c 同构)
    _b.rename(columns={c: "dyn_"+c for c in _b.columns if c != "id"}, inplace=True)
    _b.rename(columns={"id":"icustay_id"}, inplace=True)
print(f"  Block B: motor {len(b_motor)} | motor_n≥2 占比 {100*(b_motor.dyn_motor_n>=2).mean():.1f}% (MIMIC-IV 13a 锚点≈95%+)")

# GCS 首测时点探针 (2026-09-03 勘误: 旧注释"[0,24h] 静态 76.2%"系 eICU 数字错置——
# M3 v3 轮实测首测 100% ≤24h, >24h 0 人, 72h 覆盖 99.2%; SAP §3.0 窗口冻结不动)
tot72 = g.dropna(subset=["gcs_total"]); n_coh = len(coh)
first = tot72.groupby("id").offset_hr.min(); sys_first = tot72.groupby("id")["system"].first()
for sys_name in ["carevue", "metavision"]:
    sel = first[sys_first == sys_name]
    if len(sel) == 0: continue
    print(f"  [GCS 首测时点] {sys_name}: n={len(sel)} | ≤24h {100*(sel<=24).mean():.1f}% | >24h 才首测 {100*(sel>24).mean():.1f}%")
print(f"  [GCS 首测时点] 全队列: 72h 内有 total {len(first)}/{n_coh} ({100*len(first)/n_coh:.1f}%), 其中 >24h 才首测 {(first>24).sum()} 人")
mot72 = g.dropna(subset=["gcs_motor"])
n_mot24 = mot72[mot72.offset_hr <= 24].id.nunique()
print(f"  [motor 24h 内可得] {n_mot24}/{n_coh} ({100*n_mot24/n_coh:.1f}%)")

# ========== 合并 ==========
df = coh.merge(vit, on="icustay_id", how="left").merge(labs, on="icustay_id", how="left") \
       .merge(ga, on="icustay_id", how="left").merge(b_motor, on="icustay_id", how="left") \
       .merge(b_total, on="icustay_id", how="left")
df.to_parquet(OUT/"features_mimic3.parquet", index=False)
cov = com.coverage_audit(df, id_cols=("icustay_id","subject_id","hadm_id","intime","outtime","first_careunit",
                         "days_to_death","los_h","d28","died_hosp"), out_csv=REP/"07d_coverage.csv")
print(f"\n=== 07d 汇总 === features_mimic3: {df.shape[0]} × {df.shape[1]}")
print(cov.to_string(index=False))
print("DONE 07d")
