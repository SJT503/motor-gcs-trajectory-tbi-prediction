# -*- coding: utf-8 -*-
# 08i: eICU SOFA (first-day, 08z 窗口约定) — 自建映射移植
# 背景: eicu-code 官方仓库无 SOFA query (concepts 仅 demographics/labs/icustay_detail/diagnosis/pivoted)
#       → 组件阈值语义严格对齐 mimic-code sofa.sql (severityscores), 数据源映射自建, 全部决策在 JSON 披露
# 字面量来源: 08i_probe_eicu_sofa.py 实测 (reports/08i_probe_{lab,drug,uo,respchart,treatment}.csv)
# 已披露偏离 (vs mimic-code):
#   1) 窗口统一 08z 约定: 测量类 [-360,+1440] min, UO [0,+1440], (官方 M3 为 [intime,+1d] 等)
#   2) FiO2 仅取 lab 表, 每 paO2 行配最近前置 FiO2 (ASOF, 无回看上限); 值 ≤1 视为分数×100, 21-100 视为百分数
#   3) 通气为患者级 0/24h 标志 (非官方行级 ventnum 状态机); vent 患者 P/F 走 4 档梯, novent 走 <300→2/<400→1
#   4) 升压药仅纳入 base 名精确匹配 + 单位可解析行 (mcg/kg/min 直用; mcg/min ÷ 行内 patientweight,
#      缺 → 该患者 vaso 行中位体重; ml/hr / mg/kg/min / MAX / STD / 裸名等除外并计数)
#   5) CNS 复用 07a 调和产物 gcs_long_eicu.parquet (ETT verbal=0→15 已在 07a 实现), [0,24h] min
#   6) missing 组件计 0 分 (官方 coalesce 语义), 组件覆盖率全量报告
import sys, json, re, duckdb, pandas as pd, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:/TBI subtype")
EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"

# ---- 实测字面量 (08i probe 落盘, 禁止凭记忆改动) ----
LAB_PAO2, LAB_FIO2 = "paO2", "FiO2"
LAB_PLT, LAB_CR, LAB_BILI = "platelets x 1000", "creatinine", "total bilirubin"
UO_LABELS = ["Urine","URINE CATHETER","Urine Count","Urine Output (mL)-Urethral Catheter",
             "Urine Output-Foley","Urine Output-foley","SN Urine Output(ml)","OR urine","urine count"]
UO_EXCLUDED = ["Urine Occurrence","Urine Incontinence","incontinent urine","urine incontinence",
               "Mixed Urine/Stool Volume"]
VASO_BASES = {"norepinephrine":"norepi","epinephrine":"epi","dopamine":"dopa","dobutamine":"doba"}
# 通气判定字面量 (08i probe 实测落盘: reports/08i_probe_{respchart,treatment}.csv)
VENT_RESPCATS = []   # respcharttypecat 分布未查, 留空 (用 label + treatment 两源)
VENT_RCLABELS = [   # 呼吸机设定类 chart 标签 (自发呼吸患者不会出现的设定项)
  "PEEP","Tidal Volume (set)","TV/kg IBW","Vent Rate","Mean Airway Pressure",
  "Pressure Support","Peak Insp. Pressure","Exhaled TV (machine)","Exhaled MV",
  "Exhaled TV (patient)","Plateau Pressure","Flow Sensitivity","Peak Flow",
  "Pressure Control","PEEP/CPAP"]
VENT_TREAT = ["mechanical ventilation", "ventilatory support"]   # treatment LIKE 子串
# 刻意排除: FiO2/RR/Total RR/SaO2/LPM O2/O2 Device/O2 Percentage (非 MV 特异);
#           non-invasive ventilation/CPAP-PEEP therapy/oxygen therapy/ventilator weaning (非有创 MV)

coh = pd.read_parquet(OUT/"cohort_eicu.parquet")
assert "icustay_id_eicu" in coh.columns, "cohort_eicu 缺 icustay_id_eicu 列"
ids = tuple(int(x) for x in coh.icustay_id_eicu.tolist())
print(f"eICU 队列 n={len(coh)}")
if not (VENT_RESPCATS or VENT_RCLABELS or VENT_TREAT):
    sys.exit("VENT_* 字面量为空 — 先读 probe 结果再填充")

con = duckdb.connect(); con.execute("PRAGMA threads=4")
def I(lst): return "("+",".join(f"'{v}'" for v in lst)+")"

# ========== 组件: coag / liver / renal-lab (lab 表 [-360,+1440]) ==========
labwin = con.execute(f"""SELECT patientunitstayid, labname, labresultoffset tof, labresult v
FROM read_csv_auto('{(EI/'lab.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids} AND labresultoffset BETWEEN -360 AND 1440
  AND labresult IS NOT NULL
  AND labname IN ({LAB_PAO2!r},{LAB_FIO2!r},{LAB_PLT!r},{LAB_CR!r},{LAB_BILI!r})""").df()
print(f"  lab 窗口行 n={len(labwin)}")
plt_min  = labwin[labwin.labname==LAB_PLT ].groupby("patientunitstayid").v.min()
cr_max   = labwin[labwin.labname==LAB_CR  ].groupby("patientunitstayid").v.max()
bili_max = labwin[labwin.labname==LAB_BILI].groupby("patientunitstayid").v.max()
n_plt, n_cr, n_bili = len(plt_min), len(cr_max), len(bili_max)

# ========== 组件: resp (paO2 × 最近前置 FiO2 → P/F; 患者级 vent 标志) ==========
pf_rows = con.execute(f"""
WITH p AS (SELECT patientunitstayid, labresultoffset tof, labresult po2
  FROM read_csv_auto('{(EI/'lab.csv.gz').as_posix()}')
  WHERE patientunitstayid IN {ids} AND labresultoffset BETWEEN -360 AND 1440
    AND labname = {LAB_PAO2!r} AND labresult BETWEEN 1 AND 800),
f AS (SELECT patientunitstayid, labresultoffset tof,
    CASE WHEN labresult>0 AND labresult<=1 THEN labresult*100
         WHEN labresult>=21 AND labresult<=100 THEN labresult END fio2
  FROM read_csv_auto('{(EI/'lab.csv.gz').as_posix()}')
  WHERE patientunitstayid IN {ids} AND labresultoffset BETWEEN -360 AND 1440
    AND labname = {LAB_FIO2!r} AND labresult IS NOT NULL)
SELECT p.patientunitstayid, p.po2, f.fio2
FROM p ASOF LEFT JOIN f ON p.patientunitstayid=f.patientunitstayid AND p.tof>=f.tof""").df()
n_fio2_null = int(pf_rows.fio2.isna().sum())
pf = pf_rows.dropna(subset=["fio2"]).copy()
pf["pf_ratio"] = 100.0*pf.po2/pf.fio2

vent_ids = set()
if VENT_RESPCATS or VENT_RCLABELS:
    conds = []
    if VENT_RESPCATS: conds.append(f"lower(respcharttypecat) IN {I([c.lower() for c in VENT_RESPCATS])}")
    if VENT_RCLABELS: conds.append(f"lower(respchartvaluelabel) IN {I([c.lower() for c in VENT_RCLABELS])}")
    vent_ids |= set(con.execute(f"""SELECT DISTINCT patientunitstayid
FROM read_csv('{(EI/'respiratoryCharting.csv.gz').as_posix()}', quote='"')
WHERE patientunitstayid IN {ids} AND respchartoffset BETWEEN -1440 AND 1440
  AND ("""+" OR ".join(conds)+")").df().patientunitstayid.astype(int))
if VENT_TREAT:
    conds = " OR ".join(f"lower(treatmentstring) LIKE '%{t.lower()}%'" for t in VENT_TREAT)
    vent_ids |= set(con.execute(f"""SELECT DISTINCT patientunitstayid
FROM read_csv('{(EI/'treatment.csv.gz').as_posix()}', quote='"')
WHERE patientunitstayid IN {ids} AND ({conds})""").df().patientunitstayid.astype(int))
print(f"  vent 标志 n={len(vent_ids)}")

pf_min = pf.groupby("patientunitstayid").pf_ratio.min()
def resp_score(pid):
    if pid not in pf_min.index: return np.nan
    v = pf_min[pid]
    if pid in vent_ids:
        return 4 if v<100 else 3 if v<200 else 2 if v<300 else 1 if v<400 else 0
    return 2 if v<300 else 1 if v<400 else 0

# ========== 组件: cardio (MAP min + 升压药 max 剂量) ==========
mapv = con.execute(f"""SELECT patientunitstayid, min(v) mbp FROM (
SELECT patientunitstayid, systemicmean v FROM read_csv_auto('{(EI/'vitalPeriodic.csv.gz').as_posix()}')
 WHERE patientunitstayid IN {ids} AND observationoffset BETWEEN -360 AND 1440 AND systemicmean>0 AND systemicmean<300
 UNION ALL
SELECT patientunitstayid, noninvasivemean v FROM read_csv_auto('{(EI/'vitalAperiodic.csv.gz').as_posix()}')
 WHERE patientunitstayid IN {ids} AND observationoffset BETWEEN -360 AND 1440 AND noninvasivemean>0 AND noninvasivemean<300)
GROUP BY patientunitstayid""").df()
mbp_min = mapv.set_index("patientunitstayid").mbp

drugs = con.execute(f"""SELECT patientunitstayid, drugname,
  try_cast(drugrate AS DOUBLE) AS drugrate, try_cast(patientweight AS DOUBLE) AS patientweight, infusionoffset
FROM read_csv_auto('{(EI/'infusionDrug.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids} AND infusionoffset BETWEEN -360 AND 1440
  AND try_cast(drugrate AS DOUBLE) IS NOT NULL AND try_cast(drugrate AS DOUBLE) > 0
  AND (lower(drugname) LIKE 'norepinephrine%' OR lower(drugname) LIKE 'epinephrine%'
    OR lower(drugname) LIKE 'dopamine%' OR lower(drugname) LIKE 'dobutamine%')""").df()
drug_dist, excluded = {}, {}
pw_med = (drugs.groupby("patientunitstayid").patientweight.median()
          if len(drugs) else pd.Series(dtype=float))
doses = {}
for r in drugs.itertuples(index=False):
    key = str(r.drugname).strip()
    drug_dist[key] = drug_dist.get(key, 0)+1
    m = re.match(r"^([A-Za-z]+)\s*\(([^)]*)\)\s*$", key)
    if not m: excluded[key] = excluded.get(key, 0)+1; continue
    base = VASO_BASES.get(m.group(1).lower()); unit = m.group(2).strip().lower()
    if base is None or unit not in ("mcg/kg/min","mcg/min"):
        excluded[key] = excluded.get(key, 0)+1; continue
    rate = float(r.drugrate)
    if unit == "mcg/min":
        w = r.patientweight if pd.notna(r.patientweight) else pw_med.get(r.patientunitstayid, np.nan)
        if pd.isna(w) or w<=0:
            excluded[key+" [无体重]"] = excluded.get(key+" [无体重]", 0)+1; continue
        rate /= float(w)
    pid = int(r.patientunitstayid)
    doses.setdefault(pid, {})
    cls = base   # base 已是 VASO_BASES 映射后的短类名 (norepi/epi/dopa/doba)
    doses[pid][cls] = max(doses[pid].get(cls, 0.0), rate)
n_vaso = len(doses)

def cardio_score(pid):
    d = doses.get(pid, {})
    nr, ep, da, db = d.get("norepi"), d.get("epi"), d.get("dopa"), d.get("doba")
    if (da is not None and da>15) or (ep is not None and ep>0.1) or (nr is not None and nr>0.1): return 4
    if (da is not None and da>5) or (ep is not None and 0<ep<=0.1) or (nr is not None and 0<nr<=0.1): return 3
    if (da is not None and da>0) or (db is not None and db>0): return 2
    if pid in mbp_min.index: return 1 if mbp_min[pid]<70 else 0
    return np.nan

# ========== 组件: CNS (gcs_long_eicu, [0,24h] min) ==========
gcs = pd.read_parquet(OUT/"gcs_long_eicu.parquet")
gcs["id"] = gcs.id.astype(int)
inter = len(set(gcs.id) & set(coh.icustay_id_eicu.astype(int)))
assert inter >= 0.5*len(coh), f"gcs_long_eicu id 与队列交集过小 ({inter}/{len(coh)})"
mingcs = gcs[(gcs.offset_hr>=0)&(gcs.offset_hr<=24)].groupby("id").gcs_total.min()
n_gcs = len(mingcs)

# ========== 组件: renal (creatinine max + UO sum) ==========
uo_raw = con.execute(f"""SELECT patientunitstayid, celllabel, cellvaluenumeric v
FROM read_csv_auto('{(EI/'intakeOutput.csv.gz').as_posix()}')
WHERE patientunitstayid IN {ids} AND intakeoutputoffset BETWEEN 0 AND 1440
  AND cellvaluenumeric IS NOT NULL AND cellvaluenumeric > 0 AND celllabel ILIKE '%urine%'""").df()
uo_excl_counts = uo_raw[~uo_raw.celllabel.isin(UO_LABELS+UO_EXCLUDED)].celllabel.value_counts().to_dict()
sel = uo_raw[uo_raw.celllabel.isin(UO_LABELS)]
uo_by_pt = sel.groupby("patientunitstayid").v.sum().to_dict()
n_uo = len(uo_by_pt)

# ========== 汇总评分 ==========
rows = []
for pid in coh.icustay_id_eicu.astype(int):
    rs = resp_score(pid)
    if pid in mingcs.index:
        gg = mingcs[pid]; cs = 1 if gg<15 else 2 if gg<13 else 3 if gg<10 else 4 if gg<6 else 0
    else: cs = np.nan
    if pid in plt_min.index:
        pv = plt_min[pid]; ks = 4 if pv<20 else 3 if pv<50 else 2 if pv<100 else 1 if pv<150 else 0
    else: ks = np.nan
    if pid in bili_max.index:
        bv = bili_max[pid]; lv = 4 if bv>=12 else 3 if bv>=6 else 2 if bv>=2 else 1 if bv>=1.2 else 0
    else: lv = np.nan
    crv = cr_max.get(pid); u = uo_by_pt.get(pid)
    if crv is None and u is None: rn = np.nan
    else:
        u4 = u is not None and u<200; u3 = u is not None and u<500
        if   (crv is not None and crv>=5) or u4: rn = 4
        elif (crv is not None and crv>=3.5) or u3: rn = 3
        elif crv is not None and crv>=2: rn = 2
        elif crv is not None and crv>=1.2: rn = 1
        else: rn = 0
    ca_ = cardio_score(pid)
    comp = dict(respiration=rs, coagulation=ks, liver=lv, cardiovascular=ca_, cns=cs, renal=rn)
    sofa = sum(0 if pd.isna(v) else v for v in comp.values())
    rows.append(dict(icustay_id_eicu=pid, sofa=int(sofa),
                     **{k: int(v) if pd.notna(v) else -1 for k, v in comp.items()}))
out = pd.DataFrame(rows)

# ========== 硬门禁 ==========
comp_cols = ["respiration","coagulation","liver","cardiovascular","cns","renal"]
assert len(out)==len(coh)==4440, f"行数 {len(out)} != 4440"
assert out.icustay_id_eicu.is_unique, "icustay_id_eicu 不唯一"
for c in comp_cols:
    assert out[c].between(-1,4).all(), f"{c} 越界"
assert out.sofa.between(0,24).all(), "sofa 越界"
assert (out.sofa == out[comp_cols].apply(lambda r: sum(v for v in r if v>0), axis=1)).all(), "sofa != 组件和"
covered = int((out[comp_cols]>-1).any(axis=1).sum())
assert covered/len(out) >= 0.90, f"总覆盖 {covered}/{len(out)} < 90%"

out.to_parquet(OUT/"08i_sofa_eicu.parquet", index=False)
cov = {c: int((out[c]>-1).sum()) for c in comp_cols}
report = dict(
  n=len(out), source="eicu-crd-2.0 + 07b cohort_eicu (n=4440)",
  window_convention="08z: 测量[-360,+1440]min, UO[0,+1440]min, GCS[0,24h]",
  component_coverage=cov, any_component_coverage=f"{covered}/{len(out)}",
  thresholds="对齐 mimic-code sofa.sql (severityscores)",
  vent_flag=dict(n_vent=int(len(vent_ids)), level="患者级0/24h (偏离官方行级状态机, 已披露)",
                 resp_cats=VENT_RESPCATS, rc_labels=VENT_RCLABELS, treatment=VENT_TREAT),
  vaso=dict(n_scored=n_vaso,
            unit_rule="mcg/kg/min 直用; mcg/min÷行内patientweight(缺→该患者vaso行中位体重); 其余除外",
            excluded_counts=excluded, observed_literals=drug_dist),
  pf=dict(n_paO2_without_preceding_FiO2=n_fio2_null, n_pf_rows=len(pf),
          fio2_unit_rule="≤1→×100; 21-100 保留; 其余 NULL 丢弃"),
  uo=dict(labels_in=UO_LABELS, labels_known_excluded=UO_EXCLUDED, other_labels_seen=uo_excl_counts),
  cns=dict(source="gcs_long_eicu.parquet (07a; ETT verbal=0→15 已实现)", n=n_gcs,
           id_overlap_with_cohort=inter),
  labs=dict(plt=n_plt, creatinine=n_cr, bilirubin=n_bili),
  deviations_disclosed=[
    "窗口统一 08z 约定 (官方 M3 各组件窗口不同)",
    "FiO2 仅 lab 表 + 最近前置 ASOF 配对 (无回看上限) + 双单位规则 (≤1 分数/21-100 百分数)",
    "通气患者级标志 (官方为行级 ventnum 状态机); vent 患者按总体最小 P/F 走 4 档梯",
    "升压药仅 base 精确匹配 + mcg/kg/min|mcg/min; ml/hr|mg/kg/min|MAX|STD|裸名除外",
    "CNS 复用 07a gcs_long 调和产物",
    "missing 组件计 0 分 (官方 coalesce 语义), 覆盖率全量披露"],
  provenance="literals: reports/08i_probe_{lab,drug,uo,respchart,treatment}.csv (实测落盘)")
(REP/"08i_sofa_eicu.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
print("GATES PASS — 08i_sofa_eicu.parquet + JSON 已落盘")
print(cov)
print(f"SOFA 分布: mean={out.sofa.mean():.2f} median={out.sofa.median():.0f} range=[{out.sofa.min()},{out.sofa.max()}]")
