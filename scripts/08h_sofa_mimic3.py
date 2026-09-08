# -*- coding: utf-8 -*-
# 08h: MIMIC-III first-day SOFA (24h landmark) — 外库 ΔAUC 闭合前置件 (SAP §6 base5 需 sofa)
# 移植来源 (2026-09-03 实拉, 非记忆): MIT-LCP/mimic-code @main, mimic-iii 分支:
#   concepts/severityscores/sofa.sql             — vaso CV/MV itemid + 体重换算 + 组件 CASE 阈值
#   concepts/firstday/labs_first_day.sql         — bili 50885 / cr 50912 / plt 51265 + 值域
#   concepts/firstday/vitals_first_day.sql       — MeanBP itemid 456,52,6702,443,220052,220181,225312
#   concepts/firstday/urine_output_first_day.sql — 26 UO itemid + GU irrigant 227488 取负
#   concepts/firstday/gcs_first_day.sql          — ETT verbal=0→15 + 6h 结转 + min
#   concepts/firstday/blood_gas_first_day(.sql/_arterial.sql) — bg panel + FiO2 4h 回看 + specimen='ART'
#   concepts/durations/ventilation_classification(.sql)/ventilation_durations.sql — 8h 合并状态机
# 与官方的已披露偏离 (Methods 披露口径):
#   1) 窗口统一为 08z 口径: 测量类 [-6h,+24h], UO [intime,+24h], weight [-1d,+24h](官方[-1d,+1d])
#      (M3 官方 vitals/gcs/vaso 为 [intime,+24h]; 统一化使三库 SOFA 口径一致)
#   2) 动脉血气仅 specimen='ART' (官方另有 SPECIMEN_PROB>0.75 回归补判, 不采用, 缺失计数披露)
#   3) weight 的 echo_data 回退不采用 (官方 echo2 CTE); CV mcgmin 无体重可换算的 rate 记 null 并计数
#   4) vent 设定扫描窗 [-24h,+24h] (官方全住院时长; 我们只需 landmark 附近 isvent)
# 所有 itemid 运行时对 D_ITEMS/D_LABITEMS 实测存在性核证 (防抄写错位), 映射表落盘备审
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M3   = ROOT/"data/mimic-iii-1.4"
OUT  = ROOT/"07_prediction_system/data"
REP  = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
def lc(df):  # M3 CSV 表头全大写 → 统一小写 (SQL 内引用大小写不敏感, pandas 侧需小写)
    df.columns = [str(c).lower() for c in df.columns]
    return df

coh = pd.read_parquet(OUT/"cohort_mimic3.parquet")
need_cols = {"icustay_id","subject_id","intime"}
assert need_cols <= set(coh.columns), f"cohort_mimic3 缺列 {need_cols - set(coh.columns)}"
coh = coh[["icustay_id","subject_id","intime"]].copy()
coh["icustay_id"] = coh.icustay_id.astype(int); coh["subject_id"] = coh.subject_id.astype(int)
coh["intime"] = pd.to_datetime(coh.intime)

# CareVue-only 过滤 (2026-09-07): MIMIC-III MetaVision 期 (2008-2012) 与 MIMIC-IV train (2008-2017)
# 患者重叠 (PhysioNet 官方确认), 仅保留 CareVue 期 (2001-2008, 不同 EHR 系统)
_cv = lc(con.execute(f"""SELECT ICUSTAY_ID AS icustay_id, DBSOURCE AS dbsource
FROM read_csv_auto('{(M3/'ICUSTAYS.csv.gz').as_posix()}')
WHERE DBSOURCE = 'carevue'""").df())
_cv["icustay_id"] = _cv.icustay_id.astype(int)
_n_all = len(coh)
coh = coh[coh.icustay_id.isin(_cv.icustay_id)].copy()
print(f"[CareVue 过滤] {_n_all}→{len(coh)} (剔 MetaVision 重叠 {_n_all - len(coh)})")

ids = tuple(coh.icustay_id.tolist())
print(f"M3 队列 n={len(coh)} (CareVue-only, 07b 产物+CareVue 过滤)")

# ========== hadm_id (LABEVENTS 按 subject+hadm join, 官方口径) ==========
icu = lc(con.execute(f"""SELECT ICUSTAY_ID AS icustay_id, HADM_ID AS hadm_id
FROM read_csv_auto('{(M3/'ICUSTAYS.csv.gz').as_posix()}')
WHERE ICUSTAY_ID IN {ids}""").df())
assert len(icu)==len(coh) and icu.icustay_id.is_unique, f"ICUSTAYS join 异常 n={len(icu)}"
coh = coh.merge(icu, on="icustay_id", how="left")
assert coh.hadm_id.notna().all()
coh["hadm_id"] = coh.hadm_id.astype(int)
coh.to_parquet(OUT/"_tmp_08h_coh.parquet", index=False)
COH = (OUT/"_tmp_08h_coh.parquet").as_posix()

# ========== itemid 存在性实测核证 (D_ITEMS / D_LABITEMS) ==========
d_items = lc(con.execute(f"SELECT ITEMID AS itemid, LABEL AS label FROM read_csv_auto('{(M3/'D_ITEMS.csv.gz').as_posix()}')").df())
d_lab   = lc(con.execute(f"SELECT ITEMID AS itemid, LABEL AS label FROM read_csv_auto('{(M3/'D_LABITEMS.csv.gz').as_posix()}')").df())
DI = dict(zip(d_items.itemid.astype(int), d_items.label))
DL = dict(zip(d_lab.itemid.astype(int),   d_lab.label))

MEANBP = [456,52,6702,443,220052,220181,225312]
GCS_V, GCS_M, GCS_E = [723,223900],[454,223901],[184,220739]
WT_KG, WT_LB, WT_OZ = [762,763,3723,3580,226512],[3581],[3582]
FIO2_CE = [3420,190,223835,3422]
VASO_CV_NORPI, VASO_CV_EPI, VASO_CV_DA, VASO_CV_DOBA = [30047,30120],[30044,30119,30309],[30043,30307],[30042,30306]
VASO_MV = [221906,221289,221662,221653]   # norepi, epi, dopamine, dobutamine
VENT_SETTING = [720,223849,223848,445,448,449,450,1340,1486,1600,224687,
 639,654,681,682,683,684,224685,224684,224686,218,436,535,444,224697,224695,224696,224746,224747,
 221,1,1211,1655,2000,226873,224738,224419,224750,227187,543,5865,5866,224707,224709,224705,224706,
 60,437,505,506,686,220339,224700,3459,501,502,503,224702,223,667,668,669,670,671,672,224701]
VENT_EXT, VENT_O2 = [640],[468,469,470,471,227287,226732,223834,467]
PROC_EXT = [227194,225468,225477]
UO_ITEMS = [40055,43175,40069,40094,40715,40473,40085,40057,40056,40405,40428,40086,40096,40651,
 226559,226560,226561,226584,226563,226564,226565,226567,226557,226558,227488,227489]
LAB_BG   = list(range(50800,50829)) + [51545]
LAB_SOFA = [50885,50912,51265]     # bili, cr, plt

audit, bad = [], []
def verify(iids, where, concept):
    for i in iids:
        src = DI if where=="D_ITEMS" else DL
        if i not in src: bad.append((concept, i, where, "MISSING"))
        else: audit.append(dict(concept=concept, src=where, itemid=i, label=src[i]))
verify(MEANBP+GCS_V+GCS_M+GCS_E+WT_KG+WT_LB+WT_OZ+FIO2_CE+VENT_SETTING+VENT_EXT+VENT_O2
       +VASO_CV_NORPI+VASO_CV_EPI+VASO_CV_DA+VASO_CV_DOBA+VASO_MV+UO_ITEMS+PROC_EXT, "D_ITEMS", "chart")
verify(LAB_BG+LAB_SOFA, "D_LABITEMS", "lab")
pd.DataFrame(audit).to_csv(REP/"08h_itemid_audit.csv", index=False)
if bad:
    (REP/"08h_sofa_mimic3.json").write_text(json.dumps({"STATUS":"ABORT_itemid_verify_fail","bad":bad}, ensure_ascii=False, indent=1))
    print(f"[GATE FAIL] itemid 核证未过: {bad}"); sys.exit(1)
print(f"itemid 核证: {len(audit)} 条全命中 (audit → 08h_itemid_audit.csv)")

CE_ITEMS = MEANBP+GCS_V+GCS_M+GCS_E+WT_KG+WT_LB+WT_OZ+FIO2_CE+VENT_SETTING+VENT_EXT+VENT_O2
def I(lst): return "("+",".join(map(str,lst))+")"

# ========== PASS1: CHARTEVENTS 单扫物化 [intime-1d, intime+24h] ==========
print("PASS1 CHARTEVENTS 物化 ([-1d,+24h], 全概念 itemid)...")
con.execute(f"""CREATE TABLE ce_win AS
SELECT c.icustay_id, c.charttime, c.itemid, c.value, c.valuenum
FROM read_csv('{(M3/'CHARTEVENTS.csv.gz').as_posix()}', types = {{'VALUE': 'VARCHAR'}}) c
JOIN read_parquet('{COH}') k ON c.icustay_id = k.icustay_id
WHERE c.itemid IN {I(CE_ITEMS)}
  AND c.charttime BETWEEN k.intime - INTERVAL 24 HOUR AND k.intime + INTERVAL 24 HOUR
  AND (c.error IS NULL OR c.error != 1)""")
n_ce = lc(con.execute("SELECT count(*) n, count(DISTINCT icustay_id) n_stay FROM ce_win").df())
print(f"  ce_win rows={int(n_ce.n[0])} stays={int(n_ce.n_stay[0])}")

# ========== PASS2: LABEVENTS [-6h,+24h] (bg panel + SOFA labs) ==========
print("PASS2 LABEVENTS 物化...")
con.execute(f"""CREATE TABLE le_win AS
SELECT l.subject_id, l.hadm_id, l.charttime, l.itemid, l.value, l.valuenum
FROM read_csv('{(M3/'LABEVENTS.csv.gz').as_posix()}', types = {{'VALUE': 'VARCHAR'}}) l
JOIN read_parquet('{COH}') k ON l.subject_id = k.subject_id AND l.hadm_id = k.hadm_id
WHERE l.itemid IN {I(LAB_BG + LAB_SOFA)}
  AND l.charttime BETWEEN k.intime - INTERVAL 6 HOUR AND k.intime + INTERVAL 24 HOUR
  AND l.valuenum IS NOT NULL AND l.valuenum > 0""")
n_le = lc(con.execute("SELECT count(*) n, count(DISTINCT subject_id) n_pat FROM le_win").df())
print(f"  le_win rows={int(n_le.n[0])} patients={int(n_le.n_pat[0])}")

# ========== PASS3: INPUTEVENTS_CV / MV, OUTPUTEVENTS, PROCEDUREEVENTS_MV ==========
print("PASS3 输入/输出/操作事件...")
VASO_CV_ALL = VASO_CV_NORPI+VASO_CV_EPI+VASO_CV_DA+VASO_CV_DOBA
con.execute(f"""CREATE TABLE cv_win AS
SELECT c.icustay_id, c.itemid, c.rate
FROM read_csv_auto('{(M3/'INPUTEVENTS_CV.csv.gz').as_posix()}') c
JOIN read_parquet('{COH}') k ON c.icustay_id = k.icustay_id
WHERE c.itemid IN {I(VASO_CV_ALL)} AND c.rate IS NOT NULL
  AND c.charttime BETWEEN k.intime - INTERVAL 6 HOUR AND k.intime + INTERVAL 24 HOUR""")
con.execute(f"""CREATE TABLE mv_win AS
SELECT m.icustay_id, m.itemid, m.rate
FROM read_csv_auto('{(M3/'INPUTEVENTS_MV.csv.gz').as_posix()}') m
JOIN read_parquet('{COH}') k ON m.icustay_id = k.icustay_id
WHERE m.itemid IN {I(VASO_MV)} AND m.rate IS NOT NULL AND m.statusdescription != 'Rewritten'
  AND m.starttime BETWEEN k.intime - INTERVAL 6 HOUR AND k.intime + INTERVAL 24 HOUR""")
con.execute(f"""CREATE TABLE uo_win AS
SELECT o.icustay_id, o.itemid, try_cast(o.value AS DOUBLE) AS value
FROM read_csv('{(M3/'OUTPUTEVENTS.csv.gz').as_posix()}', types = {{'VALUE': 'VARCHAR'}}) o
JOIN read_parquet('{COH}') k ON o.icustay_id = k.icustay_id
WHERE o.itemid IN {I(UO_ITEMS)} AND o.value IS NOT NULL
  AND o.charttime BETWEEN k.intime AND k.intime + INTERVAL 24 HOUR""")
con.execute(f"""CREATE TABLE pe_win AS
SELECT p.icustay_id, p.starttime
FROM read_csv_auto('{(M3/'PROCEDUREEVENTS_MV.csv.gz').as_posix()}') p
WHERE p.icustay_id IN {ids} AND p.itemid IN {I(PROC_EXT)}""")
for t in ["cv_win","mv_win","uo_win","pe_win"]:
    n = lc(con.execute(f"SELECT count(*) n, count(DISTINCT icustay_id) s FROM {t}").df())
    print(f"  {t}: rows={int(n.n[0])} stays={int(n.s[0])}")

# ========== 组件: weight / meanbp / labs / uo ============
wt = lc(con.execute(f"""SELECT icustay_id, avg(CASE
  WHEN itemid IN {I(WT_KG)} THEN valuenum
  WHEN itemid IN {I(WT_LB)} THEN valuenum*0.45359237
  WHEN itemid IN {I(WT_OZ)} THEN valuenum*0.0283495231 END) weight
FROM ce_win WHERE itemid IN {I(WT_KG+WT_LB+WT_OZ)} AND valuenum IS NOT NULL AND valuenum != 0
GROUP BY icustay_id""").df())
mbp = lc(con.execute(f"""SELECT c.icustay_id AS icustay_id, min(c.valuenum) AS meanbp_min FROM ce_win c
JOIN read_parquet('{COH}') k ON c.icustay_id = k.icustay_id
WHERE c.itemid IN {I(MEANBP)} AND c.valuenum > 0 AND c.valuenum < 300
  AND c.charttime BETWEEN k.intime - INTERVAL 6 HOUR AND k.intime + INTERVAL 24 HOUR
GROUP BY c.icustay_id""").df())
labs = lc(con.execute(f"""SELECT k.icustay_id AS icustay_id,
  max(CASE WHEN l.itemid=50885 AND l.valuenum<=150 THEN l.valuenum END) bilirubin_max,
  max(CASE WHEN l.itemid=50912 AND l.valuenum<=150 THEN l.valuenum END) creatinine_max,
  min(CASE WHEN l.itemid=51265 AND l.valuenum<=10000 THEN l.valuenum END) platelet_min
FROM le_win l JOIN read_parquet('{COH}') k
  ON l.subject_id=k.subject_id AND l.hadm_id=k.hadm_id
WHERE l.itemid IN {I(LAB_SOFA)} GROUP BY k.icustay_id""").df())
uo = lc(con.execute(f"""SELECT icustay_id AS icustay_id,
  sum(CASE WHEN itemid=227488 AND value>0 THEN -value ELSE value END) AS urineoutput
FROM uo_win GROUP BY icustay_id""").df())

# ========== 组件: GCS (官方状态机: ETT verbal=0→15, 6h 结转, min) ============
gcs = lc(con.execute(f"""SELECT icustay_id AS icustay_id, charttime AS charttime,
  max(CASE WHEN itemid IN {I(GCS_M)} THEN CASE WHEN itemid=723 AND value='1.0 ET/Trach' THEN 0
         WHEN itemid=223900 AND value='No Response-ETT' THEN 0 ELSE valuenum END END) gcs_motor,
  max(CASE WHEN itemid IN {I(GCS_V)} THEN CASE WHEN itemid=723 AND value='1.0 ET/Trach' THEN 0
         WHEN itemid=223900 AND value='No Response-ETT' THEN 0 ELSE valuenum END END) gcs_verbal,
  max(CASE WHEN itemid IN {I(GCS_E)} THEN CASE WHEN itemid=723 AND value='1.0 ET/Trach' THEN 0
         WHEN itemid=223900 AND value='No Response-ETT' THEN 0 ELSE valuenum END END) gcs_eyes
FROM ce_win WHERE itemid IN {I(GCS_V+GCS_M+GCS_E)} GROUP BY icustay_id, charttime""").df())
intime_map = coh.set_index("icustay_id").intime
gcs["icustay_id"] = gcs.icustay_id.astype(int)
gcs["charttime"] = pd.to_datetime(gcs.charttime)
gcs = gcs[(gcs.charttime >= gcs.icustay_id.map(intime_map) - pd.Timedelta(hours=6)) &
          (gcs.charttime <= gcs.icustay_id.map(intime_map) + pd.Timedelta(hours=24))].copy()
gcs = gcs.sort_values(["icustay_id","charttime"]).reset_index(drop=True)
for c in ["gcs_motor","gcs_verbal","gcs_eyes"]:
    gcs[c] = pd.to_numeric(gcs[c], errors="coerce")
gcs["prev_verbal"] = gcs.groupby("icustay_id").gcs_verbal.shift(1)
gcs["prev_motor"]  = gcs.groupby("icustay_id").gcs_motor.shift(1)
gcs["prev_eyes"]   = gcs.groupby("icustay_id").gcs_eyes.shift(1)
gcs["prev_time"]   = gcs.groupby("icustay_id").charttime.shift(1)
within6 = (gcs.prev_time.notna()) & (gcs.charttime - gcs.prev_time <= pd.Timedelta(hours=6))
pv = gcs.prev_verbal.where(within6); pm = gcs.prev_motor.where(within6); pe = gcs.prev_eyes.where(within6)
def _gcs_row(v0, m0, e0, pvv, pmm, pee):
    if v0 == 0: return 15                                    # 当前插管 → 15 (官方)
    if pd.isna(v0) and pvv == 0: return 15                   # 结转自插管行 → 15 (官方)
    m = m0 if pd.notna(m0) else 6; e = e0 if pd.notna(e0) else 4
    if pvv == 0: return m + 5 + e                            # 曾插管现拔管: 不用结转 verbal, 缺省 5
    v = v0 if pd.notna(v0) else (pvv if pd.notna(pvv) else 5)
    mm = m0 if pd.notna(m0) else (pmm if pd.notna(pmm) else 6)
    ee = e0 if pd.notna(e0) else (pee if pd.notna(pee) else 4)
    return mm + v + ee
gcs["gcs_total"] = [_gcs_row(*t) for t in zip(gcs.gcs_verbal, gcs.gcs_motor, gcs.gcs_eyes, pv, pm, pe)]
mingcs = gcs.groupby("icustay_id").gcs_total.min().rename("mingcs").reset_index()
print(f"  GCS 覆盖 {len(mingcs)}/{len(coh)}")

# ========== 组件: 动脉血气 P/F (specimen 存于 VALUE 字符串列 → 单独物化 spec_win, 不做 valuenum 过滤) ============
con.execute(f"""CREATE TABLE spec_win AS
SELECT l.subject_id, l.hadm_id, l.charttime, upper(trim(l.value)) AS specimen
FROM read_csv('{(M3/'LABEVENTS.csv.gz').as_posix()}', types = {{'VALUE': 'VARCHAR'}}) l
JOIN read_parquet('{COH}') k ON l.subject_id = k.subject_id AND l.hadm_id = k.hadm_id
WHERE l.itemid = 50800
  AND l.charttime BETWEEN k.intime - INTERVAL 6 HOUR AND k.intime + INTERVAL 24 HOUR""")
spec_dist = lc(con.execute("SELECT specimen, count(*) n FROM spec_win GROUP BY 1 ORDER BY n DESC").df())
print(f"  specimen 分布: {dict(zip(spec_dist.specimen.head(6), spec_dist.n.head(6).astype(int)))}")
bg = lc(con.execute(f"""SELECT k.icustay_id AS icustay_id, b.charttime AS charttime,
  max(s.specimen) specimen, max(b.po2) po2, max(b.fio2_lab) fio2_lab
FROM (SELECT subject_id, hadm_id, charttime,
        max(CASE WHEN itemid=50821 AND valuenum<=800 THEN valuenum END) po2,
        max(CASE WHEN itemid=50816 AND valuenum BETWEEN 20 AND 100 THEN valuenum END) fio2_lab
      FROM le_win WHERE itemid IN (50816,50821) GROUP BY subject_id, hadm_id, charttime) b
LEFT JOIN spec_win s ON b.subject_id=s.subject_id AND b.hadm_id=s.hadm_id AND b.charttime=s.charttime
JOIN read_parquet('{COH}') k ON b.subject_id=k.subject_id AND b.hadm_id=k.hadm_id
GROUP BY k.icustay_id, b.charttime""").df())
bg["icustay_id"]=bg.icustay_id.astype(int); bg["charttime"]=pd.to_datetime(bg.charttime)
n_spec_null = int(bg.specimen.isna().sum())
bg = bg[bg.specimen.str.startswith("ART", na=False) & bg.po2.notna()].copy()   # 已披露偏离 2: ART 前缀匹配(含 'ART.'/'ARTERIAL'), 实测分布入 JSON
fce = lc(con.execute(f"""SELECT icustay_id AS icustay_id, charttime AS charttime, max(CASE
  WHEN itemid IN (3420,3422) THEN valuenum
  WHEN itemid=190 AND valuenum>0.20 AND valuenum<1 THEN valuenum*100
  WHEN itemid=223835 AND valuenum>0 AND valuenum<=1 THEN valuenum*100
  WHEN itemid=223835 AND valuenum>=21 AND valuenum<=100 THEN valuenum END) fio2_ce
FROM ce_win WHERE itemid IN {I(FIO2_CE)} AND valuenum IS NOT NULL
GROUP BY icustay_id, charttime""").df())
fce["icustay_id"]=fce.icustay_id.astype(int); fce["charttime"]=pd.to_datetime(fce.charttime)
bg = bg.sort_values(["icustay_id","charttime"]).reset_index(drop=True)
pf_rows = []
fce_g = dict(tuple(fce.groupby("icustay_id")))
for sid, grp in bg.groupby("icustay_id"):
    sub = fce_g.get(sid)
    for _, r in grp.iterrows():
        f = r.fio2_lab
        if pd.isna(f) and sub is not None and len(sub):
            cand = sub[(sub.charttime >= r.charttime - pd.Timedelta(hours=4)) & (sub.charttime <= r.charttime)]
            if len(cand): f = cand.sort_values("charttime").fio2_ce.iloc[-1]
        if pd.notna(f) and f > 0:
            pf_rows.append((sid, r.charttime, 100.0*r.po2/f))
pafi = pd.DataFrame(pf_rows, columns=["icustay_id","charttime","pao2fio2"])
print(f"  动脉血气: bg 组 specimen 缺失弃 {n_spec_null} 行 | ART+P/F 可算 {len(pafi)} 行, 覆盖 {pafi.icustay_id.nunique()} stays")

# ========== 组件: 通气状态机 (官方 8h 合并) → isvent @ bg 时间 ============
vc = lc(con.execute(f"""SELECT icustay_id AS icustay_id, charttime AS charttime,
  max(CASE WHEN (itemid=720 AND value!='Other/Remarks') OR itemid=223849 OR (itemid=223848 AND value!='Other')
    OR (itemid=467 AND value='Ventilator') OR itemid IN {I(VENT_SETTING)} THEN 1 ELSE 0 END) mechvent,
  max(CASE WHEN (itemid=226732 AND value IN ('Nasal cannula','Face tent','Aerosol-cool','Trach mask ','High flow neb',
      'Non-rebreather','Venti mask ','Medium conc mask ','T-piece','High flow nasal cannula','Ultrasonic neb','Vapomist'))
    OR (itemid=467 AND value IN ('Cannula','Nasal Cannula','Face Tent','Aerosol-Cool','Trach Mask','Hi Flow Neb',
      'Non-Rebreather','Venti Mask','Medium Conc Mask','Vapotherm','T-Piece','Hood','Hut','TranstrachealCat','Heated Neb','Ultrasonic Neb'))
    THEN 1 ELSE 0 END) oxytherapy,
  max(CASE WHEN itemid=640 AND value IN ('Extubated','Self Extubation') THEN 1 ELSE 0 END) extubated
FROM ce_win WHERE itemid IN {I(VENT_SETTING+VENT_EXT+VENT_O2)} GROUP BY icustay_id, charttime""").df())
vc["icustay_id"]=vc.icustay_id.astype(int); vc["charttime"]=pd.to_datetime(vc.charttime)
it = coh.set_index("icustay_id").intime
vc = vc[(vc.charttime >= vc.icustay_id.map(it) - pd.Timedelta(hours=24)) &
        (vc.charttime <= vc.icustay_id.map(it) + pd.Timedelta(hours=24))].copy()   # 已披露偏离 4
pe = lc(con.execute("SELECT icustay_id AS icustay_id, starttime AS charttime FROM pe_win").df())
if len(pe):
    pe["icustay_id"]=pe.icustay_id.astype(int); pe["charttime"]=pd.to_datetime(pe.charttime)
    pe["mechvent"]=0; pe["oxytherapy"]=0; pe["extubated"]=1
    vc = pd.concat([vc, pe[["icustay_id","charttime","mechvent","oxytherapy","extubated"]]], ignore_index=True)
vc = vc.groupby(["icustay_id","charttime"], as_index=False)[["mechvent","oxytherapy","extubated"]].max()
vc = vc.sort_values(["icustay_id","charttime"]).reset_index(drop=True)
# --- durations (官方 vd0-2 pandas 等价, 按患者分区) ---
vc["charttime_lag"] = vc.groupby(["icustay_id","mechvent"]).charttime.shift(1).where(vc.mechvent==1)
active = vc[(vc.mechvent==1)|(vc.extubated==1)].sort_values(["icustay_id","charttime","extubated"]).copy()
active["extubated_lag"] = active.groupby("icustay_id").extubated.shift(1)
vc["extubated_lag"] = pd.Series(active.extubated_lag.values, index=active.index)
def _newvent(r):
    if r.extubated_lag == 1: return 1
    if r.mechvent==0 and r.oxytherapy==1: return 1
    if pd.notna(r.charttime_lag) and r.charttime > r.charttime_lag + pd.Timedelta(hours=8): return 1
    return 0
vc["newvent"] = vc.apply(_newvent, axis=1)
act = vc.loc[(vc.mechvent==1)|(vc.extubated==1)]
vc.loc[act.index, "ventnum"] = act.groupby(act.icustay_id).newvent.cumsum()
dur = vc[vc.ventnum.notna()].groupby(["icustay_id","ventnum"]).agg(
    starttime=("charttime","min"), endtime=("charttime","max"), any_mech=("mechvent","max")).reset_index()
dur = dur[(dur.starttime!=dur.endtime) & (dur.any_mech==1)]
print(f"  通气段: {len(dur)} 段, 覆盖 {dur.icustay_id.nunique()} stays")
# isvent: 同一 bg 时间对多段取 OR (防重复行污染 novent 列)
pairs = pafi[["icustay_id","charttime"]].drop_duplicates().merge(
    dur[["icustay_id","starttime","endtime"]], on="icustay_id", how="left")
if len(pairs):
    pairs["cover"] = ((pairs.charttime>=pairs.starttime)&(pairs.charttime<=pairs.endtime)).fillna(False)
    isv = pairs.groupby(["icustay_id","charttime"], as_index=False).cover.max().rename(columns={"cover":"isvent"})
    pafi = pafi.merge(isv, on=["icustay_id","charttime"], how="left")
    pafi["isvent"] = pafi.isvent.fillna(False).astype(int)
else:
    pafi["isvent"] = 0
pf2 = pafi.groupby("icustay_id", as_index=False).agg(
    pao2fio2_vent_min=("pao2fio2", lambda s: s[pafi.loc[s.index,"isvent"]==1].min()),
    pao2fio2_novent_min=("pao2fio2", lambda s: s[pafi.loc[s.index,"isvent"]==0].min()))

# ========== 组件: 升压药 (CV 体重换算 + MV, coalesce CV>MV 官方) ============
cv_raw = lc(con.execute("SELECT icustay_id AS icustay_id, itemid AS itemid, rate AS rate FROM cv_win").df())
cv_raw["icustay_id"]=cv_raw.icustay_id.astype(int); cv_raw["rate"]=cv_raw.rate.astype(float)
n_cv_needW = int(cv_raw.itemid.isin([30047,30044]).sum())
wmap = wt.set_index("icustay_id").weight if len(wt) else pd.Series(dtype=float)
cv_raw["w"] = cv_raw.icustay_id.map(wmap)
n_cv_noW = int((cv_raw.itemid.isin([30047,30044]) & cv_raw.w.isna()).sum())
def vcv(r):
    if r.itemid==30047: return r.rate/r.w if pd.notna(r.w) and r.w>0 else None
    if r.itemid==30044: return r.rate/r.w if pd.notna(r.w) and r.w>0 else None
    if r.itemid in (30120,30119,30309,30043,30307,30042,30306): return r.rate
    return None
cv_raw["val"] = cv_raw.apply(vcv, axis=1)
vaso_cv = cv_raw.dropna(subset=["val"]).groupby("icustay_id", as_index=False).agg(
    rate_norepinephrine=("val", lambda s: s[cv_raw.loc[s.index,"itemid"].isin([30047,30120])].max()),
    rate_epinephrine=("val", lambda s: s[cv_raw.loc[s.index,"itemid"].isin([30044,30119,30309])].max()),
    rate_dopamine=("val", lambda s: s[cv_raw.loc[s.index,"itemid"].isin([30043,30307])].max()),
    rate_dobutamine=("val", lambda s: s[cv_raw.loc[s.index,"itemid"].isin([30042,30306])].max()))
mvr = lc(con.execute("""SELECT icustay_id,
  max(CASE WHEN itemid=221906 THEN rate END) rate_norepinephrine,
  max(CASE WHEN itemid=221289 THEN rate END) rate_epinephrine,
  max(CASE WHEN itemid=221662 THEN rate END) rate_dopamine,
  max(CASE WHEN itemid=221653 THEN rate END) rate_dobutamine
FROM mv_win GROUP BY icustay_id""").df())
if len(mvr): mvr["icustay_id"]=mvr.icustay_id.astype(int)
DRUGS = ["rate_norepinephrine","rate_epinephrine","rate_dopamine","rate_dobutamine"]
vaso = coh[["icustay_id"]].merge(vaso_cv, on="icustay_id", how="left", suffixes=("_cv",""))
if len(vaso_cv) and len(mvr):
    vaso = vaso.merge(mvr, on="icustay_id", how="left", suffixes=("_cv","_mv"))
    for d in DRUGS: vaso[d] = vaso[f"{d}_cv"].fillna(vaso[f"{d}_mv"])
else:
    src = vaso_cv if len(vaso_cv) else (mvr if len(mvr) else pd.DataFrame())
    if len(src):
        vaso = coh[["icustay_id"]].merge(src, on="icustay_id", how="left")
    else:
        vaso = coh[["icustay_id"]].copy()
    for d in DRUGS:
        if d not in vaso.columns: vaso[d] = float("nan")

# ========== 汇总 + 组件 CASE (官方阈值逐字) ============
sc = (coh[["icustay_id"]].merge(mbp,on="icustay_id",how="left")
      .merge(vaso[["icustay_id"]+DRUGS],on="icustay_id",how="left")
      .merge(labs,on="icustay_id",how="left").merge(pf2,on="icustay_id",how="left")
      .merge(uo,on="icustay_id",how="left").merge(mingcs,on="icustay_id",how="left"))
# respiration: <100 vent→4, <200 vent→3, <300 novent→2, <400 novent→1, 其余有值→0 (pd.cut 左闭右开)
sc["respiration"] = pd.cut(sc.pao2fio2_vent_min, [-1e9,100,200], right=False, labels=[4,3]).astype(float)
mask = sc.respiration.isna()
sc.loc[mask,"respiration"] = pd.cut(sc.loc[mask,"pao2fio2_novent_min"], [-1e9,300,400], right=False, labels=[2,1]).astype(float).values
sc.loc[sc.respiration.isna() & sc.pao2fio2_novent_min.notna() & (sc.pao2fio2_novent_min>=400), "respiration"] = 0
sc.loc[sc.respiration.isna() & sc.pao2fio2_vent_min.notna() & (sc.pao2fio2_vent_min>=200), "respiration"] = 0
sc["coagulation"] = pd.cut(sc.platelet_min, [-1e9,20,50,100,150], right=False, labels=[4,3,2,1]).astype(float)
sc.loc[sc.coagulation.isna() & sc.platelet_min.notna() & (sc.platelet_min>=150), "coagulation"] = 0
sc["liver"] = float("nan")
sc.loc[sc.bilirubin_max.notna(),"liver"] = 4
for thr, sc4 in [(12,3),(6,2),(2,1),(1.2,0)]:
    sc.loc[sc.bilirubin_max < thr, "liver"] = sc4
def cardio(r):
    if pd.notna(r.rate_dopamine) and r.rate_dopamine>15: return 4
    if pd.notna(r.rate_epinephrine) and r.rate_epinephrine>0.1: return 4
    if pd.notna(r.rate_norepinephrine) and r.rate_norepinephrine>0.1: return 4
    if pd.notna(r.rate_dopamine) and r.rate_dopamine>5: return 3
    if pd.notna(r.rate_epinephrine) and r.rate_epinephrine<=0.1: return 3
    if pd.notna(r.rate_norepinephrine) and r.rate_norepinephrine<=0.1: return 3
    if (pd.notna(r.rate_dopamine) and r.rate_dopamine>0) or (pd.notna(r.rate_dobutamine) and r.rate_dobutamine>0): return 2
    if pd.notna(r.meanbp_min) and r.meanbp_min<70: return 1
    if r[["meanbp_min"]+DRUGS].isna().all(): return float("nan")
    return 0
sc["cardiovascular"] = sc.apply(cardio, axis=1)
sc["cns"] = float("nan")
sc.loc[sc.mingcs.notna(),"cns"] = 4
sc.loc[(sc.mingcs>=6)&(sc.mingcs<=9),"cns"] = 3
sc.loc[(sc.mingcs>=10)&(sc.mingcs<=12),"cns"] = 2
sc.loc[(sc.mingcs>=13)&(sc.mingcs<=14),"cns"] = 1
sc.loc[sc.mingcs>=15,"cns"] = 0
def renal(r):
    if pd.notna(r.creatinine_max) and r.creatinine_max>=5.0: return 4
    if pd.notna(r.urineoutput) and r.urineoutput<200: return 4
    if pd.notna(r.creatinine_max) and r.creatinine_max>=3.5: return 3
    if pd.notna(r.urineoutput) and r.urineoutput<500: return 3
    if pd.notna(r.creatinine_max) and r.creatinine_max>=2.0: return 2
    if pd.notna(r.creatinine_max) and r.creatinine_max>=1.2: return 1
    if pd.isna(r.urineoutput) and pd.isna(r.creatinine_max): return float("nan")
    return 0
sc["renal"] = sc.apply(renal, axis=1)
COMPS = ["respiration","coagulation","liver","cardiovascular","cns","renal"]
sc["sofa"] = sc[COMPS].fillna(0).sum(axis=1).astype(int)
sc["sofa_raw_missing"] = sc[COMPS].isna().any(axis=1).astype(int)

# ========== 硬门禁 ============
gates = {}
gates["n_rows_eq_cohort"] = len(sc)==len(coh)
gates["icustay_unique"] = sc.icustay_id.is_unique
gates["component_range_0_4"] = bool(sc[COMPS].dropna().apply(lambda s: s.between(0,4)).all().all())
gates["sofa_eq_sum"] = bool((sc.sofa == sc[COMPS].fillna(0).sum(axis=1).astype(int)).all())
gates["sofa_range_0_24"] = bool(sc.sofa.between(0,24).all())
cov = {c: round(float(sc[c].notna().mean()),4) for c in COMPS}
# 覆盖门校准修正 (2026-09-03): 冻结参照 08z (M4) respiration 缺失 58.1%/liver 缺失 63.4%, 官方 gate 集无覆盖门
# → 弃用自设 60%/组件线; 改用 08i 同款灾难性失败底线 (任一组件 ≥90%), M3↔M4 覆盖差距降级为披露项 (见 xdb_cov)
any_cov = float(sc[COMPS].notna().any(axis=1).mean())
gates["any_component_coverage_ge_90pct"] = any_cov >= 0.90
M4_REF_COV = {"respiration":0.419, "coagulation":0.990, "liver":0.366, "cardiovascular":0.999, "cns":0.999, "renal":0.999}
xdb_cov = {c: {"m3":cov[c], "m4_ref_08z":M4_REF_COV[c]} for c in COMPS}
print("GATES:", gates); print("coverage:", cov); print(f"any_cov={any_cov:.4f} | M3 vs M4(08z): respiration {cov['respiration']:.3f} vs {M4_REF_COV['respiration']}, liver {cov['liver']:.3f} vs {M4_REF_COV['liver']}")
status = "PASS" if all(gates.values()) else "FAIL"
rep = {"status":status, "gates":gates, "coverage":cov, "n":int(len(sc)),
 "sofa_dist": {"mean":round(float(sc.sofa.mean()),3), "median":float(sc.sofa.median()),
   "p25":float(sc.sofa.quantile(.25)), "p75":float(sc.sofa.quantile(.75)), "max":int(sc.sofa.max())},
 "component_missing_raw": {c:int(sc[c].isna().sum()) for c in COMPS},
 "cross_db_coverage_vs_08z": xdb_cov,
 "cross_db_note": "M3 respiration/liver 覆盖低于 M4 (08z: 41.9%/36.6%): MIMIC-III 记录密度低于 IV + FiO2/胆红素在 [-6h,+24h] 窗内较少测量; 组件缺失计 0 的求和惯例与 08z 一致; sofa 分布对比见 sofa_dist",
 "sofa_any_component_missing_pct": round(float(sc.sofa_raw_missing.mean()),4),
 "deviations_disclosed": [
   "窗口统一 08z 口径: 测量类[-6h,+24h]/UO[0,+24h]/weight[-1d,+24h] (M3官方 vitals/gcs/vaso 为 [intime,+24h])",
   "动脉血气 specimen 取 LABEVENTS itemid=50800 的 VALUE 列 (PASS2 valuenum 过滤会滤掉该行 → spec_win 单独物化); 匹配规则 = ART 前缀 (含 'ART.'/'ARTERIAL'), 不采用官方 SPECIMEN_PROB 回归补判; 实测分布见 specimen_dist",
   "weight echo_data 回退不采用; CV mcgmin 无体重行 rate 置 null 计数见 n_cv_no_weight",
   "通气设定扫描窗 [-24h,+24h] (官方全住院时长; 仅需 landmark 附近 isvent)",
   "覆盖门校准: 初版自设 60%/组件线与冻结参照 08z 自身现实矛盾 (08z respiration/liver 覆盖 41.9%/36.6%) → 改为 08i 同款任一组件≥90%底线, M3↔M4 差异入 cross_db_coverage_vs_08z 披露"],
 "n_spec_null_dropped": n_spec_null,
 "specimen_dist": {str(r.specimen): int(r.n) for _, r in spec_dist.iterrows()},
 "n_cv_need_weight": n_cv_needW, "n_cv_no_weight": n_cv_noW,
 "provenance": "MIT-LCP/mimic-code@main mimic-iii: severityscores/sofa.sql + firstday/{labs,vitals,urine_output,gcs,blood_gas_first_day,blood_gas_first_day_arterial}.sql + durations/{ventilation_classification,ventilation_durations}.sql (2026-09-03 实拉)"}
sc[["icustay_id","sofa"]+COMPS+["mingcs","urineoutput","sofa_raw_missing"]].to_parquet(OUT/"08h_sofa_mimic3.parquet", index=False)
(REP/"08h_sofa_mimic3.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
if status!="PASS":
    print(f"[GATE FAIL] {gates}"); sys.exit(1)
print(f"DONE 08h: n={len(sc)} sofa mean={rep['sofa_dist']['mean']} → 08h_sofa_mimic3.parquet")
