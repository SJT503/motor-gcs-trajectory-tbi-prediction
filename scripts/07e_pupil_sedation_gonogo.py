# Phase1-④: 瞳孔动态 + 镇静中断响应 go/no-go 实测 (SAP_prediction §3.4 预注册)
# 判定规则(冻结): 瞳孔覆盖率 <80% → 整族退出主分析转敏感性, 记 analytic log
#               镇静中断无法操作化(无可靠中断-复评配对事件) → 放弃该特征族并记 log
# 输出: reports/07e_gonogo.json + reports/07e_gonogo_log.md (analytic decisions log 素材)
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M4 = ROOT/"data/mimic-iv-3.1"; M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
REP = ROOT/"07_prediction_system/reports"; REP.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"07_prediction_system/data"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
COH4 = (ROOT/"results/cohort_audit/02_tbi_cohort.parquet").as_posix()
qc = {}

# ========== ① 瞳孔可得性: 三库 itemid/字段动态查证 ==========
print("[①] MIMIC-IV 瞳孔 itemid 查证 (D_ITEMS)...")
p4 = con.execute(f"""SELECT itemid, label FROM read_csv_auto('{M4.as_posix()}/icu/d_items.csv.gz')
WHERE label ILIKE '%pupil%' ORDER BY itemid""").df()
print(p4.to_string(index=False))
p4_ids = ",".join(map(str, p4.itemid.tolist()))
cov4 = con.execute(f"""
WITH pe AS (SELECT DISTINCT c.stay_id FROM read_csv_auto('{M4.as_posix()}/icu/chartevents.csv.gz', types={{'valuenum':'DOUBLE','value':'VARCHAR'}}) c
  JOIN read_parquet('{COH4}') co ON c.stay_id=co.stay_id
  WHERE c.itemid IN ({p4_ids}) AND (c.valuenum IS NOT NULL OR c.value IS NOT NULL)
    AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL 72 HOUR)
SELECT count(*) FROM pe""").fetchone()[0]
n4 = con.execute(f"SELECT count(*) FROM read_parquet('{COH4}')").fetchone()[0]
qc["pupil_mimic4"] = dict(itemids=p4_ids, labels=" | ".join(p4.label.tolist()),
                          n_covered=int(cov4), n_total=int(n4), coverage_pct=round(100*cov4/n4,1))
print(f"  72h 覆盖 {cov4}/{n4} = {qc['pupil_mimic4']['coverage_pct']}%")

print("[①] MIMIC-III 瞳孔 itemid 查证 (D_ITEMS)...")
p3 = con.execute(f"""SELECT itemid AS itemid, label AS label FROM read_csv_auto('{M3.as_posix()}/D_ITEMS.csv.gz')
WHERE label ILIKE '%pupil%' ORDER BY itemid""").df()
print(p3.to_string(index=False))
m3coh = pd.read_parquet(OUT/"cohort_mimic3.parquet")
m3coh[["icustay_id","intime"]].to_parquet(OUT/"_tmp_m3_pupil.parquet", index=False)
p3_ids = ",".join(map(str, p3.itemid.tolist()))
cov3 = con.execute(f"""
WITH pe AS (SELECT DISTINCT c.icustay_id FROM read_csv_auto('{M3.as_posix()}/CHARTEVENTS.csv.gz',
    types={{'valuenum':'DOUBLE','value':'VARCHAR'}}, strict_mode=false, ignore_errors=true) c
  JOIN read_parquet('{(OUT/'_tmp_m3_pupil.parquet').as_posix()}') co ON c.icustay_id=co.icustay_id
  WHERE c.itemid IN ({p3_ids}) AND (c.valuenum IS NOT NULL OR c.value IS NOT NULL)
    AND c.charttime BETWEEN co.intime AND co.intime + INTERVAL 72 HOUR)
SELECT count(*) FROM pe""").fetchone()[0]
qc["pupil_mimic3"] = dict(itemids=p3_ids, labels=" | ".join(p3.label.tolist()),
                          n_covered=int(cov3), n_total=int(len(m3coh)), coverage_pct=round(100*cov3/len(m3coh),1))
print(f"  72h 覆盖 {cov3}/{len(m3coh)} = {qc['pupil_mimic3']['coverage_pct']}%")

print("[①] eICU 瞳孔字段查证 (nurseAssessment)...")
na_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{(EI/'nurseAssessment.csv.gz').as_posix()}')").df().column_name)
print(f"  [nurseAssessment] 列: {sorted(na_cols)}")
off_cands = [c for c in ["nursingchartoffset","celloffset","nurseassessentryoffset","assessmentoffset"] if c in na_cols]
assert len(off_cands)>=1, "nurseAssessment 无时间列 — 按上方实测列名修正脚本"
OFFC = off_cands[0]
na = con.execute(f"""SELECT celllabel, cellattribute, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{EI.as_posix()}/nurseAssessment.csv.gz')
WHERE celllabel ILIKE '%pupil%' GROUP BY 1,2 ORDER BY n DESC LIMIT 20""").df()
print(na.to_string(index=False))
# 尽调补丁 (2026-09-02 二轮): 判定规则只认 celllabel — 补查其他 celllabel 下的 cellattribute 层瞳孔子属性,
# 证据存 JSON; 若揭示大量隐藏瞳孔数据 → 报 PI 决定是否修订判定, 不自行改预注册口径
na_attr = con.execute(f"""SELECT celllabel, cellattribute, count(*) n, count(DISTINCT patientunitstayid) n_pat
FROM read_csv_auto('{EI.as_posix()}/nurseAssessment.csv.gz')
WHERE cellattribute ILIKE '%pupil%' GROUP BY 1,2 ORDER BY n DESC LIMIT 20""").df()
print("[尽调] cellattribute 层 pupil 探针 (其他 celllabel 下的瞳孔子属性):")
print(na_attr.to_string(index=False))
n_attr = con.execute(f"""SELECT count(DISTINCT patientunitstayid) FROM read_csv_auto('{EI.as_posix()}/nurseAssessment.csv.gz')
WHERE cellattribute ILIKE '%pupil%'""").fetchone()[0]
qc["pupil_eicu_cellattribute_probe"] = dict(
    rows=[f"{r.celllabel} / {r.cellattribute} / n={r.n} / n_pat={r.n_pat}" for r in na_attr.itertuples()],
    n_pat_total=int(n_attr),
    scope="n_pat_total=全表口径(未限队列/时间窗), 仅尽调参考, 非队列覆盖率")
ei_ids = tuple(int(x) for x in pd.read_parquet(OUT/"cohort_eicu.parquet").icustay_id_eicu.tolist())
covE = con.execute(f"""
SELECT count(DISTINCT patientunitstayid) FROM read_csv_auto('{EI.as_posix()}/nurseAssessment.csv.gz')
WHERE patientunitstayid IN {ei_ids} AND celllabel ILIKE '%pupil%'
  AND {OFFC} BETWEEN 0 AND 4320""").fetchone()[0]
qc["pupil_eicu"] = dict(celllabels=" | ".join(sorted(set(na.celllabel))[:8]),
                        n_covered=int(covE), n_total=int(len(ei_ids)), coverage_pct=round(100*covE/len(ei_ids),1))
print(f"  72h 覆盖 {covE}/{len(ei_ids)} = {qc['pupil_eicu']['coverage_pct']}%")

# ========== ② 镇静中断-复评配对 (discovery MIMIC-IV 深挖; 13a 锚点 itemid) ==========
print("[②] MIMIC-IV 镇静输注段 → 中断-复评配对...")
sed_ids = "222168,229420,225150,221668"   # 13a 已跑通(丙泊酚/咪达唑仑/右美/其他)
sed_lab = con.execute(f"""SELECT itemid, label FROM read_csv_auto('{M4.as_posix()}/icu/d_items.csv.gz')
WHERE itemid IN ({sed_ids}) ORDER BY itemid""").df()
print(sed_lab.to_string(index=False))     # 实测标签核验(221668 身份存疑, 打印人工确认; 标签不符即改 sed_ids 重跑)
inf = con.execute(f"""
SELECT e.stay_id, e.starttime, e.endtime
FROM read_csv_auto('{M4.as_posix()}/icu/inputevents.csv.gz') e
JOIN read_parquet('{COH4}') co ON e.stay_id=co.stay_id
WHERE e.itemid IN ({sed_ids}) AND e.starttime IS NOT NULL AND e.endtime IS NOT NULL
  AND e.statusdescription != 'Rewritten'""").df()
inf["starttime"]=pd.to_datetime(inf.starttime); inf["endtime"]=pd.to_datetime(inf.endtime)
inf = inf[inf.endtime>inf.starttime]
motor = pd.read_parquet(ROOT/"results/features/13_motor_long.parquet")   # 13a 产物: stay_id,hr,motor(1-6)
m4coh = pd.read_parquet(COH4)[["stay_id","intime"]]
motor = motor.merge(m4coh, on="stay_id", how="left")
motor["charttime"] = pd.to_datetime(motor.intime) + pd.to_timedelta(motor.hr, unit="h")
# 每 stay 合并镇静区间 → 找 ≥30min 无输注间隙 → 间隙后 2h 内有 motor 复测 = 可配对中断
motor_by_stay = {sid: g for sid, g in motor.groupby("stay_id")}
paired_stays = set()
for sid, g in inf.groupby("stay_id"):
    iv = list(g.sort_values("starttime")[["starttime","endtime"]].itertuples(index=False, name=None))  # pd.Timestamp 元组(非 numpy.datetime64, 保 total_seconds 可用)
    merged = [list(iv[0])]
    for s,e in iv[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s,e])
    gaps = [(merged[i][1], merged[i+1][0]) for i in range(len(merged)-1)
            if (merged[i+1][0]-merged[i][1]).total_seconds() >= 1800]
    if not gaps: continue
    mg = motor_by_stay.get(sid)
    if mg is None or len(mg)==0: continue
    for gap_end, nxt_start in gaps:
        mt = mg[(mg.charttime>=gap_end)&(mg.charttime<=gap_end+pd.Timedelta(hours=2))]
        if len(mt)>0: paired_stays.add(sid); break
qc["sedation_interruption"] = dict(
    definition="镇静输注合并区间间 ≥30min 间隙, 间隙结束后 2h 内有 GCS-motor 复测 = 可配对中断事件",
    n_stays_with_infusion=int(inf.stay_id.nunique()),
    n_stays_paired=int(len(paired_stays)),
    paired_share_pct=round(100*len(paired_stays)/max(1,inf.stay_id.nunique()),1))
print(f"  镇静输注 {inf.stay_id.nunique()} stays → 可配对中断 {len(paired_stays)} ({qc['sedation_interruption']['paired_share_pct']}%)")

# ========== go/no-go 判定 (SAP §3.4 冻结规则 + 预设操作化判据) ==========
# 瞳孔: 三库最低覆盖率 ≥80% = GO (外验两库也要用, 取最严口径)
# 镇静中断: 可配对 ≥200 stays 且 ≥20% 输注患者 = 可操作化 (预设工程判据, 写死防事后挪动)
pupil_min_cov = min(qc["pupil_mimic4"]["coverage_pct"], qc["pupil_eicu"]["coverage_pct"], qc["pupil_mimic3"]["coverage_pct"])
pupil_go = pupil_min_cov >= 80
sed_go = len(paired_stays) >= 200 and qc["sedation_interruption"]["paired_share_pct"] >= 20
qc["verdict"] = dict(
    pupil = "GO" if pupil_go else f"NO-GO(三库最低覆盖{pupil_min_cov}%<80% → 整族退出主分析转敏感性, 记log)",
    sedation_interruption = "GO" if sed_go else f"NO-GO(配对{len(paired_stays)}stays/{qc['sedation_interruption']['paired_share_pct']}% < 预设200/20% → 放弃该特征族并记log, 禁临时替代)")
(REP/"07e_gonogo.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
log_md = f"""# 07e go/no-go 实测记录 (analytic decisions log 素材, SAP_prediction §3.4/§11)

日期: 脚本运行日(见 JSON) | 预注册规则: 瞳孔覆盖<80%→退出主分析转敏感性; 镇静中断不可操作化→放弃并记log

## 瞳孔动态
- MIMIC-IV: {qc['pupil_mimic4']['coverage_pct']}% ({qc['pupil_mimic4']['n_covered']}/{qc['pupil_mimic4']['n_total']}) | itemids: {qc['pupil_mimic4']['itemids']}
- eICU: {qc['pupil_eicu']['coverage_pct']}% | celllabels: {qc['pupil_eicu']['celllabels']}
- MIMIC-III: {qc['pupil_mimic3']['coverage_pct']}%
- **判定: {qc['verdict']['pupil']}**

## 镇静中断响应
- 定义: {qc['sedation_interruption']['definition']}
- 可配对: {qc['sedation_interruption']['n_stays_paired']}/{qc['sedation_interruption']['n_stays_with_infusion']} ({qc['sedation_interruption']['paired_share_pct']}%)
- **判定: {qc['verdict']['sedation_interruption']}**
"""
(REP/"07e_gonogo_log.md").write_text(log_md, encoding="utf-8")
print(f"\n=== 07e 判定 ===\n  瞳孔: {qc['verdict']['pupil']}\n  镇静中断: {qc['verdict']['sedation_interruption']}")
print("DONE 07e")
