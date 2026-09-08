# -*- coding: utf-8 -*-
"""
08z — MIMIC-IV first-day SOFA (24h) 官方概念的 DuckDB 忠实移植 (仅限本项目 TBI 队列 n=2751)

用途: SAP §6 基础模型对照 (年龄+性别+入院GCS+Charlson+SOFA) 所需 SOFA24。
来源: mimic-code/mimic-iv/concepts_postgres (本地文件, 2026-09-03 逐字核证):
  firstday/first_day_sofa.sql          — 6 组分 CASE + COALESCE(0) 求和 + pafi 双列 vent 交互
  firstday/first_day_vitalsign.sql     — 窗口 [intime-6h, intime+1d]
  measurement/vitalsign.sql            — mbp itemids 220052/220181/225312, 域 0<v<300
  firstday/first_day_gcs.sql           — gcs_seq=1 (ORDER BY gcs NULLS FIRST)
  measurement/gcs.sql                  — ETT verbal=0→GCS 15 + 6h 前值 carry-forward
  measurement/bg.sql                   — labevents specimen 透视 + chartevents FiO2(223835) 4h 回溯
  treatment/ventilation.sql            — InvasiveVent 状态机 (trach>mech>NIV>HFNC>O2, 14h 分段)
  measurement/ventilator_setting.sql   — 223849/229314/223848 等 15 itemids
  measurement/oxygen_delivery.sql      — 226732 设备 + 223834/227582/227287 流量
  medication/{norepinephrine,epinephrine,dopamine,dobutamine}.sql — 221906/221289/221662/221653
  firstday/first_day_urine_output.sql  — 窗口 [intime, intime+1d]
  measurement/urine_output.sql         — 12 UO itemids + GU 冲洗液负值
  firstday/first_day_lab.sql + chemistry.sql(50912: >0,<=150) + complete_blood_count.sql(51265: >0)
  measurement/enzyme.sql               — 50885 总胆红素 (>0)
窗口口径 (官方): vitals/gcs/bg/labs = [intime-6h, intime+1d]; vaso 按 starttime ∈ 同窗; UO = [intime, intime+1d]
设计决策:
  1) 全部成分从原始 csv.gz 现抽, 不复用 03b 列 (03b chartevents 窗口为 [intime,+24h], 与官方不同;
     Methods 可辩护性: "SOFA per mimic-code first_day_sofa concept")
  2) SOFA 的 GCS 用官方 gcs 概念 (含 ETT→15 规则), 与 12_gcs_long(项目 GCS 主线) 用途不同、互不影响
  3) 概念表全为 stay 级全病程 (vent 状态机/GCS carry-forward 官方即如此), 首日窗口只在 first_day 连接时套
  4) bg 的 stg_spo2 连接对 pafi 无影响 (仅产出 spo2 列), 移植时省略并在此声明等价
  5) 大表 (chartevents/labevents) 各一次扫描物化小表, 下游全走内存 (慢任务铁律)
  6) o2_flow 行仅向 tm 贡献 charttime, 不参与分类 (设备 NULL→状态 NULL→被过滤), 未保留流量列 (声明等价)
产物: E:\\TBI subtype\\results\\features\\08z_sofa24.parquet + 07_prediction_system/reports/08z_sofa_report.json
"""
import sys, time, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import duckdb, pandas as pd

ROOT = Path(r"E:/TBI subtype")
COHORT = (ROOT / "results/cohort_audit/02_tbi_cohort.parquet").as_posix()
DATA = ROOT / "data/mimic-iv-3.1"
CHARTEVENTS = (DATA / "icu/chartevents.csv.gz").as_posix()
LABEVENTS = (DATA / "hosp/labevents.csv.gz").as_posix()
INPUTEVENTS = (DATA / "icu/inputevents.csv.gz").as_posix()
OUTPUTEVENTS = (DATA / "icu/outputevents.csv.gz").as_posix()
D_ITEMS = (DATA / "icu/d_items.csv.gz").as_posix()
D_LABITEMS = (DATA / "hosp/d_labitems.csv.gz").as_posix()
OUT = ROOT / "results/features"; OUT.mkdir(parents=True, exist_ok=True)
REP = ROOT / "07_prediction_system/reports"; REP.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)

con = duckdb.connect(); con.execute("PRAGMA threads=4")
con.execute(f"CREATE TABLE cohort AS SELECT subject_id,hadm_id,stay_id,CAST(intime AS TIMESTAMP) intime FROM read_parquet('{COHORT}')")
n_cohort = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
log(f"cohort: {n_cohort} stays")

# ===== 0) itemid 运行时核证 (d_items/d_labitems label 必须命中关键词, 禁记忆硬编码) =====
EXPECT_ICU = {  # chartevents / inputevents / outputevents 的 itemid: (label 须含关键词, 可空=存在即可)
    220052: ("Arterial Blood Pressure mean",), 220181: ("Non Invasive Blood Pressure mean",), 225312: ("ART BP Mean",),
    223900: ("GCS - Verbal Response",), 223901: ("GCS - Motor Response",), 220739: ("GCS - Eye Opening",),
    224688: ("Respiratory Rate (Set)",), 224689: ("Respiratory Rate (spontaneous)",), 224690: ("Respiratory Rate (Total)",),
    224687: ("minute volume",), 224684: (), 224685: (), 224686: ("tidal volume",), 224696: ("plateau",),
    220339: (), 224700: ("PEEP",), 223835: ("Inspired O2 Fraction",), 223849: ("ventilator mode",), 229314: ("ventilator mode",),
    223848: ("ventilator type",), 224691: ("Flow Rate",),
    223834: ("o2 flow",), 227582: ("bipap o2 flow",), 227287: ("o2 flow",), 226732: ("O2 Delivery Device",),
    221906: ("Norepinephrine",), 221289: ("Epinephrine",), 221662: ("Dopamine",), 221653: ("Dobutamine",),
    226559: ("Foley",), 226560: ("Void",), 226561: ("Condom Cath",), 226584: ("Ileoconduit",), 226563: ("Suprapubic",),
    226564: ("Nephrostomy",), 226565: ("Nephrostomy",), 226567: ("Straight Cath",), 226557: ("Ureteral Stent",),
    226558: ("Ureteral Stent",), 227488: ("GU Irrigant Volume In",), 227489: ("GU Irrigant/Urine Volume Out",),
}
EXPECT_HOSP = {
    52033: ("specimen",), 50816: ("oxygen",), 50821: ("pO2",),
    50912: ("Creatinine",), 51265: ("platelet",), 50885: ("bilirubin",),
}
CE_ITEMS = [220052,220181,225312,223900,223901,220739,224688,224689,224690,224687,224684,224685,224686,
            224696,220339,224700,223835,223849,229314,223848,224691,223834,227582,227287,226732]
di = con.execute(f"SELECT itemid, label FROM read_csv_auto('{D_ITEMS}')").df()
dl = con.execute(f"SELECT itemid, label FROM read_csv_auto('{D_LABITEMS}')").df()
di_map = dict(zip(di.itemid.astype(int), di.label.astype(str)))
dl_map = dict(zip(dl.itemid.astype(int), dl.label.astype(str)))
fails = []
for iid, kws in EXPECT_ICU.items():
    lab_txt = di_map.get(iid)
    if lab_txt is None: fails.append(f"icu itemid {iid} 不在 d_items")
    elif kws and not any(k.lower() in lab_txt.lower() for k in kws): fails.append(f"icu {iid} label='{lab_txt}' 未命中 {kws}")
for iid, kws in EXPECT_HOSP.items():
    lab_txt = dl_map.get(iid)
    if lab_txt is None: fails.append(f"hosp itemid {iid} 不在 d_labitems")
    elif kws and not any(k.lower() in lab_txt.lower() for k in kws): fails.append(f"hosp {iid} label='{lab_txt}' 未命中 {kws}")
if fails:
    for f_ in fails: log(f"[itemid核证FAIL] {f_}")
    sys.exit(1)
log(f"itemid 核证 PASS: icu {len(EXPECT_ICU)} + hosp {len(EXPECT_HOSP)} 全命中")

# ===== 1) chartevents 一次扫描物化 (vitals-mbp + GCS + vent_setting + O2_delivery; stay 级全病程) =====
t0 = time.time()
log(f"[PASS1] chartevents 全病程 ({len(CE_ITEMS)} itemids) ...")
con.execute(f"""
CREATE TABLE ce AS
SELECT ce.stay_id, CAST(ce.charttime AS TIMESTAMP) charttime, ce.itemid,
       TRY_CAST(ce.valuenum AS DOUBLE) valuenum, ce.value
FROM read_csv_auto('{CHARTEVENTS}', types={{'valuenum':'VARCHAR','value':'VARCHAR'}}) ce
JOIN cohort c ON ce.stay_id=c.stay_id
WHERE ce.itemid IN ({",".join(map(str, CE_ITEMS))})
""")
n_ce = con.execute("SELECT count(*) FROM ce").fetchone()[0]
log(f"  ce 物化 {n_ce} 行 ({time.time()-t0:.0f}s)")

# --- 1a) vitalsign 概念: mbp = AVG per (stay, charttime), 域 0<v<300 (vitalsign.sql L25) ---
con.execute("""CREATE TABLE vit_mbp AS
SELECT stay_id, charttime, AVG(valuenum) mbp
FROM ce WHERE itemid IN (220052,220181,225312) AND valuenum>0 AND valuenum<300
GROUP BY stay_id, charttime""")

# --- 1b) GCS 概念 (gcs.sql 全链: base 透视 → rn 自连接 6h 前值 → ETT verbal=0→15) ---
con.execute("""CREATE TABLE gcs_base AS
SELECT subject_id, stay_id, charttime,
  MAX(CASE WHEN itemid=223901 THEN valuenum END) gcsmotor,
  MAX(CASE WHEN itemid=223900 AND value='No Response-ETT' THEN 0
           WHEN itemid=223900 THEN valuenum END) gcsverbal,
  MAX(CASE WHEN itemid=220739 THEN valuenum END) gcseyes,
  MAX(CASE WHEN itemid=223900 AND value='No Response-ETT' THEN 1 ELSE 0 END) endotrachflag,
  ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime ASC NULLS FIRST) rn
FROM (SELECT c.subject_id, ce.* FROM ce ce JOIN cohort c ON ce.stay_id=c.stay_id
      WHERE ce.itemid IN (223900,223901,220739))
GROUP BY subject_id, stay_id, charttime""")
con.execute("""CREATE TABLE gcs_concept AS
SELECT b.subject_id, b.stay_id, b.charttime,
  CASE
    WHEN b.gcsverbal = 0 THEN 15
    WHEN b.gcsverbal IS NULL AND b2.gcsverbal = 0 THEN 15
    WHEN b2.gcsverbal = 0 THEN COALESCE(b.gcsmotor,6)+COALESCE(b.gcsverbal,5)+COALESCE(b.gcseyes,4)
    ELSE COALESCE(b.gcsmotor, COALESCE(b2.gcsmotor,6)) + COALESCE(b.gcsverbal, COALESCE(b2.gcsverbal,5)) + COALESCE(b.gcseyes, COALESCE(b2.gcseyes,4))
  END gcs
FROM gcs_base b LEFT JOIN gcs_base b2
  ON b.stay_id=b2.stay_id AND b.rn=b2.rn+1 AND b2.charttime > b.charttime - INTERVAL 6 HOUR""")

# --- 1c) ventilator_setting 概念 (ventilator_setting.sql: 15 itemids, value 非空) ---
con.execute("""CREATE TABLE vent_setting AS
SELECT subject_id, MAX(stay_id) stay_id, charttime,
  MAX(CASE WHEN itemid=223849 THEN value END) ventilator_mode,
  MAX(CASE WHEN itemid=229314 THEN value END) ventilator_mode_hamilton
FROM (SELECT c.subject_id, ce.* FROM ce ce JOIN cohort c ON ce.stay_id=c.stay_id
      WHERE ce.itemid IN (224688,224689,224690,224687,224685,224684,224686,224696,220339,224700,223835,223849,229314,223848,224691)
        AND ce.value IS NOT NULL)
GROUP BY subject_id, charttime""")

# --- 1d) oxygen_delivery 概念 (oxygen_delivery.sql; 只保留分类所需的 device_1 列, 见设计决策 6) ---
con.execute("""CREATE TABLE o2_delivery AS
WITH ce_stg2 AS (
  SELECT c.subject_id, ce.stay_id, ce.charttime,
    CASE WHEN ce.itemid IN (223834,227582) THEN 223834 ELSE ce.itemid END itemid,
    ROW_NUMBER() OVER (PARTITION BY c.subject_id, ce.charttime, ce.itemid ORDER BY ce.value) rn0
  FROM ce ce JOIN cohort c ON ce.stay_id=c.stay_id
  WHERE ce.itemid IN (223834,227582,227287) AND ce.value IS NOT NULL),
o2 AS (
  SELECT c.subject_id, ce.stay_id, ce.charttime, ce.value o2_device,
    ROW_NUMBER() OVER (PARTITION BY c.subject_id, ce.charttime ORDER BY ce.value NULLS FIRST) rn
  FROM ce ce JOIN cohort c ON ce.stay_id=c.stay_id WHERE ce.itemid=226732)
SELECT COALESCE(a.subject_id,b.subject_id) subject_id, COALESCE(a.stay_id,b.stay_id) stay_id,
       COALESCE(a.charttime,b.charttime) charttime, b.o2_device o2_delivery_device_1
FROM (SELECT * FROM ce_stg2 WHERE rn0=1) a
FULL OUTER JOIN (SELECT * FROM o2 WHERE rn=1) b
  ON a.subject_id=b.subject_id AND a.charttime=b.charttime""")

# --- 1e) ventilation 状态机 (ventilation.sql: trach>mech>NIV>HFNC>O2 优先级; 14h 分段; HAVING MIN<>MAX) ---
con.execute("""CREATE TABLE vent_status AS
WITH tm AS (
  SELECT stay_id, charttime FROM vent_setting
  UNION
  SELECT stay_id, charttime FROM o2_delivery),
vs AS (
  SELECT tm.stay_id, tm.charttime, od.o2_delivery_device_1,
    COALESCE(vs2.ventilator_mode, vs2.ventilator_mode_hamilton) vent_mode,
    CASE
      WHEN od.o2_delivery_device_1 IN ('Tracheostomy tube','Trach mask ') THEN 'Tracheostomy'
      WHEN od.o2_delivery_device_1 IN ('Endotracheal tube')
        OR vs2.ventilator_mode IN ('(S) CMV','APRV','APRV/Biphasic+ApnPress','APRV/Biphasic+ApnVol','APV (cmv)','Ambient','Apnea Ventilation','CMV','CMV/ASSIST','CMV/ASSIST/AutoFlow','CMV/AutoFlow','CPAP/PPS','CPAP/PSV','CPAP/PSV+Apn TCPL','CPAP/PSV+ApnPres','CPAP/PSV+ApnVol','MMV','MMV/AutoFlow','MMV/PSV','MMV/PSV/AutoFlow','P-CMV','PCV+','PCV+/PSV','PCV+Assist','PRES/AC','PRVC/AC','PRVC/SIMV','PSV/SBT','SIMV','SIMV/AutoFlow','SIMV/PRES','SIMV/PSV','SIMV/PSV/AutoFlow','SIMV/VOL','SYNCHRON MASTER','SYNCHRON SLAVE','VOL/AC')
        OR vs2.ventilator_mode_hamilton IN ('APRV','APV (cmv)','Ambient','(S) CMV','P-CMV','SIMV','APV (simv)','P-SIMV','VS','ASV')
        THEN 'InvasiveVent'
      WHEN od.o2_delivery_device_1 IN ('Bipap mask ','CPAP mask ')
        OR vs2.ventilator_mode_hamilton IN ('DuoPaP','NIV','NIV-ST') THEN 'NonInvasiveVent'
      WHEN od.o2_delivery_device_1 IN ('High flow nasal cannula') THEN 'HFNC'
      WHEN od.o2_delivery_device_1 IN ('Non-rebreather','Face tent','Aerosol-cool','Venti mask ','Medium conc mask ','Ultrasonic neb','Vapomist','Oxymizer','High flow neb','Nasal cannula') THEN 'SupplementalOxygen'
      WHEN od.o2_delivery_device_1 IN ('None') THEN 'None'
      ELSE NULL END ventilation_status
  FROM tm
  LEFT JOIN vent_setting vs2 ON tm.stay_id=vs2.stay_id AND tm.charttime=vs2.charttime
  LEFT JOIN o2_delivery od ON tm.stay_id=od.stay_id AND tm.charttime=od.charttime),
vd0 AS (
  SELECT stay_id, charttime,
    LAG(charttime,1) OVER (PARTITION BY stay_id, ventilation_status ORDER BY charttime NULLS FIRST) charttime_lag,
    LEAD(charttime,1) OVER w charttime_lead,
    ventilation_status,
    LAG(ventilation_status,1) OVER w ventilation_status_lag
  FROM vs WHERE NOT ventilation_status IS NULL
  WINDOW w AS (PARTITION BY stay_id ORDER BY charttime NULLS FIRST)),
vd1 AS (
  SELECT stay_id, charttime, charttime_lead, ventilation_status,
    CASE WHEN ventilation_status_lag IS NULL THEN 1
         WHEN EXTRACT(EPOCH FROM (charttime - charttime_lag))/3600.0 >= 14 THEN 1
         WHEN ventilation_status_lag <> ventilation_status THEN 1 ELSE 0 END new_ventilation_event
  FROM vd0),
vd2 AS (
  SELECT stay_id, charttime, charttime_lead, ventilation_status,
    SUM(new_ventilation_event) OVER (PARTITION BY stay_id ORDER BY charttime NULLS FIRST) vent_seq
  FROM vd1)
SELECT stay_id, MIN(charttime) starttime,
  MAX(CASE WHEN charttime_lead IS NULL OR EXTRACT(EPOCH FROM (charttime_lead - charttime))/3600.0 >= 14
           THEN charttime ELSE charttime_lead END) endtime,
  MAX(ventilation_status) ventilation_status
FROM vd2 GROUP BY stay_id, vent_seq HAVING MIN(charttime) <> MAX(charttime)""")
n_vent = con.execute("SELECT count(*) FROM vent_status").fetchone()[0]
log(f"  ventilation 状态机: {n_vent} 段 ({time.time()-t0:.0f}s)")

# ===== 2) labevents 一次扫描物化 (bg: 52033/50816/50821 + cr 50912 + plt 51265 + bili 50885) =====
t1 = time.time(); log("[PASS2] labevents (bg + creatinine + platelet + bilirubin) ...")
con.execute(f"""
CREATE TABLE le AS
SELECT le.subject_id, CAST(le.charttime AS TIMESTAMP) charttime, le.itemid, le.specimen_id,
       TRY_CAST(le.valuenum AS DOUBLE) valuenum, le.value
FROM read_csv_auto('{LABEVENTS}', types={{'valuenum':'VARCHAR','value':'VARCHAR'}}) le
JOIN cohort c ON le.subject_id=c.subject_id
WHERE le.itemid IN (52033,50816,50821,50912,51265,50885)
""")
# chartevents FiO2 (bg.sql stg_fio2: 223835, 0.2-1 ×100, 1-20 NULL, 20-100 保留; 先建 — bg_concept 依赖它)
con.execute("""CREATE TABLE stg_fio2 AS
SELECT c.subject_id, ce.charttime,
  MAX(CASE WHEN ce.valuenum>0.2 AND ce.valuenum<=1 THEN ce.valuenum*100
           WHEN ce.valuenum>1 AND ce.valuenum<20 THEN NULL
           WHEN ce.valuenum>=20 AND ce.valuenum<=100 THEN ce.valuenum END) fio2_chartevents
FROM ce ce JOIN cohort c ON ce.stay_id=c.stay_id
WHERE ce.itemid=223835 AND ce.valuenum>0 AND ce.valuenum<=100
GROUP BY c.subject_id, ce.charttime""")
# bg 透视 (bg.sql: per specimen MAX; fio2 单位修正 >20..100 保留, 0.2..1.0 ×100)
con.execute("""CREATE TABLE bg_concept AS
WITH spec AS (
  SELECT specimen_id, MAX(subject_id) subject_id, MAX(charttime) charttime,
    MAX(CASE WHEN itemid=52033 THEN value END) specimen,
    MAX(CASE WHEN itemid=50821 THEN valuenum END) po2,
    MAX(CASE WHEN itemid=50816 THEN
          CASE WHEN valuenum>20 AND valuenum<=100 THEN valuenum
               WHEN valuenum>0.2 AND valuenum<=1.0 THEN valuenum*100.0 END END) fio2
  FROM le WHERE itemid IN (52033,50816,50821) GROUP BY specimen_id),
stg3 AS (
  SELECT g.*, s.fio2_chartevents,
    ROW_NUMBER() OVER (PARTITION BY g.subject_id, g.charttime ORDER BY s.charttime DESC NULLS LAST) lastrowfio2
  FROM (SELECT * FROM spec WHERE po2 IS NOT NULL) g
  LEFT JOIN stg_fio2 s ON g.subject_id=s.subject_id
    AND s.charttime >= g.charttime - INTERVAL 4 HOUR AND s.charttime <= g.charttime AND s.fio2_chartevents > 0)
SELECT subject_id, charttime, po2,
  CASE WHEN po2 IS NULL THEN NULL
       WHEN fio2 IS NOT NULL THEN 100.0*po2/fio2
       WHEN fio2_chartevents IS NOT NULL THEN 100.0*po2/fio2_chartevents
       ELSE NULL END pao2fio2ratio
FROM stg3 WHERE lastrowfio2=1""")
# labs 透视 (chemistry 50912: >0,<=150; cbc 51265: >0; enzyme 50885: >0; per specimen MAX(subject/charttime) 同官方)
con.execute("""CREATE TABLE lab_concept AS
SELECT MAX(subject_id) subject_id, MAX(charttime) charttime,
  MAX(CASE WHEN itemid=50912 AND valuenum>0 AND valuenum<=150 THEN valuenum END) creatinine,
  MAX(CASE WHEN itemid=51265 AND valuenum>0 THEN valuenum END) platelet,
  MAX(CASE WHEN itemid=50885 AND valuenum>0 THEN valuenum END) bilirubin_total
FROM le WHERE itemid IN (50912,51265,50885) GROUP BY specimen_id""")
log(f"  labs 物化 ({time.time()-t1:.0f}s)")

# ===== 3) inputevents 升压药 (4 itemids; 去甲肾 mg/kg/min×1000 修正; MAX rate) =====
t2 = time.time(); log("[PASS3] inputevents 升压药 ...")
con.execute(f"""
CREATE TABLE vaso_mv AS
WITH vaso AS (
  SELECT c.stay_id,
    CASE ce.itemid WHEN 221906 THEN 'norepinephrine' WHEN 221289 THEN 'epinephrine'
                   WHEN 221662 THEN 'dopamine' WHEN 221653 THEN 'dobutamine' END treatment,
    CASE WHEN ce.itemid=221906 THEN
      CASE WHEN ce.rateuom='mg/kg/min' AND TRY_CAST(ce.patientweight AS DOUBLE)=1 THEN TRY_CAST(ce.rate AS DOUBLE)
           WHEN ce.rateuom='mg/kg/min' THEN TRY_CAST(ce.rate AS DOUBLE)*1000.0
           ELSE TRY_CAST(ce.rate AS DOUBLE) END
    ELSE TRY_CAST(ce.rate AS DOUBLE) END vaso_rate
  FROM read_csv_auto('{INPUTEVENTS}') ce
  JOIN cohort c ON ce.stay_id=c.stay_id
  WHERE ce.itemid IN (221906,221289,221662,221653)
    AND CAST(ce.starttime AS TIMESTAMP) >= c.intime - INTERVAL 6 HOUR
    AND CAST(ce.starttime AS TIMESTAMP) <= c.intime + INTERVAL 24 HOUR)
SELECT stay_id,
  MAX(CASE WHEN treatment='norepinephrine' THEN vaso_rate END) rate_norepinephrine,
  MAX(CASE WHEN treatment='epinephrine' THEN vaso_rate END) rate_epinephrine,
  MAX(CASE WHEN treatment='dopamine' THEN vaso_rate END) rate_dopamine,
  MAX(CASE WHEN treatment='dobutamine' THEN vaso_rate END) rate_dobutamine
FROM vaso GROUP BY stay_id""")

# ===== 4) outputevents 尿量 ([intime, intime+1d]; GU 冲洗液入量负值) =====
log("[PASS4] outputevents 尿量 ...")
con.execute(f"""
CREATE TABLE uo AS
SELECT c.stay_id, SUM(CASE WHEN ce.itemid=227488 AND TRY_CAST(ce.value AS DOUBLE)>0
                           THEN -1*TRY_CAST(ce.value AS DOUBLE) ELSE TRY_CAST(ce.value AS DOUBLE) END) urineoutput
FROM read_csv_auto('{OUTPUTEVENTS}') ce
JOIN cohort c ON ce.stay_id=c.stay_id
WHERE ce.itemid IN (226559,226560,226561,226584,226563,226564,226565,226567,226557,226558,227488,227489)
  AND CAST(ce.charttime AS TIMESTAMP) >= c.intime
  AND CAST(ce.charttime AS TIMESTAMP) <= c.intime + INTERVAL 24 HOUR
GROUP BY c.stay_id""")

# ===== 5) first_day 聚合 + pafi 双列 vent 交互 + 6 组分 + SOFA (first_day_sofa.sql 逐句) =====
log("[PASS5] first_day 聚合 + SOFA 计算 ...")
con.execute("""CREATE TABLE fd_sofa AS
WITH fd_v AS (
  SELECT c.stay_id, MIN(v.mbp) mbp_min
  FROM cohort c LEFT JOIN vit_mbp v ON c.stay_id=v.stay_id
    AND v.charttime >= c.intime - INTERVAL 6 HOUR AND v.charttime <= c.intime + INTERVAL 24 HOUR
  GROUP BY c.stay_id),
fd_gcs AS (
  SELECT c.stay_id, g.gcs gcs_min
  FROM cohort c LEFT JOIN (
    SELECT g.stay_id, g.gcs, ROW_NUMBER() OVER (PARTITION BY g.stay_id ORDER BY g.gcs NULLS FIRST) gcs_seq
    FROM gcs_concept g JOIN cohort c2 ON g.stay_id=c2.stay_id
    WHERE g.charttime >= c2.intime - INTERVAL 6 HOUR AND g.charttime <= c2.intime + INTERVAL 24 HOUR) g
    ON c.stay_id=g.stay_id AND g.gcs_seq=1),
fd_lab AS (
  SELECT c.stay_id, MAX(l.creatinine) creatinine_max, MAX(l.bilirubin_total) bilirubin_total_max, MIN(l.platelet) platelets_min
  FROM cohort c LEFT JOIN lab_concept l ON c.subject_id=l.subject_id
    AND l.charttime >= c.intime - INTERVAL 6 HOUR AND l.charttime <= c.intime + INTERVAL 24 HOUR
  GROUP BY c.stay_id),
pafi1 AS (
  SELECT c.stay_id, bg.pao2fio2ratio, CASE WHEN vd.stay_id IS NOT NULL THEN 1 ELSE 0 END isvent
  FROM cohort c
  LEFT JOIN bg_concept bg ON c.subject_id=bg.subject_id
    AND bg.charttime >= c.intime - INTERVAL 6 HOUR AND bg.charttime <= c.intime + INTERVAL 24 HOUR
  LEFT JOIN vent_status vd ON c.stay_id=vd.stay_id
    AND bg.charttime >= vd.starttime AND bg.charttime <= vd.endtime AND vd.ventilation_status='InvasiveVent'),
pafi2 AS (
  SELECT stay_id, MIN(CASE WHEN isvent=0 THEN pao2fio2ratio END) pao2fio2_novent_min,
         MIN(CASE WHEN isvent=1 THEN pao2fio2ratio END) pao2fio2_vent_min
  FROM pafi1 GROUP BY stay_id),
scorecomp AS (
  SELECT c.stay_id, v.mbp_min, mv.rate_norepinephrine, mv.rate_epinephrine, mv.rate_dopamine, mv.rate_dobutamine,
    l.creatinine_max, l.bilirubin_total_max bilirubin_max, l.platelets_min platelet_min,
    pf.pao2fio2_novent_min, pf.pao2fio2_vent_min, uo.urineoutput, g.gcs_min
  FROM cohort c
  LEFT JOIN vaso_mv mv ON c.stay_id=mv.stay_id
  LEFT JOIN pafi2 pf ON c.stay_id=pf.stay_id
  LEFT JOIN fd_v v ON c.stay_id=v.stay_id
  LEFT JOIN fd_lab l ON c.stay_id=l.stay_id
  LEFT JOIN uo ON c.stay_id=uo.stay_id
  LEFT JOIN fd_gcs g ON c.stay_id=g.stay_id),
scorecalc AS (
  SELECT stay_id,
    CASE WHEN pao2fio2_vent_min < 100 THEN 4
         WHEN pao2fio2_vent_min < 200 THEN 3
         WHEN pao2fio2_novent_min < 300 THEN 2
         WHEN pao2fio2_novent_min < 400 THEN 1
         WHEN COALESCE(pao2fio2_vent_min, pao2fio2_novent_min) IS NULL THEN NULL
         ELSE 0 END respiration,
    CASE WHEN platelet_min < 20 THEN 4
         WHEN platelet_min < 50 THEN 3
         WHEN platelet_min < 100 THEN 2
         WHEN platelet_min < 150 THEN 1
         WHEN platelet_min IS NULL THEN NULL ELSE 0 END coagulation,
    CASE WHEN bilirubin_max >= 12.0 THEN 4
         WHEN bilirubin_max >= 6.0 THEN 3
         WHEN bilirubin_max >= 2.0 THEN 2
         WHEN bilirubin_max >= 1.2 THEN 1
         WHEN bilirubin_max IS NULL THEN NULL ELSE 0 END liver,
    CASE WHEN rate_dopamine > 15 OR rate_epinephrine > 0.1 OR rate_norepinephrine > 0.1 THEN 4
         WHEN rate_dopamine > 5 OR rate_epinephrine <= 0.1 OR rate_norepinephrine <= 0.1 THEN 3
         WHEN rate_dopamine > 0 OR rate_dobutamine > 0 THEN 2
         WHEN mbp_min < 70 THEN 1
         WHEN COALESCE(mbp_min, rate_dopamine, rate_dobutamine, rate_epinephrine, rate_norepinephrine) IS NULL THEN NULL
         ELSE 0 END cardiovascular,
    CASE WHEN gcs_min >= 13 AND gcs_min <= 14 THEN 1
         WHEN gcs_min >= 10 AND gcs_min <= 12 THEN 2
         WHEN gcs_min >= 6 AND gcs_min <= 9 THEN 3
         WHEN gcs_min < 6 THEN 4
         WHEN gcs_min IS NULL THEN NULL ELSE 0 END cns,
    CASE WHEN creatinine_max >= 5.0 THEN 4
         WHEN urineoutput < 200 THEN 4
         WHEN creatinine_max >= 3.5 AND creatinine_max < 5.0 THEN 3
         WHEN urineoutput < 500 THEN 3
         WHEN creatinine_max >= 2.0 AND creatinine_max < 3.5 THEN 2
         WHEN creatinine_max >= 1.2 AND creatinine_max < 2.0 THEN 1
         WHEN COALESCE(urineoutput, creatinine_max) IS NULL THEN NULL
         ELSE 0 END renal,
    mbp_min, gcs_min, creatinine_max, bilirubin_max, platelet_min,
    pao2fio2_novent_min, pao2fio2_vent_min, urineoutput,
    rate_norepinephrine, rate_epinephrine, rate_dopamine, rate_dobutamine
  FROM scorecomp)
SELECT stay_id,
  COALESCE(respiration,0)+COALESCE(coagulation,0)+COALESCE(liver,0)+COALESCE(cardiovascular,0)+COALESCE(cns,0)+COALESCE(renal,0) sofa,
  respiration, coagulation, liver, cardiovascular, cns, renal,
  mbp_min, gcs_min, creatinine_max, bilirubin_max, platelet_min,
  pao2fio2_novent_min, pao2fio2_vent_min, urineoutput,
  rate_norepinephrine, rate_epinephrine, rate_dopamine, rate_dobutamine
FROM scorecalc""")

df = con.execute("SELECT * FROM fd_sofa ORDER BY stay_id").df()

# ===== 6) 硬门禁 =====
gates = []
def gate(name, ok, detail=""):
    gates.append({"gate": name, "pass": bool(ok), "detail": str(detail)})
    log(f"  [gate] {name}: {'PASS' if ok else 'FAIL'} {detail}")

gate("n_rows", len(df) == n_cohort, f"{len(df)} vs {n_cohort}")
gate("unique_stay", df.stay_id.nunique() == n_cohort)
comp_cols = ["respiration","coagulation","liver","cardiovascular","cns","renal"]
ok_rng = all(df[c].dropna().between(0,4).all() for c in comp_cols)
gate("component_range_0_4", ok_rng)
recomp = sum(df[c].fillna(0) for c in comp_cols)
gate("sofa_equals_sum", bool((recomp == df.sofa).all()), f"max diff {(recomp-df.sofa).abs().max()}")
gate("sofa_range_0_24", bool(df.sofa.between(0,24).all()), f"min {df.sofa.min()} max {df.sofa.max()}")

# 覆盖率 + 分布 (审计报告)
cov = {c: round(float(df[c].notna().mean()*100),1) for c in
       ["mbp_min","gcs_min","creatinine_max","bilirubin_max","platelet_min",
        "pao2fio2_novent_min","pao2fio2_vent_min","urineoutput",
        "rate_norepinephrine","rate_epinephrine","rate_dopamine","rate_dobutamine"]}
dist = {c: {int(k): int(v) for k,v in df[c].value_counts().sort_index().items()} for c in comp_cols}
report = {
    "script": "08z_sofa.py", "n_stays": int(len(df)),
    "source": "mimic-code/mimic-iv/concepts_postgres first_day_sofa + 12 依赖概念 (DuckDB 忠实移植, 2026-09-03 逐字核证)",
    "itemid_verification": {"icu_checked": len(EXPECT_ICU), "hosp_checked": len(EXPECT_HOSP), "fails": fails},
    "windows": {"vitals_gcs_bg_labs": "[intime-6h, intime+24h]", "vaso_starttime": "[intime-6h, intime+24h]", "uo": "[intime, intime+24h]"},
    "ingredient_coverage_pct": cov,
    "component_distributions": dist,
    "sofa": {"mean": round(float(df.sofa.mean()),2), "median": float(df.sofa.median()),
             "min": int(df.sofa.min()), "max": int(df.sofa.max()),
             "component_missing_pct": {c: round(float(df[c].isna().mean()*100),1) for c in comp_cols}},
    "gates": gates,
}
df.to_parquet(OUT / "08z_sofa24.parquet", index=False)
(REP / "08z_sofa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
all_pass = all(g["pass"] for g in gates)
log(f"SOFA 分布: mean {report['sofa']['mean']} median {report['sofa']['median']} | 组分缺失% {report['sofa']['component_missing_pct']}")
log(f"产物: {OUT/'08z_sofa24.parquet'} | {REP/'08z_sofa_report.json'}")
log("08Z_SOFA_DONE gates=" + ("ALL_PASS" if all_pass else "HAS_FAIL"))
if not all_pass: sys.exit(1)
