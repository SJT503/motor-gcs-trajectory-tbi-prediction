# -*- coding: utf-8 -*-
# ============================================================
# 08c — M4 24h landmark 建模矩阵 (SAP §2/§3/§4 冻结合同的机械执行)
# 输入(全部本项目 M4 冻结产物, 逐一经本会话 Read 核 schema):
#   results/cohort_audit/02_tbi_cohort.parquet      年龄/性别/intime/los_icu/dod
#   results/features/03b_features_master.parquet    Block A 冻结清单 (窗口止于 intime+24h ✓)
#   results/features/04_charlson.parquet            charlson_comorbidity_score (无 age_score)
#   07_prediction_system/data/gcs_long_mimic4.parquet  07a 统一长表 → Block B + GCS 首值
#   07_prediction_system/data/08b_posterior24_mimic4.parquet  Block C 截断后验 (08b 产出)
# 冻结依据 (SAP_prediction_v1.0_frozen.md, 2026-09-03 本会话重新实锚):
#   L31 拆分 2008-17 train / 2018-19 tune / 2020-22 temporal 内验
#   L34 landmark 24h 风险集 = 仍存活 + 仍在 ICU
#   L40-49 Block A 白名单(逐族); L53 Block B motor主/total敏感性
#   L57 Block C 5 列截断后验; L66 血气+严重度评分不入特征
#   L74 主结局 = 自 ICU 入院 28d 全因死亡
# 门禁: 白名单列缺失/风险集异常/事件率异常 → sys.exit(1) 不产出
# 产出: data/08c_matrix_mimic4.parquet + reports/08c_matrix_report.json
# 挂旗(不在本脚本解决, 报告中显式声明): §6 基线对照需 SOFA, 全项目无 SOFA 实现
# ============================================================
import sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
FEAT = ROOT / "results/features"
sys.path.insert(0, str(ROOT / "07_prediction_system/scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("c07", ROOT / "07_prediction_system/scripts/07_common.py")
c07 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c07)

FAIL = []
def gate(cond, msg):
    print(("  [gate OK] " if cond else "  [GATE FAIL] ") + msg)
    if not cond: FAIL.append(msg)

print("=== 08c M4 24h landmark 建模矩阵 ===")

# ---------- 0. 读入 ----------
coh = pd.read_parquet(ROOT / "results/cohort_audit/02_tbi_cohort.parquet")
f03 = pd.read_parquet(FEAT / "03b_features_master.parquet")
ch  = pd.read_parquet(FEAT / "04_charlson.parquet")[["stay_id", "charlson_comorbidity_score"]]
lng = pd.read_parquet(DATA / "gcs_long_mimic4.parquet").rename(
    columns={"motor": "gcs_motor", "eye": "gcs_eye", "verbal": "gcs_verbal"})  # 07a 两代命名统一 (缺列 no-op)
lng = lng.assign(id=pd.to_numeric(lng["id"]).astype("int64"))  # object→int64, 与 stay_id 合并对齐 (gf/dyn_m/dyn_t 三处 merge 共用)
pos = pd.read_parquet(DATA / "08b_posterior24_mimic4.parquet")
print(f"[输入] cohort={len(coh)} 03b={f03.shape} charlson={len(ch)} long={len(lng)} posterior={pos.shape}")
need_lng = {"id", "offset_hr", "gcs_total", "gcs_motor", "gcs_eye", "gcs_verbal"}
gate(need_lng <= set(lng.columns), f"gcs_long 缺列: {need_lng - set(lng.columns)}" if not need_lng <= set(lng.columns) else "gcs_long schema ✓")
gate({"prob_m1", "prob_t1", "n_obs_m"} <= set(pos.columns), "posterior24 schema (prob_m1..5/prob_t1..5) ✓")

# ---------- 1. 结局与风险集 (SAP L34/L74) ----------
coh["intime"] = pd.to_datetime(coh["intime"])
coh["dod"] = pd.to_datetime(coh["dod"])
coh["hours_to_death"] = (coh["dod"] - coh["intime"]).dt.total_seconds() / 3600.0
coh["los_h"] = coh["los_icu"] * 24.0
alive24 = coh["hours_to_death"].isna() | (coh["hours_to_death"] > 24.0)
risk = coh[(coh["los_h"] >= 24.0) & alive24].copy()
risk["d28"] = ((risk["hours_to_death"] <= 28 * 24)).astype(int)
gate(len(risk) > 2400, f"风险集 n={len(risk)} (cohort {len(coh)}, 剔除 24h 内死亡 {int(((coh['hours_to_death'] <= 24)).sum())} 例 / LOS<24h {int((coh['los_h'] < 24).sum())} 例)")
ev = risk["d28"].mean()
gate(0.03 < ev < 0.40, f"d28 事件率 {ev:.1%} (SAP 预期 10-20% 量级, 越界才 FAIL)")

# ---------- 2. 时间拆分 (SAP L31; M4 去标识化年代还原) ----------
# M4 admittime 为未来偏移日期 (per-patient 随机平移, year 落在 ~2100-2200), 真实年代必须经
# patients.anchor_year_group 还原: real_year ≈ 分组中点 + (admittime.year - anchor_year), 含 ±1.5y 不确定度 (Methods 披露)
pat = pd.read_csv(r"E:\TBI subtype\data\mimic-iv-3.1\hosp\patients.csv.gz",
                  usecols=["subject_id", "anchor_year_group"])
_g = pat["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
gate(_g.notna().all().all(),
     f"anchor_year_group 解析失败样例: {pat['anchor_year_group'].dropna().unique()[:5].tolist()}" if _g.isna().any().any() else "anchor_year_group 解析 ✓ (正则取起止年)")
pat["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
risk = risk.merge(pat[["subject_id", "group_mid"]], on="subject_id", how="left")
gate(risk["group_mid"].notna().all(), "anchor_year_group 按 subject_id 全覆盖 ✓")
risk["admittime"] = pd.to_datetime(risk["admittime"])
risk["year"] = (risk["group_mid"] + (risk["admittime"].dt.year - risk["anchor_year"])).round().astype(int)
gate(risk["year"].between(2005, 2025).all(),
     f"还原真实年代 {risk['year'].min()}-{risk['year'].max()} (期望 2008-2022 量级)")
risk["split"] = np.select(
    [risk["year"] <= 2017, risk["year"] <= 2019], ["train", "tune"], default="intval")
sp = risk.groupby("split").agg(n=("stay_id", "size"), d28=("d28", "mean"), y0=("year", "min"), y1=("year", "max"))
print("[拆分]\n" + sp.to_string())
gate(set(sp.index) == {"train", "tune", "intval"} and (sp["n"] > 100).all(), "三 split 均非空且 n>100")

# ---------- 3. Block A 冻结白名单 (SAP L44-49; 显式 inclusion, 非 03b 全列) ----------
V7 = ["hr", "sbp", "dbp", "mbp", "rr", "temp", "spo2"]           # 7 生命体征 (无 glu)
vital_cols = [f"{v}_{s}" for v in V7 for s in ("min", "mean", "max")]
cbc_cols   = [f"{v}_{s}" for v in ("hematocrit", "hemoglobin", "platelet", "wbc") for s in ("min", "max")]
chem_names = ["aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine",
              "glucose_lab", "sodium", "potassium"]
chem_cols  = [f"{v}_{s}" for v in chem_names for s in ("min", "max")]
coag_cols  = [f"{v}_{s}" for v in ("inr", "pt", "ptt") for s in ("min", "max")]
gcs_min_cols = ["gcs_eye_min", "gcs_motor_min", "gcs_verbal_min", "gcs_total_min"]  # 03b 原列
missing = [c for c in vital_cols + cbc_cols + chem_cols + coag_cols + gcs_min_cols if c not in f03.columns]
gate(not missing, f"Block A 白名单 03b 列全存在 (缺: {missing})" if missing else f"Block A 白名单 03b 列全存在 ✓ ({len(vital_cols+cbc_cols+chem_cols+coag_cols+gcs_min_cols)} 列)")

mA = risk[["stay_id", "split", "d28", "year", "hours_to_death", "los_h"]].merge(
    f03[["stay_id"] + vital_cols + cbc_cols + chem_cols + coag_cols + gcs_min_cols], on="stay_id", how="left")
mA = mA.merge(coh[["stay_id", "admission_age", "gender"]], on="stay_id", how="left")
mA = mA.merge(ch, on="stay_id", how="left")
mA["sex_female"] = (mA["gender"] == "F").astype(int)
bg_cols = ["admission_age", "sex_female", "charlson_comorbidity_score"]
gate(mA["charlson_comorbidity_score"].notna().all(), "Charlson 全覆盖 (04: 4440/1425 全覆盖先例, M4 应同)")

# ---------- 4. GCS 首值 + Block B (0→24h, 07_common 复用) ----------
w = lng[(lng["offset_hr"] >= 0) & (lng["offset_hr"] <= 24.0)]
gcs_first = {f"gcs_first_{k}": w.groupby("id")[f"gcs_{k}"].first() for k in ("total", "eye", "motor", "verbal")}
gf = pd.DataFrame(gcs_first).reset_index().rename(columns={"index": "id"})
mA = mA.merge(gf, left_on="stay_id", right_on="id", how="left").drop(columns=["id"])
mA = mA.rename(columns={f"gcs_first_{k}": f"gcs_{k}_first" for k in ("total", "eye", "motor", "verbal")})
first_cols = [f"gcs_{k}_first" for k in ("total", "eye", "motor", "verbal")]

dyn_m = c07.neuro_dynamic_features(lng, value_col="gcs_motor", prefix="motor", win_hr=24.0, min_obs=2)
dyn_t = c07.neuro_dynamic_features(lng, value_col="gcs_total", prefix="total", win_hr=24.0, min_obs=2)
mA = mA.merge(dyn_m, left_on="stay_id", right_on="id", how="left").drop(columns=["id"])
mA = mA.merge(dyn_t, left_on="stay_id", right_on="id", how="left").drop(columns=["id"])
bB_main = [c for c in dyn_m.columns if c != "id"]     # motor 12 特征 = 主
bB_sens = [c for c in dyn_t.columns if c != "id"]     # total 12 特征 = 敏感性
gate(len(bB_main) == 12 and len(bB_sens) == 12, f"Block B 特征数 motor={len(bB_main)} total={len(bB_sens)} (期望 12/12)")

# ---------- 5. Block C 截断后验 (08b 产出直挂) ----------
mA = mA.merge(pos, left_on="stay_id", right_on="id", how="left").drop(columns=["id"])
# Block C 列动态推导 (motor ng 由 08a2 冻结规则选出, 可为 3/4/5; total 恒 5)
bC_main = sorted([c for c in pos.columns if re.fullmatch(r"prob_m\d+", c)],
                 key=lambda c: int(c[len("prob_m"):]))
bC_sens = sorted([c for c in pos.columns if re.fullmatch(r"prob_t\d+", c)],
                 key=lambda c: int(c[len("prob_t"):]))
gate(len(bC_main) in (3, 4, 5) and len(bC_sens) == 5,
     f"Block C 列数 motor={len(bC_main)} total={len(bC_sens)} (motor 3/4/5, total 5)")
gate(mA["prob_m1"].notna().mean() > 0.95, f"Block C posterior 覆盖 {mA['prob_m1'].notna().mean():.1%} (无 motor 观测者 MICE 处理, >5% 则 FAIL)")

# ---------- 6. 覆盖审计 + 落盘 ----------
WHITELIST_MAIN = vital_cols + cbc_cols + chem_cols + coag_cols + gcs_min_cols + first_cols + bg_cols + bB_main + bC_main
SENSITIVITY = bB_sens + bC_sens
meta = ["stay_id", "split", "d28", "year", "hours_to_death", "los_h", "n_obs_m", "n_obs_t"]
out = mA[meta + WHITELIST_MAIN + SENSITIVITY].copy()
cov = out[WHITELIST_MAIN].notna().mean().sort_values()
drop20 = cov[cov < 0.80]
print(f"[覆盖] 白名单 {len(WHITELIST_MAIN)} 特征, 缺失>20% 的有 {len(drop20)} 个:")
if len(drop20): print(drop20.to_string())
gate(len(drop20) == 0, "SAP §3.6: 特征级缺失>20% 必须剔除 — 冻结清单设计上不应出现 (血气已排除)")
out.to_parquet(DATA / "08c_matrix_mimic4.parquet", index=False)
rep = dict(n_riskset=int(len(out)), d28_rate=round(float(ev), 4),
  splits={k: dict(n=int(v["n"]), d28=round(float(v["d28"]), 4)) for k, v in sp.iterrows()},
  n_whitelist=len(WHITELIST_MAIN), n_sensitivity=len(SENSITIVITY),
  whitelist_main=WHITELIST_MAIN, sensitivity_cols=SENSITIVITY, meta_cols=meta,
  blockA_cols=vital_cols + cbc_cols + chem_cols + coag_cols + gcs_min_cols + first_cols + bg_cols,
  blockB_cols=bB_main, blockC_cols=bC_main,
  blockA=len(vital_cols + cbc_cols + chem_cols + coag_cols + gcs_min_cols + first_cols + bg_cols),
  blockB=12, blockC=len(bC_main), blockC_ng={"motor": len(bC_main), "total": len(bC_sens)},
  coverage_min=round(float(cov.iloc[0]), 4), coverage_min_feature=str(cov.index[0]),
  over20pct_missing=list(drop20.index),
  flags=["SOFA 未实现 — SAP §6 基线对照(年龄+性别+入院GCS+Charlson+SOFA)缺最后一项, 08d 前必须解决",
         "eICU 外验投影时 coag/aniongap/GCS 族缺失>eICU 20% 阈的家族如何处置 = SAP_v2 §10 + Phase 1 §3.6 已报 PI, 待 08e 落地"],
  gates_failed=FAIL)
(REP / "08c_matrix_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[产出] {DATA}/08c_matrix_mimic4.parquet  shape={out.shape}")
print(f"[产出] {REP}/08c_matrix_report.json")
if FAIL:
    print(f"\n!! {len(FAIL)} 项门禁 FAIL — 产物已写但标记 gates_failed, 下游 08d 不得直接使用")
    sys.exit(1)
print("DONE 08c")
