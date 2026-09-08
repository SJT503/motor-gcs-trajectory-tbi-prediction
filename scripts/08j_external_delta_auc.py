# -*- coding: utf-8 -*-
# 08j — 外库基线对照 ΔAUC (DeLong 配对): full (08e 冻结模型) vs base5 / trad2
# 基线 LR 拟合于 M4 train (与 08f 完全同方法学: z-score 用 train 统计, LR max_iter=2000 rs=42)
# 评估集 = 08e 风险集 (died24/short 已排除); 外库 SOFA = 08h (M3) / 08i (eICU) 移植
# 输出: data/08j_external_delta.json + reports/_08j_summary 打印
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

t0 = time.time()
ROOT = Path(r"E:/TBI subtype")
OUT = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

BASE_VARS = ["admission_age", "sex_female", "gcs_total_first", "charlson_comorbidity_score", "sofa"]
TRAD_VARS = ["sofa", "gcs_total_first"]

# ---------- 1. M4 train 拟合基线 LR (08f 同款) ----------
m4 = pd.read_parquet(OUT / "08c_matrix_mimic4.parquet")
sofa4 = pd.read_parquet(ROOT / "results/features/08z_sofa24.parquet")[["stay_id", "sofa"]]
tr = m4[m4.split == "train"].merge(sofa4, on="stay_id", how="left")
lg(f"M4 train n={len(tr)} (ev {tr.d28.mean():.1%}, sofa 缺失 {int(tr.sofa.isna().sum())})")

def fit_lr(cols):
    trc = tr.dropna(subset=cols + ["d28"])
    mu, sd = trc[cols].mean(), trc[cols].std(ddof=0).replace(0, 1.0)
    lr_ = LogisticRegression(max_iter=2000, random_state=42).fit(
        ((trc[cols] - mu) / sd).values, trc.d28.values)
    return mu, sd, lr_, len(trc)

models = {v: fit_lr(vs) for v, vs in dict(base5=BASE_VARS, trad2=TRAD_VARS).items()}
lg(f"基线拟合完成: {[(k, v[3]) for k, v in models.items()]}")

# ---------- 2. DeLong 配对 (08f 自含实现逐字复用) ----------
def delong_compare(y, p1, p2):
    from scipy.stats import norm
    y = np.asarray(y, int)
    def partials(p):
        p = np.asarray(p, float)
        pos, neg = p[y == 1], p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos])
        V01 = np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg])
        return V10, V01
    P1, P2 = partials(p1), partials(p2)
    a1, a2 = P1[0].mean(), P2[0].mean()
    v1 = np.var(P1[0], ddof=1) / len(P1[0]) + np.var(P1[1], ddof=1) / len(P1[1])
    v2 = np.var(P2[0], ddof=1) / len(P2[0]) + np.var(P2[1], ddof=1) / len(P2[1])
    cov = (np.cov(P1[0], P2[0], ddof=1)[0, 1] / len(P1[0])
           + np.cov(P1[1], P2[1], ddof=1)[0, 1] / len(P1[1]))
    d = a1 - a2; se = np.sqrt(max(v1 + v2 - 2 * cov, 1e-18)); z = d / se
    return dict(AUC_full=round(a1, 4), AUC_base=round(a2, 4), delta=round(float(d), 4),
                se=round(float(se), 4), z=round(float(z), 3),
                p=round(float(2 * (1 - norm.cdf(abs(z)))), 5))

# ---------- 3. 外库评估集组装 ----------
def build(db):
    if db == "eicu":
        fe = pd.read_parquet(OUT / "features_eicu.parquet").rename(columns={"age": "admission_age"})
        fe["sex_female"] = 1 - fe["male"]
        preds = pd.read_parquet(OUT / "08e_preds_eicu.parquet")          # icustay_id_eicu, p_full, d28, post_avail
        sof = pd.read_parquet(OUT / "08i_sofa_eicu.parquet")[["icustay_id_eicu", "sofa"]]
        key = "icustay_id_eicu"
    else:
        fe = pd.read_parquet(OUT / "features_mimic3.parquet").rename(columns={"age": "admission_age"})
        fe["sex_female"] = 1 - fe["male"]
        preds = pd.read_parquet(OUT / "08e_preds_mimic3.parquet")        # icustay_id, p_full, d28, post_avail
        sof = pd.read_parquet(OUT / "08h_sofa_mimic3.parquet")[["icustay_id", "sofa"]]
        key = "icustay_id"
    d = preds.merge(fe[[key] + BASE_VARS[:-1]], on=key, how="left").merge(sof, on=key, how="left")
    assert d[key].is_unique
    return d

res, gates = {}, {}
for db in ["eicu", "mimic3"]:
    d = build(db)
    n_full = len(d)
    miss = {c: int(d[c].isna().sum()) for c in BASE_VARS}
    for tag, cols in dict(base5=BASE_VARS, trad2=TRAD_VARS).items():
        dc = d.dropna(subset=cols + ["p_full", "d28"]).copy()
        mu, sd, lr_, _ = models[tag]
        dc["p_base"] = lr_.predict_proba(((dc[cols] - mu) / sd).values)[:, 1]
        r = delong_compare(dc.d28.values, dc.p_full.values, dc.p_base.values)
        r.update(n=n_full, n_eval=len(dc), n_died=int(dc.d28.sum()), base_missing=miss)
        res[f"{db}_{tag}"] = r
        lg(f"[{db} {tag}] n={n_full}→可评 {len(dc)} | full {r['AUC_full']} vs base {r['AUC_base']} | Δ={r['delta']} p={r['p']}")
    # 门 = 配对 DeLong 可计算 (两类均非零); 比例不作硬门 (与 08f complete-case 惯例一致), 缺失与子集偏差全披露
    nd = res[f"{db}_base5"]["n_died"]
    gates[f"{db}_paired_computable"] = res[f"{db}_base5"]["n_eval"] > 0 and 0 < nd < res[f"{db}_base5"]["n_eval"]

REF_08E = {"eicu": 0.830, "mimic3": 0.8823}   # 08e 全风险集 AUROC (CareVue-only 2026-09-07, reports/08e_external_validation.json 本轮实读)
out = dict(
    method="base LR 拟合于 M4 train (08f 同方法学); full = 08e 冻结模型外验预测; 外库 SOFA = 08h(M3)/08i(eICU) 移植; 评估集 = 08e 风险集",
    inputs=dict(sofa_eicu="08i_sofa_eicu.parquet (sofa 恒有值, 缺失组件计0)", sofa_m3="08h_sofa_mimic3.parquet (同)"),
    results=res, gates=gates,
    auc_full_riskset_08e=REF_08E,
    evaluable_subset_note=(
        "complete-case 配对 DeLong (与 08f 内验基线同惯例)。eICU gcs_total_first 缺 1055/4440 (23.8%) = eICU GCS 记录稀疏, "
        "与 08b 后验不可得人群高度重叠 (见 08g: 后验可得 0.8793 vs 插补 0.7063); "
        "AUC_full 在该子集高于 08e 全风险集 0.830 — 与 08g '轨迹特征价值集中于有 GCS 记录者' 同向, "
        "Methods/Discussion 需同时报告两口径 (全集 0.830 与 complete-case 0.8657)"),
    train=dict(n=len(tr), base5_n=models["base5"][3], trad2_n=models["trad2"][3]))
(REP / "08j_external_delta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
lg("JSON → reports/08j_external_delta.json")
if not all(gates.values()):
    print(f"[GATE FAIL] {gates}"); sys.exit(1)
print(f"DONE 08j: eicu base5 Δ={res['eicu_base5']['delta']} (p={res['eicu_base5']['p']}) | m3 base5 Δ={res['mimic3_base5']['delta']} (p={res['mimic3_base5']['p']})")
