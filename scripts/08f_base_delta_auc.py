# -*- coding: utf-8 -*-
# ============================================================
# 08f — SAP §6 基线对照 ΔAUC (DeLong): full (08d) vs base / 传统评分档
#   base    = 年龄 + 性别 + 入院GCS(gcs_total_first) + Charlson + SOFA(08z)
#   传统档  = SOFA + 入院GCS 单点 (SAP L112 第一档)
#   full    = 08d_preds_mimic4.parquet 的 p_full (intval 时段验证)
# LR 拟合于 train, 在 intval 上评估 (与 08d 同一评估集, zero-touch)
# 产出: reports/08f_base_delta_auc.json
# 挂旗: 统计ML档(L1-LR/RF/XGBoost)与 DL 档(LSTM, 仅Supp)不在本脚本 — 独立 benchmark 工作包
# ============================================================
import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

print("=== 08f 基线对照 ΔAUC (SAP §6 次级 endpoint 主链) ===")

# ---------- 1. 输入 ----------
m = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
sofa = pd.read_parquet(ROOT / "results/features/08z_sofa24.parquet")[["stay_id", "sofa"]]
preds = pd.read_parquet(DATA / "08d_preds_mimic4.parquet")
lg(f"matrix={m.shape} sofa={len(sofa)} preds={len(preds)} (sofa 缺失 {int(sofa.sofa.isna().sum())})")

BASE_VARS = ["admission_age", "sex_female", "gcs_total_first", "charlson_comorbidity_score", "sofa"]
TRAD_VARS = ["sofa", "gcs_total_first"]

df = m.merge(sofa, on="stay_id", how="left").merge(preds[["stay_id", "p_full"]], on="stay_id", how="left")
iv = df[df.split == "intval"].copy()
tr = df[df.split == "train"].copy()
if iv.p_full.isna().any():
    lg(f"!! intval 有 {int(iv.p_full.isna().sum())} 例无 08d 预测 — 检查 08d_preds 产出"); sys.exit(1)
y = iv.d28.values

# ---------- 2. DeLong 配对检验 (DeLong et al. 1988, 自含实现, 结值 0.5 处理) ----------
def delong_compare(y, p1, p2):
    """配对 DeLong: 同一样本两模型
       V10_i = (Σ_j [p_i>p_j]+0.5[p_i=p_j])/n0  (case 侧) ; V01_j 对称 (control 侧)
       AUC=mean(V10); Var=Var(V10)/n1+Var(V01)/n0; Cov=Cov(V10a,V10b)/n1+Cov(V01a,V01b)/n0
       z = (A1-A2)/sqrt(Var1+Var2-2Cov)"""
    from scipy.stats import norm
    y = np.asarray(y, int)
    def partials(p):
        p = np.asarray(p, float)
        pos, neg = p[y == 1], p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos])
        V01 = np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg])
        return V10, V01
    P1 = partials(p1); P2 = partials(p2)
    a1 = P1[0].mean(); a2 = P2[0].mean()
    v1 = np.var(P1[0], ddof=1) / len(P1[0]) + np.var(P1[1], ddof=1) / len(P1[1])
    v2 = np.var(P2[0], ddof=1) / len(P2[0]) + np.var(P2[1], ddof=1) / len(P2[1])
    c10 = np.cov(P1[0], P2[0], ddof=1)[0, 1] / len(P1[0])
    c01 = np.cov(P1[1], P2[1], ddof=1)[0, 1] / len(P1[1])
    cov = c10 + c01
    d = a1 - a2
    se = np.sqrt(max(v1 + v2 - 2 * cov, 1e-18))
    z = d / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return dict(AUC_full=round(a1, 4), AUC_base=round(a2, 4), delta=round(float(d), 4),
                se=round(float(se), 4), z=round(float(z), 3), p=round(float(pval), 5),
                var_full=round(float(v1), 6), var_base=round(float(v2), 6), cov=round(float(cov), 6))

# ---------- 3. 两个基线档拟合 (train) → intval 预测 ----------
def fit_predict_lr(cols, label):
    trc = tr.dropna(subset=cols + ["d28"])
    ivc = iv.dropna(subset=cols)
    mu, sd = trc[cols].mean(), trc[cols].std(ddof=0).replace(0, 1.0)
    Xtr = ((trc[cols] - mu) / sd).values
    Xiv = ((ivc[cols] - mu) / sd).values
    lr_ = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, trc.d28.values)
    lg(f"[{label}] train n={len(trc)} (ev {trc.d28.mean():.1%}) | intval 可评 n={len(ivc)}/{len(iv)} (基线缺失 {len(iv)-len(ivc)})")
    return ivc.stay_id.values, lr_.predict_proba(Xiv)[:, 1]

res = {"eval_set": "mimic4 intval (temporal 内验, 与 08d 同集)", "n_intval": int(len(iv)),
       "events": int(y.sum()),
       "flags": ["统计ML档(L1-LR/RF/XGBoost)与DL档(LSTM 仅Supp)未在本脚本 — 独立 benchmark 工作包 (SAP L112-113)"]}

from sklearn.metrics import roc_auc_score, brier_score_loss
ids_b, p_base = fit_predict_lr(BASE_VARS, "base5")
ids_t, p_trad = fit_predict_lr(TRAD_VARS, "trad2")
al = pd.DataFrame({"stay_id": iv.stay_id.values, "p_full": iv.p_full.values, "d28": y})
al_b = al.merge(pd.DataFrame({"stay_id": ids_b, "p_base": p_base}), on="stay_id")
res["base5_vars"] = BASE_VARS; res["trad2_vars"] = TRAD_VARS
res["base5"] = delong_compare(al_b.d28.values, al_b.p_full.values, al_b.p_base.values)
res["base5"]["AUC_base_alone"] = round(float(roc_auc_score(al_b.d28, al_b.p_base)), 4)
res["base5"]["Brier_base"] = round(float(brier_score_loss(al_b.d28, al_b.p_base)), 4)
res["base5"]["n_eval"] = int(len(al_b))
al_t = al.merge(pd.DataFrame({"stay_id": ids_t, "p_trad": p_trad}), on="stay_id")
res["trad2"] = delong_compare(al_t.d28.values, al_t.p_full.values, al_t.p_trad.values)
res["trad2"]["AUC_trad_alone"] = round(float(roc_auc_score(al_t.d28, al_t.p_trad)), 4)
res["trad2"]["Brier_trad"] = round(float(brier_score_loss(al_t.d28, al_t.p_trad)), 4)
res["trad2"]["n_eval"] = int(len(al_t))

lg(f"ΔAUC full vs base5 : {res['base5']['delta']:+.4f} (p={res['base5']['p']})")
lg(f"ΔAUC full vs trad2 : {res['trad2']['delta']:+.4f} (p={res['trad2']['p']})")
(REP / "08f_base_delta_auc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"产出: {REP}/08f_base_delta_auc.json")
print("DONE 08f")
