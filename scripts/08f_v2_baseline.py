# -*- coding: utf-8 -*-
# 08f v2 — 25 特征版基线对照 ΔAUC DeLong (与 08f v1 同方法学)
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"

screen = json.loads(open(f"{REP}/10c_feature_screen.json", encoding="utf-8").read())
SEL = screen["final_features"]
rep0 = json.loads(open(f"{REP}/08c_matrix_report.json", encoding="utf-8").read())
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, iv = m[m.split == "train"], m[m.split == "intval"]
ytr, yiv = tr.d28.values.astype(int), iv.d28.values.astype(int)

# SOFA
sofa = pd.read_parquet(f"{ROOT}/results/features/08z_sofa24.parquet")[["stay_id", "sofa"]]
tr = tr.merge(sofa, on="stay_id", how="left")
iv = iv.merge(sofa, on="stay_id", how="left")

# 25 特征主模型预测 (v2 冻结管线)
preds = pd.read_parquet(f"{DATA}/08d_v2_preds_mimic4.parquet")
iv["p_full"] = iv.stay_id.map(preds.set_index("stay_id").p_full)
assert not iv.p_full.isna().any()

def delong(y, p1, p2):
    y = np.asarray(y, int); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    def partials(p):
        pos, neg = p[y == 1], p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos])
        V01 = np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg])
        return V10, V01
    P1, P2 = partials(p1), partials(p2)
    a1, a2 = P1[0].mean(), P2[0].mean()
    v1 = float(np.var(P1[0], ddof=1)) / len(P1[0]) + float(np.var(P1[1], ddof=1)) / len(P1[1])
    v2 = float(np.var(P2[0], ddof=1)) / len(P2[0]) + float(np.var(P2[1], ddof=1)) / len(P2[1])
    c10 = float(np.cov(P1[0], P2[0], ddof=1)[0, 1]) / len(P1[0])
    c01 = float(np.cov(P1[1], P2[1], ddof=1)[0, 1]) / len(P1[1])
    tot = v1 + v2 - 2 * (c10 + c01)
    d_ = a1 - a2; se = tot ** 0.5
    return dict(AUC_full=round(a1, 4), AUC_base=round(a2, 4), delta=round(float(d_), 4),
                se=round(float(se), 4), p=round(float(2 * (1 - norm.cdf(abs(d_ / se)))), 5))

res = {"version": "08f_v2_25feat", "eval_set": "mimic4 intval n=483 ev=81"}
for tag, cols in [("base5", ["admission_age", "sex_female", "gcs_total_first", "charlson_comorbidity_score", "sofa"]),
                  ("trad2", ["sofa", "gcs_total_first"])]:
    dc = tr.dropna(subset=cols + ["d28"])
    mu, sd = dc[cols].mean(), dc[cols].std()
    lr_ = LogisticRegression(max_iter=5000, random_state=42)
    lr_.fit(((dc[cols] - mu) / sd).values, dc.d28.values)
    p_base = lr_.predict_proba(((iv[cols] - mu) / sd).values)[:, 1]
    r = delong(yiv, iv.p_full.values, p_base)
    r["n_eval"] = len(iv)
    res[tag] = r
    print(f"[{tag}] full {r['AUC_full']} vs base {r['AUC_base']} | Δ={r['delta']} p={r['p']}")

json.dump(res, open(f"{REP}/08f_v2_base_delta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DONE 08f v2")
