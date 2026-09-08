# -*- coding: utf-8 -*-
# 10b — 三级部署级联 (SAP §12 post-hoc companion, 2026-09-06 PI 指令)
#   F = 79 全量 (=08d 主模型, 直接引用其结果作参照, 不重跑)
#   B = 47 无化验档 (砍 32 化验/凝血)
#   G = 25 纯 GCS 档 (再砍 21 生命体征; 留 GCS 8 + 年龄/性别 2 + B 12 + C 3)
# 规则: 可得性分层 (先化验后体征), 与性能无关; 同嵌套协议 (5×5, seed 42, grid 32)
# 插补: 复用 08d 五套 MICE (插补模型同, 取特征子集)
# 外验: eICU/M3 zero-touch (08k 建库逻辑逐字复刻)
import json, time, warnings
import numpy as np
import pandas as pd
import joblib, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"
t0 = time.time()
def lg(m):
    print(m, flush=True)

rep0 = json.loads(open(f"{REP}/08c_matrix_report.json", encoding="utf-8").read())
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
bA = rep0["blockA_cols"]
VITALS = [c for c in bA if c.split("_")[0] in ("hr","sbp","dbp","mbp","rr","temp","spo2")]
LABS = [c for c in bA if c not in VITALS and c not in ("admission_age","sex_female","charlson_comorbidity_score")
        and not c.startswith("gcs_")]
GCSA = [c for c in bA if c.startswith("gcs_")]
DEMOG = ["admission_age", "sex_female"]
TIER_B = sorted(VITALS + GCSA + DEMOG + ["charlson_comorbidity_score"] + rep0["blockB_cols"] + rep0["blockC_cols"])
TIER_G = sorted(GCSA + ["admission_age", "sex_female"] + rep0["blockB_cols"] + rep0["blockC_cols"])
assert len(LABS) == 32, f"labs={len(LABS)}"
assert len(TIER_B) == 47 and len(TIER_G) == 25, (len(TIER_B), len(TIER_G))
lg(f"[tiers] F=79 B={len(TIER_B)} G={len(TIER_G)} | labs cut={len(LABS)}, vitals cut={len(VITALS)}")

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, iv = m[m.split == "train"], m[m.split == "intval"]
ytr, yiv = tr.d28.values, iv.d28.values
preds = pd.read_parquet(f"{DATA}/08d_preds_mimic4.parquet")
iv["p_full"] = iv.stay_id.map(preds.set_index("stay_id").p_full)
assert not iv.p_full.isna().any()

imp_sets = {j: pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib").transform(m[FULL]),
                            columns=FULL, index=m.index) for j in range(5)}
tr_idx, iv_idx = tr.index.to_numpy(), iv.index.to_numpy()
Xtr, Xiv = {j: imp_sets[j].loc[tr_idx] for j in range(5)}, {j: imp_sets[j].loc[iv_idx] for j in range(5)}

GRID = [dict(nl=nl, md=md, lr=lr, col=col, sub=sub, mcs=mcs)
        for nl in (15, 31, 63) for md in (3, 5, -1) for lr in (0.03, 0.05)
        for col in (0.8, 1.0) for sub in (0.8,) for mcs in (20, 40)]
lg(f"[grid] {len(GRID)} configs")

def make_lgb(c, spw):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=c["lr"], num_leaves=c["nl"],
                              max_depth=c["md"], colsample_bytree=c["col"], subsample=c["sub"],
                              subsample_freq=1, min_child_samples=c["mcs"],
                              scale_pos_weight=spw, random_state=42, n_jobs=4, verbose=-1)

def nested_fit(cols, tag):
    X1 = Xtr[0][cols]
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    spw = (len(ytr) - ytr.sum()) / ytr.sum()
    best, oof = None, []
    for k, (itr, ivo) in enumerate(outer.split(X1, ytr)):
        inner = StratifiedKFold(5, shuffle=True, random_state=42 + 100 + k)
        Xa, ya = X1.iloc[itr], ytr[itr]
        bb, bu = None, -1
        for c in GRID:
            aucs = []
            for it2, iv2 in inner.split(Xa, ya):
                mdl = make_lgb(c, spw).fit(Xa.iloc[it2], ya[it2])
                aucs.append(roc_auc_score(ya[iv2], mdl.predict_proba(Xa.iloc[iv2])[:, 1]))
            mu = float(np.mean(aucs))
            if mu > bu: bu, bb = mu, c
        mdl = make_lgb(bb, spw).fit(Xa, ya)
        oof.append(roc_auc_score(ytr[ivo], mdl.predict_proba(X1.iloc[ivo])[:, 1]))
        best = bb
    lg(f"[{tag}] best_cfg={best} oof={np.mean(oof):.4f}")
    p_iv = np.zeros(len(iv))
    for j in range(5):
        mdl = make_lgb(best, spw).fit(Xtr[j][cols], ytr)
        p_iv += mdl.predict_proba(Xiv[j][cols])[:, 1] / 5
    return best, p_iv

# --- 08k 建库逻辑逐字复刻 (外验) ---
def strip_dyn(df):
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)
NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})
def build(tag):
    fe = pd.read_parquet(f"{DATA}/features_{tag}.parquet").rename(columns={"age": "admission_age"})
    fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
    ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
    if ic: fe = NULLABLE(fe, ic)
    pk = "icustay_id_eicu" if tag == "eicu" else "icustay_id"
    post = pd.read_parquet(f"{DATA}/08b_posterior24_{'eicu' if tag=='eicu' else 'mimic3'}.parquet").rename(columns={"id": pk})
    d = fe.merge(post, on=pk, how="left")
    if tag == "eicu":
        died24 = ((d.died_hosp == 1) & (d.los_h <= 24)); short = d.los_h < 24
        d = d[~(died24 | short)].copy()
        y = d.died_hosp_28d.astype(int).values
    else:
        died24 = (d.d28 == 1) & (d.days_to_death < 1.0); short = d.los_h < 24
        d = d[~(died24 | short)].copy()
        y = d.d28.astype(int).values
    miss = [c for c in FULL if c not in d.columns]
    assert not miss, f"[{tag}] missing {miss}"
    assert len(d) > 1000 and len(y) == len(d), f"[{tag}] build异常 n={len(d)}"
    return d, y, pk

d_ei, y_ei, k_ei = build("eicu")
d_m3, y_m3, k_m3 = build("mimic3")

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
    from scipy.stats import norm
    d_ = a1 - a2; se = tot ** 0.5
    return round(a1, 4), round(a2, 4), round(float(d_), 4), round(float(se), 4), round(float(2 * (1 - norm.cdf(abs(d_ / se)))), 5)

spw = (len(ytr) - ytr.sum()) / ytr.sum()
res = {"design": "deployment tier cascade (availability-driven; F=79 frozen reference from 08d, B=47 no-labs, G=25 GCS-core); SAP §12 2026-09-06",
       "tiers": {"B_n_features": len(TIER_B), "G_n_features": len(TIER_G),
                 "labs_removed": len(LABS), "vitals_removed": len(VITALS)},
       "reference_F": {"intval": {"AUROC": 0.9054}}}
# F 档外验参照 (08e p_full 复算, 与 08k 同法)
for tag, dd, yy, pk, refname in [("eicu", d_ei, y_ei, k_ei, "08e_preds_eicu.parquet"),
                                  ("mimic3", d_m3, y_m3, k_m3, "08e_preds_mimic3.parquet")]:
    pass  # p_full 参照直接读 08e preds (与 08k 相同)
ref_ei = pd.read_parquet(f"{DATA}/08e_preds_eicu.parquet")
ref_m3 = pd.read_parquet(f"{DATA}/08e_preds_mimic3.parquet")
d_ei = d_ei.merge(ref_ei[["icustay_id_eicu", "p_full"]], on="icustay_id_eicu", how="left")
d_m3 = d_m3.merge(ref_m3[["icustay_id", "p_full"]], on="icustay_id", how="left")

for cols, tag in [(TIER_B, "B47"), (TIER_G, "G25")]:
    best, p_iv = nested_fit(cols, tag)
    blk = {"n_features": len(cols), "best_cfg": {k: str(v) for k, v in best.items()},
           "intval": {"AUROC": round(float(roc_auc_score(yiv, p_iv)), 4),
                      "AUPRC": round(float(average_precision_score(yiv, p_iv)), 4),
                      "Brier": round(float(brier_score_loss(yiv, p_iv)), 4)}}
    dl = delong(yiv, iv.p_full.values, p_iv)
    blk["intval"]["delong_vs_full"] = {"AUC_full": dl[0], "AUC_tier": dl[1], "delta": dl[2], "p": dl[4]}
    for tagx, dd, yy in [("eicu", d_ei, y_ei), ("mimic3", d_m3, y_m3)]:
        p_ext = np.zeros(len(dd))
        for j in range(5):
            Xe_j = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib").transform(dd[FULL]), columns=FULL)
            mdl = make_lgb(best, spw).fit(Xtr[j][cols], ytr)
            p_ext += mdl.predict_proba(Xe_j[cols])[:, 1] / 5
        blk[tagx] = {"AUROC": round(float(roc_auc_score(yy, p_ext)), 4),
                     "AUPRC": round(float(average_precision_score(yy, p_ext)), 4)}
        dlx = delong(yy, dd.p_full.values, p_ext)
        blk[tagx]["delong_vs_full"] = {"AUC_full": dlx[0], "AUC_tier": dlx[1], "delta": dlx[2], "p": dlx[4]}
        lg(f"[{tag}/{tagx}] AUROC={blk[tagx]['AUROC']} vs full {dlx[0]} (p={dlx[4]})")
    res[tag] = blk
res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(f"{REP}/10b_deployment_tiers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lg(f"[DONE] {round(time.time()-t0,1)}s -> reports/10b_deployment_tiers.json")
