# -*- coding: utf-8 -*-
# 10d — 25 特征筛选版主模型 + 全链重跑 (SAP §12 post-hoc companion)
#   与 79 特征冻结版并列报告 (Opus 5 评审方案 A: 主模型=79 不变, 筛选版=parsimony analysis)
#   用完全相同协议: LightGBM 嵌套 5×5 CV 种子 42, MICE m=5 train-only
#   外验: eICU + M3 CareVue zero-touch
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
def lg(m): print(m, flush=True)

# 加载筛选结果
screen = json.loads(open(f"{REP}/10c_feature_screen.json", encoding="utf-8").read())
SEL = screen["final_features"]
lg(f"[10d] 筛选版 {len(SEL)} 特征 | 规则: {screen['rule_applied']}")

rep0 = json.loads(open(f"{REP}/08c_matrix_report.json", encoding="utf-8").read())
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, tu, iv = (m[m.split == s].copy() for s in ("train", "tune", "intval"))
ytr, yiv = tr.d28.values, iv.d28.values

# 79 特征版参照 (08e 冻结管线预测, 已有)
preds = pd.read_parquet(f"{DATA}/08d_preds_mimic4.parquet")
iv["p_full"] = iv.stay_id.map(preds.set_index("stay_id").p_full)
assert not iv.p_full.isna().any()

# MICE: 复用 08d 五套 (插补模型同, 取特征子集)
imp_sets = {}
for j in range(5):
    full_imp = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib").transform(m[FULL]),
                            columns=FULL, index=m.index)
    imp_sets[j] = full_imp[SEL]  # 只取筛选后列
tr_idx, iv_idx = tr.index.to_numpy(), iv.index.to_numpy()
Xtr, Xiv = {j: imp_sets[j].loc[tr_idx] for j in range(5)}, {j: imp_sets[j].loc[iv_idx] for j in range(5)}

GRID = [dict(nl=nl, md=md, lr=lr, col=col, sub=sub, mcs=mcs)
        for nl in (15, 31, 63) for md in (3, 5, -1) for lr in (0.03, 0.05)
        for col in (0.8, 1.0) for sub in (0.8,) for mcs in (20, 40)]

def make_lgb(c, spw):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=c["lr"], num_leaves=c["nl"],
                              max_depth=c["md"], colsample_bytree=c["col"], subsample=c["sub"],
                              subsample_freq=1, min_child_samples=c["mcs"],
                              scale_pos_weight=spw, random_state=42, n_jobs=4, verbose=-1)

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

# ===== 嵌套 CV + 终模型 =====
X1 = Xtr[0]
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
lg(f"[10d] best_cfg={best} oof={np.mean(oof):.4f}")

p_iv = np.zeros(len(iv))
for j in range(5):
    mdl = make_lgb(best, spw).fit(Xtr[j], ytr)
    p_iv += mdl.predict_proba(Xiv[j])[:, 1] / 5

res = {
    "design": f"25-feature screened model (Zhang-style dual screen, {screen['rule_applied']}); same protocol as 08d",
    "n_features": len(SEL),
    "features": SEL,
    "best_cfg": {k2: str(v2) for k2, v2 in best.items()},
    "nested_cv_oof_auc": round(float(np.mean(oof)), 4),
    "intval": {
        "AUROC": round(float(roc_auc_score(yiv, p_iv)), 4),
        "AUPRC": round(float(average_precision_score(yiv, p_iv)), 4),
        "Brier": round(float(brier_score_loss(yiv, p_iv)), 4),
    },
    "reference_79feat": {"AUROC": 0.9054},
}
dl = delong(yiv, iv.p_full.values, p_iv)
res["intval"]["delong_vs_79feat"] = {"AUC_79": dl[0], "AUC_25": dl[1], "delta": dl[2], "p": dl[4]}
lg(f"[10d intval] AUROC={res['intval']['AUROC']} vs 79feat 0.9054 (delong p={dl[4]})")

# ===== 外验 (zero-touch, CareVue M3) =====
def strip_dyn(df):
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)
NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

def build(tag):
    if tag == "eicu":
        fe = pd.read_parquet(f"{DATA}/features_eicu.parquet").rename(columns={"age": "admission_age"})
        fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
        ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
        if ic: fe = NULLABLE(fe, ic)
        post = pd.read_parquet(f"{DATA}/08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
        d = fe.merge(post, on="icustay_id_eicu", how="left")
        died24 = ((d.died_hosp == 1) & (d.los_h <= 24)); short = d.los_h < 24
        d = d[~(died24 | short)].copy()
        y = d.died_hosp_28d.astype(int).values
    else:
        fe = pd.read_parquet(f"{DATA}/features_mimic3.parquet").rename(columns={"age": "admission_age"})
        fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
        ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
        if ic: fe = NULLABLE(fe, ic)
        post = pd.read_parquet(f"{DATA}/08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
        d = fe.merge(post, on="icustay_id", how="left")
        # CareVue 过滤
        import duckdb as _dd
        _cv = _dd.connect().execute("""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
        FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true)
        WHERE DBSOURCE = 'carevue'""").df()
        d = d[d.icustay_id.isin(_cv.icustay_id)].copy()
        died24 = (d.d28 == 1) & (d.days_to_death < 1.0); short = d.los_h < 24
        d = d[~(died24 | short)].copy()
        y = d.d28.astype(int).values
    return d, y

for tag in ("eicu", "mimic3"):
    dd, yy = build(tag)
    # 参照 (08e 冻结管线预测, 已有 parquet)
    ref_name = f"08e_preds_{tag}.parquet"
    ref = pd.read_parquet(f"{DATA}/{ref_name}")
    idcol = "icustay_id_eicu" if tag == "eicu" else "icustay_id"
    dd = dd.merge(ref[[idcol, "p_full"]], on=idcol, how="left")

    p_ext = np.zeros(len(dd))
    for j in range(5):
        Xe_j = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib").transform(dd[FULL]), columns=FULL)
        mdl = make_lgb(best, spw).fit(Xtr[j], ytr)
        p_ext += mdl.predict_proba(Xe_j[SEL])[:, 1] / 5

    auc25 = round(float(roc_auc_score(yy, p_ext)), 4)
    dlx = delong(yy, dd.p_full.values, p_ext)
    res[tag] = {"n": len(dd), "events": int(yy.sum()),
                "AUROC_25": auc25, "AUROC_79": dlx[0],
                "delong_vs_79": {"delta": dlx[2], "p": dlx[4]}}
    lg(f"[10d {tag}] AUROC={auc25} vs 79feat {dlx[0]} (p={dlx[4]})")

res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(f"{REP}/10d_screened_model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lg(f"[DONE] {round(time.time()-t0,1)}s -> reports/10d_screened_model.json")
