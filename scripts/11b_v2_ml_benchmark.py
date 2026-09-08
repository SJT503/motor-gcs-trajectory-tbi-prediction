# -*- coding: utf-8 -*-
# 11b — statistical ML benchmark tier (L1-LR / RF / XGBoost) for the v2 25-feature main model.
#   08k logic verbatim, with the feature set = SEL (25) and reference predictions = frozen v2 (08d_v2 / 08e_v2).
#   Nested 5×5 StratifiedKFold (outer rs=42, inner rs=42+100+k), MICE m=5 frozen imputers (transform only), 5-imputation averaging.
# Output: reports/11b_v2_ml_benchmark.json + data/11b_v2_preds_{intval,eicu,mimic3}.parquet
import json, sys, time, warnings
import joblib, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"
t0 = time.time(); SEED = 42
def lg(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

card = json.load(open(f"{MODELS}/08d_v2_model_card.json", encoding="utf-8")); SEL = card["features"]
rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8")); FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, iv = m[m.split == "train"].copy(), m[m.split == "intval"].copy()
ytr, yiv = tr.d28.values.astype(int), iv.d28.values.astype(int)
spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
iv["p_full"] = iv.stay_id.map(pd.read_parquet(f"{DATA}/08d_v2_preds_mimic4.parquet").set_index("stay_id").p_full)
assert not iv.p_full.isna().any()
lg(f"train={len(tr)} intval={len(iv)} | SEL={len(SEL)} | anchor v2 intval AUROC={roc_auc_score(yiv, iv.p_full):.4f}")
imputers = {j: joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib") for j in range(5)}
imp = {j: pd.DataFrame(imputers[j].transform(m[FULL]), columns=FULL, index=m.index)[SEL] for j in range(5)}
tr_idx, iv_idx = tr.index.to_numpy(), iv.index.to_numpy()

def make_lr(C, cw): return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight=cw, max_iter=5000, random_state=SEED))])
def make_rf(md, msl, mf, cw): return RandomForestClassifier(n_estimators=300, max_depth=md, min_samples_leaf=msl, max_features=mf, class_weight=cw, random_state=SEED, n_jobs=-1)
def make_xgb(md, lr_, sub, col, w): return XGBClassifier(n_estimators=400, max_depth=md, learning_rate=lr_, subsample=sub, colsample_bytree=col, scale_pos_weight=w, tree_method="hist", eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0)
LEARNERS = {"L1_LR": (make_lr, [dict(C=c, cw=cw) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0) for cw in (None, "balanced")]),
            "RF": (make_rf, [dict(md=md, msl=msl, mf=mf, cw=cw) for md in (None, 8) for msl in (5, 20) for mf in ("sqrt", 0.3) for cw in (None, "balanced_subsample")]),
            "XGB": (make_xgb, [dict(md=md, lr_=lr_, sub=sub, col=col, w=w) for md in (3, 6) for lr_ in (0.05, 0.1) for sub in (0.8, 1.0) for col in (0.8, 1.0) for w in (1.0, spw)])}
Xtr1 = imp[0].loc[tr_idx]
def nested_select(make, grid):
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED); oof, best_cfg, inner_log = [], None, []
    for k, (itr, ivo) in enumerate(outer.split(Xtr1, ytr)):
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 100 + k); Xa, ya = Xtr1.iloc[itr], ytr[itr]; best, best_auc = None, -1
        for cfg in grid:
            aucs = [roc_auc_score(ya[iv2], make(**cfg).fit(Xa.iloc[it2], ya[it2]).predict_proba(Xa.iloc[iv2])[:, 1]) for it2, iv2 in inner.split(Xa, ya)]
            mu = float(np.mean(aucs))
            if mu > best_auc: best_auc, best = mu, cfg
        oof.append(roc_auc_score(ytr[ivo], make(**best).fit(Xa, ya).predict_proba(Xtr1.iloc[ivo])[:, 1])); best_cfg = best; inner_log.append(round(best_auc, 4))
    return best_cfg, float(np.mean(oof)), inner_log
def delong(y, p1, p2):
    y = np.asarray(y, int).ravel(); p1 = np.asarray(p1, float).ravel(); p2 = np.asarray(p2, float).ravel()
    def partials(p):
        pos, neg = p[y == 1], p[y == 0]; n1, n0 = len(pos), len(neg)
        return (np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos]), np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg]))
    P1, P2 = partials(p1), partials(p2); a1, a2 = float(P1[0].mean()), float(P2[0].mean())
    v1 = float(np.var(P1[0], ddof=1)) / len(P1[0]) + float(np.var(P1[1], ddof=1)) / len(P1[1]); v2 = float(np.var(P2[0], ddof=1)) / len(P2[0]) + float(np.var(P2[1], ddof=1)) / len(P2[1])
    c10 = float(np.cov(P1[0], P2[0], ddof=1)[0, 1]) / len(P1[0]); c01 = float(np.cov(P1[1], P2[1], ddof=1)[0, 1]) / len(P1[1])
    se = float(np.sqrt(max(v1 + v2 - 2.0 * (c10 + c01), 1e-18))); d = a1 - a2
    return dict(AUC_full=round(a1, 4), AUC_base=round(a2, 4), delta=round(float(d), 4), se=round(se, 4), p=float(2 * (1 - norm.cdf(abs(d / se)))))
def metrics_block(y, p, seed=SEED):
    yy, pp = np.asarray(y, int), np.asarray(p, float); rng = np.random.default_rng(seed); pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]; aucs = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]); aucs.append(roc_auc_score(yy[idx], pp[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return dict(AUROC=round(float(roc_auc_score(yy, pp)), 4), AUROC_lo=round(float(lo), 4), AUROC_hi=round(float(hi), 4), AUPRC=round(float(average_precision_score(yy, pp)), 4), Brier=round(float(brier_score_loss(yy, pp)), 4))

res = {"version": "11b_v2_25feat", "eval_set": "mimic4 intval (same as 08d_v2)", "n_intval": int(len(iv)), "events_intval": int(yiv.sum()),
       "anchor_v2_full": round(float(roc_auc_score(yiv, iv.p_full)), 4), "learners": {}}
store = {"stay_id": iv.stay_id.values}
for name, (make, grid) in LEARNERS.items():
    t1 = time.time(); best_cfg, oof_auc, inner_log = nested_select(make, grid); p_iv = np.zeros(len(iv))
    for j in range(5): p_iv += make(**best_cfg).fit(imp[j].loc[tr_idx], ytr).predict_proba(imp[j].loc[iv_idx])[:, 1] / 5
    mb = metrics_block(yiv, p_iv); dl = delong(yiv, iv.p_full.values, p_iv)
    if name == "L1_LR":
        cz = make(**best_cfg).fit(imp[0].loc[tr_idx], ytr).named_steps["lr"].coef_[0]
        res["l1_sparsity_check"] = dict(n_coef=int(len(cz)), n_zero=int((cz == 0).sum()), frac_zero=round(float((cz == 0).mean()), 4))
    res["learners"][name] = dict(best_cfg={k: (str(v) if v is None else v) for k, v in best_cfg.items()}, nested_cv_oof_auc=round(oof_auc, 4), inner_best_by_fold=inner_log, intval=mb, delong_vs_full=dl, elapsed_s=round(time.time() - t1, 1))
    store[f"p_{name}"] = p_iv
    lg(f"[{name}] OOF={oof_auc:.4f} | intval {mb['AUROC']} [{mb['AUROC_lo']},{mb['AUROC_hi']}] | Δ vs v2 full={dl['delta']:+.4f} p={dl['p']:.4f}")
pd.DataFrame(store).to_parquet(f"{DATA}/11b_v2_preds_intval.parquet", index=False)

# external (10d build logic)
def strip_dyn(df):
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)
NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})
def build(tag):
    if tag == "eicu":
        fe = pd.read_parquet(f"{DATA}/features_eicu.parquet").rename(columns={"age": "admission_age"}); fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
        ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
        if ic: fe = NULLABLE(fe, ic)
        d = fe.merge(pd.read_parquet(f"{DATA}/08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"}), on="icustay_id_eicu", how="left")
        d = d[~(((d.died_hosp == 1) & (d.los_h <= 24)) | (d.los_h < 24))].copy(); return d, d.died_hosp_28d.astype(int).values, "icustay_id_eicu", "08e_v2_preds_eicu.parquet"
    fe = pd.read_parquet(f"{DATA}/features_mimic3.parquet").rename(columns={"age": "admission_age"}); fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
    ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
    if ic: fe = NULLABLE(fe, ic)
    d = fe.merge(pd.read_parquet(f"{DATA}/08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"}), on="icustay_id", how="left")
    import duckdb as _dd
    cv = _dd.connect().execute("SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'").df()
    d = d[d.icustay_id.isin(cv.icustay_id)].copy(); d = d[~(((d.d28 == 1) & (d.days_to_death < 1.0)) | (d.los_h < 24))].copy()
    return d, d.d28.astype(int).values, "icustay_id", "08e_v2_preds_mimic3.parquet"
res["external"] = {}
for tag in ("eicu", "mimic3"):
    d, y_ext, idc, refname = build(tag); d = d.reset_index(drop=True)
    ref = pd.read_parquet(f"{DATA}/{refname}").set_index(idc).loc[d[idc], "p_full"].values
    blk = {"n": int(len(d)), "events": int(y_ext.sum()), "AUROC_v2_full_ref": round(float(roc_auc_score(y_ext, ref)), 4)}; st = {idc: d[idc].values}
    for name, (make, grid) in LEARNERS.items():
        cfg = dict(res["learners"][name]["best_cfg"])
        for kk in ("cw", "md"):
            if kk in cfg and cfg[kk] == "None": cfg[kk] = None
        p_ext = np.zeros(len(d))
        for j in range(5):
            Xe = pd.DataFrame(imputers[j].transform(d[FULL]), columns=FULL)[SEL]
            p_ext += make(**cfg).fit(imp[j].loc[tr_idx], ytr).predict_proba(Xe)[:, 1] / 5
        blk[name] = dict(AUROC=round(float(roc_auc_score(y_ext, p_ext)), 4), AUPRC=round(float(average_precision_score(y_ext, p_ext)), 4), delong_vs_full=delong(y_ext, ref, p_ext))
        st[f"p_{name}"] = p_ext
        lg(f"[{tag} {name}] AUROC={blk[name]['AUROC']} (v2 ref {blk['AUROC_v2_full_ref']}) Δ={blk[name]['delong_vs_full']['delta']:+.4f} p={blk[name]['delong_vs_full']['p']:.4f}")
    res["external"][tag] = blk; pd.DataFrame(st).to_parquet(f"{DATA}/11b_v2_preds_{tag}.parquet", index=False)
json.dump(res, open(f"{REP}/11b_v2_ml_benchmark.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lg(f"DONE 11b | {time.time()-t0:.0f}s")
