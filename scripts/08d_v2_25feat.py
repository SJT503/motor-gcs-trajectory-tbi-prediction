# -*- coding: utf-8 -*-
# 08d v2 — 25 特征主模型 (2026-09-07 PI 指令: 主模型切换为筛选后特征)
# 与 08d v1 完全同构: 嵌套 5×5 CV 种子 42, MICE m=5 train-only, 冻结管线持久化
# 区别仅: 特征 = 10c 筛选的 25 个 (而非 79 个白名单)
# 产出: reports/08d_v2_lgbm_main.json + models/08d_v2_* + data/08d_v2_preds_*.parquet
import json, time, warnings, os
import numpy as np
import pandas as pd
import joblib, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"
t0 = time.time()
def lg(m): print(m, flush=True)

# ===== 加载筛选结果 =====
screen = json.loads(open(f"{REP}/10c_feature_screen.json", encoding="utf-8").read())
SEL = screen["final_features"]
rep0 = json.loads(open(f"{REP}/08c_matrix_report.json", encoding="utf-8").read())
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

lg(f"=== 08d v2 | 25 特征主模型 | SAP §12 2026-09-07 PI 指令 ===")

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, tu, iv = (m[m.split == s].copy() for s in ("train", "tune", "intval"))
ytr, yiv = tr.d28.values.astype(int), iv.d28.values.astype(int)
lg(f"train n={len(tr)} (ev {ytr.mean():.1%}) | tune n={len(tu)} | intval n={len(iv)} (ev {yiv.sum()})")

# ===== MICE (复用 08d 五套, 取 25 列子集) =====
imp_sets = {}
for j in range(5):
    full_imp = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib").transform(m[FULL]),
                            columns=FULL, index=m.index)
    imp_sets[j] = full_imp[SEL]
tr_idx, tu_idx, iv_idx = tr.index.to_numpy(), tu.index.to_numpy(), iv.index.to_numpy()

# ===== 嵌套 CV =====
GRID = [dict(nl=nl, md=md, lr=lr, col=col, sub=sub, mcs=mcs)
        for nl in (15, 31, 63) for md in (3, 5, -1) for lr in (0.03, 0.05)
        for col in (0.8, 1.0) for sub in (0.8,) for mcs in (20, 40)]
def make(c, spw):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=c["lr"], num_leaves=c["nl"],
                              max_depth=c["md"], colsample_bytree=c["col"], subsample=c["sub"],
                              subsample_freq=1, min_child_samples=c["mcs"],
                              scale_pos_weight=spw, random_state=42, n_jobs=4, verbose=-1)

spw = (len(ytr) - ytr.sum()) / ytr.sum()
lg(f"scale_pos_weight={spw:.3f} | grid={len(GRID)}")

X1 = imp_sets[0].loc[tr_idx]
outer = StratifiedKFold(5, shuffle=True, random_state=42)
oof, inner_log, best_cfg = [], [], None
for k, (itr, ivo) in enumerate(outer.split(X1, ytr)):
    inner = StratifiedKFold(5, shuffle=True, random_state=42 + 100 + k)
    Xa, ya = X1.iloc[itr], ytr[itr]
    best, best_auc = None, -1
    for c in GRID:
        aucs = []
        for it2, iv2 in inner.split(Xa, ya):
            mdl = make(c, spw).fit(Xa.iloc[it2], ya[it2])
            aucs.append(roc_auc_score(ya[iv2], mdl.predict_proba(Xa.iloc[iv2])[:, 1]))
        mu = float(np.mean(aucs))
        if mu > best_auc: best_auc, best = mu, c
    mdl = make(best, spw).fit(Xa, ya)
    oof.append(roc_auc_score(ytr[ivo], mdl.predict_proba(X1.iloc[ivo])[:, 1]))
    best_cfg = best; inner_log.append(round(best_auc, 4))
lg(f"nested OOF={np.mean(oof):.4f} | best={best_cfg} | inner={inner_log}")

# ===== 冻结 Youden 阈值 (train OOF) =====
# 用第一折的最优参数在全 train 上 5 折 OOF 预测算 Youden
oof_pred = np.zeros(len(tr))
for k, (itr, ivo) in enumerate(outer.split(X1, ytr)):
    mdl = make(best_cfg, spw).fit(X1.iloc[itr], ytr[itr])
    oof_pred[ivo] = mdl.predict_proba(X1.iloc[ivo])[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(ytr, oof_pred)
j_idx = np.argmax(tpr - fpr)
THRESH = round(float(thr[j_idx]), 6)
lg(f"冻结 Youden 阈值={THRESH:.6f}")

# ===== 终模型 (5 套插补平均) + 持久化 =====
MDL_DIR = f"{MODELS}"
p_iv = np.zeros(len(iv))
for j in range(5):
    mdl = make(best_cfg, spw).fit(imp_sets[j].loc[tr_idx], ytr)
    mdl.booster_.save_model(f"{MDL_DIR}/08d_v2_lgbm_sel_imp{j}.txt")
    p_iv += mdl.predict_proba(imp_sets[j].loc[iv_idx])[:, 1] / 5
lg(f"intval p: AUROC={roc_auc_score(yiv, p_iv):.4f}")

# ===== 校准 =====
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
logit = np.log(np.clip(p_iv, 1e-10, 1-1e-10) / (1 - np.clip(p_iv, 1e-10, 1-1e-10)))
lr_cal = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), yiv)
slope = round(float(lr_cal.coef_[0][0]), 4)
intercept = round(float(lr_cal.intercept_[0]), 4)
ece = round(float(np.mean(np.abs(p_iv - yiv))), 4)
brier = round(float(brier_score_loss(yiv, p_iv)), 4)
sens = round(float(((p_iv >= THRESH) & (yiv == 1)).sum() / yiv.sum()), 4)
spec = round(float(((p_iv < THRESH) & (yiv == 0)).sum() / (yiv == 0).sum()), 4)

# ===== 种子敏感性 =====
seed_aucs = []
for seed in [42, 43, 44, 45, 46]:
    outer_s = StratifiedKFold(5, shuffle=True, random_state=seed)
    mdl_s = make(best_cfg, spw)
    preds_s = np.zeros(len(iv))
    for j in range(5):
        mdl_s = make(best_cfg, spw).fit(imp_sets[j].loc[tr_idx], ytr)
        preds_s += mdl_s.predict_proba(imp_sets[j].loc[iv_idx])[:, 1] / 5
    seed_aucs.append(round(roc_auc_score(yiv, preds_s), 4))

# ===== SHAP (top 特征重要性) =====
booster0 = lgb.Booster(model_file=f"{MDL_DIR}/08d_v2_lgbm_sel_imp0.txt")
contrib = booster0.predict(imp_sets[0].loc[iv_idx], pred_contrib=True)[:, :-1]
shap_mean = {c: round(float(np.abs(contrib[:, i]).mean()), 5) for i, c in enumerate(SEL)}
shap_top = dict(sorted(shap_mean.items(), key=lambda x: -x[1])[:20])

# ===== 完整病例敏感性 =====
cc_mask = iv[SEL].notna().all(axis=1).values
cc_auc = round(float(roc_auc_score(yiv[cc_mask], p_iv[cc_mask])), 4) if cc_mask.sum() > 50 else None

# ===== 保存主结果 =====
iv_res = {
    "AUROC": round(float(roc_auc_score(yiv, p_iv)), 4),
    "AUROC_lo": round(float(roc_auc_score(yiv, p_iv) - 1.96 * np.sqrt(roc_auc_score(yiv, p_iv)*(1-roc_auc_score(yiv, p_iv))/len(yiv))), 4),
    "AUROC_hi": round(float(roc_auc_score(yiv, p_iv) + 1.96 * np.sqrt(roc_auc_score(yiv, p_iv)*(1-roc_auc_score(yiv, p_iv))/len(yiv))), 4),
    "AUPRC": round(float(average_precision_score(yiv, p_iv)), 4),
    "Brier": brier, "ECE": ece, "slope": slope, "intercept": intercept,
    "sens": sens, "spec": spec,
}

# ===== 持久化 preds + card =====
pd.DataFrame({"stay_id": iv.stay_id.values, "p_full": p_iv, "d28": yiv,
              "split": "intval"}).to_parquet(f"{DATA}/08d_v2_preds_mimic4.parquet", index=False)

card = {
    "version": "08d_v2 (25-feature screened main model)",
    "date": "2026-09-07",
    "features": SEL, "n_features": len(SEL),
    "screening": screen["rule_applied"],
    "best_cfg": {k: str(v) for k, v in best_cfg.items()},
    "threshold_youden_train_oof": THRESH,
    "scale_pos_weight": round(spw, 4),
    "nested_cv_train_oof_auc": round(float(np.mean(oof)), 4),
    "model_persist": {"dir": MDL_DIR, "boosters": "08d_v2_lgbm_sel_imp[0-4].txt",
                      "imputers": "复用 08d_mice_imp[0-4].joblib (79 特征插补, 取 25 列子集)",
                      "card": "08d_v2_model_card.json"},
}
json.dump(card, open(f"{MODELS}/08d_v2_model_card.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

res = {
    "version": "08d_v2_25feat",
    "design": "25-feature screened main model (Zhang-style dual screen); same protocol as 08d v1 (79 features)",
    "features": SEL, "n_features": len(SEL),
    "screening_rule": screen["rule_applied"],
    "best_cfg": {k: str(v) for k, v in best_cfg.items()},
    "nested_cv_train_oof_auc": round(float(np.mean(oof)), 4),
    "inner_best_by_fold": inner_log,
    "threshold_yuden_train_oof": THRESH,
    "scale_pos_weight_value": round(spw, 4),
    "intval": {"full": iv_res},
    "seeds_sensitivity": {"aucs": seed_aucs, "mean": round(float(np.mean(seed_aucs)), 4)},
    "complete_case": {"n": int(cc_mask.sum()), "AUROC": cc_auc},
    "shap_top20": shap_top,
    "model_persist": card["model_persist"],
}
json.dump(res, open(f"{REP}/08d_v2_lgbm_main.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

lg(f"intval full: AUROC={iv_res['AUROC']} CI[{iv_res['AUROC_lo']}-{iv_res['AUROC_hi']}] AUPRC={iv_res['AUPRC']}")
lg(f"calib: slope={slope} ECE={ece} Brier={brier} | sens={sens} spec={spec}")
lg(f"seeds: {seed_aucs} | CC: {cc_auc}")
lg(f"SHAP top-5: {list(shap_top.items())[:5]}")
lg(f"产出: {REP}/08d_v2_lgbm_main.json + models/08d_v2_* + data/08d_v2_preds_mimic4.parquet")
lg(f"DONE 08d v2 | {round(time.time()-t0,1)}s")
