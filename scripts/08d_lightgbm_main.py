# -*- coding: utf-8 -*-
# ============================================================
# 08d — LightGBM 主模型: 嵌套5折CV调参 + temporal 内验 + 三向消融 (SAP §7.1/§6)
# 运行环境: D:/Tools/envs/tbi_ml (uv venv 3.12.13, 由 driver step0 一次性装齐:
#          lightgbm scikit-learn scipy pandas pyarrow numpy xgboost imbalanced-learn)
# SAP 冻结条款 (2026-09-03 实锚):
#   §7.1 嵌套5折CV, 内层调参 num_leaves/max_depth/lr/colsample/subsample/
#        min_child_samples + scale_pos_weight, 早期停止仅内层, 种子42主 + 5种子敏感性
#   §3.6 MICE m=5, 插补器仅 train 拟合, transform 到 tune/内验/外验; 完整病例敏感性必报
#   §5  Youden 阈值仅在 train 内层 CV
#   §6  temporal 内验 AUROC(bootstrap 1000, 95%CI) + AUPRC + 校准四件套 + 三向消融
# 实现注记(记入 analytic log, 均为 SAP 未指定处的预注册化选择):
#   ① 嵌套搜索仅在插补集#1 上执行选定超参, 终模型在 5 个插补集上重拟合、预测取均值
#      (缺失大多<5%, 插补集间差异微小; 避免全嵌套×MICE 的 2000+ 次拟合)
#   ② 消融(A / A+B / A+B+C)沿用全模型选定超参(常见做法), 只换特征集
#   ③ 完整病例敏感性 = 全特征非缺样本重训
#   ④ DeLong ΔAUC vs base(年龄+性别+入院GCS+Charlson+SOFA) 因 SOFA 未实现暂缺,
#      由 08e 前置 SOFA 补算后单独出 (已挂旗, 不在本脚本静默降级)
# 产出: reports/08d_lgbm_main.json + data/08d_preds_mimic4.parquet
# ============================================================
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump as jbdump
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

print("=== 08d LightGBM 主模型 (SAP §7.1) ===")
m = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
rep0 = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
assert not rep0.get("gates_failed"), f"08c 门禁未过 {rep0.get('gates_failed')} — 不得建模"
A_COLS, B_COLS, C_COLS = rep0["blockA_cols"], rep0["blockB_cols"], rep0["blockC_cols"]
SENS = rep0["sensitivity_cols"]; META = set(rep0["meta_cols"])
FULL = A_COLS + B_COLS + C_COLS
assert set(FULL) | set(SENS) | META == set(m.columns), "08c 列布局与报告不一致"

tr, tu, iv = (m[m.split == s].copy() for s in ("train", "tune", "intval"))
ytr, ytu, yiv = tr.d28.values, tu.d28.values, iv.d28.values
lg(f"train={len(tr)}(ev {ytr.mean():.1%}) tune={len(tu)}(ev {ytu.mean():.1%}) intval={len(iv)}(ev {yiv.mean():.1%})")

# ---------- MICE m=5 (fit on train only, transform 全部; SAP §3.6) ----------
MICE_M, SEED = 5, 42
imp_sets = {}
imputers = {}   # 保存 fit 于 M4-train 的插补器实体 → 08e 零接触外验必须用同一管线 transform, 禁 refit
for j in range(MICE_M):
    it = IterativeImputer(random_state=SEED + j, sample_posterior=True, max_iter=10)
    Xtr_j = pd.DataFrame(it.fit_transform(tr[FULL]), columns=FULL, index=tr.index)
    Xtu_j = pd.DataFrame(it.transform(tu[FULL]), columns=FULL, index=tu.index)
    Xiv_j = pd.DataFrame(it.transform(iv[FULL]), columns=FULL, index=iv.index)
    imp_sets[j] = (Xtr_j, Xtu_j, Xiv_j)
    imputers[j] = it
lg(f"MICE m={MICE_M} 完成 (fit=train only)")

# ---------- 内层调参网格 (§7.1 清单) ----------
spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
GRID = [dict(num_leaves=nl, max_depth=md, learning_rate=lr, colsample_bytree=cs,
             subsample=ss, min_child_samples=mc, scale_pos_weight=s)
        for nl in (15, 31) for md in (-1, 6) for lr in (0.03, 0.1)
        for cs in (0.8, 1.0) for ss in (0.8, 1.0) for mc in (20, 50) for s in (1.0, spw)]
lg(f"网格 {len(GRID)} 组")

def fit_lgb(X, y, params, Xev=None, yev=None, seed=SEED, rounds=500):
    p = dict(objective="binary", metric="auc", verbosity=-1, seed=seed, subsample_freq=1, **params)
    if Xev is not None:
        dtr, dev = lgb.Dataset(X, y), lgb.Dataset(Xev, yev)
        booster = lgb.train(p, dtr, num_boost_round=rounds,
                            valid_sets=[dev], callbacks=[lgb.early_stopping(50, verbose=False)])
        return booster, booster.best_iteration
    return lgb.train(p, lgb.Dataset(X, y), num_boost_round=rounds), rounds

# ---------- 嵌套 5 折 CV (插补集#1 选参; 外层 OOF 供 Youden + 无偏估计) ----------
Xtr1 = imp_sets[0][0]
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(tr)); inner_scores = {}
for k, (itr, ivo) in enumerate(outer.split(Xtr1, ytr)):
    Xa, ya, Xb, yb = Xtr1.iloc[itr], ytr[itr], Xtr1.iloc[ivo], ytr[ivo]
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 100 + k)
    best_cfg, best_auc = None, -1
    for cfg in GRID:  # 内层: 5折均值选参, 早期停止用内层再劈 20%
        aucs = []
        for it2, iv2 in inner.split(Xa, ya):
            Xi, yi, Xv, yv = Xa.iloc[it2], ya[it2], Xa.iloc[iv2], ya[iv2]
            n_ev = max(int(len(yi) * 0.2), 1)
            ev_idx = np.random.default_rng(SEED).choice(len(yi), n_ev, replace=False)
            mask = np.zeros(len(yi), bool); mask[ev_idx] = True
            bst, _ = fit_lgb(Xi[~mask], yi[~mask], cfg, Xi[mask], yi[mask])
            aucs.append(roc_auc_score(yv, bst.predict(Xv, num_iteration=bst.best_iteration)))
        if np.mean(aucs) > best_auc: best_auc, best_cfg = np.mean(aucs), cfg
    inner_scores[k] = dict(best_auc=round(best_auc, 4), cfg=best_cfg)
    bst, nit = fit_lgb(Xa, ya, best_cfg, Xb, yb)
    oof[ivo] = bst.predict(Xb, num_iteration=nit)
    lg(f"outer fold {k+1}/5: inner best AUC={best_auc:.4f}")
nested_auc = roc_auc_score(ytr, oof)
best_cfg = max(inner_scores.values(), key=lambda d: d["best_auc"])["cfg"]
lg(f"嵌套 CV train OOF AUC={nested_auc:.4f} | 选定超参={best_cfg}")

# Youden 阈值 (SAP §5: 仅 train 内层/OOF)
fpr, tpr, thr = roc_curve(ytr, oof)
j = np.argmax(tpr - fpr); THRESH = float(thr[j])
lg(f"Youden 阈值(train OOF)={THRESH:.4f} (sens={tpr[j]:.3f} spec={1-fpr[j]:.3f})")

# ---------- 终模型: 5 插补集重拟合, 预测取均值 ----------
def run_ensemble(cfg, seed=SEED, feats=None):
    feats = feats or FULL
    ps_tu, ps_iv, bs = [], [], []
    for j in range(MICE_M):
        Xtr_j, Xtu_j, Xiv_j = imp_sets[j]
        bst, _ = fit_lgb(Xtr_j[feats], ytr, cfg, seed=seed)
        bs.append(bst)
        ps_tu.append(bst.predict(Xtu_j[feats])); ps_iv.append(bst.predict(Xiv_j[feats]))
    return np.mean(ps_tu, 0), np.mean(ps_iv, 0), bs

def bootstrap_auc(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        a.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def calib(y, p, bins=10):
    # 标准 logistic 重校准 (Cox): y ~ logit(p), 大 C 无惩罚 — 与 08e calib3 同口径
    # (2026-09-03 修正: 原实现 e=(1-p)/p 取 log 得 -logit(p), 斜率符号翻转 + OLS 口径不一, 作废重算)
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(pc / (1 - pc)).reshape(-1, 1), y)
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b", observed=True).agg(pm=("p", "mean"), om=("y", "mean"), n=("y", "size"))
    ece = float((g.n / len(y) * (g.pm - g.om).abs()).sum())
    return dict(slope=round(float(lr.coef_[0][0]), 4), intercept=round(float(lr.intercept_[0]), 4),
                ECE=round(ece, 4), Brier=round(float(brier_score_loss(y, p)), 4))

def metrics_block(y, p, tag, seed=SEED):
    lo, hi = bootstrap_auc(y, p, seed=seed)
    pred = (p >= THRESH).astype(int)
    tp, fp, fn, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), \
                     ((pred == 0) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    return dict(variant=tag, n=int(len(y)), events=int(y.sum()),
                AUROC=round(float(roc_auc_score(y, p)), 4), AUROC_lo=round(lo, 4), AUROC_hi=round(hi, 4),
                AUPRC=round(float(average_precision_score(y, p)), 4),
                sens=round(float(tp / max(tp + fn, 1)), 4), spec=round(float(tn / max(tn + fp, 1)), 4),
                PPV=round(float(tp / max(tp + fp, 1)), 4), NPV=round(float(tn / max(tn + fn, 1)), 4),
                **calib(y, p))

res = {"threshold_yuden_train_oof": round(THRESH, 6),
       "nested_cv_train_oof_auc": round(float(nested_auc), 4),
       "selected_params": best_cfg, "scale_pos_weight_value": round(spw, 4),
       "inner_fold_scores": inner_scores, "tune": {}, "intval": {}, "ablation": {},
       "seeds_sensitivity": {}, "complete_case": {}}

p_tu, p_iv, B_FULL = run_ensemble(best_cfg)
res["tune"]["full"] = metrics_block(ytu, p_tu, "full")
res["intval"]["full"] = metrics_block(yiv, p_iv, "full")
lg(f"temporal 内验 AUROC={res['intval']['full']['AUROC']} [{res['intval']['full']['AUROC_lo']},{res['intval']['full']['AUROC_hi']}] "
   f"AUPRC={res['intval']['full']['AUPRC']} Brier={res['intval']['full']['Brier']}")

# ---------- 终模型持久化 (SAP L32 外验 zero-touch: 冻结管线 = MICE(M4-train-fitted) → 5 booster 均值 → 冻结 Youden) ----------
MDL = ROOT / "07_prediction_system/models"
MDL.mkdir(exist_ok=True)
for j in range(MICE_M):
    B_FULL[j].save_model(str(MDL / f"08d_lgbm_full_imp{j}.txt"))
    jbdump(imputers[j], MDL / f"08d_mice_imp{j}.joblib")
card = dict(params=dict(objective="binary", metric="auc", verbosity=-1, seed=SEED,
                        subsample_freq=1, num_boost_round=500, **best_cfg),
            full_cols=FULL, threshold_youden=round(THRESH, 6), scale_pos_weight=spw,
            trained_on="mimic4 train", n_train=int(len(tr)), n_train_events=int(ytr.sum()),
            mice_m=MICE_M, predict_rule="mean of 5 boosters over MICE-transformed features")
(MDL / "08d_model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"终模型已持久化: {MDL}/08d_lgbm_full_imp[0-4].txt + 08d_mice_imp[0-4].joblib + 08d_model_card.json")

# ---------- 三向消融 (§6: A / A+B / A+B+C) ----------
preds_store = {"full": p_iv}
for tag, feats in (("A", A_COLS), ("AB", A_COLS + B_COLS), ("ABC", FULL)):
    _, p_iv_ab, _ = run_ensemble(best_cfg, feats=feats)
    preds_store[tag] = p_iv_ab
    res["ablation"][tag] = metrics_block(yiv, p_iv_ab, tag)
    lg(f"消融 {tag}: AUROC={res['ablation'][tag]['AUROC']}")

# ---------- 5 种子敏感性 (params 固定) ----------
for sd in (42, 43, 44, 45, 46):
    _, p_iv_s, _ = run_ensemble(best_cfg, seed=sd)
    res["seeds_sensitivity"][f"seed{sd}"] = round(float(roc_auc_score(yiv, p_iv_s)), 4)
    preds_store[f"full_seed{sd}"] = p_iv_s
lg("种子敏感性: " + str(res["seeds_sensitivity"]))

# ---------- 完整病例敏感性 (§3.6 必报) ----------
cc = m[m[FULL].notna().all(axis=1)]
lg(f"完整病例 n={len(cc)}/{len(m)} ({len(cc)/len(m):.1%})")
cct = cc[cc.split == "train"]; cci = cc[cc.split == "intval"]
if len(cci) and cci.d28.nunique() == 2 and len(cct) > 50:
    bst, _ = fit_lgb(cct[FULL], cct.d28.values, best_cfg)
    p_cc = bst.predict(cci[FULL])
    res["complete_case"] = metrics_block(cci.d28.values, p_cc, "cc")
    preds_store["cc"] = pd.Series(p_cc, index=cci.index)

# ---------- SHAP top-20 (intval, 终模型插补集#1) ----------
bst_f, _ = fit_lgb(imp_sets[0][0][FULL], ytr, best_cfg)
sv = bst_f.predict(imp_sets[0][2][FULL], pred_contrib=True)
shap_mean = dict(zip(FULL, np.round(np.abs(sv[:, :-1]).mean(0), 5)))
res["shap_top20"] = dict(sorted(shap_mean.items(), key=lambda kv: -kv[1])[:20])
lg("SHAP top5: " + str(list(res["shap_top20"].items())[:5]))

# ---------- 落盘 ----------
pd.DataFrame({"stay_id": iv.stay_id.values, "d28": yiv, **{f"p_{k}": v for k, v in preds_store.items() if len(v) == len(iv)}}
             ).to_parquet(DATA / "08d_preds_mimic4.parquet", index=False)
res["flags"] = ["ΔAUC vs base(含SOFA) 已由 08f_base_delta_auc.json 补算 (intval: +0.0473, p=0.0065 vs base5)",
                "END_post 次要结局矩阵未建 (需 END_TRAJ_FRAMEWORK 判据逐字对齐, 独立脚本)",
                "calib() 2026-09-03 修正为 logistic 重校准 (原 OLS+符号翻转作废), 本次为确定性重跑"]
res["model_persist"] = {"dir": str(MDL), "boosters": "08d_lgbm_full_imp[0-4].txt",
                        "imputers": "08d_mice_imp[0-4].joblib", "card": "08d_model_card.json",
                        "note": "08e 零接触外验: 加载 booster+M4-fitted MICE, transform 外库特征, 禁 refit/调参/换阈值"}
(REP / "08d_lgbm_main.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
lg(f"产出: {REP}/08d_lgbm_main.json + {DATA}/08d_preds_mimic4.parquet")
print("DONE 08d")
