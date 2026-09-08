# -*- coding: utf-8 -*-
# ============================================================
# 08k — SAP §7.2 统计 ML 基准档: L1-LR / RF / XGBoost (三学习器 vs 主模型 08d)
#   口径逐字对齐 08d: 08c 矩阵 FULL 79 / 08d 冻结 MICE m=5 imputer (joblib 加载, transform only, zero-touch)
#   / 嵌套 5×5 StratifiedKFold 同种子 (outer rs=42, inner rs=42+100+k; 内层调参用插补集#1, 终模型 5 套平均)
#   评估: M4 intval (与 08d 同集) AUROC(bootstrap 1000)+AUPRC+Brier+校准 + 配对 DeLong vs 08d p_full (08f 同款)
#   外验: eICU/M3 zero-touch (08e build_eicu/build_m3 口径复用 + 08d imputer transform)
#   产出: reports/08k_ml_benchmark.json + data/08k_preds_intval.parquet + 08k_preds_eicu.parquet + 08k_preds_mimic3.parquet
#   注: DL 档 (LSTM) 仅 Supp 探索性, 不在本脚本 (SAP L112)
# ============================================================
import json, sys, time, os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
MODELS = ROOT / "07_prediction_system/models"
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

SEED = 42
print("=== 08k 统计 ML 基准档 (SAP §7.2 第二档: L1-LR / RF / XGBoost) ===")

# ---------- 1. 输入 (08d 同源) ----------
m = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
rep0 = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
assert not rep0.get("gates_failed"), "08c 门禁未过 — 不得建模"
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
tr, tu, iv = (m[m.split == s].copy() for s in ("train", "tune", "intval"))
ytr, yiv = tr.d28.values, iv.d28.values
spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
lg(f"train={len(tr)}(ev {ytr.mean():.1%}) intval={len(iv)}(ev {yiv.mean():.1%}) | FULL={len(FULL)} | spw={spw:.2f}")

preds = pd.read_parquet(DATA / "08d_preds_mimic4.parquet")
# run2 缺陷修复 (diag8 实证): 原写法 iv.merge(preds) 会把 iv 索引重置为 0..n-1, 而 L58 iv_idx 在其后取
#   → imp_sets.loc[iv_idx] 只取到 83/483 正确行 + 294 个 train 行 (p_iv 全为错行预测, AUROC 0.494 vs 正确行 0.887)
#   改 map 挂列 (索引/行序原样保留); preds.stay_id 唯一 (diag8: 483 行 483 唯一) → map 安全
iv["p_full"] = iv.stay_id.map(preds.set_index("stay_id").p_full)
assert not iv.p_full.isna().any(), "intval 缺 08d 预测"
auc_full_anchor = round(float(roc_auc_score(iv.d28, iv.p_full)), 4)
lg(f"08d intval AUROC 复现锚 = {auc_full_anchor} (期望 0.9054±0.0002)")

# ---------- 2. 冻结 MICE m=5 transform (08d 持久化实体, zero-touch) ----------
imp_sets, imputers = {}, {}
for j in range(5):
    it = joblib.load(MODELS / f"08d_mice_imp{j}.joblib")
    imputers[j] = it
    imp_sets[j] = pd.DataFrame(it.transform(m[FULL]), columns=FULL, index=m.index)
lg("MICE m=5 transform 完成 (08d 冻结实体, 无 refit)")
tr_idx, iv_idx = tr.index.to_numpy(), iv.index.to_numpy()
Xiv_sets = {j: imp_sets[j].loc[iv_idx] for j in range(5)}

# ---------- 3. 三学习器 + 网格 (SAP §7.2 清单) ----------
def make_lr(C, cw):
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                               class_weight=cw, max_iter=5000, random_state=SEED))])
def make_rf(md, msl, mf, cw):
    return RandomForestClassifier(n_estimators=300, max_depth=md, min_samples_leaf=msl,
                                  max_features=mf, class_weight=cw, random_state=SEED, n_jobs=-1)
def make_xgb(md, lr_, sub, col, w):
    return XGBClassifier(n_estimators=400, max_depth=md, learning_rate=lr_, subsample=sub,
                         colsample_bytree=col, scale_pos_weight=w, tree_method="hist",
                         eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0)

LEARNERS = {
    "L1_LR": (make_lr,  [dict(C=c, cw=cw) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
                          for cw in (None, "balanced")]),
    "RF":    (make_rf,  [dict(md=md, msl=msl, mf=mf, cw=cw) for md in (None, 8) for msl in (5, 20)
                          for mf in ("sqrt", 0.3) for cw in (None, "balanced_subsample")]),
    "XGB":   (make_xgb, [dict(md=md, lr_=lr_, sub=sub, col=col, w=w) for md in (3, 6)
                          for lr_ in (0.05, 0.1) for sub in (0.8, 1.0) for col in (0.8, 1.0)
                          for w in (1.0, spw)]),
}
lg("网格: " + ", ".join(f"{k}={len(v[1])}" for k, v in LEARNERS.items()))

# ---------- 4. 嵌套 5×5 CV (08d 同骨架; 内层用插补集#1) ----------
Xtr1 = imp_sets[0].loc[tr_idx]
def nested_select(make, grid):
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof, best_cfg, inner_log = [], None, []
    for k, (itr, ivo) in enumerate(outer.split(Xtr1, ytr)):
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 100 + k)
        Xa, ya = Xtr1.iloc[itr], ytr[itr]
        best, best_auc = None, -1
        for cfg in grid:
            aucs = []
            for it2, iv2 in inner.split(Xa, ya):
                mdl = make(**cfg).fit(Xa.iloc[it2], ya[it2])
                aucs.append(roc_auc_score(ya[iv2], mdl.predict_proba(Xa.iloc[iv2])[:, 1]))
            mu = float(np.mean(aucs))
            if mu > best_auc: best_auc, best = mu, cfg
        mdl = make(**best).fit(Xa, ya)  # run1 崩点修复: itr 是对全 train(1661) 的位置索引, ya 已是该折子集(=ytr[itr]) — 误写 ya[itr] 越界
        oof.append(roc_auc_score(ytr[ivo], mdl.predict_proba(Xtr1.iloc[ivo])[:, 1]))
        best_cfg = best; inner_log.append(round(best_auc, 4))
    return best_cfg, float(np.mean(oof)), inner_log

res = {"eval_set": "mimic4 intval (与 08d 同集)", "n_intval": int(len(iv)), "events_intval": int(yiv.sum()),
       "anchor_08d_full": auc_full_anchor, "learners": {}, "flags": []}

def metrics_block(y, p, seed=SEED):
    rng = np.random.default_rng(seed); aucs = []
    yy, pp = np.asarray(y, int), np.asarray(p, float)
    pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
    for _ in range(1000):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(yy[idx], pp[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return dict(AUROC=round(float(roc_auc_score(yy, pp)), 4),
                AUROC_lo=round(float(lo), 4), AUROC_hi=round(float(hi), 4),
                AUPRC=round(float(average_precision_score(yy, pp)), 4),
                Brier=round(float(brier_score_loss(yy, pp)), 4))

def delong_compare(y, p1, p2):   # 08f 逐字复用 + run2 崩防 (L137 数组真值错: 输入法证转储 + 整形, diag8 三组真实输入无法复现 → 自带证据)
    (REP / "_08k_delong_forensic.json").write_text(json.dumps(
        {k: f"{type(v).__name__} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}"
         for k, v in (("y", y), ("p1", p1), ("p2", p2))}, ensure_ascii=False, indent=1), encoding="utf-8")
    y = np.asarray(y, int).ravel()
    p1 = np.asarray(p1, float).ravel()
    p2 = np.asarray(p2, float).ravel()
    def partials(p):
        p = np.asarray(p, float)
        pos, neg = p[y == 1], p[y == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos])
        V01 = np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg])
        return V10, V01
    P1, P2 = partials(p1), partials(p2)
    a1, a2 = float(P1[0].mean()), float(P2[0].mean())
    v1 = float(np.var(P1[0], ddof=1)) / len(P1[0]) + float(np.var(P1[1], ddof=1)) / len(P1[1])
    v2 = float(np.var(P2[0], ddof=1)) / len(P2[0]) + float(np.var(P2[1], ddof=1)) / len(P2[1])
    c10 = float(np.cov(P1[0], P2[0], ddof=1)[0, 1]) / len(P1[0])
    c01 = float(np.cov(P1[1], P2[1], ddof=1)[0, 1]) / len(P1[1])   # 根因修复: 此行原漏 [0,1] 索引 → c01 成 2×2 矩阵 → max() 数组真值崩 (run2/run3 同点必崩的确定性根因, diag9 健康值即修复后预期)
    tot = float(v1) + float(v2) - 2.0 * (float(c10) + float(c01))   # 纯 python float 算术 → max() 结构性安全 (diag9 锚: 同环境全 0-d 标量, 数学不变)
    se = float(np.sqrt(max(tot, 1e-18)))
    d = a1 - a2
    return dict(AUC_full=round(a1, 4), AUC_base=round(a2, 4), delta=round(float(d), 4),
                se=round(float(se), 4), p=round(float(2 * (1 - norm.cdf(abs(d / se)))), 5))

preds_store = {"stay_id": iv.stay_id.values}
for name, (make, grid) in LEARNERS.items():
    t1 = time.time()
    best_cfg, oof_auc, inner_log = nested_select(make, grid)
    # 终模型: 5 套插补各 fit, 概率平均 (08d run_ensemble 同式)
    p_iv = np.zeros(len(iv))
    for j in range(5):
        mdl = make(**best_cfg).fit(imp_sets[j].loc[tr_idx], ytr)
        p_iv += mdl.predict_proba(Xiv_sets[j])[:, 1] / 5
    mb = metrics_block(iv.d28.values, p_iv)
    dl = delong_compare(iv.d28.values, iv.p_full.values, p_iv)
    if name == "L1_LR":  # sklearn 1.8 penalty 弃用告警 → 系数稀疏度实证 L1 语义 (liblinear+标准化后 L1 应产生精确零系数)
        probe = make(**best_cfg).fit(imp_sets[0].loc[tr_idx], ytr)
        cz = probe.named_steps["lr"].coef_[0]
        res["l1_sparsity_check"] = dict(n_coef=int(len(cz)), n_zero=int((cz == 0).sum()),
                                        frac_zero=round(float((cz == 0).mean()), 4),
                                        note="frac_zero>0 即 L1 生效; L2/无罚不会产生精确零系数")
    res["learners"][name] = dict(best_cfg={k: (str(v) if v is None else v) for k, v in best_cfg.items()},
                                 nested_cv_oof_auc=round(oof_auc, 4), inner_best_by_fold=inner_log,
                                 intval=mb, delong_vs_full=dl, elapsed_s=round(time.time() - t1, 1))
    preds_store[f"p_{name}"] = p_iv
    lg(f"[{name}] 嵌套OOF={oof_auc:.4f} | intval AUROC={mb['AUROC']} [{mb['AUROC_lo']},{mb['AUROC_hi']}] "
       f"AUPRC={mb['AUPRC']} | Δvs full={dl['delta']:+.4f} (p={dl['p']})")

pd.DataFrame(preds_store).to_parquet(DATA / "08k_preds_intval.parquet", index=False)

# ---------- 5. 外验 eICU/M3 zero-touch (08e build 口径复用) ----------
def strip_dyn(df):
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)

NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

def build_eicu():
    fe = pd.read_parquet(DATA / "features_eicu.parquet").rename(columns={"age": "admission_age"})
    fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
    ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
    if ic: fe = NULLABLE(fe, ic)
    post = pd.read_parquet(DATA / "08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
    d = fe.merge(post, on="icustay_id_eicu", how="left")
    died24 = ((d.died_hosp == 1) & (d.los_h <= 24)); short = d.los_h < 24
    return d[~(died24 | short)].copy(), int(died24.sum()), int(short.sum())

def build_m3():
    fm = pd.read_parquet(DATA / "features_mimic3.parquet").rename(columns={"age": "admission_age"})
    fm["sex_female"] = 1 - fm["male"]; fm = strip_dyn(fm)
    ic = [c for c in fm.columns if str(fm[c].dtype) == "Int64"]
    if ic: fm = NULLABLE(fm, ic)
    post = pd.read_parquet(DATA / "08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
    d = fm.merge(post, on="icustay_id", how="left")
    # CareVue-only 过滤 (2026-09-07): 同 08e/08h/09d
    import duckdb as _dd
    _cv3 = _dd.connect().execute("""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
    FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true)
    WHERE DBSOURCE = 'carevue'""").df()
    d = d[d.icustay_id.isin(_cv3.icustay_id)].copy()
    died24 = (d.d28 == 1) & (d.days_to_death < 1.0); short = d.los_h < 24
    return d[~(died24 | short)].copy(), int(died24.sum()), int(short.sum())

res["external"] = {}
for tag, builder, ycol, refname, idcol in [("eicu", build_eicu, "died_hosp_28d", "08e_preds_eicu.parquet", "icustay_id_eicu"),   # run5 修复: run4 误用裸 died_hosp (events 530 vs 冻结 520); 08e L178/08g L92/09d L241 均为 died_hosp_28d (07b L45: 院内死亡×hospitaldischargeoffset≤28d)
                                            ("mimic3", build_m3, "d28", "08e_preds_mimic3.parquet", "icustay_id")]:
    d, n_died24, n_short = builder()
    missing = [c for c in FULL if c not in d.columns]
    assert not missing, f"[{tag}] FULL 缺列: {missing}"
    ref = pd.read_parquet(DATA / refname)
    pcol = "p_full" if "p_full" in ref.columns else ref.columns[ref.columns.str.startswith("p_")][0]
    d = d.merge(ref[[idcol, pcol]].rename(columns={idcol: "_id", pcol: "p_full_ref"}), left_on=idcol, right_on="_id", how="left")
    y_ext = d[ycol].astype(int).values
    blk = {"n": int(len(d)), "events": int(y_ext.sum()), "excl_died24": n_died24, "excl_short": n_short}
    store = {idcol: d[idcol].values}
    for name, (make, grid) in LEARNERS.items():
        cfg = res["learners"][name]["best_cfg"]
        cfg_typed = dict(cfg)
        if "cw" in cfg_typed and cfg_typed["cw"] == "None": cfg_typed["cw"] = None
        if "md" in cfg_typed and cfg_typed["md"] == "None": cfg_typed["md"] = None
        p_ext = np.zeros(len(d))
        for j in range(5):
            Xe_j = pd.DataFrame(imputers[j].transform(d[FULL]), columns=FULL)  # 冻结 imputer transform (08e 同范式)
            mdl = make(**cfg_typed).fit(imp_sets[j].loc[tr_idx], ytr)   # 重新 fit (与 intval 同实体)
            p_ext += mdl.predict_proba(Xe_j)[:, 1] / 5
        blk[name] = dict(AUROC=round(float(roc_auc_score(y_ext, p_ext)), 4),
                         AUPRC=round(float(average_precision_score(y_ext, p_ext)), 4),
                         AUROC_08d_full_ref=round(float(roc_auc_score(y_ext, d.p_full_ref)), 4),
                         delong_vs_full=delong_compare(y_ext, d.p_full_ref.values, p_ext))
        store[f"p_{name}"] = p_ext
        lg(f"[{tag} {name}] AUROC={blk[name]['AUROC']} (08d full ref={blk[name]['AUROC_08d_full_ref']}) "
           f"Δ={blk[name]['delong_vs_full']['delta']:+.4f} p={blk[name]['delong_vs_full']['p']}")
    res["external"][tag] = blk
    pd.DataFrame(store).to_parquet(DATA / f"08k_preds_{tag}.parquet", index=False)

(REP / "08k_ml_benchmark.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"产出 reports/08k_ml_benchmark.json | 总耗时 {time.time()-t0:.0f}s")
