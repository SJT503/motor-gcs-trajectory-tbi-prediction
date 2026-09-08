# -*- coding: utf-8 -*-
# ============================================================
# 08e — 三库外验 (SAP L32/L100-103 zero-touch 红线)
#   eICU (co-primary 估计对象) + MIMIC-III (Holm 序内报告)
#   冻结管线: MICE(M4-train-fitted) transform → 5 booster 均值 → 冻结 Youden
#   禁: 任何调参 / 重选阈值 / 重选特征 (L32)
#   次级: AUPRC / 校准三件套+ECE+Brier / DCA@0.10/0.20/0.30 / 外验三向消融(A,AB 冻结参数在 M4-train 重训, 无调参)
#   敏感性: eICU 后验可得 vs 后验被插补 分层 (08b 644 例家族缺失 PI 决策点 → 用数据回答)
# 产出: reports/08e_external_validation.json + data/08e_preds_eicu.parquet + data/08e_preds_mimic3.parquet
# 挂旗: 外验 ΔAUC vs base 需 eICU/M3 SOFA 移植 (独立工作包, 本脚本不做)
# ============================================================
import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from joblib import load as jbload
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import norm

ROOT = Path(r"E:/TBI subtype")
M3 = ROOT / "data/mimic-iii-1.4"
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
MDL = ROOT / "07_prediction_system/models"
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

print("=== 08e 三库外验 (zero-touch, SAP L32/L100) ===")

# ---------- 1. 加载冻结管线 ----------
card = json.loads((MDL / "08d_model_card.json").read_text(encoding="utf-8"))
FULL = list(card["full_cols"]); THRESH = float(card["threshold_youden"])
PARAMS = dict(card["params"]); MICE_M = int(card["mice_m"])
lgb_params = {k: v for k, v in PARAMS.items() if k not in ("num_boost_round",)}   # 含 scale_pos_weight (grid 内选定)
NR = int(PARAMS.get("num_boost_round", 500))
# Block 权威定义 = 08c report (与 08d 消融逐字同源, 不自行推断)
rep0 = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
A_COLS, B_COLS, C_COLS = rep0["blockA_cols"], rep0["blockB_cols"], rep0["blockC_cols"]
assert A_COLS + B_COLS + C_COLS == FULL, "08c blocks 拼接 ≠ card full_cols — 冻结管线自洽性破坏"
BL = {"A": A_COLS, "B": B_COLS, "C": C_COLS}
boosters = [lgb.Booster(model_file=str(MDL / f"08d_lgbm_full_imp{j}.txt")) for j in range(MICE_M)]
imputers = [jbload(MDL / f"08d_mice_imp{j}.joblib") for j in range(MICE_M)]
assert all(list(it.feature_names_in_) == FULL for it in imputers), "MICE imputer 特征名与 card full_cols 不一致 — 冻结管线被破坏"
lg(f"冻结管线加载: FULL={len(FULL)} (A{len(A_COLS)}/B{len(B_COLS)}/C{len(C_COLS)}), Youden={THRESH:.4f}, MICE m={MICE_M}")

def frozen_predict(Xdf, feats=None, refit_on=None):
    """Xdf: 外库特征 DataFrame(已对齐列名). feats+refit_on 给定时: 在 M4-train 用冻结参数重训消融器(无调参)"""
    ps = []
    for j in range(MICE_M):
        Xj = imputers[j].transform(Xdf[FULL])
        if feats is None:
            p = boosters[j].predict(Xj)
        else:
            idx = [FULL.index(c) for c in feats]
            bst = lgb.train(lgb_params, lgb.Dataset(refit_on[j][:, idx], label=refit_on["y"]),
                            num_boost_round=NR)
            p = bst.predict(Xj[:, idx])
        ps.append(p)
    return np.mean(ps, 0)

# ---------- 2. M4 train 重建 (供外验消融 A/AB 在 M4-train 用冻结参数重训, 无调参) ----------
m4 = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
tr4 = m4[m4.split == "train"]
tr_sets = {"y": tr4.d28.values.astype(int)}
for j in range(MICE_M):
    tr_sets[j] = imputers[j].transform(tr4[FULL])
lg(f"M4 train 重建: n={len(tr4)} (ev {tr4.d28.mean():.1%})")

# ---------- 3. 外库特征矩阵构建 ----------
NULLABLE = lambda df, cols: df.assign(**{c: df[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

def strip_dyn(df):
    """dyn_motor_X→motor_X ; dyn_gcs_total_X→total_X (显式双前缀 — 防 dyn_gcs_total_min 撞静态 gcs_total_min)"""
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)

def build_eicu():
    fe = pd.read_parquet(DATA / "features_eicu.parquet")
    fe = fe.rename(columns={"age": "admission_age"})
    fe["sex_female"] = 1 - fe["male"]
    fe = strip_dyn(fe)
    intcols = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
    if intcols: fe = NULLABLE(fe, intcols)
    post = pd.read_parquet(DATA / "08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
    d = fe.merge(post, on="icustay_id_eicu", how="left")
    d["post_avail"] = d["prob_m1"].notna()
    died24 = ((d.died_hosp == 1) & (d.los_h <= 24))
    short = d.los_h < 24
    d = d[~(died24 | short)].copy()   # 风险集: 24h 可观测 (M4 同规则; eICU 无精确死亡时刻, 用 los_h 近似, 报告披露)
    return d, int(died24.sum()), int(short.sum())

def build_m3():
    fm = pd.read_parquet(DATA / "features_mimic3.parquet").rename(columns={"age": "admission_age"})
    fm["sex_female"] = 1 - fm["male"]
    fm = strip_dyn(fm)
    intcols = [c for c in fm.columns if str(fm[c].dtype) == "Int64"]
    if intcols: fm = NULLABLE(fm, intcols)
    post = pd.read_parquet(DATA / "08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
    d = fm.merge(post, on="icustay_id", how="left")
    d["post_avail"] = d["prob_m1"].notna()
    # CareVue-only 过滤 (2026-09-07): MIMIC-III MetaVision 期 (2008-2012) 与 MIMIC-IV train (2008-2017)
    # 患者重叠 (PhysioNet 官方确认), 仅保留 CareVue 期 (2001-2008, 不同 EHR 系统, 真独立外验)
    import duckdb
    con = duckdb.connect()
    cv_ids = con.execute(f"""
        SELECT DISTINCT CAST(i.icustay_id AS BIGINT) AS icustay_id
        FROM read_csv('{M3.as_posix()}/ICUSTAYS.csv.gz', all_varchar=true) i
        WHERE i.dbsource = 'carevue'""").df()
    n_before = len(d)
    d = d[d.icustay_id.isin(cv_ids.icustay_id)].copy()
    lg(f"[M3 CareVue 过滤] {n_before}→{len(d)} (剔 MetaVision 重叠 {n_before - len(d)})")
    died24 = (d.d28 == 1) & (d.days_to_death < 1.0)
    short = d.los_h < 24
    d = d[~(died24 | short)].copy()
    return d, int(died24.sum()), int(short.sum())

def align_check(d, name):
    missing = [c for c in FULL if c not in d.columns]
    assert not missing, f"[{name}] FULL 缺列: {missing} — 特征对齐失败"
    lg(f"[{name}] FULL {len(FULL)} 列全对齐 ✓")

# ---------- 4. 评价函数 ----------
def bootstrap_auc(y, p, n=1000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(q), 4) for q in np.percentile(aucs, [2.5, 50, 97.5])]

def calib3(y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    X = np.log(pc / (1 - pc)).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)   # 大 C = 无惩罚校准回归
    # ECE 10 等频 bin
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "b": q})
    g = df.groupby("b", observed=True).agg(pm=("p", "mean"), om=("y", "mean"), n=("y", "size"))
    ece = float((g.n / len(df) * (g.pm - g.om).abs()).sum())
    return dict(slope=round(float(lr.coef_[0][0]), 3), intercept=round(float(lr.intercept_[0]), 3),
                ECE=round(ece, 4), Brier=round(float(brier_score_loss(y, p)), 4))

def net_benefit(y, p, t):
    y = np.asarray(y, int); p = np.asarray(p, float)
    pos = (p >= t)
    return float((y[pos].sum() - pos.sum() * t / (1 - t)) / len(y))

def dca_block(y, p):
    ts = np.round(np.arange(0.05, 0.5001, 0.01), 3)
    curve = [{"t": float(t), "model": round(net_benefit(y, p, t), 5),
              "treat_all": round(float(y.mean() - (1 - y.mean()) * t / (1 - t)), 5)} for t in ts]
    pts = {f"nb@{int(t*100)}": round(net_benefit(y, p, t), 5) for t in (0.10, 0.20, 0.30)}
    return {"curve": curve, "points": pts}

def metrics_block(y, p, thr, label):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    auci = bootstrap_auc(y, p)
    r = dict(n=int(len(y)), events=int(y.sum()), event_rate=round(float(y.mean()), 4),
             AUROC=round(float(roc_auc_score(y, p)), 4), AUROC_boot95CI=auci,
             AUPRC=round(float(average_precision_score(y, p)), 4),
             sens_at_youden=round(tp / max(tp + fn, 1), 4), spec_at_youden=round(tn / max(tn + fp, 1), 4),
             PPV=round(tp / max(tp + fp, 1), 4), NPV=round(tn / max(tn + fn, 1), 4),
             threshold_frozen=round(float(thr), 6), **{f"calib_{k}": v for k, v in calib3(y, p).items()})
    if y.sum() < 100:
        r["SAP_L129_warning"] = "外验事件 <100 → CI 宽度预警, 不单独下强结论"
    lg(f"[{label}] AUROC={r['AUROC']} CI{auci} AUPRC={r['AUPRC']} sens={r['sens_at_youden']} spec={r['spec_at_youden']} n={r['n']} ev={r['events']}")
    if r["AUROC"] < 0.70:
        r["SAP_L103_flag"] = "AUROC<0.70 → 定性'性能受限', 走 §9 漂移叙事 (禁止回头调参)"
    return r

res = {"frozen_pipeline": {"n_features": len(FULL), "youden_threshold": THRESH, "blocks": {b: len(v) for b, v in BL.items()},
                           "source": "08d_model_card.json + 08d_lgbm_full_imp[0-4] + 08d_mice_imp[0-4]"},
       "zero_touch": "无调参/无阈值重选/无特征重选 (SAP L32)",
       "flags": ["外验 ΔAUC vs base (年龄+性别+GCS+Charlson+SOFA) 需 eICU/M3 SOFA 移植 — 独立工作包未做",
                  "eICU 无精确死亡时刻, 风险集 24h 内死亡用 los_h≤24&died_hosp 近似 — Methods 披露"]}

# ---------- 5. eICU (co-primary) ----------
ei, n_died24, n_short = build_eicu()
align_check(ei, "eICU")
lg(f"eICU 风险集: {len(ei)}/4440 (剔 24h 内死亡≈{n_died24}, LOS<24h={n_short}) | 后验可得 {int(ei.post_avail.sum())} ({ei.post_avail.mean():.1%})")
ye, pe = ei.died_hosp_28d.values.astype(int), frozen_predict(ei)
pd.DataFrame({"icustay_id_eicu": ei.icustay_id_eicu.values, "p_full": pe, "d28": ye,
              "post_avail": ei.post_avail.values}).to_parquet(DATA / "08e_preds_eicu.parquet", index=False)
res["eicu"] = metrics_block(ye, pe, THRESH, "eICU")
res["eicu"]["risk_set_note"] = f"4440 → {len(ei)} (剔 24h 死亡≈{int(n_died24)} + LOS<24h {int(n_short)}); SAP_v2 估计 ~6681 vs 实际 4440 — 披露"
res["eicu"]["dca"] = dca_block(ye, pe)
# 敏感性: 后验可得 vs 后验被插补
for lbl, msk in [("posterior_available", ei.post_avail.values), ("posterior_imputed", ~ei.post_avail.values)]:
    if msk.sum() >= 30 and ye[msk].sum() >= 5:
        res["eicu"][f"sens_{lbl}"] = dict(n=int(msk.sum()), events=int(ye[msk].sum()),
                                          AUROC=round(float(roc_auc_score(ye[msk], pe[msk])), 4))
# 外验消融 (冻结参数 M4-train 重训)
for bname, feats in (("ablation_A", BL["A"]), ("ablation_AB", BL["A"] + BL["B"])):
    p_ab = frozen_predict(ei, feats=feats, refit_on=tr_sets)
    res["eicu"][bname] = dict(n_feats=len(feats), AUROC=round(float(roc_auc_score(ye, p_ab)), 4))
    lg(f"eICU {bname}: AUROC={res['eicu'][bname]['AUROC']} ({len(feats)} feats)")

# ---------- 6. MIMIC-III (Holm 序内报告) ----------
m3, n_d24, n_sh = build_m3()
align_check(m3, "MIMIC-III")
lg(f"M3 风险集: {len(m3)}/1425 (剔 24h 死亡≈{n_d24}, LOS<24h={n_sh}) | 后验可得 {int(m3.post_avail.sum())}")
y3, p3 = m3.d28.values.astype(int), frozen_predict(m3)
pd.DataFrame({"icustay_id": m3.icustay_id.values, "p_full": p3, "d28": y3,
              "post_avail": m3.post_avail.values}).to_parquet(DATA / "08e_preds_mimic3.parquet", index=False)
res["mimic3"] = metrics_block(y3, p3, THRESH, "MIMIC-III")
res["mimic3"]["dca"] = dca_block(y3, p3)
for bname, feats in (("ablation_A", BL["A"]), ("ablation_AB", BL["A"] + BL["B"])):
    p_ab = frozen_predict(m3, feats=feats, refit_on=tr_sets)
    res["mimic3"][bname] = dict(n_feats=len(feats), AUROC=round(float(roc_auc_score(y3, p_ab)), 4))

(REP / "08e_external_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"产出: {REP}/08e_external_validation.json")
print("DONE 08e")
