# -*- coding: utf-8 -*-
# ============================================================
# 08g — eICU 后验可得亚组消融 (PI 2026-09-03 拍板 ②: 回答"轨迹块在 eICU 不加分
#       是不是被 35% 被插补患者稀释的")
#   设计: 只在 post_avail=True (n=2889) 内重做 A / AB / full 三向消融
#   口径: 与 08e 逐字同源 — frozen_predict(冻结参数 M4-train 重训消融器, 无调参);
#         build_eicu / tr_sets / bootstrap_auc 从 08e 复制; delong_compare 从 08f 复制
#   门禁: 先复现 08e 四个锚点 (全库 full 0.83 / avail 0.8793 / imputed 0.7063 /
#         ablation A 0.8319 / AB 0.8308), 复现失败即 STOP — 不带病出数
#   定位: post-hoc 敏感性分析延伸 (exploratory, 服务 Discussion, 非确证性)
# 产出: reports/08g_post_avail_ablation.json + data/08g_preds_eicu_ablation.parquet
# ============================================================
import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from joblib import load as jbload
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"
MDL = ROOT / "07_prediction_system/models"
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

print("=== 08g eICU 后验可得亚组消融 (post-hoc 敏感性, PI 拍板 2026-09-03) ===")

# ---------- 1. 加载冻结管线 (与 08e 逐字同源) ----------
card = json.loads((MDL / "08d_model_card.json").read_text(encoding="utf-8"))
FULL = list(card["full_cols"]); THRESH = float(card["threshold_youden"])
PARAMS = dict(card["params"]); MICE_M = int(card["mice_m"])
lgb_params = {k: v for k, v in PARAMS.items() if k not in ("num_boost_round",)}
NR = int(PARAMS.get("num_boost_round", 500))
rep0 = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
A_COLS, B_COLS, C_COLS = rep0["blockA_cols"], rep0["blockB_cols"], rep0["blockC_cols"]
assert A_COLS + B_COLS + C_COLS == FULL
BL = {"A": A_COLS, "B": B_COLS, "C": C_COLS}
boosters = [lgb.Booster(model_file=str(MDL / f"08d_lgbm_full_imp{j}.txt")) for j in range(MICE_M)]
imputers = [jbload(MDL / f"08d_mice_imp{j}.joblib") for j in range(MICE_M)]
assert all(list(it.feature_names_in_) == FULL for it in imputers)
lg(f"冻结管线加载: FULL={len(FULL)} (A{len(A_COLS)}/B{len(B_COLS)}/C{len(C_COLS)}), Youden={THRESH:.4f}, MICE m={MICE_M}")

def frozen_predict(Xdf, feats=None, refit_on=None):
    """[08e 逐字复制] Xdf: 外库特征 DataFrame(已对齐列名). feats+refit_on: M4-train 冻结参数重训消融器"""
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

# ---------- 2. M4 train 重建 (08e 逐字) ----------
m4 = pd.read_parquet(DATA / "08c_matrix_mimic4.parquet")
tr4 = m4[m4.split == "train"]
tr_sets = {"y": tr4.d28.values.astype(int)}
for j in range(MICE_M):
    tr_sets[j] = imputers[j].transform(tr4[FULL])
lg(f"M4 train 重建: n={len(tr4)} (ev {tr4.d28.mean():.1%})")

# ---------- 3. eICU 队列构建 (08e 逐字) ----------
NULLABLE = lambda df, cols: df.assign(**{c: df[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

def strip_dyn(df):
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)

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
d = d[~(died24 | short)].copy()
missing = [c for c in FULL if c not in d.columns]
assert not missing, f"FULL 缺列: {missing}"
ye = d.died_hosp_28d.values.astype(int)
lg(f"eICU 风险集 n={len(d)} ev={ye.sum()} | post_avail={int(d.post_avail.sum())} ({d.post_avail.mean():.1%})")

# ---------- 4. 三向预测: full / A / AB ----------
p_full = frozen_predict(d)
p_A = frozen_predict(d, feats=BL["A"], refit_on=tr_sets)
p_AB = frozen_predict(d, feats=BL["A"] + BL["B"], refit_on=tr_sets)
auc_full = round(float(roc_auc_score(ye, p_full)), 4)
auc_A = round(float(roc_auc_score(ye, p_A)), 4)
auc_AB = round(float(roc_auc_score(ye, p_AB)), 4)
lg(f"全库复算: full={auc_full} A={auc_A} AB={auc_AB}")

# ---------- 5. 08e 锚点复现门禁 (失败即 STOP) ----------
e8 = json.loads((REP / "08e_external_validation.json").read_text(encoding="utf-8"))
anchor = {
    "full":  (auc_full, e8["eicu"]["AUROC"]),
    "A":     (auc_A,    e8["eicu"]["ablation_A"]["AUROC"]),
    "AB":    (auc_AB,   e8["eicu"]["ablation_AB"]["AUROC"]),
    "avail": (round(float(roc_auc_score(ye[d.post_avail.values], p_full[d.post_avail.values])), 4),
              e8["eicu"]["sens_posterior_available"]["AUROC"]),
    "imputed": (round(float(roc_auc_score(ye[~d.post_avail.values], p_full[~d.post_avail.values])), 4),
                e8["eicu"]["sens_posterior_imputed"]["AUROC"]),
}
repro = {k: {"ours": v[0], "08e": v[1], "match": bool(v[0] == v[1])} for k, v in anchor.items()}
lg(f"锚点复现: { {k: v['match'] for k, v in repro.items()} }")
if not all(v["match"] for v in repro.values()):
    (REP / "08g_post_avail_ablation.json").write_text(json.dumps(
        {"STATUS": "ABORT_anchor_mismatch", "reproduce_08e": repro}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    lg("!! 08e 锚点复现失败 — STOP, 不出亚组数字"); sys.exit(1)

# ---------- 6. DeLong (08f 逐字复制) + bootstrap CI (08e 逐字) ----------
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
    P1 = partials(p1); P2 = partials(p2)
    a1 = P1[0].mean(); a2 = P2[0].mean()
    v1 = np.var(P1[0], ddof=1) / len(P1[0]) + np.var(P1[1], ddof=1) / len(P1[1])
    v2 = np.var(P2[0], ddof=1) / len(P2[0]) + np.var(P2[1], ddof=1) / len(P2[1])
    c10 = np.cov(P1[0], P2[0], ddof=1)[0, 1] / len(P1[0])
    c01 = np.cov(P1[1], P2[1], ddof=1)[0, 1] / len(P1[1])
    cov = c10 + c01
    dlt = a1 - a2
    se = np.sqrt(max(v1 + v2 - 2 * cov, 1e-18))
    z = dlt / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return dict(AUC_1=round(a1, 4), AUC_2=round(a2, 4), delta=round(float(dlt), 4),
                se=round(float(se), 4), z=round(float(z), 3), p=round(float(pval), 5))

def bootstrap_auc(y, p, n=1000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(q), 4) for q in np.percentile(aucs, [2.5, 50, 97.5])]

def block(y, p, label):
    r = dict(n=int(len(y)), events=int(y.sum()), AUROC=round(float(roc_auc_score(y, p)), 4),
             AUROC_boot95CI=bootstrap_auc(y, p))
    lg(f"[{label}] AUROC={r['AUROC']} CI{r['AUROC_boot95CI']} n={r['n']} ev={r['events']}")
    return r

# ---------- 7. 亚组消融 ----------
av = d.post_avail.values
res = {"purpose": "eICU 后验可得亚组消融 — 回答'轨迹块 eICU 零增益是否被 35% 被插补患者稀释' (PI 2026-09-03 拍板②)",
       "pi_decisions_2026_09_03": {"framing": "B — '轨迹价值集中于连续GCS监测患者'升格为核心 Discussion 点(部署洞见)",
                                    "analysis": "只在 post_avail 亚组内重做 A/AB/full 消融",
                                    "sap_disclosure": "A — Methods 一句披露 SAP 估计 ~6681 vs 实际 4440"},
       "methodology": "与 08e 逐字同源: frozen_predict 冻结参数 M4-train 重训消融器(无调参/无阈值重选/无特征重选); "
                      "post-hoc 敏感性(exploratory), 非确证性推断",
       "reproduce_08e": repro}

sub = {"A": block(ye[av], p_A[av], "avail·A"),
       "AB": block(ye[av], p_AB[av], "avail·AB"),
       "full": block(ye[av], p_full[av], "avail·full"),
       "delong_AB_vs_A": delong_compare(ye[av], p_AB[av], p_A[av]),
       "delong_full_vs_A": delong_compare(ye[av], p_full[av], p_A[av])}
res["subgroup_post_avail"] = sub
lg(f"亚组关键: ΔAUC(AB−A)={sub['delong_AB_vs_A']['delta']:+.4f} p={sub['delong_AB_vs_A']['p']} | "
   f"ΔAUC(full−A)={sub['delong_full_vs_A']['delta']:+.4f} p={sub['delong_full_vs_A']['p']}")

imp = {"A": block(ye[~av], p_A[~av], "imputed·A"),
       "AB": block(ye[~av], p_AB[~av], "imputed·AB"),
       "full": block(ye[~av], p_full[~av], "imputed·full"),
       "delong_AB_vs_A": delong_compare(ye[~av], p_AB[~av], p_A[~av])}
res["subgroup_post_imputed"] = imp

res["flags"] = ["post-hoc 敏感性分析 (exploratory) — Discussion 引用时注明, 不进确证性主链",
                "消融器为冻结参数 M4-train 重训 (与 08e 外验消融同一披露口径)",
                "亚组按 prob_m1 可得性定义 (08b 后验引擎, 与 08e sens 分层同源)"]

pd.DataFrame({"icustay_id_eicu": d.icustay_id_eicu.values, "died_hosp_28d": ye,
              "post_avail": av, "p_full": p_full, "p_A": p_A, "p_AB": p_AB}
             ).to_parquet(DATA / "08g_preds_eicu_ablation.parquet", index=False)
(REP / "08g_post_avail_ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"产出: {REP}/08g_post_avail_ablation.json + {DATA}/08g_preds_eicu_ablation.parquet")
print("DONE 08g")
