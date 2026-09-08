# -*- coding: utf-8 -*-
# ============================================================
# 11a — v2 (25-feature main model) consistency reruns that were never done after the
#       2026-09-07 model switch (SAP §12): 08g (block ablation + eICU documentation
#       subgroup), 08j (external ΔAUC vs base models), proper seed sensitivity, tune-set
#       (2018-2019) performance, Holm family table, eICU hospital-level clustering of
#       motor-GCS absence. Anchors: refit-full must reproduce 08d_v2 / 08e_v2 preds.
# Outputs: reports/11a_v2_ablation_subgroup.json, reports/11a_v2_external_delta.json,
#          reports/11a_v2_seeds_tune.json, data/11a_v2_ablation_preds_{eicu,mimic3,intval}.parquet
# ============================================================
import json, time, warnings, sys
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"
t0 = time.time()
def lg(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

card = json.load(open(f"{MODELS}/08d_v2_model_card.json", encoding="utf-8"))
SEL = card["features"]; THR = float(card["threshold_youden_train_oof"]); SPW = float(card["scale_pos_weight"])
cfg = {k: (int(v) if k in ("nl", "md", "mcs") else float(v)) for k, v in card["best_cfg"].items()}
rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
A_SEL = [c for c in SEL if c in rep0["blockA_cols"]]
B_SEL = [c for c in SEL if c in rep0["blockB_cols"]]
C_SEL = [c for c in SEL if c in rep0["blockC_cols"]]
assert len(A_SEL) + len(B_SEL) + len(C_SEL) == 25, (A_SEL, B_SEL, C_SEL)
VARIANTS = {"A": A_SEL, "AB": A_SEL + B_SEL, "ABC": SEL}
lg(f"blocks A={len(A_SEL)} B={B_SEL} C={C_SEL} | cfg={cfg} spw={SPW:.4f} thr={THR}")

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr, tu, iv = (m[m.split == s].copy() for s in ("train", "tune", "intval"))
ytr, ytu, yiv = (d.d28.values.astype(int) for d in (tr, tu, iv))
imputers = [joblib.load(f"{MODELS}/08d_mice_imp{j}.joblib") for j in range(5)]
imp = {j: pd.DataFrame(imputers[j].transform(m[FULL]), columns=FULL, index=m.index)[SEL] for j in range(5)}
tr_idx, tu_idx, iv_idx = tr.index.to_numpy(), tu.index.to_numpy(), iv.index.to_numpy()
lg(f"train {len(tr)} (ev {ytr.sum()}) | tune {len(tu)} (ev {ytu.sum()}) | intval {len(iv)} (ev {yiv.sum()})")

def make(seed=42):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=cfg["lr"], num_leaves=cfg["nl"], max_depth=cfg["md"],
                              colsample_bytree=cfg["col"], subsample=cfg["sub"], subsample_freq=1,
                              min_child_samples=cfg["mcs"], scale_pos_weight=SPW, random_state=seed, n_jobs=4, verbose=-1)

# ---------- external cohorts (10d/08g build, verbatim logic) ----------
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
        d = d[~(died24 | short)].copy(); y = d.died_hosp_28d.astype(int).values; idc = "icustay_id_eicu"
    else:
        fe = pd.read_parquet(f"{DATA}/features_mimic3.parquet").rename(columns={"age": "admission_age"})
        fe["sex_female"] = 1 - fe["male"]; fe = strip_dyn(fe)
        ic = [c for c in fe.columns if str(fe[c].dtype) == "Int64"]
        if ic: fe = NULLABLE(fe, ic)
        post = pd.read_parquet(f"{DATA}/08b_posterior24_mimic3.parquet").rename(columns={"id": "icustay_id"})
        d = fe.merge(post, on="icustay_id", how="left")
        import duckdb as _dd
        _cv = _dd.connect().execute("""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
        FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'""").df()
        d = d[d.icustay_id.isin(_cv.icustay_id)].copy()
        died24 = (d.d28 == 1) & (d.days_to_death < 1.0); short = d.los_h < 24
        d = d[~(died24 | short)].copy(); y = d.d28.astype(int).values; idc = "icustay_id"
    d["post_avail"] = d["prob_m1"].notna()
    Xs = {j: pd.DataFrame(imputers[j].transform(d[FULL]), columns=FULL)[SEL] for j in range(5)}
    return d.reset_index(drop=True), y, Xs, idc
ext = {tag: build(tag) for tag in ("eicu", "mimic3")}
lg(f"eICU n={len(ext['eicu'][0])} ev={ext['eicu'][1].sum()} avail={int(ext['eicu'][0].post_avail.sum())} | "
   f"M3 n={len(ext['mimic3'][0])} ev={ext['mimic3'][1].sum()} avail={int(ext['mimic3'][0].post_avail.sum())}")

# ---------- fit each variant once per imputation, predict all evaluation sets ----------
EVAL = {"intval": {j: imp[j].loc[iv_idx] for j in range(5)}, "tune": {j: imp[j].loc[tu_idx] for j in range(5)},
        "eicu": ext["eicu"][2], "mimic3": ext["mimic3"][2]}
Y = {"intval": yiv, "tune": ytu, "eicu": ext["eicu"][1], "mimic3": ext["mimic3"][1]}
P = {v: {k: np.zeros(len(Y[k])) for k in EVAL} for v in VARIANTS}
for v, feats in VARIANTS.items():
    for j in range(5):
        mdl = make(42).fit(imp[j].loc[tr_idx, feats], ytr)
        for k in EVAL:
            P[v][k] += mdl.predict_proba(EVAL[k][j][feats])[:, 1] / 5
    lg(f"variant {v} ({len(feats)} feats): " + " | ".join(f"{k} {roc_auc_score(Y[k], P[v][k]):.4f}" for k in EVAL))

# ---------- anchors: refit-full must reproduce frozen v2 predictions ----------
ref_iv = pd.read_parquet(f"{DATA}/08d_v2_preds_mimic4.parquet").set_index("stay_id").loc[iv.stay_id, "p_full"].values
ref_e = pd.read_parquet(f"{DATA}/08e_v2_preds_eicu.parquet").set_index("icustay_id_eicu").loc[ext["eicu"][0].icustay_id_eicu, "p_full"].values
ref_3 = pd.read_parquet(f"{DATA}/08e_v2_preds_mimic3.parquet").set_index("icustay_id").loc[ext["mimic3"][0].icustay_id, "p_full"].values
anch = {k: dict(auroc_refit=round(float(roc_auc_score(Y[k], P["ABC"][k])), 4), auroc_frozen=round(float(roc_auc_score(Y[k], r)), 4),
                max_abs_dp=round(float(np.max(np.abs(P["ABC"][k] - r))), 6))
        for k, r in (("intval", ref_iv), ("eicu", ref_e), ("mimic3", ref_3))}
lg(f"anchors: {anch}")
if any(abs(a["auroc_refit"] - a["auroc_frozen"]) > 1e-3 for a in anch.values()):
    lg("!! anchor mismatch > 0.001 — investigate before using ablation numbers"); ANCHOR_OK = False
else: ANCHOR_OK = True

def delong(y, p1, p2):
    y = np.asarray(y, int); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    def partials(p):
        pos, neg = p[y == 1], p[y == 0]; n1, n0 = len(pos), len(neg)
        V10 = np.array([(np.sum(neg < pv) + 0.5 * np.sum(neg == pv)) / n0 for pv in pos])
        V01 = np.array([(np.sum(pos < nv) + 0.5 * np.sum(pos == nv)) / n1 for nv in neg])
        return V10, V01
    P1, P2 = partials(p1), partials(p2); a1, a2 = P1[0].mean(), P2[0].mean()
    v1 = float(np.var(P1[0], ddof=1)) / len(P1[0]) + float(np.var(P1[1], ddof=1)) / len(P1[1])
    v2 = float(np.var(P2[0], ddof=1)) / len(P2[0]) + float(np.var(P2[1], ddof=1)) / len(P2[1])
    c10 = float(np.cov(P1[0], P2[0], ddof=1)[0, 1]) / len(P1[0]); c01 = float(np.cov(P1[1], P2[1], ddof=1)[0, 1]) / len(P1[1])
    se = max(v1 + v2 - 2 * (c10 + c01), 1e-18) ** 0.5; d = a1 - a2
    return dict(AUC_1=round(float(a1), 4), AUC_2=round(float(a2), 4), delta=round(float(d), 4), se=round(float(se), 4),
                p=float(2 * (1 - norm.cdf(abs(d / se)))))
def boot(y, p, B=1000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float); rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]); out.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(x), 4) for x in np.percentile(out, [2.5, 97.5])]
def block(y, p):
    return dict(n=int(len(y)), events=int(np.sum(y)), AUROC=round(float(roc_auc_score(y, p)), 4), AUROC_boot95=boot(y, p),
                AUPRC=round(float(average_precision_score(y, p)), 4), Brier=round(float(brier_score_loss(y, p)), 4))

# ---------- 08g_v2: ablation (all sets) + eICU/M3 documentation subgroups ----------
abl = {"anchors": anch, "anchor_ok": ANCHOR_OK, "blocks": {"A_static": A_SEL, "B_motor_dynamics": B_SEL, "C_posteriors": C_SEL},
       "sets": {}}
for k in EVAL:
    abl["sets"][k] = {v: block(Y[k], P[v][k]) for v in VARIANTS}
    abl["sets"][k]["delong_AB_vs_A"] = delong(Y[k], P["AB"][k], P["A"][k])
    abl["sets"][k]["delong_ABC_vs_AB"] = delong(Y[k], P["ABC"][k], P["AB"][k])
    abl["sets"][k]["delong_ABC_vs_A"] = delong(Y[k], P["ABC"][k], P["A"][k])
for tag in ("eicu", "mimic3"):
    d, y = ext[tag][0], ext[tag][1]; av = d.post_avail.values
    for lab, mask in (("motor_documented", av), ("motor_undocumented", ~av)):
        if mask.sum() < 30 or len(np.unique(y[mask])) < 2: continue
        sub = {v: block(y[mask], P[v][tag][mask]) for v in VARIANTS}
        sub["delong_AB_vs_A"] = delong(y[mask], P["AB"][tag][mask], P["A"][tag][mask])
        sub["delong_ABC_vs_AB"] = delong(y[mask], P["ABC"][tag][mask], P["AB"][tag][mask])
        sub["delong_ABC_vs_A"] = delong(y[mask], P["ABC"][tag][mask], P["A"][tag][mask])
        abl["sets"][f"{tag}_{lab}"] = sub
        lg(f"[{tag} {lab}] n={mask.sum()} ev={y[mask].sum()} A={sub['A']['AUROC']} AB={sub['AB']['AUROC']} ABC={sub['ABC']['AUROC']} "
           f"ΔAB-A={sub['delong_AB_vs_A']['delta']:+.4f} p={sub['delong_AB_vs_A']['p']:.4f}")
# eICU hospital clustering of motor documentation
d_e = ext["eicu"][0]
hosp = d_e.groupby("hospitalid").agg(n=("post_avail", "size"), documented=("post_avail", "mean"))
abl["eicu_hospital_clustering"] = dict(n_hospitals=int(len(hosp)),
    hospitals_0pct=int((hosp.documented == 0).sum()), patients_in_0pct_hospitals=int(hosp.n[hosp.documented == 0].sum()),
    hospitals_lt10pct=int((hosp.documented < 0.10).sum()), patients_in_lt10pct=int(hosp.n[hosp.documented < 0.10].sum()),
    hospitals_gt90pct=int((hosp.documented > 0.90).sum()), patients_in_gt90pct=int(hosp.n[hosp.documented > 0.90].sum()),
    hospitals_100pct=int((hosp.documented == 1).sum()), patients_in_100pct=int(hosp.n[hosp.documented == 1].sum()),
    share_of_undocumented_patients_in_lt10pct_hospitals=float(hosp.n[hosp.documented < 0.10].sum() * 1.0 / max((~d_e.post_avail).sum(), 1)
                                                               * (1 - hosp.documented[hosp.documented < 0.10]).mean() if (hosp.documented < 0.10).any() else 0.0),
    documented_share_by_unittype={str(k): round(float(v), 3) for k, v in d_e.groupby("unittype").post_avail.mean().items()},
    n_by_unittype={str(k): int(v) for k, v in d_e.unittype.value_counts().items()})
# exact share of undocumented patients that sit in hospitals with <10% documentation
und = d_e[~d_e.post_avail]; abl["eicu_hospital_clustering"]["share_undocumented_in_lt10pct_hospitals"] = float(
    und.hospitalid.isin(hosp.index[hosp.documented < 0.10]).mean())
abl["eicu_hospital_clustering"]["share_undocumented_in_0pct_hospitals"] = float(und.hospitalid.isin(hosp.index[hosp.documented == 0]).mean())
lg(f"hospital clustering: {abl['eicu_hospital_clustering']}")
json.dump(abl, open(f"{REP}/11a_v2_ablation_subgroup.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame({"icustay_id_eicu": d_e.icustay_id_eicu.values, "y": Y["eicu"], "post_avail": d_e.post_avail.values,
              "p_A": P["A"]["eicu"], "p_AB": P["AB"]["eicu"], "p_ABC": P["ABC"]["eicu"]}).to_parquet(f"{DATA}/11a_v2_ablation_preds_eicu.parquet", index=False)
pd.DataFrame({"icustay_id": ext["mimic3"][0].icustay_id.values, "y": Y["mimic3"], "post_avail": ext["mimic3"][0].post_avail.values,
              "p_A": P["A"]["mimic3"], "p_AB": P["AB"]["mimic3"], "p_ABC": P["ABC"]["mimic3"]}).to_parquet(f"{DATA}/11a_v2_ablation_preds_mimic3.parquet", index=False)
pd.DataFrame({"stay_id": iv.stay_id.values, "y": yiv, "p_A": P["A"]["intval"], "p_AB": P["AB"]["intval"], "p_ABC": P["ABC"]["intval"]}).to_parquet(
    f"{DATA}/11a_v2_ablation_preds_intval.parquet", index=False)

# ---------- seeds (proper: LightGBM random_state varies; MICE/CV fixed) + tune set ----------
seeds = {}
for s in (42, 43, 44, 45, 46):
    p = np.zeros(len(yiv))
    for j in range(5):
        p += make(s).fit(imp[j].loc[tr_idx], ytr).predict_proba(imp[j].loc[iv_idx])[:, 1] / 5
    seeds[str(s)] = round(float(roc_auc_score(yiv, p)), 4)
tune_blk = block(ytu, P["ABC"]["tune"]); tune_blk["note"] = "2018-2019 model-selection set; not used by nested CV (hyperparameters chosen by inner CV on training folds); reported as an additional temporal hold-out"
st = {"seeds_intval_auroc": seeds, "seeds_mean": round(float(np.mean(list(seeds.values()))), 4), "seeds_range": [min(seeds.values()), max(seeds.values())],
      "note_seeds": "08d_v2 seed loop was vacuous (random_state hard-coded 42 → identical 0.9054×5); this is the corrected sensitivity",
      "tune_2018_2019": tune_blk,
      "intval_boot95": boot(yiv, ref_iv), "eicu_boot95": boot(Y["eicu"], ref_e), "mimic3_boot95": boot(Y["mimic3"], ref_3)}
lg(f"seeds {seeds} | tune {tune_blk['AUROC']} {tune_blk['AUROC_boot95']}")
json.dump(st, open(f"{REP}/11a_v2_seeds_tune.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 08j_v2: external ΔAUC vs base models (08j methodology, v2 preds) ----------
BASE = ["admission_age", "sex_female", "gcs_total_first", "charlson_comorbidity_score", "sofa"]; TRAD = ["sofa", "gcs_total_first"]
sofa4 = pd.read_parquet(f"{ROOT}/results/features/08z_sofa24.parquet")[["stay_id", "sofa"]]
trs = tr.merge(sofa4, on="stay_id", how="left")
def fit_lr(cols):
    trc = trs.dropna(subset=cols + ["d28"]); mu, sd = trc[cols].mean(), trc[cols].std(ddof=0).replace(0, 1.0)
    return mu, sd, LogisticRegression(max_iter=2000, random_state=42).fit(((trc[cols] - mu) / sd).values, trc.d28.values)
LRS = {"base5": fit_lr(BASE), "trad2": fit_lr(TRAD)}
dj = {"method": "08j methodology (base LR fitted on MIMIC-IV train; z-scored with train stats); full = frozen v2 (25-feature) external predictions; complete-case pairs on base variables", "results": {}}
for tag in ("eicu", "mimic3"):
    d, y = ext[tag][0].copy(), ext[tag][1]; idc = ext[tag][3]
    sof = pd.read_parquet(f"{DATA}/08i_sofa_eicu.parquet")[["icustay_id_eicu", "sofa"]] if tag == "eicu" else pd.read_parquet(f"{DATA}/08h_sofa_mimic3.parquet")[["icustay_id", "sofa"]]
    d = d.merge(sof, on=idc, how="left"); d["p_full"] = ref_e if tag == "eicu" else ref_3; d["y"] = y
    for lab, cols in (("base5", BASE), ("trad2", TRAD)):
        dc = d.dropna(subset=cols + ["p_full"]).copy(); mu, sd, lr_ = LRS[lab]
        dc["p_base"] = lr_.predict_proba(((dc[cols] - mu) / sd).values)[:, 1]
        r = delong(dc.y.values, dc.p_full.values, dc.p_base.values); r.update(n_total=int(len(d)), n_eval=int(len(dc)), events_eval=int(dc.y.sum()),
            base_missing={c: int(d[c].isna().sum()) for c in cols})
        dj["results"][f"{tag}_{lab}"] = r
        lg(f"[08j_v2 {tag} {lab}] n_eval={len(dc)} full {r['AUC_1']} vs base {r['AUC_2']} Δ={r['delta']:+.4f} p={r['p']:.5f}")
# Holm family (confirmatory ΔAUC chain, 24-h landmark): intval base5/trad2 (08f_v2) + external 4
f2 = json.load(open(f"{REP}/08f_v2_base_delta.json", encoding="utf-8"))
fam = {"intval_base5": f2["base5"]["p"], "intval_trad2": f2["trad2"]["p"]}
fam.update({k: v["p"] for k, v in dj["results"].items()})
order = sorted(fam, key=lambda k: fam[k]); mlen = len(order); adj = {}; run = 0.0
for i, k in enumerate(order):
    run = max(run, fam[k] * (mlen - i)); adj[k] = min(run, 1.0)
dj["holm"] = {k: {"p_raw": fam[k], "p_holm": adj[k], "significant_0.05": adj[k] < 0.05} for k in fam}
json.dump(dj, open(f"{REP}/11a_v2_external_delta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lg("DONE 11a")
