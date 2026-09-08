# -*- coding: utf-8 -*-
# Supp Fig S3 v1 — SHAP dependence for the frozen v2 (25-feature) model (the v0 figure had used the 79-feature booster); numbered by first citation in the manuscript
# two stages: `compute` (tbi_ml env: lightgbm/joblib) -> parquet; `plot` (Python314: matplotlib)
import sys, json, numpy as np, pandas as pd
stage = sys.argv[1]; ROOT = "E:/TBI subtype"; REP = f"{ROOT}/07_prediction_system/reports"; MODELS = f"{ROOT}/07_prediction_system/models"; FIGD = f"{ROOT}/07_prediction_system/manuscript/figures"
FEATS = ["motor_last", "motor_slope", "prob_m2"]
if stage == "compute":
    import joblib, lightgbm as lgb
    card = json.load(open(f"{MODELS}/08d_v2_model_card.json", encoding="utf-8")); SEL = card["features"]
    rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8")); FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
    m = pd.read_parquet(f"{ROOT}/07_prediction_system/data/08c_matrix_mimic4.parquet"); iv = m[m.split == "intval"]
    X = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp0.joblib").transform(iv[FULL]), columns=FULL, index=iv.index)[SEL]
    bst = lgb.Booster(model_file=f"{MODELS}/08d_v2_lgbm_sel_imp0.txt"); contrib = bst.predict(X, pred_contrib=True)[:, :-1]
    out = pd.DataFrame({"stay_id": iv.stay_id.values})
    for j, f in enumerate(SEL):
        if f in FEATS: out[f] = X[f].values; out[f"shap_{f}"] = contrib[:, j]
    out.to_parquet(f"{FIGD}/_supp_shapdep_v1_data.parquet", index=False)
    print("[compute v2] mean|shap|", {f: round(float(np.abs(out[f'shap_{f}']).mean()), 4) for f in FEATS}, "| anchor 08d_v2 shap_top20 motor_last 0.39914 / motor_slope 0.13509 / prob_m2 0.24337")
elif stage == "plot":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    df = pd.read_parquet(f"{FIGD}/_supp_shapdep_v1_data.parquet")
    titles = {"motor_last": "Last motor GCS (admission to 24 h)", "motor_slope": "Motor GCS slope (per 10 h)", "prob_m2": "Posterior probability, high-stable phenotype"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, f in zip(axes, FEATS):
        x, y = df[f].values, df[f"shap_{f}"].values
        sc = ax.scatter(x, y, c=y, cmap="coolwarm", s=7, vmin=-abs(y).max(), vmax=abs(y).max(), linewidths=0, alpha=0.85)
        ax.axhline(0, color="#999999", lw=0.6, ls="--"); ax.set_xlabel(titles[f]); ax.set_ylabel("SHAP value (log-odds)"); ax.spines[["top", "right"]].set_visible(False)
        o = np.argsort(x); k = 40; ax.plot(np.convolve(x[o], np.ones(k)/k, mode="valid"), np.convolve(y[o], np.ones(k)/k, mode="valid"), color="#222222", lw=1.1, zorder=3)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.17, wspace=0.34)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.015); cb.set_label("SHAP value", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(f"{FIGD}/SuppFig_S3_shapdep_v1.png", dpi=300); fig.savefig(f"{FIGD}/SuppFig_S3_shapdep_v1.pdf"); print("[saved] SuppFig_S3_shapdep_v1.png")
