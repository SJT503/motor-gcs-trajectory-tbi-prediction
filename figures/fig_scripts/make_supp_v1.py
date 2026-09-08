# -*- coding: utf-8 -*-
# Supplementary figures S1 (hospital-level motor charting), S2 (END rolling AUROC), S4 (fairness forest) — numbered by first citation in the manuscript
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path("E:/TBI subtype"); REP = ROOT / "07_prediction_system/reports"; D = ROOT / "07_prediction_system/data"; OUT = ROOT / "07_prediction_system/manuscript/figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C = {"mimic4_intval": "#0072B2", "eicu": "#D55E00", "mimic3_carevue": "#009E73"}
# S2 fairness
f = json.load(open(REP / "11c_v2_fairness_subgroups.json", encoding="utf-8"))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
labs = [("female", "Women"), ("male", "Men"), ("age_lt65", "Age < 65"), ("age_ge65", "Age ≥ 65"), ("age_ge80", "Age ≥ 80")]
for ax, (db, title) in zip(axes, [("mimic4_intval", "MIMIC-IV temporal"), ("eicu", "eICU-CRD"), ("mimic3_carevue", "MIMIC-III CareVue")]):
    ys = np.arange(len(labs))[::-1]
    for y, (k, lab) in zip(ys, labs):
        r = f[db][k]; ax.plot(r["boot95"], [y, y], color=C[db], lw=1.6, solid_capstyle="butt"); ax.plot(r["AUROC"], y, "o", color=C[db], ms=4, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.text(1.005, y, f"{r['AUROC']:.3f} (n = {r['n']:,}; {r['events']} deaths)", fontsize=5.4, va="center", color="#333333", clip_on=False)
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in labs]); ax.set_xlim(0.65, 1.0); ax.set_title(title, pad=3); ax.set_xlabel("AUROC (95% CI)"); ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.18, wspace=1.35)
fig.savefig(OUT / "SuppFig_S4_fairness_v1.png", dpi=300); fig.savefig(OUT / "SuppFig_S4_fairness_v1.pdf"); print("[saved] S4 (fairness)")
# S3 hospital charting
fe = pd.read_parquet(D / "features_eicu.parquet"); post = pd.read_parquet(D / "08b_posterior24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
d = fe.merge(post[["icustay_id_eicu", "prob_m1"]], on="icustay_id_eicu", how="left"); d = d[~(((d.died_hosp == 1) & (d.los_h <= 24)) | (d.los_h < 24))]
h = d.groupby("hospitalid").agg(n=("prob_m1", "size"), share=("prob_m1", lambda s: s.notna().mean()))
fig, ax = plt.subplots(figsize=(3.6, 2.6)); bins = np.linspace(0, 1, 11)
ax.hist(h.share, bins=bins, color="#D55E00", edgecolor="white", lw=0.5); ax.set_xlabel("Share of TBI patients with a motor-GCS score charted in first 24 h"); ax.set_ylabel("Hospitals, n")
ax.text(0.02, 0.95, f"{int((h.share==0).sum())} hospitals: none charted\n{int((h.share==1).sum())} hospitals: all charted\n{len(h)} hospitals, {len(d):,} patients", fontsize=6.0, va="top", transform=ax.transAxes)
ax.spines[["top", "right"]].set_visible(False); fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.2)
fig.savefig(OUT / "SuppFig_S1_hospital_v1.png", dpi=300); fig.savefig(OUT / "SuppFig_S1_hospital_v1.pdf"); print("[saved] S1 (hospital)", int((h.share==0).sum()), int((h.share==1).sum()), len(h))
# S4 END rolling
r = pd.read_csv(REP / "09e_fig2_endpost.csv"); fig, ax = plt.subplots(figsize=(3.6, 2.6))
for db, col, lab in [("mimic4", C["mimic4_intval"], "MIMIC-IV (temporal)"), ("eicu", C["eicu"], "eICU-CRD"), ("mimic3", C["mimic3_carevue"], "MIMIC-III CareVue")]:
    s = r[r.db == db].sort_values("t"); ax.plot(s.t, s.auroc, color=col, lw=1.0, label=lab); ax.fill_between(s.t, s.lo, s.hi, color=col, alpha=0.15, lw=0)
ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":"); ax.axvline(24, color="#BBBBBB", lw=0.7, ls="--"); ax.set_xlim(6, 72); ax.set_ylim(0.35, 0.8); ax.set_xticks([6, 12, 24, 36, 48, 60, 72])
ax.set_xlabel("Hours since ICU admission"); ax.set_ylabel("AUROC, END within 48 h"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.0, columnspacing=0.9, fontsize=6.0)
ax.spines[["top", "right"]].set_visible(False); fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.30)
fig.savefig(OUT / "SuppFig_S2_endroll_v1.png", dpi=300); fig.savefig(OUT / "SuppFig_S2_endroll_v1.pdf"); print("[saved] S2 (endroll)")
