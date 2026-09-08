# -*- coding: utf-8 -*-
# Fig 3 v1 — (a) END warning lead time (layout fixed: no truncation, db names as ticks), (b) FAC, (c) SHAP top-15 with readable names
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
ROOT = Path("E:/TBI subtype"); REP = ROOT / "07_prediction_system/reports"; OUT = ROOT / "07_prediction_system/manuscript/figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C = {"m4": "#0072B2", "eicu": "#D55E00", "m3": "#009E73", "traj": "#0072B2", "band": "#BBBBBB"}
e = json.load(open(REP / "09e_rolling_report.json", encoding="utf-8")); d8 = json.load(open(REP / "08d_v2_lgbm_main.json", encoding="utf-8")); W = e["warning"]
dbs = [("mimic4", "MIMIC-IV", C["m4"]), ("eicu", "eICU-CRD", C["eicu"]), ("mimic3", "MIMIC-III CareVue", C["m3"])]
fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.50, wspace=0.42)
axa = fig.add_subplot(gs[0, :]); centers = []
for i, (db, lab, col) in enumerate(dbs):
    base = i * 2.6
    for j, (rule, alpha) in enumerate([("end_single", 1.0), ("end_two_consec", 0.45)]):
        x = base + j * 0.85; wt = W[db][rule]["warning_time_hr"]; det = W[db][rule]["detection_rate"]
        axa.plot([x, x], [wt["q1"], wt["q3"]], color=col, lw=5.5, alpha=alpha, solid_capstyle="butt", zorder=2)
        axa.plot(x, wt["median"], "o", color=col, ms=4.5 if j == 0 else 3.5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        axa.text(x, wt["q3"] + 1.8, f"{det*100:.0f}%", fontsize=5.6, ha="center", color="#333333")
    centers.append(base + 0.425)
    axa.text(base + 0.425, -4.5, f"28-d mortality lead {W[db]['d28_single']['lead_time_hr']['median']:.0f} h", fontsize=5.6, ha="center", color="#666666")
axa.axhline(12, color=C["band"], lw=0.7, ls="--"); axa.text(6.55, 12.8, "12 h", fontsize=5.5, color="#888888", ha="right")
axa.set_xlim(-0.6, 6.6); axa.set_ylim(-7.5, 52); axa.set_xticks(centers); axa.set_xticklabels([lab for _, lab, _ in dbs]); axa.tick_params(axis="x", length=0)
axa.set_yticks([0, 10, 20, 30, 40, 50]); axa.set_ylabel("Warning lead time before END (h)")
axa.plot([], [], color="#333333", lw=5.5, label="single crossing (dark) / two consecutive (light): IQR; dot = median; % = events flagged")
axa.legend(frameon=False, loc="upper right", handlelength=1.4, borderpad=0.2, fontsize=5.8); fig.text(0.012, 0.975, "a", fontsize=9, fontweight="bold")
axb = fig.add_subplot(gs[1, 0]); x = np.arange(len(dbs)); w = 0.34
single = [W[db]["fac_end_single"]["false_alarms_per_100_patient_days"] for db, _, _ in dbs]; two = [W[db]["fac_end_two_consec"]["false_alarms_per_100_patient_days"] for db, _, _ in dbs]
b1 = axb.bar(x - w/2, single, w, color=[c for _, _, c in dbs], label="Single crossing"); b2 = axb.bar(x + w/2, two, w, color=[c for _, _, c in dbs], alpha=0.45, label="Two consecutive")
for bars in (b1, b2):
    for b in bars: axb.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", fontsize=5.6, color="#333333")
axb.set_xticks(x); axb.set_xticklabels(["MIMIC-IV", "eICU-CRD", "MIMIC-III\nCareVue"], fontsize=6.3); axb.set_ylim(0, 64); axb.set_ylabel("False alarms / 100 event-free patient-days")
axb.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", loc="upper left", handlelength=1.1, borderpad=0.4); fig.text(0.012, 0.50, "b", fontsize=9, fontweight="bold")
axc = fig.add_subplot(gs[1, 1]); top = list(d8["shap_top20"].items())[:15]; names = [k for k, _ in top][::-1]; vals = [v for _, v in top][::-1]
nice = {"admission_age": "Age", "gcs_eye_first": "GCS eye (first)", "gcs_total_first": "GCS total (first)", "motor_last": "Motor GCS (last)", "pt_min": "PT (min)", "wbc_min": "WBC (min)",
        "prob_m2": "Posterior, high-stable", "prob_m3": "Posterior, low-declining", "prob_m1": "Posterior, moderate-improving", "hemoglobin_min": "Haemoglobin (min)",
        "hemoglobin_max": "Haemoglobin (max)", "rr_mean": "Respiratory rate (mean)", "bun_min": "Urea (min)", "motor_slope": "Motor GCS slope", "motor_max": "Motor GCS (max)",
        "platelet_max": "Platelets (max)", "platelet_min": "Platelets (min)", "temp_min": "Temperature (min)", "creatinine_min": "Creatinine (min)", "pt_max": "PT (max)", "motor_sd": "Motor GCS SD"}
cols = [C["traj"] if (k.startswith("motor") or k.startswith("prob")) else "#AAAAAA" for k in names]
axc.barh(np.arange(len(names)), vals, color=cols, height=0.62, edgecolor="white", lw=0.3); axc.set_yticks(np.arange(len(names))); axc.set_yticklabels([nice.get(k, k) for k in names], fontsize=5.8)
for yv, v in enumerate(vals): axc.text(v + max(vals)*0.015, yv, f"{v:.2f}", fontsize=5.2, va="center", color="#333333")
axc.set_xlim(0, max(vals) * 1.17); axc.set_xlabel("Mean |SHAP value| (log-odds)")
axc.legend(handles=[Patch(fc=C["traj"], label="Trajectory-derived"), Patch(fc="#AAAAAA", label="Static")], frameon=False, loc="lower right", fontsize=6.0, handlelength=1.0)
axc.spines[["top", "right"]].set_visible(False); axc.text(-0.62, 1.05, "c", fontsize=9, fontweight="bold", transform=axc.transAxes)
fig.subplots_adjust(left=0.10, right=0.975, top=0.955, bottom=0.08)
fig.savefig(OUT / "Fig3_v1.png", dpi=300); fig.savefig(OUT / "Fig3_v1.pdf"); print("[saved]", OUT / "Fig3_v1.png")
