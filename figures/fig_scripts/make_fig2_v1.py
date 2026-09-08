# -*- coding: utf-8 -*-
# Fig 2 v1 — (a) v2 AUROC forest with bootstrap CIs, (b) rolling curve, (c) v2 block ablation incl. eICU documented/undocumented + M3,
#            (d) eICU discrimination by number of charted motor assessments in first 24 h. All numbers read from frozen JSON/parquet.
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path("E:/TBI subtype"); REP = ROOT / "07_prediction_system/reports"; OUT = ROOT / "07_prediction_system/manuscript/figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C = {"m4": "#0072B2", "eicu": "#D55E00", "m3": "#009E73", "band": "#BBBBBB", "static": "#BBBBBB", "dyn": "#56B4E9", "full": "#009E73"}
st = json.load(open(REP / "11a_v2_seeds_tune.json", encoding="utf-8")); ab = json.load(open(REP / "11a_v2_ablation_subgroup.json", encoding="utf-8"))
roll = pd.read_csv(REP / "09e_fig2_d28.csv"); dens = pd.read_parquet("E:/tbitemp_bashdiag/_eicu_density_v2.parquet")
fig = plt.figure(figsize=(7.2, 6.0)); gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.34)

# (a)
axa = fig.add_subplot(gs[0, 0]); S = ab["sets"]
rows = [("MIMIC-IV temporal\n(n = 483; 81 deaths)", S["intval"]["ABC"]["AUROC"], *st["intval_boot95"], C["m4"]),
        ("eICU-CRD\n(n = 4,440; 520 deaths)", 0.8176, *st["eicu_boot95"], C["eicu"]),
        ("MIMIC-III CareVue\n(n = 839; 155 deaths)", 0.8839, *st["mimic3_boot95"], C["m3"])]
ys = [2, 1, 0]
for y, (lab, med, lo, hi, col) in zip(ys, rows):
    axa.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="butt"); axa.plot(med, y, "o", color=col, ms=5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    axa.text(0.995, y + 0.30, f"{med:.3f} [{lo:.3f}–{hi:.3f}]", fontsize=6.0, ha="right", va="center", color="#333333")
axa.axvline(0.80, color=C["band"], lw=0.6, ls=":"); axa.set_yticks(ys); axa.set_yticklabels([r[0] for r in rows], fontsize=6.2)
axa.set_xlim(0.78, 1.0); axa.set_ylim(-0.6, 2.75); axa.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00]); axa.set_xlabel("AUROC, 28-day mortality (24-h landmark)")
axa.spines[["top", "right"]].set_visible(False); fig.text(0.012, 0.965, "a", fontsize=9, fontweight="bold")

# (b)
axb = fig.add_subplot(gs[0, 1]); dbc = {"mimic4": (C["m4"], "MIMIC-IV (temporal)"), "eicu": (C["eicu"], "eICU-CRD"), "mimic3": (C["m3"], "MIMIC-III CareVue")}
for db, (col, lab) in dbc.items():
    sub = roll[roll.db == db].sort_values("t"); axb.plot(sub.t, sub.auroc, color=col, lw=1.0, label=lab); axb.fill_between(sub.t, sub.lo, sub.hi, color=col, alpha=0.15, lw=0)
axb.axvline(24, color=C["band"], lw=0.7, ls="--"); axb.set_xlim(6, 72); axb.set_xlabel("Hours since ICU admission"); axb.set_ylabel("AUROC (28-day mortality)")
axb.set_ylim(roll["lo"].min() - 0.05, roll["hi"].max() + 0.012); axb.text(24.6, axb.get_ylim()[0] + 0.006, "confirmatory\nlandmark 24 h", fontsize=5.5, color="#888888")
axb.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", loc="lower right", handlelength=1.4, borderpad=0.4); axb.set_xticks([6, 12, 24, 36, 48, 60, 72])
axb.text(-0.22, 1.04, "b", fontsize=9, fontweight="bold", transform=axb.transAxes)

# (c)
axc = fig.add_subplot(gs[1, 0])
groups = [("Internal\n(n = 483)", "intval"), ("eICU-CRD\nmotor charted\n(n = 2,889)", "eicu_motor_documented"),
          ("eICU-CRD\nmotor not charted\n(n = 1,551)", "eicu_motor_undocumented"), ("MIMIC-III\nCareVue\n(n = 839)", "mimic3")]
x = np.arange(len(groups)); w = 0.25; cols3 = [C["static"], C["dyn"], C["full"]]; labs3 = ["Static only", "+ Motor dynamics", "+ Posteriors (full)"]
for i, v in enumerate(["A", "AB", "ABC"]):
    vals = [S[key][v]["AUROC"] for _, key in groups]
    bars = axc.bar(x + (i - 1) * w, vals, w, color=cols3[i], label=labs3[i], edgecolor="white", lw=0.4)
    for bb, val in zip(bars, vals): axc.text(bb.get_x() + bb.get_width()/2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", rotation=90, fontsize=5.0, color="#333333")
axc.set_xticks(x); axc.set_xticklabels([g[0] for g in groups], fontsize=5.9); axc.set_ylim(0.60, 1.0); axc.set_ylabel("AUROC, 28-day mortality")
axc.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=3, frameon=False, fontsize=6.0, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
fig.text(0.012, 0.50, "c", fontsize=9, fontweight="bold")

# (d) strata
axd = fig.add_subplot(gs[1, 1]); strata = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2–3"), (4, 6, "4–6"), (7, 12, "7–12"), (13, 999, "≥13")]
rng = np.random.default_rng(42); pts = []
for lo, hi, lab in strata:
    s = dens[(dens.n_motor >= lo) & (dens.n_motor <= hi)]; y = s.died_hosp_28d.values.astype(int); p = s.p_v2.values
    auc = roc_auc_score(y, p); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; bs = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]); bs.append(roc_auc_score(y[idx], p[idx]))
    pts.append((lab, len(s), int(y.sum()), auc, *np.percentile(bs, [2.5, 97.5])))
xd = np.arange(len(pts)); ax2 = axd.twinx()
ax2.bar(xd, [p[1] for p in pts], 0.6, color="#E8E8E8", zorder=1); ax2.set_ylabel("Patients, n", color="#888888"); ax2.tick_params(axis="y", colors="#888888"); ax2.set_ylim(0, 3400)
for xi, (lab, n, ev, auc, lo, hi) in zip(xd, pts):
    axd.plot([xi, xi], [lo, hi], color=C["eicu"], lw=1.4, zorder=3); axd.plot(xi, auc, "o", color=C["eicu"], ms=4.5, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    axd.text(xi, hi + 0.012, f"{auc:.2f}", ha="center", fontsize=5.4, color="#333333", zorder=5)
axd.set_zorder(ax2.get_zorder() + 1); axd.patch.set_visible(False)
axd.set_xticks(xd); axd.set_xticklabels([p[0] for p in pts]); axd.set_xlabel("Motor-GCS assessments charted in first 24 h (eICU-CRD)")
axd.set_ylabel("AUROC, full model"); axd.set_ylim(0.55, 0.97); axd.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
axd.text(-0.22, 1.04, "d", fontsize=9, fontweight="bold", transform=axd.transAxes)
fig.subplots_adjust(left=0.165, right=0.93, top=0.95, bottom=0.11)
print("[strata]", [(p[0], p[1], p[2], round(p[3], 3)) for p in pts])
fig.savefig(OUT / "Fig2_v1.png", dpi=300); fig.savefig(OUT / "Fig2_v1.pdf"); print("[saved]", OUT / "Fig2_v1.png")
