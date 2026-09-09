# -*- coding: utf-8 -*-
# Fig 1 v1 — (a) corrected cohort flow (CareVue n = 839; exclusions; elbow connectors), (b) phenotype curves with names, (c) stability forest
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path("E:/TBI subtype"); OUT = ROOT / "07_prediction_system/manuscript/figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C = {"c1": "#D55E00", "c2": "#0072B2", "c3": "#009E73", "grey": "#666666", "crit": "#BBBBBB"}
fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.15, wspace=0.50)

# ---------- (a) flow ----------
axa = fig.add_subplot(gs[0, :]); axa.axis("off"); axa.set_xlim(0, 10.5); axa.set_ylim(0.55, 6.35)
B = {}
def box(key, x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=6.3, lw=0.7):
    axa.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, fc=fc, ec=ec, lw=lw, zorder=2))
    axa.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.3); B[key] = (x, y, w, h)
def down(k1, k2, label=None, lx=0.08):
    x1, y1, w1, h1 = B[k1]; x2, y2, w2, h2 = B[k2]
    axa.annotate("", xy=(x2, y2 + h2/2), xytext=(x1, y1 - h1/2), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=0, shrinkB=0), zorder=1)
    if label: axa.text(x1 + lx, (y1 - h1/2 + y2 + h2/2)/2, label, fontsize=5.6, color=C["grey"], va="center", ha="left")
def fan(k1, keys):
    x1, y1, w1, h1 = B[k1]; ybar = y1 - h1/2 - 0.28
    axa.plot([x1, x1], [y1 - h1/2, ybar], color="#333333", lw=0.7, zorder=1)
    xs = [B[k][0] for k in keys]; axa.plot([min(xs), max(xs)], [ybar, ybar], color="#333333", lw=0.7, zorder=1)
    for k in keys:
        x2, y2, w2, h2 = B[k]
        axa.annotate("", xy=(x2, y2 + h2/2), xytext=(x2, ybar), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=0, shrinkB=0), zorder=1)
# development column
box("m4", 2.55, 5.7, 3.6, 0.72, "MIMIC-IV v3.1 — TBI ICU admissions\nn = 2,751", fc="#F2F2F2")
box("risk", 2.55, 4.35, 3.6, 0.72, "24-h landmark risk set\nn = 2,700 (402 deaths)")
down("m4", "risk", "51 died or left the ICU\nbefore 24 h", lx=0.12)
box("train", 0.88, 2.95, 1.62, 0.82, "Training\n2008–2017\nn = 1,661 (248)", fs=6.0)
box("tune", 2.55, 2.95, 1.50, 0.82, "Model selection\n2018–2019\nn = 556 (73)", fs=6.0)
box("intval", 4.22, 2.95, 1.62, 0.82, "Internal validation\n2020–2022\nn = 483 (81)", fs=6.0)
fan("risk", ["train", "tune", "intval"])
box("gbtm", 2.75, 1.2, 5.4, 0.95, "Latent-class trajectory model (motor GCS, 3 classes; unsupervised; n = 2,749)\ntruncated posteriors + motor dynamics from admission-to-checkpoint data only",
    fc="#FFFFFF", ec=C["c2"], fs=5.8)
axa.annotate("", xy=(2.55, 1.2 + 0.95/2), xytext=(2.55, 2.95 - 0.82/2), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C["c2"], shrinkA=0, shrinkB=0), zorder=1)
# external columns
box("e0", 6.65, 5.7, 2.35, 0.72, "eICU-CRD v2.0\n6,677 TBI ICU stays", fc="#F2F2F2")
box("e1", 6.65, 4.35, 2.35, 0.72, "24-h risk set\nn = 4,440 (520 deaths)")
down("e0", "e1", "adults, first ICU\nstay, LOS ≥ 24 h,\nalive at 24 h", lx=0.08)
box("e2", 6.65, 2.95, 2.35, 0.82, "130 hospitals\nmotor GCS charted in\n65% of patients", fs=6.0)
down("e1", "e2")
box("m0", 9.05, 5.7, 1.85, 0.72, "MIMIC-III v1.4\n1,950 TBI ICU stays", fc="#F2F2F2")
box("m1", 9.05, 4.35, 1.85, 0.72, "CareVue era 2001–2008\nn = 841 of 1,425")
down("m0", "m1", "MetaVision era\nexcluded (overlap\nwith MIMIC-IV)", lx=0.08)
box("m2", 9.05, 2.95, 1.85, 0.82, "24-h risk set\nn = 839 (155 deaths)", fs=6.0)
down("m1", "m2", "2 died < 24 h", lx=0.1)
box("zt", 7.85, 1.2, 4.25, 0.95, "Zero-touch external validation\nfrozen pipeline: no refitting, no threshold or feature changes", fc="#FFFFFF", ec=C["c3"], fs=6.0)
for k in ("e2", "m2"):
    x, y, w, h = B[k]; axa.annotate("", xy=(x, 1.2 + 0.95/2), xytext=(x, y - h/2), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C["c3"], shrinkA=0, shrinkB=0), zorder=1)
axa.text(0.02, 6.22, "a", fontsize=9, fontweight="bold")

# ---------- (b) phenotype curves ----------
axb = fig.add_subplot(gs[1, 0])
b = pd.read_parquet(ROOT / "results/features/13_motor_binned6h.parquet"); cls = pd.read_csv(ROOT / "results/cluster/13_motor/13b_motor_classes.csv")
d = b.merge(cls, on="stay_id")
names = {1: "Moderate-improving", 2: "High-stable", 3: "Low-declining"}
for k, col in zip([1, 2, 3], [C["c1"], C["c2"], C["c3"]]):
    sub = d[d["class"] == k]; g = sub.groupby("t")["motor"].agg(["mean", "sem"]); n_k = sub["stay_id"].nunique()
    axb.plot(g.index, g["mean"], color=col, lw=1.0, label=f"{names[k]} (n = {n_k:,})")
    axb.fill_between(g.index, g["mean"] - 1.96*g["sem"], g["mean"] + 1.96*g["sem"], color=col, alpha=0.15, lw=0)
axb.set_xlabel("Hours since ICU admission"); axb.set_ylabel("Motor GCS (mean, 95% CI)"); axb.set_xlim(0, 72); axb.set_ylim(0, 6.3)
axb.set_xticks([0, 12, 24, 36, 48, 60, 72]); axb.set_yticks([0, 1, 2, 3, 4, 5, 6])
axb.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", loc="upper center", bbox_to_anchor=(0.5, -0.21), handlelength=1.4, borderpad=0.4)
axb.text(-0.20, 1.04, "b", fontsize=9, fontweight="bold", transform=axb.transAxes)

# ---------- (c) stability forest ----------
axc = fig.add_subplot(gs[1, 1])
h12 = json.load(open(ROOT / "results/cluster/13_motor/12h_motor_gbtm_bootstrap.json", encoding="utf-8"))
g12 = json.load(open(ROOT / "05_END_traj/results/12_gcs_course/12g_gbtm_bootstrap.json", encoding="utf-8"))
rows = [("Motor, 3 classes", h12["summary"]["3"], C["c2"], True), ("  2 classes", h12["summary"]["2"], C["c2"], False),
        ("  4 classes", h12["summary"]["4"], C["c2"], False), ("Total GCS, 5 classes", g12["summary"]["5"], C["c3"], True),
        ("  4 classes", g12["summary"]["4"], C["c3"], False), ("  6 classes", g12["summary"]["6"], C["c3"], False)]
ys = np.arange(len(rows))[::-1]
for y, (lab, s, col, main) in zip(ys, rows):
    med, q25, q75 = s["median_ari"], s["q25_ari"], s["q75_ari"]
    axc.plot([q25, q75], [y, y], color=col, lw=1.6 if main else 1.0, alpha=1.0 if main else 0.5, solid_capstyle="butt")
    axc.plot(med, y, "o", color=col, ms=4.5 if main else 3.2, markeredgecolor="white", markeredgewidth=0.5, alpha=1.0 if main else 0.6, zorder=3)
    axc.text(1.03, y, f"{med:.3f} [{q25:.3f}–{q75:.3f}]", fontsize=5.8, va="center", color="#333333", clip_on=False)
axc.axvline(0.75, color=C["crit"], lw=0.7, ls="--", zorder=1)
axc.text(0.75, -0.9, "stability threshold 0.75", fontsize=5.5, color="#888888", ha="center", va="top")
axc.set_yticks(ys); axc.set_yticklabels([r[0] for r in rows], fontsize=6.0)
axc.set_xlim(0.3, 1.02); axc.set_ylim(-1.4, len(rows) - 0.4); axc.set_xticks([0.4, 0.6, 0.8, 1.0])
axc.set_xlabel("Bootstrap adjusted Rand index\n(median, IQR; 100 main / 50 sensitivity replicates)")
axc.spines[["top", "right"]].set_visible(False)
axc.text(-0.50, 1.04, "c", fontsize=9, fontweight="bold", transform=axc.transAxes)
fig.subplots_adjust(left=0.07, right=0.86, top=0.985, bottom=0.17)
issues = []
# legend placement check: must sit fully below panel b's data area and inside the figure
fig.canvas.draw()
lb = axb.get_legend().get_window_extent(fig.canvas.get_renderer()).expanded(1.0, 1.0)
if lb.y1 > axb.bbox.y0 + 1: issues.append(f"LEGEND intrudes into panel b data area (legend top {lb.y1:.1f} >= axes bottom {axb.bbox.y0:.1f})")
if lb.y0 < 0: issues.append(f"LEGEND clipped at figure bottom (y0 {lb.y0:.1f})")
print(f"[legend-check] legend bbox y {lb.y0:.1f}-{lb.y1:.1f} px; panel b axes bottom {axb.bbox.y0:.1f} px; figure bottom 0")
for txt in axc.get_xticklabels() + [axc.xaxis.label]:
    tb = txt.get_window_extent(fig.canvas.get_renderer())
    if lb.overlaps(tb): issues.append(f"LEGEND overlaps panel c x-label text")
if lb.x1 > axc.get_window_extent().x0 - 8: issues.append(f"LEGEND too close to panel c (x1 {lb.x1:.1f})")
for k1 in B:
    for k2 in B:
        if k1 < k2:
            x1, y1, w1, h1 = B[k1]; x2, y2, w2, h2 = B[k2]
            if abs(x1 - x2) - (w1 + w2)/2 < 0.05 and abs(y1 - y2) - (h1 + h2)/2 < 0.05: issues.append(f"FLOW OVERLAP {k1}-{k2}")
print("[collision-check]", "PASS" if not issues else issues)
fig.savefig(OUT / "Fig1_v1.png", dpi=300); fig.savefig(OUT / "Fig1_v1.pdf"); print("[saved]", OUT / "Fig1_v1.png")
