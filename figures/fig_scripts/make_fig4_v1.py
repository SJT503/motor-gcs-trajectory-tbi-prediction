# -*- coding: utf-8 -*-
# Fig 4 — (a) calibration deciles ×3 db (frozen v2), (b) held-out-half calibration before/after logistic recalibration (eICU, M3),
#         (c) decision curves before/after in the held-out halves. Also prints the eICU threshold where original NB crosses zero (full set).
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ROOT = Path("E:/TBI subtype"); D = ROOT / "07_prediction_system/data"; OUT = ROOT / "07_prediction_system/manuscript/figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
                     "legend.fontsize": 6.0, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C = {"m4": "#0072B2", "eicu": "#D55E00", "m3": "#009E73", "grey": "#999999", "recal": "#333333"}
sets = {"MIMIC-IV temporal": (pd.read_parquet(D / "08d_v2_preds_mimic4.parquet"), C["m4"]), "eICU-CRD": (pd.read_parquet(D / "08e_v2_preds_eicu.parquet"), C["eicu"]),
        "MIMIC-III CareVue": (pd.read_parquet(D / "08e_v2_preds_mimic3.parquet"), C["m3"])}
def deciles(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); qs = np.unique(np.percentile(p, np.linspace(0, 100, 11))); idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    return np.array([p[idx == b].mean() for b in range(len(qs) - 1) if (idx == b).any()]), np.array([y[idx == b].mean() for b in range(len(qs) - 1) if (idx == b).any()])
def slope_citl(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(lg.reshape(-1, 1), y); return float(lr.coef_[0][0]), float(np.log(y.mean()/(1-y.mean())) - np.log(p.mean()/(1-p.mean())))
def nb_curve(y, p, ts):
    y = np.asarray(y, int); p = np.asarray(p, float); n = len(y); out = []
    for t in ts:
        pos = p >= t; out.append(((pos & (y == 1)).sum() - (pos & (y == 0)).sum() * t / (1 - t)) / n)
    return np.array(out)
ts = np.round(np.arange(0.05, 0.501, 0.01), 2)
fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 12, hspace=0.55, wspace=2.2)
# (a)
for i, (name, (df, col)) in enumerate(sets.items()):
    ax = fig.add_subplot(gs[0, 4*i:4*i+4]); x, o = deciles(df.d28, df.p_full); s, c = slope_citl(df.d28, df.p_full)
    ax.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.7); ax.plot(x, o, "o-", color=col, lw=1.0, ms=3.5, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(name, pad=3); ax.set_xlabel("Predicted probability")
    if i == 0: ax.set_ylabel("Observed frequency")
    ax.text(0.04, 0.93, f"slope {s:.2f}\ncal.-in-the-large {c:+.2f}", fontsize=5.8, va="top", transform=ax.transAxes, color="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0: fig.text(0.012, 0.965, "a", fontsize=9, fontweight="bold")
# (b) + (c)
cross = None
for j, (name, col, key) in enumerate([("eICU-CRD", C["eicu"], "eICU-CRD"), ("MIMIC-III CareVue", C["m3"], "MIMIC-III CareVue")]):
    df = sets[key][0]; y = df.d28.values.astype(int); p = np.clip(df.p_full.values, 1e-6, 1 - 1e-6)
    if key == "eICU-CRD":
        nb_full = nb_curve(y, p, ts); neg = np.where(nb_full < 0)[0]; cross = float(ts[neg[0]]) if len(neg) else None
    ia, ib = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    la, lb = np.log(p[ia]/(1-p[ia])), np.log(p[ib]/(1-p[ib])); pl = LogisticRegression(C=1e6, max_iter=2000).fit(la.reshape(-1, 1), y[ia]); pr = pl.predict_proba(lb.reshape(-1, 1))[:, 1]
    axb = fig.add_subplot(gs[1, 6*j:6*j+3]); axb.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.7)
    x0, o0 = deciles(y[ib], p[ib]); x1, o1 = deciles(y[ib], pr)
    axb.plot(x0, o0, "o-", color=col, lw=1.0, ms=3.2, markeredgecolor="white", markeredgewidth=0.5, label="original"); axb.plot(x1, o1, "s-", color=C["recal"], lw=1.0, ms=3.0, markeredgecolor="white", markeredgewidth=0.5, label="recalibrated")
    axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.set_title(f"{name}: held-out half", pad=3); axb.set_xlabel("Predicted probability")
    if j == 0: axb.set_ylabel("Observed frequency")
    axb.legend(frameon=False, loc="upper left", handlelength=1.2); axb.spines[["top", "right"]].set_visible(False)
    if j == 0: fig.text(0.012, 0.47, "b", fontsize=9, fontweight="bold")
    axc = fig.add_subplot(gs[1, 6*j+3:6*j+6]); yb = y[ib]; prev = yb.mean()
    axc.plot(ts, nb_curve(yb, p[ib], ts), color=col, lw=1.0, label="original"); axc.plot(ts, nb_curve(yb, pr, ts), color=C["recal"], lw=1.0, label="recalibrated")
    axc.plot(ts, prev - (1 - prev) * ts / (1 - ts), color=C["grey"], lw=0.8, ls="--", label="treat all"); axc.axhline(0, color="#555555", lw=0.6, label="treat none")
    axc.set_xlim(0.05, 0.50); axc.set_ylim(-0.06, max(prev, 0.15) + 0.03); axc.set_xticks([0.05, 0.20, 0.35, 0.50]); axc.set_xlabel("Threshold probability"); axc.set_title(f"{name}: net benefit", pad=3)
    if j == 0: axc.set_ylabel("Net benefit")
    axc.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=5.6); axc.spines[["top", "right"]].set_visible(False)
    if j == 0: fig.text(0.255, 0.47, "c", fontsize=9, fontweight="bold")
fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.10)
print("[eICU original NB crosses zero at threshold]", cross)
fig.savefig(OUT / "Fig4_v1.png", dpi=300); fig.savefig(OUT / "Fig4_v1.pdf"); print("[saved]", OUT / "Fig4_v1.png")
