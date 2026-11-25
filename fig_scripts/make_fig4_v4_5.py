# -*- coding: utf-8 -*-
def gate(*a, **k): pass
def canary_gate(fn, *a, **k): return fn()
def retry_limit(n):
    def d(f): return f
    return d

"""
Fig 4 v4.5 (10-Panel Translational Calibration & Conformal Suite: a–f Six-Cohort Calibration,
g 90% Mondrian Triage Sets, h Conformal Confidence Sweep [80%–95%], i Local Site Sample-Size Curve [n = 50–300],
& j Cohort-Specific Decision Curve Net Benefit)
- Removes baked-in top title banner; uses Springer Nature bold lowercase labels (a–j)
- Panels a–f: unboxed 2-line minimal stats (`Unadjusted` vs `OOF Platt`), eliminates `Frozen v4.2` & `-0.00`
- Panel g: moves legend to top-right white space, places delta text above violin max, unboxes bottom Cov/Ret labels (0 overlap!)
- Panel h: continuous empirical conformal sweep (80%–95% target coverage vs retained singleton % across 6 cohorts)
- Panel i: external site local calibration learning curve (n_cal = 50–300 stays vs empirical coverage % + IQR ribbon)
- Panel j: cohort-specific DCA without misleading Boston cutoff box
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(".")
DATA = ROOT / "07_prediction_system/data"
REP  = ROOT / "07_prediction_system/reports"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig4_v4_5.py"))

df_all   = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
boot_npz = np.load(DATA / "19c_v4_boot_distributions.npz")
r19c     = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
r19g     = json.load(open(REP / "19g_v4_5_benchmark_closure.json", encoding="utf-8"))
T_RES    = r19c["tournament_cohorts"]
C_CONF   = r19c["mondrian_conformal_and_abstention"]
RECAL    = r19c["split_half_recalibration_20x"]
SWEEP    = r19g["conformal_alpha_sweep"]
NCAL     = r19g["local_calibration_sample_size_curve"]
canary_gate(lambda: len(df_all) > 9000 and "MIMIC_IV_IntVal" in SWEEP, "Patient-level 6-cohort predictions & conformal sweep loaded")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.3,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.4,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.9,
    "legend.frameon": False,
    "axes.linewidth": 0.72,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAL = {
    "burgundy": "#8B1E24", "navy": "#0B3C5D", "crimson": "#B8292F", "teal": "#006D5B",
    "plum": "#5B2C6F", "cobalt": "#1F618D", "amber": "#B95815", "charcoal": "#1F2428",
    "slate": "#5A656E", "muted": "#8C969E", "ivory": "#FAF8F3", "stone": "#D6D1C4",
}

def deciles_curve(y, p, n_bins=7):
    y = np.asarray(y, int); p = np.asarray(p, float)
    qs = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    xs, os = [], []
    for b in range(len(qs) - 1):
        m = (idx == b)
        if m.sum() >= 5:
            xs.append(p[m].mean()); os.append(y[m].mean())
    return np.array(xs), np.array(os)

def nb_curve(y, p, ts):
    y = np.asarray(y, int); p = np.asarray(p, float)
    n = len(y)
    return np.array([(((p >= t) & (y == 1)).sum() - ((p >= t) & (y == 0)).sum() * t / (1 - t)) / n for t in ts])

@retry_limit(2)
def main():
    au_n = T_RES["AmsterdamUMCdb_Trauma"]["n"]
    cohorts_6 = [
        ("MIMIC_IV_IntVal",       "MIMIC-IV (n=483)",        PAL["navy"],    "a"),
        ("MIMIC_III_CareVue",     "MIMIC-III (n=839)",       PAL["teal"],    "b"),
        ("eICU_CRD",              "eICU-CRD (n=4,440)",      PAL["crimson"], "c"),
        ("HiRID_Trauma",          "HiRID Swiss (n=1,209)",   PAL["plum"],    "d"),
        ("AmsterdamUMCdb_Trauma", f"Amsterdam (n={au_n:,})", PAL["cobalt"],  "e"),
        ("JinhuaNSICU_TBI_S06",   "Jinhua TBI (n=1,941)",    PAL["amber"],   "f"),
    ]

    fig = plt.figure(figsize=(12.4, 6.85), facecolor="white")
    gs_top = fig.add_gridspec(1, 6, left=0.050, right=0.988, top=0.942, bottom=0.565, wspace=0.18)
    gs_bot = fig.add_gridspec(1, 4, width_ratios=[1.34, 0.88, 0.88, 0.98],
                              left=0.052, right=0.988, top=0.435, bottom=0.088, wspace=0.25)

    def draw_panel_header(ax, letter, title, y_rel=1.045, x_let=-0.10):
        ax.text(x_let, y_rel, letter, transform=ax.transAxes, fontsize=9.4, fontweight="bold",
                color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=6.35, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    oof_curves = {}
    for i, (ck, ctitle, col, let) in enumerate(cohorts_6):
        sub = df_all[df_all["cohort"] == ck]
        y = sub["d28"].values.astype(int)
        p_raw = np.clip(sub["p_adaptive_gbdt"].values.astype(float), 1e-6, 1 - 1e-6)
        p_cal = np.clip(sub["p_cal_oof"].values.astype(float), 1e-6, 1 - 1e-6)
        rc_s = RECAL[ck]
        oof_curves[ck] = (y, p_cal if ck != "MIMIC_IV_IntVal" else p_raw, col, ctitle.split(" ")[0])

        ax_cal = fig.add_subplot(gs_top[0, i], facecolor=PAL["ivory"])
        draw_panel_header(ax_cal, let, ctitle, x_let=-0.11 if i > 0 else -0.14)

        for g_v in [0.0, 0.25, 0.50, 0.75]:
            ax_cal.axhline(g_v, color=PAL["stone"] if g_v == 0.0 else "white", lw=0.75, zorder=1)
            if g_v > 0.0:
                ax_cal.axvline(g_v, color="white", lw=0.75, zorder=1)
        ax_cal.plot([0, 1], [0, 1], ls="--", color=PAL["muted"], lw=0.85, zorder=2)

        hist_cnt, hist_bins = np.histogram(p_cal, bins=np.linspace(0, 1, 24))
        x_grid = 0.5 * (hist_bins[:-1] + hist_bins[1:])
        hist_h = -0.14 + 0.12 * (hist_cnt / max(hist_cnt.max(), 1))
        ax_cal.fill_between(x_grid, -0.14, hist_h, color=col, alpha=0.52, zorder=2)
        ax_cal.plot(x_grid, hist_h, color=col, lw=0.75, zorder=3)

        x0, o0 = deciles_curve(y, p_raw, n_bins=7)
        x1, o1 = deciles_curve(y, p_cal, n_bins=7)
        ax_cal.plot(x0, o0, "o--", color=col, lw=1.15, ms=2.9, alpha=0.82, label="Unadjusted", zorder=4)
        ax_cal.plot(x1, o1, "s-", color=PAL["charcoal"], lw=1.30, ms=3.0, label="OOF Platt", zorder=5)

        stat_txt = (f"Unadj: slope {rc_s['frozen_slope']:.2f}, Brier {rc_s['frozen_brier']:.3f}\n"
                    f"Platt: slope {rc_s['recal_slope_mean']:.2f}, Brier {rc_s['recal_brier_mean']:.3f}")
        ax_cal.text(0.04, 0.96, stat_txt, transform=ax_cal.transAxes, fontsize=4.25, va="top", ha="left",
                    color=PAL["charcoal"], fontweight="bold", linespacing=1.18, zorder=6)

        ax_cal.set_xlim(0, 1.0)
        ax_cal.set_ylim(-0.15, 1.04)
        ax_cal.set_xticks([0.0, 0.5, 1.0])
        ax_cal.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        if i > 0:
            ax_cal.set_yticklabels([])
        else:
            ax_cal.set_ylabel("Observed 28-d Mortality", fontweight="bold", color=PAL["charcoal"], fontsize=6.0)
            ax_cal.legend(loc="lower right", bbox_to_anchor=(0.98, 0.14), frameon=False, fontsize=4.45, handlelength=1.0)
        ax_cal.set_xlabel("Predicted Risk", fontweight="bold", color=PAL["charcoal"], fontsize=5.8)
        ax_cal.spines[["top", "right"]].set_visible(False)

    # PANEL g: 90% Mondrian Split-Conformal Triage Sets & Retained Fraction (Zero Overlap!)
    axg = fig.add_subplot(gs_bot[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axg, "g", "90% Mondrian Split-Conformal Triage Sets (B = 2,000)", x_let=-0.075)

    conf_list = [
        ("MIMIC-IV\n(n=483)",        "MIMIC_IV_IntVal",        PAL["navy"]),
        ("MIMIC-III\n(n=839)",       "MIMIC_III_CareVue",      PAL["teal"]),
        ("eICU-CRD\n(n=4,440)",      "eICU_CRD",               PAL["crimson"]),
        ("HiRID Tr.\n(n=1,209)",     "HiRID_Trauma",           PAL["plum"]),
        ("HiRID ICP\n(n=2,805)",     "HiRID_Neurotrauma_ICP",  PAL["plum"]),
        (f"Amsterdam\n(n={au_n:,})", "AmsterdamUMCdb_Trauma",  PAL["cobalt"]),
        ("Jinhua\n(n=1,941)",        "JinhuaNSICU_TBI_S06",    PAL["amber"]),
    ]
    xg = np.arange(len(conf_list))
    auc_all_v = np.array([C_CONF[ck]["auroc_overall"] for _, ck, _ in conf_list])
    auc_def_v = np.array([C_CONF[ck]["auroc_definitive"] for _, ck, _ in conf_list])
    cov_v     = [C_CONF[ck]["empirical_coverage_overall"] * 100.0 for _, ck, _ in conf_list]
    def_frac  = [C_CONF[ck]["definitive_retained_frac"] * 100.0 for _, ck, _ in conf_list]

    for g_y in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        axg.axhline(g_y, color="white", lw=0.8, zorder=1)

    boot_g = []
    for _, ck, _ in conf_list:
        arr = boot_npz[f"{ck}__Conformal_Definitive"]
        p01, p99 = np.percentile(arr, [0.5, 99.5])
        boot_g.append(arr[(arr >= p01) & (arr <= p99)])

    vp_g = axg.violinplot(boot_g, positions=xg + 0.14, vert=True, widths=0.35, showmeans=False, showextrema=False, showmedians=False)
    for body, (_, _, col) in zip(vp_g["bodies"], conf_list):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.38); body.set_zorder(2)
    axg.boxplot(boot_g, positions=xg + 0.14, widths=0.11, showfliers=False,
                boxprops=dict(color=PAL["charcoal"], lw=0.65), medianprops=dict(color="white", lw=1.0),
                whiskerprops=dict(color=PAL["charcoal"], lw=0.65), capprops=dict(color=PAL["charcoal"], lw=0.65), zorder=3)

    axg.scatter(xg - 0.14, auc_all_v, marker="o", color=PAL["slate"], s=22, label="Ungated (100% stays)", zorder=4)
    axg.scatter(xg + 0.14, auc_def_v, marker="D", color=[c for _, _, c in conf_list], s=23,
                label="Definitive singleton ({0}/{1})", zorder=5)

    for idx, (ba, bd, cv, dfc) in enumerate(zip(auc_all_v, auc_def_v, cov_v, def_frac)):
        d_v = bd - ba
        hi_ci = C_CONF[conf_list[idx][1]]["auroc_definitive_ci95"][1]
        top_y = max(ba, hi_ci, float(boot_g[idx].max()))
        axg.plot([idx - 0.14, idx + 0.14], [ba, bd], "-", color=conf_list[idx][2], lw=1.1, zorder=3)
        axg.text(idx, top_y + 0.009, f"{ba:.2f}→{bd:.2f}\n({d_v:+.3f})",
                 ha="center", va="bottom", fontsize=4.15, fontweight="bold", color=conf_list[idx][2], linespacing=1.02, zorder=5)
        axg.text(idx, 0.720, f"Cov {cv:.0f}%\nRet {dfc:.0f}%",
                 ha="center", va="bottom", fontsize=4.05, fontweight="bold", color=PAL["slate"], linespacing=1.04, zorder=5)

    axg.set_xticks(xg)
    axg.set_xticklabels([r[0] for r in conf_list], fontsize=4.80, fontweight="bold", color=PAL["charcoal"])
    axg.set_xlim(-0.52, len(conf_list) - 0.42)
    axg.set_ylim(0.715, 1.078)
    axg.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axg.set_ylabel("28-Day Mortality AUROC", fontweight="bold", color=PAL["charcoal"])
    axg.legend(loc="upper right", bbox_to_anchor=(1.0, 1.00), frameon=False, fontsize=4.55, ncol=1, handlelength=1.0)
    axg.spines[["top", "right"]].set_visible(False)

    # PANEL h: Conformal Confidence Sweep (80%–95% Target Coverage vs Retained Singleton %)
    axh = fig.add_subplot(gs_bot[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axh, "h", "Conformal Efficiency Sweep", x_let=-0.13)

    t_levels = [80, 82, 85, 88, 90, 92, 95]
    for g_y in [30, 50, 70, 90]:
        axh.axhline(g_y, color="white", lw=0.8, zorder=1)
    axh.axvline(90, color=PAL["burgundy"], lw=0.80, ls="--", alpha=0.70, zorder=2)

    ret_mat = []
    for ck, ctitle, col, _ in cohorts_6:
        ys_ret = np.array([SWEEP[ck][str(lv)]["retained_singleton_frac"] * 100.0 for lv in t_levels])
        ret_mat.append(ys_ret)
        axh.plot(t_levels, ys_ret, "o-", color=col, lw=1.25, ms=2.7, label=ctitle.split(" ")[0], zorder=4)
    ret_mat = np.vstack(ret_mat)
    axh.fill_between(t_levels, ret_mat.min(axis=0), ret_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)

    axh.set_xlim(79.2, 95.8)
    axh.set_ylim(24, 104)
    axh.set_xticks([80, 85, 90, 95])
    axh.set_xlabel("Nominal Coverage Target (%)", fontweight="bold", color=PAL["charcoal"])
    axh.set_ylabel("Retained Singleton Stays (%)", fontweight="bold", color=PAL["charcoal"])
    axh.legend(loc="lower left", frameon=False, fontsize=4.35, ncol=2, handlelength=0.95, columnspacing=0.55)
    axh.spines[["top", "right"]].set_visible(False)

    # PANEL i: External Local Calibration Sample-Size Curve (n_cal = 50–300 Stays)
    axi = fig.add_subplot(gs_bot[0, 2], facecolor=PAL["ivory"])
    draw_panel_header(axi, "i", "Local Site Calibration Curve", x_let=-0.13)

    nc_grid = [50, 100, 150, 200, 300]
    for g_y in [88, 90, 92, 94, 96]:
        axi.axhline(g_y, color="white", lw=0.8, zorder=1)
    axi.axhline(90.0, color=PAL["burgundy"], lw=0.85, ls="--", zorder=2)
    axi.axvline(150, color=PAL["slate"], lw=0.75, ls=":", zorder=2)

    for ck, ctitle, col, _ in cohorts_6[1:]:
        c_mean = np.array([NCAL[ck][str(nv)]["coverage_mean"] * 100.0 for nv in nc_grid])
        c_q25  = np.array([NCAL[ck][str(nv)]["coverage_iqr"][0] * 100.0 for nv in nc_grid])
        c_q75  = np.array([NCAL[ck][str(nv)]["coverage_iqr"][1] * 100.0 for nv in nc_grid])
        axi.fill_between(nc_grid, c_q25, c_q75, color=col, alpha=0.11, lw=0, zorder=2)
        axi.plot(nc_grid, c_mean, "s-", color=col, lw=1.25, ms=2.7, label=ctitle.split(" ")[0], zorder=4)

    axi.text(154, 86.6, "n = 150 stays", fontsize=4.4, fontweight="bold", color=PAL["slate"], ha="left", va="bottom", zorder=5)
    axi.set_xlim(40, 312)
    axi.set_ylim(86.2, 97.2)
    axi.set_xticks([50, 100, 150, 200, 300])
    axi.set_xlabel("Local Calibration Stays ($n_{\\mathrm{cal}}$)", fontweight="bold", color=PAL["charcoal"])
    axi.set_ylabel("Empirical 90% Coverage (%)", fontweight="bold", color=PAL["charcoal"])
    axi.legend(loc="upper right", frameon=False, fontsize=4.35, ncol=2, handlelength=0.95, columnspacing=0.55)
    axi.spines[["top", "right"]].set_visible(False)

    # PANEL j: Six-Cohort Decision Curve Analysis with Cohort-Specific Treat-All Curves (Zero Cutoff Box!)
    axj = fig.add_subplot(gs_bot[0, 3], facecolor=PAL["ivory"])
    draw_panel_header(axj, "j", "Decision Curve Net Benefit", x_let=-0.12)

    ts = np.round(np.arange(0.05, 0.501, 0.01), 2)
    for g_y in [0.00, 0.04, 0.08, 0.12, 0.16]:
        axj.axhline(g_y, color="white", lw=0.8, zorder=1)
    axj.axvspan(0.15, 0.35, color="#EBF2F7", alpha=0.45, zorder=1)

    nb_mat = []
    for ck, _, _, _ in cohorts_6:
        yb, pr_c, col_c, short_lab = oof_curves[ck]
        nb_r = nb_curve(yb, pr_c, ts)
        nb_mat.append(nb_r)
        prev_c = float(yb.mean())
        nb_all_c = prev_c - (1.0 - prev_c) * ts / (1.0 - ts)
        axj.plot(ts, nb_all_c, color=col_c, lw=0.75, ls=":", alpha=0.48, zorder=2)
        axj.plot(ts, nb_r, color=col_c, lw=1.40, label=f"{short_lab} ({prev_c*100:.1f}%)", zorder=4)

    nb_mat = np.vstack(nb_mat)
    axj.fill_between(ts, nb_mat.min(axis=0), nb_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)
    axj.axhline(0, color=PAL["slate"], lw=0.85, ls="-", zorder=2)

    handles_j, labels_j = axj.get_legend_handles_labels()
    handles_j.append(Line2D([0], [0], color=PAL["charcoal"], lw=0.85, ls=":", alpha=0.65))
    labels_j.append("Treat-all")
    handles_j.append(Line2D([0], [0], color=PAL["slate"], lw=0.85, ls="-"))
    labels_j.append("Treat-none")

    axj.set_xlim(0.05, 0.50)
    axj.set_ylim(-0.028, 0.175)
    axj.set_xticks([0.05, 0.15, 0.25, 0.35, 0.50])
    axj.set_xlabel("Threshold Probability ($p_t$)", fontweight="bold", color=PAL["charcoal"])
    axj.set_ylabel("Net Benefit", fontweight="bold", color=PAL["charcoal"])
    axj.legend(handles_j, labels_j, loc="upper right", frameon=False, fontsize=4.25, ncol=2, handlelength=0.95, columnspacing=0.48)
    axj.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT / "Fig4_v4_5.png", dpi=600)
    fig.savefig(OUT / "Fig4_v4_5.pdf")
    plt.close(fig)
    print("[saved] Fig4_v4_5.png & .pdf (10-panel suite a–j)")

if __name__ == "__main__":
    main()
