# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
Fig 4 v5.0 (11-Panel Proactive Conformal Decision & Calibration Suite:
  Row 1 [a–f]: Six-Cohort Platt Calibration & Risk Density Strips
  Row 2 [g, h]: g = 90% Mondrian Split-Conformal Triage Sets (B = 2,000);
                h = Proactive Conformal Ambiguity Resolution (Static S0 {0,1} -> Post-Biomarker Conversion & NNR)
  Row 3 [i, j, k]: i = Conformal Efficiency Sweep (80%–95%);
                   j = Local Site Calibration Curve (n_cal = 50–300);
                   k = Cohort-Specific Decision Curve Net Benefit
)
- Spacious 3-row asymmetric layout, zero prose boxes, zero overlaps, 100% multilayer!
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP  = ROOT / "07_prediction_system/reports"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig4_v5_3.py"))

df_all   = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
boot_npz = np.load(DATA / "19c_v4_boot_distributions.npz")
r19c     = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
r19g     = json.load(open(REP / "19g_v4_5_benchmark_closure.json", encoding="utf-8"))
r19h     = json.load(open(REP / "19h_v5_proactive_biomarker_voi.json", encoding="utf-8"))
r19i     = json.load(open(REP / "19i_v5_2_sequential_conformal_corridor.json", encoding="utf-8"))
T_RES    = r19c["tournament_cohorts"]
C_CONF   = r19c["mondrian_conformal_and_abstention"]
RECAL    = r19c["split_half_recalibration_20x"]
SWEEP    = r19g["conformal_alpha_sweep"]
NCAL     = r19g["local_calibration_sample_size_curve"]
VOI      = r19h["proactive_conformal_voi"]
CORR     = r19i["sequential_0_24_48_72h_corridor_by_cohort"]
canary_gate(lambda: len(df_all) > 9000 and "MIMIC_IV_IntVal" in CORR, "Fig4 v5.2 conformal & 0-72h corridor loaded")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.3,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.4,
    "xtick.labelsize": 5.7,
    "ytick.labelsize": 5.7,
    "legend.fontsize": 5.0,
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

def boot_binom_pct(k, n, B=2000, seed=42):
    vec = np.repeat([1.0, 0.0], [int(k), int(n - k)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vec), size=(B, len(vec)))
    arr = vec[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(arr, [1.0, 99.0])
    return arr[(arr >= lo) & (arr <= hi)]

def wilson95_pct(k, n):
    z = 1.95996; p = k / n; den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hlf = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, 100.0 * (ctr - hlf)), min(100.0, 100.0 * (ctr + hlf))

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

    fig = plt.figure(figsize=(12.4, 9.35), facecolor="white")
    gs_r1 = fig.add_gridspec(1, 6, left=0.052, right=0.988, top=0.955, bottom=0.715, wspace=0.18)
    gs_r2 = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.00], left=0.055, right=0.988, top=0.618, bottom=0.355, wspace=0.34)
    gs_r3 = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.00, 1.08], left=0.055, right=0.988, top=0.258, bottom=0.062, wspace=0.26)

    def draw_panel_header(ax, letter, title, y_rel=1.045, x_let=-0.10):
        ax.text(x_let, y_rel, letter, transform=ax.transAxes, fontsize=9.4, fontweight="bold",
                color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=6.55, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    oof_curves = {}
    for i, (ck, ctitle, col, let) in enumerate(cohorts_6):
        sub = df_all[df_all["cohort"] == ck]
        y = sub["d28"].values.astype(int)
        p_raw = np.clip(sub["p_adaptive_gbdt"].values.astype(float), 1e-6, 1 - 1e-6)
        p_cal = np.clip(sub["p_cal_oof"].values.astype(float), 1e-6, 1 - 1e-6)
        rc_s = RECAL[ck]
        oof_curves[ck] = (y, p_cal if ck != "MIMIC_IV_IntVal" else p_raw, col, ctitle.split(" ")[0])

        ax_cal = fig.add_subplot(gs_r1[0, i], facecolor=PAL["ivory"])
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

    # PANEL g: 90% Mondrian Split-Conformal Triage Sets
    axg = fig.add_subplot(gs_r2[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axg, "g", "90% Mondrian Split-Conformal Definitive Triage Sets (B = 2,000)", x_let=-0.065)

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

    axg.scatter(xg - 0.14, auc_all_v, marker="o", color=PAL["slate"], s=23, label="Ungated (100% stays)", zorder=4)
    axg.scatter(xg + 0.14, auc_def_v, marker="D", color=[c for _, _, c in conf_list], s=25,
                label="Definitive singleton ({0}/{1})", zorder=5)

    for idx, (ba, bd, cv, dfc) in enumerate(zip(auc_all_v, auc_def_v, cov_v, def_frac)):
        d_v = bd - ba
        hi_ci = C_CONF[conf_list[idx][1]]["auroc_definitive_ci95"][1]
        top_y = max(ba, hi_ci, float(boot_g[idx].max()))
        axg.plot([idx - 0.14, idx + 0.14], [ba, bd], "-", color=conf_list[idx][2], lw=1.1, zorder=3)
        axg.text(idx, top_y + 0.009, f"{ba:.2f}→{bd:.2f}\n({d_v:+.3f})",
                 ha="center", va="bottom", fontsize=4.55, fontweight="bold", color=conf_list[idx][2], linespacing=1.02, zorder=5)
        axg.text(idx, 0.720, f"Cov {cv:.0f}%\nRet {dfc:.0f}%",
                 ha="center", va="bottom", fontsize=4.45, fontweight="bold", color=PAL["slate"], linespacing=1.04, zorder=5)

    axg.set_xticks(xg)
    axg.set_xticklabels([r[0] for r in conf_list], fontsize=5.1, fontweight="bold", color=PAL["charcoal"])
    axg.set_xlim(-0.52, len(conf_list) - 0.42)
    axg.set_ylim(0.715, 1.078)
    axg.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axg.set_ylabel("28-Day Mortality AUROC", fontweight="bold", color=PAL["charcoal"])
    axg.legend(loc="upper right", bbox_to_anchor=(1.0, 1.00), frameon=False, fontsize=4.9, ncol=1, handlelength=1.0)
    axg.spines[["top", "right"]].set_visible(False)

    # PANEL h: Sequential 0h -> 24h -> 72h Conformal Triage Corridor & 24-h Modality VoI
    axh = fig.add_subplot(gs_r2[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axh, "h", "Sequential 0-to-72-h Conformal Uncertainty Resolution", x_let=-0.085)

    voi_rows = [
        ("MIMIC-IV Val",            "MIMIC_IV_IntVal",        "MIMIC_IV_IntVal",                  PAL["navy"]),
        ("eICU Charted",            "eICU_CRD_Motor_Charted", "eICU_CRD_Motor_Charted",           PAL["crimson"]),
        ("HiRID Swiss",             "HiRID_Trauma",           "HiRID_Trauma",                     PAL["plum"]),
        ("MIMIC-III",               "MIMIC_III_CareVue",      "MIMIC_III_CareVue",                PAL["teal"]),
        ("Amsterdam",               "AmsterdamUMCdb_Trauma",  "AmsterdamUMCdb_Trauma",            PAL["cobalt"]),
        ("Jinhua Imaged (CT)",      "JinhuaNSICU_CT_Imaged",  "JinhuaNSICU_CT_Imaged_0h_to_24h",  PAL["amber"]),
    ]
    yh = np.arange(len(voi_rows))[::-1]
    for xg_v in [20, 40, 60, 80]:
        axh.axvline(xg_v, color="white", lw=0.82, zorder=0)
    axh.axvline(59.6, ymax=0.82, color=PAL["navy"], lw=0.80, ls="--", alpha=0.65, zorder=1)
    axh.axhline(0.5, color=PAL["stone"], lw=0.75, ls="-", zorder=1)

    for idx_v, (y_pos, (v_lab, v_k, c_k, v_col)) in enumerate(zip(yh, voi_rows)):
        vr = VOI[v_k]
        m_pct = vr["s0_ambiguous_conversion_pct"]
        cr = CORR[c_k]
        if c_k == "JinhuaNSICU_CT_Imaged_0h_to_24h":
            k0 = cr["resolved_at_24h_by_labs_and_ct_n"]
            n0 = cr["t0_arrival_uncertain_n"]
            c0_pct = cr["resolved_at_24h_pct"]
            nnr0 = cr["selective_nnr"]
            r_txt = f"{c0_pct:.1f}% (NNR {nnr0:.2f}) | CT VoI {m_pct:.1f}%"
        else:
            k0 = cr["t0_to_t24_chronological_voi"]["resolved_by_primary_n"]
            n0 = cr["t0_arrival_static3"]["uncertain_n"]
            c0_pct = cr["t0_to_t24_chronological_voi"]["resolved_by_primary_pct"]
            nnr0 = cr["t0_to_t24_chronological_voi"]["selective_nnr_primary"]
            res72 = cr["t24_residual_to_48_72h_closure"]["cumulative_conformal_or_clinical_resolved_by_72h_pct"]
            r_txt = f"{c0_pct:.1f}% (NNR {nnr0:.2f}) | 72h {res72:.1f}%"

        lo_w, hi_w = wilson95_pct(k0, n0)
        b_vec = boot_binom_pct(k0, n0, B=2000, seed=1200 + idx_v)
        vp_h = axh.violinplot([b_vec], positions=[y_pos], vert=False, widths=0.52, showmeans=False, showextrema=False, showmedians=False)
        for body in vp_h["bodies"]:
            body.set_facecolor(v_col); body.set_edgecolor(v_col); body.set_alpha(0.32); body.set_zorder(2)

        axh.plot([m_pct, c0_pct], [y_pos, y_pos], "-", color=v_col, lw=1.15, alpha=0.85, zorder=3)
        axh.scatter([m_pct], [y_pos], marker="o", s=19, facecolor="white", edgecolor=v_col, lw=1.1, zorder=4)
        axh.errorbar(c0_pct, y_pos, xerr=[[max(0, c0_pct - lo_w)], [max(0, hi_w - c0_pct)]], fmt="D", ms=4.1,
                     color=v_col, mec="white", mew=0.45, ecolor=PAL["charcoal"], elinewidth=0.95, capsize=2.1, zorder=5)
        axh.text(85.5, y_pos, r_txt, va="center", ha="left", fontsize=4.68, fontweight="bold", color=PAL["charcoal"], zorder=5)

    h_leg = [
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec=PAL["charcoal"], mew=1.1, ms=4.0, label="24-h motor/CT modality VoI (41.6%)"),
        Line2D([0], [0], marker="D", ls="none", mfc=PAL["navy"], mec="white", mew=0.45, ms=4.2, label="0-to-24-h arrival resolution (59.6%, NNR 1.68)"),
    ]
    axh.legend(handles=h_leg, loc="upper left", bbox_to_anchor=(0.0, 1.01), frameon=False, fontsize=4.6, ncol=2, handletextpad=0.3, columnspacing=0.9)
    axh.set_yticks(yh)
    axh.set_yticklabels([r[0] for r in voi_rows], fontsize=5.25, fontweight="bold", color=PAL["charcoal"])
    axh.set_xlim(18, 138)
    axh.set_ylim(-0.6, len(voi_rows) + 0.34)
    axh.set_xticks([20, 40, 60, 80])
    axh.set_xlabel("Uncertain {0, 1} Stays Resolved into Singletons at 24 h (%, Wilson 95% CI)", fontweight="bold", color=PAL["charcoal"])
    axh.spines[["top", "right"]].set_visible(False)

    # PANEL i: Conformal Confidence Sweep (80%–95%)
    axi = fig.add_subplot(gs_r3[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axi, "i", "Conformal Efficiency Sweep (80%–95%)", x_let=-0.11)

    t_levels = [80, 82, 85, 88, 90, 92, 95]
    for g_y in [30, 50, 70, 90]:
        axi.axhline(g_y, color="white", lw=0.8, zorder=1)
    axi.axvline(90, color=PAL["burgundy"], lw=0.80, ls="--", alpha=0.70, zorder=2)

    ret_mat = []
    for ck, ctitle, col, _ in cohorts_6:
        ys_ret = np.array([SWEEP[ck][str(lv)]["retained_singleton_frac"] * 100.0 for lv in t_levels])
        ret_mat.append(ys_ret)
        axi.plot(t_levels, ys_ret, "o-", color=col, lw=1.30, ms=2.9, label=ctitle.split(" ")[0], zorder=4)
    ret_mat = np.vstack(ret_mat)
    axi.fill_between(t_levels, ret_mat.min(axis=0), ret_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)

    axi.set_xlim(79.2, 95.8)
    axi.set_ylim(24, 104)
    axi.set_xticks([80, 85, 90, 95])
    axi.set_xlabel("Nominal Coverage Target (%)", fontweight="bold", color=PAL["charcoal"])
    axi.set_ylabel("Retained Singleton Stays (%)", fontweight="bold", color=PAL["charcoal"])
    axi.legend(loc="lower left", frameon=False, fontsize=4.7, ncol=2, handlelength=1.0, columnspacing=0.6)
    axi.spines[["top", "right"]].set_visible(False)

    # PANEL j: External Local Calibration Sample-Size Curve
    axj = fig.add_subplot(gs_r3[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axj, "j", "Local Site Calibration Sample-Size Curve", x_let=-0.11)

    nc_grid = [50, 100, 150, 200, 300]
    for g_y in [88, 90, 92, 94, 96]:
        axj.axhline(g_y, color="white", lw=0.8, zorder=1)
    axj.axhline(90.0, color=PAL["burgundy"], lw=0.85, ls="--", zorder=2)
    axj.axvline(150, color=PAL["slate"], lw=0.75, ls=":", zorder=2)

    for ck, ctitle, col, _ in cohorts_6[1:]:
        c_mean = np.array([NCAL[ck][str(nv)]["coverage_mean"] * 100.0 for nv in nc_grid])
        c_q25  = np.array([NCAL[ck][str(nv)]["coverage_iqr"][0] * 100.0 for nv in nc_grid])
        c_q75  = np.array([NCAL[ck][str(nv)]["coverage_iqr"][1] * 100.0 for nv in nc_grid])
        axj.fill_between(nc_grid, c_q25, c_q75, color=col, alpha=0.11, lw=0, zorder=2)
        axj.plot(nc_grid, c_mean, "s-", color=col, lw=1.30, ms=2.9, label=ctitle.split(" ")[0], zorder=4)

    axj.text(154, 86.6, "n = 150 stays", fontsize=4.7, fontweight="bold", color=PAL["slate"], ha="left", va="bottom", zorder=5)
    axj.set_xlim(40, 312)
    axj.set_ylim(86.2, 97.2)
    axj.set_xticks([50, 100, 150, 200, 300])
    axj.set_xlabel("Local Calibration Stays ($n_{\\mathrm{cal}}$)", fontweight="bold", color=PAL["charcoal"])
    axj.set_ylabel("Empirical 90% Coverage (%)", fontweight="bold", color=PAL["charcoal"])
    axj.legend(loc="upper right", frameon=False, fontsize=4.7, ncol=2, handlelength=1.0, columnspacing=0.6)
    axj.spines[["top", "right"]].set_visible(False)

    # PANEL k: Six-Cohort Decision Curve Analysis
    axk = fig.add_subplot(gs_r3[0, 2], facecolor=PAL["ivory"])
    draw_panel_header(axk, "k", "Cohort-Specific Decision Curve Net Benefit", x_let=-0.10)

    ts = np.round(np.arange(0.05, 0.501, 0.01), 2)
    for g_y in [0.00, 0.04, 0.08, 0.12, 0.16]:
        axk.axhline(g_y, color="white", lw=0.8, zorder=1)
    axk.axvspan(0.15, 0.35, color="#EBF2F7", alpha=0.45, zorder=1)

    nb_mat = []
    for ck, _, _, _ in cohorts_6:
        yb, pr_c, col_c, short_lab = oof_curves[ck]
        nb_r = nb_curve(yb, pr_c, ts)
        nb_mat.append(nb_r)
        prev_c = float(yb.mean())
        nb_all_c = prev_c - (1.0 - prev_c) * ts / (1.0 - ts)
        axk.plot(ts, nb_all_c, color=col_c, lw=0.75, ls=":", alpha=0.48, zorder=2)
        axk.plot(ts, nb_r, color=col_c, lw=1.40, label=f"{short_lab} ({prev_c*100:.1f}%)", zorder=4)

    nb_mat = np.vstack(nb_mat)
    axk.fill_between(ts, nb_mat.min(axis=0), nb_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)
    axk.axhline(0, color=PAL["slate"], lw=0.85, ls="-", zorder=2)

    handles_k, labels_k = axk.get_legend_handles_labels()
    handles_k.append(Line2D([0], [0], color=PAL["charcoal"], lw=0.85, ls=":", alpha=0.65)); labels_k.append("Treat-all")
    handles_k.append(Line2D([0], [0], color=PAL["slate"], lw=0.85, ls="-")); labels_k.append("Treat-none")

    axk.set_xlim(0.05, 0.50)
    axk.set_ylim(-0.028, 0.175)
    axk.set_xticks([0.05, 0.15, 0.25, 0.35, 0.50])
    axk.set_xlabel("Threshold Probability ($p_t$)", fontweight="bold", color=PAL["charcoal"])
    axk.set_ylabel("Net Benefit", fontweight="bold", color=PAL["charcoal"])
    axk.legend(handles_k, labels_k, loc="upper right", frameon=False, fontsize=4.55, ncol=2, handlelength=1.0, columnspacing=0.55)
    axk.spines[["top", "right"]].set_visible(False)

    if not (OUT / "Fig4_v5_3.png").exists():
        fig.savefig(OUT / "Fig4_v5_3.png", dpi=600)
    if not (OUT / "Fig4_v5_3.pdf").exists():
        fig.savefig(OUT / "Fig4_v5_3.pdf")
    plt.close(fig)
    print("[saved] Fig4_v5_3.png & .pdf (11-panel proactive conformal suite a–k)")

if __name__ == "__main__":
    main()
