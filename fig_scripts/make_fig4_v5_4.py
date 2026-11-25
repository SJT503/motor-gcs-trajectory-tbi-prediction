# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("E:/TBI subtype")
REP  = ROOT / "07_prediction_system/reports"
DATA = ROOT / "07_prediction_system/data"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig4_v5_4.py"))

r19c   = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
r19g   = json.load(open(REP / "19g_v4_5_benchmark_closure.json", encoding="utf-8"))
r19h   = json.load(open(REP / "19h_v5_proactive_biomarker_voi.json", encoding="utf-8"))
r19i   = json.load(open(REP / "19i_v5_2_sequential_conformal_corridor.json", encoding="utf-8"))
r19j   = json.load(open(REP / "19j_v5_3_corridor_and_model_audit.json", encoding="utf-8"))
df_all = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
boot_npz = np.load(DATA / "19c_v4_boot_distributions.npz")

T_RES  = r19c["tournament_cohorts"]
RECAL  = r19c["split_half_recalibration_20x"]
C_CONF = r19c["mondrian_conformal_and_abstention"]
SWEEP  = r19g["conformal_alpha_sweep"]
NC_CV  = r19g["local_calibration_sample_size_curve"]
VOI    = r19h["proactive_conformal_voi"]
CORR   = r19i["sequential_0_24_48_72h_corridor_by_cohort"]
DECOMP = r19j["by_cohort_decomposition"]

canary_gate(lambda: DECOMP["MIMIC_IV_IntVal"]["pure_conformal_singleton_by_72h_pct"] > 80.0, "Fig4 v5.4 inputs loaded")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.4,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAL = {
    "burgundy": "#8B1E24", "navy": "#0B3C5D", "crimson": "#B8292F", "teal": "#006D5B",
    "plum": "#5B2C6F", "cobalt": "#1F618D", "amber": "#B95815", "charcoal": "#1F2428",
    "slate": "#5A656E", "muted": "#8C969E", "ivory": "#FAF8F3", "stone": "#D6D1C4",
}

def wilson95_prop(k, n):
    z = 1.95996; p = k / max(n, 1); den = 1.0 + z * z / max(n, 1)
    ctr = (p + z * z / (2 * max(n, 1))) / den
    hlf = z * math.sqrt((p * (1 - p) + z * z / (4 * max(n, 1))) / max(n, 1)) / den
    return max(0.0, ctr - hlf), min(1.0, ctr + hlf)

def deciles_curve_with_ci(y, p, n_bins=7):
    y = np.asarray(y, int); p = np.asarray(p, float)
    qs = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    xs, os, los, his = [], [], [], []
    for b in range(len(qs) - 1):
        m = (idx == b)
        if m.sum() >= 5:
            k_ev, n_m = int(y[m].sum()), int(m.sum())
            lo, hi = wilson95_prop(k_ev, n_m)
            xs.append(p[m].mean()); os.append(y[m].mean()); los.append(lo); his.append(hi)
    return np.array(xs), np.array(os), np.array(los), np.array(his)

def nb_curve(y, p, ts):
    y = np.asarray(y, int); p = np.asarray(p, float); n = len(y)
    return np.array([(((p >= t) & (y == 1)).sum() - ((p >= t) & (y == 0)).sum() * t / (1 - t)) / n for t in ts])

def boot_binom_pct(k, n, B=2000, seed=42):
    vec = np.repeat([1.0, 0.0], [int(k), int(n - k)])
    rng = np.random.default_rng(seed)
    arr = vec[rng.integers(0, len(vec), size=(B, len(vec)))].mean(axis=1) * 100.0
    lo, hi = np.percentile(arr, [1.0, 99.0])
    return arr[(arr >= lo) & (arr <= hi)]

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

    # Full-page double-column canvas (9.2 x 10.8 in -> 183 x 215 mm at 0.783x scale -> all fonts >= 5.3-8.0 pt!)
    fig = plt.figure(figsize=(9.2, 10.8), facecolor="white")
    gs_r1 = fig.add_gridspec(2, 3, left=0.072, right=0.985, top=0.968, bottom=0.595, wspace=0.22, hspace=0.36)
    gs_r2 = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.04], left=0.072, right=0.985, top=0.518, bottom=0.292, wspace=0.33)
    gs_r3 = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.00, 1.06], left=0.072, right=0.985, top=0.214, bottom=0.048, wspace=0.28)

    def draw_panel_header(ax, letter, title, y_rel=1.045, x_let=-0.10):
        ax.text(x_let, y_rel, letter, transform=ax.transAxes, fontsize=10.2, fontweight="bold",
                color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=7.6, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    oof_curves = {}
    for i, (ck, ctitle, col, let) in enumerate(cohorts_6):
        r_idx, c_idx = divmod(i, 3)
        sub = df_all[df_all["cohort"] == ck]
        y = sub["d28"].values.astype(int)
        p_raw = np.clip(sub["p_adaptive_gbdt"].values.astype(float), 1e-6, 1 - 1e-6)
        p_cal = np.clip(sub["p_cal_oof"].values.astype(float), 1e-6, 1 - 1e-6)
        rc_s = RECAL[ck]
        oof_curves[ck] = (y, p_cal, col, ctitle.split(" ")[0])

        ax_cal = fig.add_subplot(gs_r1[r_idx, c_idx], facecolor=PAL["ivory"])
        draw_panel_header(ax_cal, let, ctitle, x_let=-0.12 if c_idx == 0 else -0.09)

        for g_v in [0.0, 0.25, 0.50, 0.75]:
            ax_cal.axhline(g_v, color=PAL["stone"] if g_v == 0.0 else "white", lw=0.75, zorder=1)
            if g_v > 0.0:
                ax_cal.axvline(g_v, color="white", lw=0.75, zorder=1)
        ax_cal.plot([0, 1], [0, 1], ls="--", color=PAL["muted"], lw=0.90, zorder=2)

        hist_cnt, hist_bins = np.histogram(p_cal, bins=np.linspace(0, 1, 24))
        x_grid = 0.5 * (hist_bins[:-1] + hist_bins[1:])
        hist_h = -0.14 + 0.12 * (hist_cnt / max(hist_cnt.max(), 1))
        ax_cal.fill_between(x_grid, -0.14, hist_h, color=col, alpha=0.50, zorder=2)
        ax_cal.plot(x_grid, hist_h, color=col, lw=0.80, zorder=3)

        x0, o0, _, _ = deciles_curve_with_ci(y, p_raw, n_bins=7)
        x1, o1, lo1, hi1 = deciles_curve_with_ci(y, p_cal, n_bins=7)
        ax_cal.plot(x0, o0, "o--", color=col, lw=1.25, ms=3.5, alpha=0.85, label="Unadjusted", zorder=4)
        ax_cal.errorbar(x1, o1, yerr=[np.maximum(0, o1 - lo1), np.maximum(0, hi1 - o1)], fmt="s-",
                        color=PAL["charcoal"], lw=1.35, ms=3.5, elinewidth=0.75, capsize=1.8, label="OOF Platt (95% CI)", zorder=5)

        stat_txt = (f"Unadj: slp {rc_s['frozen_slope']:.2f}, CITL {rc_s['frozen_citl']:+.2f}, Brier {rc_s['frozen_brier']:.3f}\n"
                    f"Platt: slp {rc_s['recal_slope_mean']:.2f}, CITL {rc_s['recal_citl_mean']:+.2f}, Brier {rc_s['recal_brier_mean']:.3f}")
        ax_cal.text(0.035, 0.965, stat_txt, transform=ax_cal.transAxes, fontsize=5.8, va="top", ha="left",
                    color=PAL["charcoal"], fontweight="bold", linespacing=1.20, zorder=6)

        ax_cal.set_xlim(0, 1.0)
        ax_cal.set_ylim(-0.15, 1.06)
        ax_cal.set_xticks([0.0, 0.5, 1.0])
        ax_cal.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        if c_idx > 0:
            ax_cal.set_yticklabels([])
        else:
            ax_cal.set_ylabel("Observed 28-d Mortality", fontweight="bold", color=PAL["charcoal"], fontsize=7.0)
        if i == 0:
            ax_cal.legend(loc="lower right", bbox_to_anchor=(0.99, 0.14), frameon=False, fontsize=5.8, handlelength=1.2)
        if r_idx == 1:
            ax_cal.set_xlabel("Predicted 28-d Mortality Risk", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
        ax_cal.spines[["top", "right"]].set_visible(False)

    # PANEL g: 90% Mondrian Split-Conformal Triage Sets
    axg = fig.add_subplot(gs_r2[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axg, "g", "90% Mondrian Split-Conformal Singleton Sets", x_let=-0.10)

    conf_list = [
        ("MIMIC-IV\n(483)",        "MIMIC_IV_IntVal",        PAL["navy"]),
        ("MIMIC-III\n(839)",       "MIMIC_III_CareVue",      PAL["teal"]),
        ("eICU-CRD\n(4,440)",      "eICU_CRD",               PAL["crimson"]),
        ("HiRID Tr.\n(1,209)",     "HiRID_Trauma",           PAL["plum"]),
        ("HiRID ICP\n(2,805)",     "HiRID_Neurotrauma_ICP",  PAL["plum"]),
        (f"Amsterdam\n({au_n:,})", "AmsterdamUMCdb_Trauma",  PAL["cobalt"]),
        ("Jinhua\n(1,941)",        "JinhuaNSICU_TBI_S06",    PAL["amber"]),
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

    vp_g = axg.violinplot(boot_g, positions=xg + 0.14, vert=True, widths=0.36, showmeans=False, showextrema=False, showmedians=False)
    for body, (_, _, col) in zip(vp_g["bodies"], conf_list):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.38); body.set_zorder(2)
    axg.boxplot(boot_g, positions=xg + 0.14, widths=0.11, showfliers=False,
                boxprops=dict(color=PAL["charcoal"], lw=0.65), medianprops=dict(color="white", lw=1.0),
                whiskerprops=dict(color=PAL["charcoal"], lw=0.65), capprops=dict(color=PAL["charcoal"], lw=0.65), zorder=3)

    axg.scatter(xg - 0.14, auc_all_v, marker="o", color=PAL["slate"], s=24, label="Ungated (100% stays)", zorder=4)
    axg.scatter(xg + 0.14, auc_def_v, marker="D", color=[c for _, _, c in conf_list], s=26,
                label="Definitive singleton ({0}/{1})", zorder=5)

    for idx, (ba, bd, cv, dfc) in enumerate(zip(auc_all_v, auc_def_v, cov_v, def_frac)):
        d_v = bd - ba
        hi_ci = C_CONF[conf_list[idx][1]]["auroc_definitive_ci95"][1]
        top_y = max(ba, hi_ci, float(boot_g[idx].max()))
        axg.plot([idx - 0.14, idx + 0.14], [ba, bd], "-", color=conf_list[idx][2], lw=1.1, zorder=3)
        axg.text(idx, top_y + 0.008, f"{d_v:+.3f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=conf_list[idx][2], zorder=5)
        axg.text(idx, 0.718, f"{cv:.0f}% cov\n{dfc:.0f}% ret", ha="center", va="bottom", fontsize=5.4, fontweight="bold", color=PAL["slate"], linespacing=1.06, zorder=5)

    axg.set_xticks(xg)
    axg.set_xticklabels([r[0] for r in conf_list], fontsize=6.0, fontweight="bold", color=PAL["charcoal"])
    axg.set_xlim(-0.55, len(conf_list) - 0.40)
    axg.set_ylim(0.712, 1.068)
    axg.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axg.set_ylabel("28-Day Mortality AUROC", fontweight="bold", color=PAL["charcoal"])
    axg.legend(loc="upper right", bbox_to_anchor=(1.0, 1.01), frameon=False, fontsize=5.8, ncol=1, handlelength=1.0)
    axg.spines[["top", "right"]].set_visible(False)

    # PANEL h: Sequential 0-to-72-h Conformal Corridor (reporting PURE 72-h conformal % cleanly!)
    axh = fig.add_subplot(gs_r2[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axh, "h", "0-to-72-h Conformal Uncertainty Resolution", x_let=-0.11)

    voi_rows = [
        ("MIMIC-IV Val",       "MIMIC_IV_IntVal",        "MIMIC_IV_IntVal",                  PAL["navy"]),
        ("eICU Charted",       "eICU_CRD_Motor_Charted", "eICU_CRD_Motor_Charted",           PAL["crimson"]),
        ("HiRID Swiss",        "HiRID_Trauma",           "HiRID_Trauma",                     PAL["plum"]),
        ("MIMIC-III",          "MIMIC_III_CareVue",      "MIMIC_III_CareVue",                PAL["teal"]),
        ("Amsterdam",          "AmsterdamUMCdb_Trauma",  "AmsterdamUMCdb_Trauma",            PAL["cobalt"]),
        ("Jinhua Imaged",      "JinhuaNSICU_CT_Imaged",  "JinhuaNSICU_CT_Imaged_0h_to_24h",  PAL["amber"]),
    ]
    yh = np.arange(len(voi_rows))[::-1]
    for xg_v in [20, 40, 60, 80]:
        axh.axvline(xg_v, color="white", lw=0.82, zorder=0)
    axh.axvline(59.6, ymax=0.80, color=PAL["navy"], lw=0.85, ls="--", alpha=0.65, zorder=1)
    axh.axhline(0.5, color=PAL["stone"], lw=0.75, ls="-", zorder=1)

    for idx_v, (y_pos, (v_lab, v_k, c_k, v_col)) in enumerate(zip(yh, voi_rows)):
        vr = VOI[v_k]
        m_pct = vr["s0_ambiguous_conversion_pct"]
        cr = CORR[c_k]
        if c_k == "JinhuaNSICU_CT_Imaged_0h_to_24h":
            k0 = cr["resolved_at_24h_by_labs_and_ct_n"]; n0 = cr["t0_arrival_uncertain_n"]
            c0_pct = cr["resolved_at_24h_pct"]; nnr0 = cr["selective_nnr"]
            r_txt = f"{c0_pct:.1f}% (NNR {nnr0:.2f}) | CT {m_pct:.1f}%"
        else:
            k0 = cr["t0_to_t24_chronological_voi"]["resolved_by_primary_n"]
            n0 = cr["t0_arrival_static3"]["uncertain_n"]
            c0_pct = cr["t0_to_t24_chronological_voi"]["resolved_by_primary_pct"]
            nnr0 = cr["t0_to_t24_chronological_voi"]["selective_nnr_primary"]
            pure72 = DECOMP[c_k]["pure_conformal_singleton_by_72h_pct"]
            r_txt = f"{c0_pct:.1f}% (NNR {nnr0:.2f}) | 72h {pure72:.1f}%"

        lo_p, hi_p = wilson95_prop(k0, n0)
        lo_w, hi_w = 100.0 * lo_p, 100.0 * hi_p
        b_vec = boot_binom_pct(k0, n0, B=2000, seed=1200 + idx_v)
        vp_h = axh.violinplot([b_vec], positions=[y_pos], vert=False, widths=0.52, showmeans=False, showextrema=False, showmedians=False)
        for body in vp_h["bodies"]:
            body.set_facecolor(v_col); body.set_edgecolor(v_col); body.set_alpha(0.32); body.set_zorder(2)

        axh.plot([m_pct, c0_pct], [y_pos, y_pos], "-", color=v_col, lw=1.20, alpha=0.88, zorder=3)
        axh.scatter([m_pct], [y_pos], marker="o", s=22, facecolor="white", edgecolor=v_col, lw=1.15, zorder=4)
        axh.errorbar(c0_pct, y_pos, xerr=[[max(0, c0_pct - lo_w)], [max(0, hi_w - c0_pct)]], fmt="D", ms=4.3,
                     color=v_col, mec="white", mew=0.45, ecolor=PAL["charcoal"], elinewidth=0.95, capsize=2.2, zorder=5)
        axh.text(85.5, y_pos, r_txt, va="center", ha="left", fontsize=5.65, fontweight="bold", color=PAL["charcoal"], zorder=5)

    h_leg = [
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec=PAL["charcoal"], mew=1.15, ms=4.2, label="24-h marginal VoI (41.6%)"),
        Line2D([0], [0], marker="D", ls="none", mfc=PAL["navy"], mec="white", mew=0.45, ms=4.4, label="0-to-24-h resolved (59.6%, NNR 1.68)"),
    ]
    axh.legend(handles=h_leg, loc="upper left", bbox_to_anchor=(-0.02, 1.01), frameon=False, fontsize=5.55, ncol=2, handletextpad=0.25, columnspacing=0.65)
    axh.set_yticks(yh)
    axh.set_yticklabels([r[0] for r in voi_rows], fontsize=6.4, fontweight="bold", color=PAL["charcoal"])
    axh.set_xlim(18, 146)
    axh.set_ylim(-0.6, len(voi_rows) + 0.42)
    axh.set_xticks([20, 40, 60, 80])
    axh.set_xlabel("Uncertain {0, 1} Stays Resolved at 24 h (%, Wilson 95% CI)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axh.spines[["top", "right"]].set_visible(False)

    # PANEL i: Conformal Confidence Sweep (80%-95%)
    axi = fig.add_subplot(gs_r3[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axi, "i", "Efficiency Sweep (80%–95%)", x_let=-0.12)
    t_levels = [80, 82, 85, 88, 90, 92, 95]
    for g_y in [30, 50, 70, 90]:
        axi.axhline(g_y, color="white", lw=0.8, zorder=1)
    axi.axvline(90, color=PAL["burgundy"], lw=0.85, ls="--", alpha=0.72, zorder=2)
    ret_mat = []
    for ck, ctitle, col, _ in cohorts_6:
        ys_ret = np.array([SWEEP[ck][str(lv)]["retained_singleton_frac"] * 100.0 for lv in t_levels])
        ret_mat.append(ys_ret)
        axi.plot(t_levels, ys_ret, "o-", color=col, lw=1.35, ms=3.2, label=ctitle.split(" ")[0], zorder=4)
    ret_mat = np.vstack(ret_mat)
    axi.fill_between(t_levels, ret_mat.min(axis=0), ret_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)
    axi.set_xlim(79.2, 95.8); axi.set_ylim(24, 104); axi.set_xticks([80, 85, 90, 95])
    axi.set_xlabel("Nominal Coverage Target (%)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axi.set_ylabel("Retained Singleton Stays (%)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axi.legend(loc="lower left", frameon=False, fontsize=5.6, ncol=2, handlelength=1.0, columnspacing=0.5)
    axi.spines[["top", "right"]].set_visible(False)

    # PANEL j: Local Calibration Sample-Size Curve
    axj = fig.add_subplot(gs_r3[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axj, "j", "Local Calibration Curve", x_let=-0.12)
    nc_grid = [50, 100, 150, 200, 300]
    for g_y in [88, 90, 92, 94, 96]:
        axj.axhline(g_y, color="white", lw=0.8, zorder=1)
    axj.axhline(90.0, color=PAL["burgundy"], lw=0.85, ls="--", zorder=2)
    axj.axvline(150, color=PAL["charcoal"], lw=0.85, ls=":", alpha=0.75, zorder=2)
    axj.text(155, 86.7, "n = 150 stays", fontsize=5.8, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left", zorder=5)
    for ck, clab, col in [
        ("MIMIC_III_CareVue", "MIMIC-III", PAL["teal"]),
        ("eICU_CRD", "eICU-CRD", PAL["crimson"]),
        ("HiRID_Trauma", "HiRID", PAL["plum"]),
        ("AmsterdamUMCdb_Trauma", "Amsterdam", PAL["cobalt"]),
        ("JinhuaNSICU_TBI_S06", "Jinhua", PAL["amber"]),
    ]:
        ym  = np.array([NC_CV[ck][str(nv)]["coverage_mean"] * 100.0 for nv in nc_grid])
        q25 = np.array([NC_CV[ck][str(nv)]["coverage_iqr"][0] * 100.0 for nv in nc_grid])
        q75 = np.array([NC_CV[ck][str(nv)]["coverage_iqr"][1] * 100.0 for nv in nc_grid])
        axj.fill_between(nc_grid, q25, q75, color=col, alpha=0.14, lw=0, zorder=2)
        axj.plot(nc_grid, ym, "s-", color=col, lw=1.30, ms=3.1, label=clab, zorder=4)
    axj.set_xlim(40, 312); axj.set_ylim(86.2, 99.8); axj.set_xticks([50, 100, 150, 200, 300])
    axj.set_yticks([88, 90, 92, 94, 96, 98])
    axj.set_xlabel("Local Calibration Stays ($n_{\\mathrm{cal}}$)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axj.set_ylabel("Empirical 90% Coverage (%)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axj.legend(loc="upper right", bbox_to_anchor=(0.99, 1.00), frameon=False, fontsize=5.4, ncol=3, handlelength=0.85, columnspacing=0.45)
    axj.spines[["top", "right"]].set_visible(False)

    # PANEL k: Cohort-Specific Decision Curve Net Benefit
    axk = fig.add_subplot(gs_r3[0, 2], facecolor=PAL["ivory"])
    draw_panel_header(axk, "k", "Decision Curve Net Benefit", x_let=-0.12)
    ts = np.linspace(0.05, 0.50, 46)
    for g_y in [0.00, 0.05, 0.10, 0.15]:
        axk.axhline(g_y, color=PAL["stone"] if g_y == 0.0 else "white", lw=0.80, zorder=1)
    axk.axvspan(0.15, 0.35, color=PAL["cobalt"], alpha=0.06, zorder=0)
    for ck, ctitle, col, _ in cohorts_6:
        y_c, p_c, _, sh_n = oof_curves[ck]
        prev = float(y_c.mean())
        nb_m = nb_curve(y_c, p_c, ts)
        nb_all = prev - (1.0 - prev) * ts / (1.0 - ts)
        axk.plot(ts, nb_all, ":", color=col, lw=0.75, alpha=0.45, zorder=2)
        axk.plot(ts, nb_m, "-", color=col, lw=1.35, label=f"{sh_n} ({prev*100:.1f}%)", zorder=4)
    axk.plot(ts, np.zeros_like(ts), "-", color=PAL["charcoal"], lw=0.90, label="Treat-none", zorder=3)
    axk.set_xlim(0.05, 0.50); axk.set_ylim(-0.025, 0.178); axk.set_xticks([0.05, 0.15, 0.25, 0.35, 0.50])
    axk.set_xlabel("Threshold Probability ($p_t$)", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axk.set_ylabel("Net Benefit", fontweight="bold", color=PAL["charcoal"], fontsize=6.8)
    axk.legend(loc="upper right", frameon=False, fontsize=5.35, ncol=2, handlelength=0.9, columnspacing=0.45)
    axk.spines[["top", "right"]].set_visible(False)

    for ext in ("png", "pdf"):
        out_f = OUT / f"Fig4_v5_5.{ext}"
        if not out_f.exists():
            fig.savefig(out_f, dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("[saved] Fig4_v5_5.png & .pdf (2x3 print-legible calibration + zero-overlap 4j + pure 72h conformal)")

if __name__ == "__main__":
    main()
