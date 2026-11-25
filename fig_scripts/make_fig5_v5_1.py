# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
Fig 5 v5.0 (Five-Cohort Leak-Free 67-Hour Rolling Horizon, Grouped & Patient-Level TreeSHAP,
and Harmonized v4 Robustness Across Sedation, Code Status, Hospital-Split eICU, Invasive ICP & IMPACT-Core)
- Removes baked-in top title banner; uses Springer Nature bold lowercase labels (a–d)
- Panel a: removes boxed 5-number string & prose annotation inside plot
- Panel b: removes 4 boxed `Motor = xx%` pills & cleans legend (`Latent trajectory posteriors`)
- Panel c: enforces US English (`Hemoglobin`) & consistent `Motor change` legend label
- Panel d: harmonizes all MIMIC-IV sensitivities (`0.908` sedative censoring, `0.909` DNR/CMO) to v4 with 95% CIs,
  adds eICU 130-hospital leave-hospitals-out conformal (`0.919`), and fixes misattached `(NNE=2.51)` on HiRID Deep RASS!
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path("E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP  = ROOT / "07_prediction_system/reports"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig5_v5_1.py"))

roll_us    = pd.read_csv(REP / "09e_v3_fig2_d28.csv")
roll_ei_ch = pd.read_csv(REP / "19d_v4_rolling_eicu_charted.csv")
roll_hi    = pd.read_csv(REP / "09e_v3_fig2_d28_hirid.csv")
roll_au    = pd.read_csv(REP / "19d_v4_rolling_amsterdam.csv")

df_shap  = pd.read_parquet(DATA / "19c_v4_intval_shap_matrix.parquet")
boot_npz = np.load(DATA / "19c_v4_boot_distributions.npz")
gshap    = json.load(open(REP / "19e_v4_grouped_shap.json", encoding="utf-8"))
g12      = json.load(open(REP / "12d_gcs12_refit_impact.json", encoding="utf-8"))
hir      = json.load(open(REP / "08f_v3_hirid_validation.json", encoding="utf-8"))
r19c     = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
r19d     = json.load(open(REP / "19d_v4_rolling_end_reclass.json", encoding="utf-8"))
r19g     = json.load(open(REP / "19g_v4_5_benchmark_closure.json", encoding="utf-8"))
canary_gate(lambda: len(df_shap) == 483, "Exact 483x25 patient-level SHAP matrix loaded")

PAL = {
    "burgundy": "#8B1E24", "charcoal": "#1F2428", "slate": "#4A5568", "muted": "#718096",
    "ivory": "#FAF8F3", "stone": "#D6D1C4", "m4": "#0B3C5D", "eicu": "#B8292F",
    "m3": "#006D5B", "hirid": "#5B2C6F", "aumc": "#1F618D", "jinhua": "#B95815",
    "static": "#94A3B8", "level": "#0B3C5D", "change": "#2E86AB", "post": "#006D5B", "comp": "#64748B",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.6,
    "axes.titlesize": 7.3,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.4,
    "legend.frameon": False,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

@retry_limit(2)
def main():
    fig = plt.figure(figsize=(10.8, 7.30), facecolor="white")
    gs_top = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.0], wspace=0.28, left=0.075, right=0.980, top=0.940, bottom=0.565)
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.12], wspace=0.56, left=0.160, right=0.982, top=0.430, bottom=0.068)

    def draw_panel_header(ax, letter, title, y_rel=1.042, x_let=-0.068):
        ax.text(x_let, y_rel, letter, transform=ax.transAxes,
                fontsize=9.6, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=7.0, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    # PANEL a: Leak-Free Rolling Discrimination (6–72 h, Five Cohorts)
    axa = fig.add_subplot(gs_top[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axa, "a", "Leak-Free Rolling Discrimination (6–72 h, Five Cohorts)")

    for g_y in [0.75, 0.80, 0.85, 0.90, 0.95]:
        axa.axhline(g_y, color="white", lw=0.8, zorder=1)
    axa.axvspan(6, 24, color="#ECE6D8", alpha=0.50, zorder=1)

    RS = r19d["rolling_cohorts"]
    def p_lbl(nm, rk):
        d24 = RS[rk]["paired_24h_vs_6h"]["delta_24h_vs_6h"]
        pv  = RS[rk]["paired_24h_vs_6h"]["p_paired_boot"]
        ps  = "P<0.001" if pv < 0.001 else f"P={pv:.3f}"
        return f"{nm} (24h: {RS[rk]['auroc_24h']:.3f}, Δ24–6h {d24:+.3f}, {ps})"

    roll_series = [
        (roll_us[roll_us.db == "mimic4"], PAL["m4"],    p_lbl("MIMIC-IV", "MIMIC_IV")),
        (roll_us[roll_us.db == "mimic3"], PAL["m3"],    p_lbl("MIMIC-III", "MIMIC_III")),
        (roll_ei_ch,                      PAL["eicu"],  p_lbl("eICU Charted", "eICU_CRD_Charted")),
        (roll_hi,                         PAL["hirid"], p_lbl("HiRID Swiss", "HiRID")),
        (roll_au,                         PAL["aumc"],  p_lbl("Amsterdam", "AmsterdamUMCdb_Trauma")),
    ]
    all_min_lo = 0.75
    for sub, col, lab in roll_series:
        sub = sub.sort_values("t")
        axa.plot(sub.t, sub.auroc, color=col, lw=1.45, label=lab, zorder=4)
        if "lo" in sub.columns and sub["lo"].notna().any():
            axa.fill_between(sub.t, sub.lo, sub.hi, color=col, alpha=0.11, lw=0, zorder=2)
            all_min_lo = min(all_min_lo, float(sub["lo"].min()))
        pt24 = sub[sub.t == 24]
        if len(pt24):
            axa.scatter([24], [pt24.auroc.values[0]], color=col, s=20, edgecolor="white", linewidth=0.5, zorder=5)

    axa.axvline(24, color=PAL["burgundy"], lw=0.90, ls="--", zorder=3)
    axa.text(24.9, 0.968, "24-h landmark", fontsize=5.1, fontweight="bold", color=PAL["burgundy"], va="top", ha="left", zorder=5)
    y_lo = min(0.71, round(all_min_lo - 0.02, 2))

    axa.set_xlim(6, 72)
    axa.set_ylim(y_lo, 0.980)
    axa.set_xticks([6, 12, 24, 36, 48, 60, 72])
    axa.set_xlabel("Hours Since ICU Admission (67 Leak-Free Checkpoints)", fontweight="bold", color=PAL["charcoal"])
    axa.set_ylabel("AUROC for 28-Day Mortality", fontweight="bold", color=PAL["charcoal"])
    axa.legend(frameon=False, loc="lower right", fontsize=4.55, handlelength=1.15, labelspacing=0.20)
    axa.spines[["top", "right"]].set_visible(False)

    # PANEL b: Grouped |SHAP| Attribution Stream Across TBI Severity
    axb = fig.add_subplot(gs_top[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axb, "b", "Grouped |SHAP| Attribution Stream Across TBI Severity")

    strata_order = [
        (f"Overall\nAll TBI\n(n = {gshap['overall']['n']})",             gshap["overall"]["blocks"]),
        (f"Mild\nGCS 13–15\n(n = {gshap['by_stratum']['13-15']['n']})",  gshap["by_stratum"]["13-15"]["blocks"]),
        (f"Moderate\nGCS 9–12\n(n = {gshap['by_stratum']['9-12']['n']})",gshap["by_stratum"]["9-12"]["blocks"]),
        (f"Severe\nGCS 3–8\n(n = {gshap['by_stratum']['3-8']['n']})",    gshap["by_stratum"]["3-8"]["blocks"]),
    ]
    xb = np.arange(len(strata_order))
    b_keys = [
        ("S_static", "Static physiology & baseline (18)", PAL["static"]),
        ("LEVEL",    "Motor level (last, max)",           PAL["level"]),
        ("CHANGE",   "Motor change (slope, SD)",          PAL["change"]),
        ("POST",     "Latent trajectory posteriors",      PAL["post"]),
    ]
    for g_y in [20, 40, 60, 80, 100]:
        axb.axhline(g_y, color="white", lw=0.8, zorder=1)

    y_stacks = [np.array([b[bk]["pct_of_total"] for _, b in strata_order]) for bk, _, _ in b_keys]
    axb.stackplot(xb, *y_stacks, labels=[blab for _, blab, _ in b_keys],
                  colors=[bcol for _, _, bcol in b_keys], alpha=0.88, edgecolor="white", lw=0.7, zorder=2)

    bottom = np.zeros(len(strata_order))
    for (bk, blab, bcol), vals in zip(b_keys, y_stacks):
        top_edge = bottom + vals
        axb.scatter(xb, top_edge, color="white", edgecolor=bcol, s=11, zorder=4)
        for i, v in enumerate(vals):
            if bk == "CHANGE":
                axb.text(xb[i] + 0.12, bottom[i] + v / 2, f"{v:.1f}%",
                         ha="left", va="center", fontsize=4.8, fontweight="bold", color=PAL["charcoal"], zorder=5)
            elif v >= 6.5:
                txt_col = "white" if bk in ("LEVEL", "POST") else PAL["charcoal"]
                x_pos = xb[i] - 0.14 if i == len(strata_order) - 1 else (xb[i] + 0.14 if i == 0 else xb[i])
                axb.text(x_pos, bottom[i] + v / 2, f"{v:.1f}%",
                         ha="center", va="center", fontsize=5.0, color=txt_col, fontweight="bold", zorder=5)
        bottom += vals

    for i, (_, b) in enumerate(strata_order):
        m_tot = 100.0 - b["S_static"]["pct_of_total"]
        axb.text(xb[i], 102.0, f"Motor {m_tot:.1f}%", ha="center", va="bottom",
                 fontsize=4.85, fontweight="bold", color=PAL["level"])

    axb.set_xticks(xb)
    axb.set_xticklabels([s[0] for s in strata_order], fontsize=5.5, fontweight="bold", color=PAL["charcoal"])
    axb.set_xlim(-0.35, 3.45)
    axb.set_ylim(0, 114)
    axb.set_ylabel("Share of Total |SHAP| Attribution (%)", fontweight="bold", color=PAL["charcoal"])
    axb.legend(loc="upper center", bbox_to_anchor=(0.48, -0.16), ncol=2, frameon=False, fontsize=4.9, handlelength=1.0, columnspacing=0.8)
    axb.spines[["top", "right"]].set_visible(False)

    # PANEL c: Top-15 Predictors by Patient-Level |TreeSHAP| (n = 483, US English)
    axc = fig.add_subplot(gs_bot[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axc, "c", "Top-15 Predictors by Patient-Level |SHAP| (n = 483)", y_rel=1.032, x_let=-0.085)

    feat_cols = [c for c in df_shap.columns if c not in ("stay_id", "d28")]
    abs_shap = df_shap[feat_cols].abs()
    mean_s = abs_shap.mean().sort_values(ascending=False).head(15)
    names = list(mean_s.index)[::-1]
    vals  = list(mean_s.values)[::-1]

    nice = {
        "admission_age": "Age at admission", "gcs_eye_first": "GCS eye (first)",
        "gcs_total_first": "GCS total (first)", "motor_last": "Motor GCS (last, 24 h)",
        "pt_min": "Prothrombin time (min)", "pt_max": "Prothrombin time (max)",
        "wbc_min": "WBC count (min)", "prob_m1": "Class 1 posterior (high-stable)",
        "prob_m2": "Class 2 posterior (low-declining)", "prob_m3": "Class 3 posterior (mod.-improving)",
        "hemoglobin_min": "Hemoglobin (min)", "hemoglobin_max": "Hemoglobin (max)",
        "rr_mean": "Respiratory rate (mean)", "bun_min": "Blood urea nitrogen (min)",
        "motor_slope": "Motor GCS slope (per 10 h)", "motor_max": "Motor GCS (max, 24 h)",
        "platelet_max": "Platelet count (max)", "platelet_min": "Platelet count (min)",
        "sbp_min": "Systolic BP (min)", "spo2_min": "Oxygen saturation (min)", "temp_max": "Temperature (max)",
    }
    def feat_color(k):
        if k in ("motor_last", "motor_max"): return PAL["level"]
        if k in ("motor_slope", "motor_sd"): return PAL["change"]
        if k.startswith("prob_m"): return PAL["post"]
        return PAL["static"]

    for x_g in [0.25, 0.50, 0.75, 1.00, 1.25]:
        axc.axvline(x_g, color="white", lw=0.8, zorder=1)

    yc = np.arange(len(names))
    cols_c = [feat_color(k) for k in names]
    real_shap_vecs = [abs_shap[k].values[abs_shap[k].values <= float(np.percentile(abs_shap[k].values, 95))] for k in names]
    vp_c = axc.violinplot(real_shap_vecs, positions=yc, vert=False, widths=0.72, showmeans=False, showextrema=False, showmedians=False)
    for body, col in zip(vp_c["bodies"], cols_c):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.45); body.set_zorder(2)

    q25_v = [float(np.percentile(abs_shap[k].values, 25)) for k in names]
    q75_v = [float(np.percentile(abs_shap[k].values, 75)) for k in names]
    for yv, lo_q, hi_q, col in zip(yc, q25_v, q75_v, cols_c):
        axc.plot([lo_q, hi_q], [yv, yv], color=col, lw=1.8, zorder=3)
    axc.scatter(vals, yc, marker="D", color=cols_c, edgecolor="white", linewidth=0.5, s=22, zorder=4)

    for yv, k, v, vvec in zip(yc, names, vals, real_shap_vecs):
        is_mot = k.startswith("motor") or k.startswith("prob_m")
        x_txt = max(float(vvec.max()), v) + 0.025
        axc.text(x_txt, yv, f"{v:.3f}", fontsize=4.85, va="center", ha="left",
                 fontweight="bold" if is_mot else "normal", color=PAL["charcoal"])

    axc.set_yticks(yc)
    axc.set_yticklabels([nice.get(k, k.replace("_", " ")) for k in names], fontsize=5.3)
    for tick, k in zip(axc.get_yticklabels(), names):
        if k.startswith("motor") or k.startswith("prob_m"):
            tick.set_fontweight("bold"); tick.set_color(PAL["level"])

    axc.set_xlim(0.0, 2.38)
    axc.set_ylim(-0.6, len(names) - 0.4)
    axc.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axc.set_xlabel("|SHAP Value| (Patient Violins, IQR & Mean Diamond)", fontweight="bold", color=PAL["charcoal"])
    axc.legend(handles=[
        Patch(fc=PAL["level"],  label="Motor level"),
        Patch(fc=PAL["post"],   label="Latent posteriors"),
        Patch(fc=PAL["change"], label="Motor change"),
        Patch(fc=PAL["static"], label="Static physiology"),
    ], frameon=False, loc="lower right", fontsize=4.80, handlelength=1.0)
    axc.spines[["top", "right"]].set_visible(False)

    # PANEL d: Harmonized v4 Clinical Robustness, Hospital-Split eICU, Invasive ICP, RASS & IMPACT-Core
    axd = fig.add_subplot(gs_bot[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axd, "d", "Clinical Sensitivities, ICP, RASS & IMPACT-Core", y_rel=1.032, x_let=-0.075)

    T = r19c["tournament_cohorts"]
    C = r19c["mondrian_conformal_and_abstention"]
    NNE = r19c["deployment_nne_metrics"]
    END = r19d["secondary_endpoints_end_icp"]
    HS  = r19g["mimic4_v4_harmonized_sensitivities"]
    LHO = r19g["eicu_leave_hospitals_out_conformal"]
    au_n = T["AmsterdamUMCdb_Trauma"]["n"]
    au_icp_n = END["AmsterdamUMCdb_Invasive_ICP_Subgroup"]["n_icp_monitored"]
    imp_m4 = g12["sets"]["mimic4_intval_gcs12"]

    rass_cell = hir["trauma_by_icp_and_rass"]["rass_deep_le_m3"]
    rass_ci = rass_cell.get("ci95") or rass_cell.get("boot95")

    sens_rows = [
        ("MIMIC-IV Primary Model (n=483)",            T["MIMIC_IV_IntVal"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], T["MIMIC_IV_IntVal"]["models"]["Modality_Adaptive_GBDT"]["ci95"], f" (NNE={NNE['MIMIC_IV_IntVal']['NNE']:.2f})", PAL["m4"], True, "MIMIC_IV_IntVal__Modality_Adaptive_GBDT"),
        (f"MIMIC-IV Definitive (Ret {C['MIMIC_IV_IntVal']['definitive_retained_frac']*100:.0f}%)", C["MIMIC_IV_IntVal"]["auroc_definitive"], C["MIMIC_IV_IntVal"]["auroc_definitive_ci95"], "", PAL["m4"], False, "MIMIC_IV_IntVal__Conformal_Definitive"),
        ("Sedative-censored motor (−25.5%)",          HS["sedative_infusion_censored_adaptive"]["auroc"], HS["sedative_infusion_censored_adaptive"]["ci95"], "", PAL["m4"], False, None),
        ("Excluding DNR/CMO orders (n=465)",          HS["excl_any_code_limit"]["auroc"], HS["excl_any_code_limit"]["ci95"], "", PAL["m4"], False, None),
        (f"MIMIC-IV Composite END+Death (n={END['MIMIC_IV_Composite_END_Mortality']['n']})", END["MIMIC_IV_Composite_END_Mortality"]["auroc_composite"], END["MIMIC_IV_Composite_END_Mortality"].get("ci95"), f" ({END['MIMIC_IV_Composite_END_Mortality']['composite_events']} ev)", PAL["m4"], False, None),
        ("MIMIC-III CareVue Primary (n=839)",         T["MIMIC_III_CareVue"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], T["MIMIC_III_CareVue"]["models"]["Modality_Adaptive_GBDT"]["ci95"], f" (NNE={NNE['MIMIC_III_CareVue']['NNE']:.2f})", PAL["m3"], True, "MIMIC_III_CareVue__Modality_Adaptive_GBDT"),
        ("eICU Hospital-Split Definitive (130 hosps)",LHO["auroc_definitive_mean"], None, f" (Cov {LHO['empirical_coverage_overall']*100:.1f}%)", PAL["eicu"], True, "eICU_CRD__Conformal_Definitive"),
        ("HiRID Trauma Primary (CH, n=1,209)",        T["HiRID_Trauma"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], T["HiRID_Trauma"]["models"]["Modality_Adaptive_GBDT"]["ci95"], f" (NNE={NNE['HiRID_Trauma']['NNE']:.2f})", PAL["hirid"], True, "HiRID_Trauma__Modality_Adaptive_GBDT"),
        ("HiRID Deep RASS ≤ -3 (CH, n=854)",          rass_cell["AUROC"], rass_ci, "", PAL["hirid"], False, None),
        ("HiRID ABI ICP > 20 mmHg (n=1,340)",         END["HiRID_Invasive_ICP_Subgroup"]["auroc_icp_gt20"], END["HiRID_Invasive_ICP_Subgroup"].get("ci95_icp_gt20"), " (ABI ICP)", PAL["hirid"], False, None),
        (f"Amsterdam Trauma Primary (NL, n={au_n:,})",T["AmsterdamUMCdb_Trauma"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], T["AmsterdamUMCdb_Trauma"]["models"]["Modality_Adaptive_GBDT"]["ci95"], f" (NNE={NNE['AmsterdamUMCdb_Trauma']['NNE']:.2f})", PAL["aumc"], True, "AmsterdamUMCdb_Trauma__Modality_Adaptive_GBDT"),
        (f"Amsterdam ABI ICP 24 h (NL, n={au_icp_n})",END["AmsterdamUMCdb_Invasive_ICP_Subgroup"]["auroc_icp_monitored"], END["AmsterdamUMCdb_Invasive_ICP_Subgroup"].get("ci95_icp_monitored"), " (ABI ICP)", PAL["aumc"], False, None),
        ("Jinhua TBI Routed (CN, n=1,941)",           T["JinhuaNSICU_TBI_S06"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], T["JinhuaNSICU_TBI_S06"]["models"]["Modality_Adaptive_GBDT"]["ci95"], f" (NNE={NNE['JinhuaNSICU_TBI_S06']['NNE']:.2f})", PAL["jinhua"], True, "JinhuaNSICU_TBI_S06__Modality_Adaptive_GBDT"),
        (f"GCS ≤ 12: IMPACT-core LR (n={imp_m4['impact_core']['n_complete']})", imp_m4["impact_core"]["AUROC"], imp_m4["impact_core"].get("ci95"), f" (vs {imp_m4['impact_completecase_allcohort_AUROC']:.3f})", PAL["comp"], False, None),
    ]

    for x_g in [0.65, 0.75, 0.85, 0.95]:
        axd.axvline(x_g, color="white", lw=0.8, zorder=0)
    yd = np.arange(len(sens_rows))[::-1]
    axd.fill_betweenx([-0.6, len(sens_rows) - 0.4], 0.80, 0.99, color="#EBF2F7", alpha=0.40, zorder=0)
    axd.axvline(T["MIMIC_IV_IntVal"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], color=PAL["burgundy"], lw=0.80, ls="--", alpha=0.70, zorder=1)
    axd.axvline(0.995, color=PAL["stone"], lw=0.75, ls="-", zorder=2)

    for y_pos, (lab, v, ci_pair, sfx, col, is_bold, npz_k) in zip(yd, sens_rows):
        if npz_k is not None and npz_k in boot_npz.files:
            bvec = boot_npz[npz_k]
            lo_b, hi_b = float(np.percentile(bvec, 2.5)), float(np.percentile(bvec, 97.5))
            bsub = bvec[(bvec >= lo_b) & (bvec <= hi_b)]
            vp = axd.violinplot([bsub if len(bsub) >= 10 else bvec], positions=[y_pos], vert=False, widths=0.68, showmeans=False, showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.42); body.set_zorder(2)
        if ci_pair is not None and len(ci_pair) == 2:
            axd.errorbar(v, y_pos, xerr=[[max(0, v - ci_pair[0])], [max(0, ci_pair[1] - v)]],
                         fmt="none", ecolor=col, elinewidth=1.0, capsize=2.1, zorder=3)
        axd.scatter([v], [y_pos], marker="D" if is_bold else "s", color=col, edgecolor="white", linewidth=0.5, s=19 if is_bold else 14, zorder=4)
        axd.text(1.004, y_pos, f"{v:.3f}{sfx}", fontsize=4.55, va="center", ha="left",
                 fontweight="bold" if (is_bold or sfx) else "normal", color=col if is_bold else PAL["charcoal"], zorder=5)

    axd.set_yticks(yd)
    axd.set_yticklabels([r[0] for r in sens_rows], fontsize=4.85)
    for tick, (*_, col, is_bold, _) in zip(axd.get_yticklabels(), sens_rows):
        if is_bold:
            tick.set_fontweight("bold"); tick.set_color(col)
    axd.set_xlim(0.64, 1.26)
    axd.set_ylim(-0.6, len(sens_rows) - 0.4)
    axd.set_xticks([0.65, 0.75, 0.85, 0.95])
    axd.set_xlabel("AUROC & Operational NNE Across Six Cohorts", fontweight="bold", color=PAL["charcoal"])
    axd.spines[["top", "right"]].set_visible(False)

    if not (OUT / "Fig5_v5_1.png").exists():
        fig.savefig(OUT / "Fig5_v5_1.png", dpi=600)
    if not (OUT / "Fig5_v5_1.pdf").exists():
        fig.savefig(OUT / "Fig5_v5_1.pdf")
    plt.close(fig)
    print("[saved] Fig5_v5_1.png & .pdf")

if __name__ == "__main__":
    main()
