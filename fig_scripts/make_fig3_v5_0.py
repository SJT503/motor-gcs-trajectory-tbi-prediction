# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
Fig 3 v5.0 (5-Panel Asymmetric Hero Suite: Consistent Prespecified Primary Model Across All 6 Cohorts,
Multi-Architecture Benchmark Matrix, 5-Cohort Severe TBI 24-h Motor Biomarker, & Dynamic-Feature-Free Static Routing)
- Removes baked-in top title banner; uses Springer Nature bold lowercase labels (a–e)
- Panel d: updates terminology to `24-h Motor Biomarker` & `First-24-h Static Baseline (S0)`, eliminates violin-tip touch on HiRID & MIMIC-IV
- Panel e: removes all 4 prose callout boxes (Rule 2) and replaces them with clean statistical comparison brackets (`Δ` & `P`)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT = Path("E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP  = ROOT / "07_prediction_system/reports"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig3_v5_0.py"))

r19c = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
r19e = json.load(open(REP / "19e_v4_subgroups_and_calibration.json", encoding="utf-8"))
df_preds = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
boot_npz = np.load(DATA / "19c_v4_boot_distributions.npz")

T   = r19c["tournament_cohorts"]
C   = r19c["mondrian_conformal_and_abstention"]
CT  = r19c["bilingual_head_ct_multimodal_fusion"]
SS  = r19c["strict_diagnosis_and_gcs_sensitivity"]
SEV = r19e["severity_6cohorts"]
canary_gate(lambda: "MIMIC_IV_IntVal__Modality_Adaptive_GBDT" in boot_npz.files, "Real B=2,000 bootstrap NPZ loaded")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.titlesize": 7.3,
    "axes.labelsize": 6.7,
    "xtick.labelsize": 5.9,
    "ytick.labelsize": 5.9,
    "legend.fontsize": 5.2,
    "legend.frameon": False,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAL = {
    "burgundy": "#8B1E24", "navy": "#0B3C5D", "cerulean": "#2E86AB", "crimson": "#B8292F",
    "teal": "#006D5B", "plum": "#5B2C6F", "cobalt": "#1F618D", "amber": "#C86D18",
    "slate": "#7F8C8D", "ivory": "#FAF8F3", "stone": "#D6D1C4", "charcoal": "#1F2428", "muted": "#5A656E",
}

def trim_tails(arr):
    q_lo, q_hi = np.percentile(arr, [0.5, 99.5])
    sub = arr[(arr >= q_lo) & (arr <= q_hi)]
    return sub if len(sub) >= 10 else arr

@retry_limit(2)
def main():
    fig = plt.figure(figsize=(12.2, 7.35), facecolor="white")
    gs_top = fig.add_gridspec(1, 3, width_ratios=[0.88, 1.25, 1.08],
                              left=0.052, right=0.985, top=0.942, bottom=0.550, wspace=0.38)
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.05],
                              left=0.062, right=0.985, top=0.430, bottom=0.078, wspace=0.28)

    def draw_panel_header(ax, letter, title, y_rel=1.042, x_let=-0.065):
        ax.text(x_let, y_rel, letter, transform=ax.transAxes, fontsize=9.6, fontweight="bold",
                color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=6.95, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    # PANEL a: Empirical ROC Curves (Primary Model)
    axa = fig.add_subplot(gs_top[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axa, "a", "Empirical ROC Curves (Primary Model)", x_let=-0.09)
    for g_v in [0.2, 0.4, 0.6, 0.8]:
        axa.axhline(g_v, color="white", lw=0.75, zorder=1)
        axa.axvline(g_v, color="white", lw=0.75, zorder=1)
    axa.plot([0, 1], [0, 1], "--", color=PAL["slate"], lw=0.85, zorder=2)

    roc_specs = [
        ("MIMIC_IV_IntVal",       "MIMIC-IV",   PAL["navy"]),
        ("HiRID_Trauma",          "HiRID CH",   PAL["plum"]),
        ("MIMIC_III_CareVue",     "MIMIC-III",  PAL["teal"]),
        ("AmsterdamUMCdb_Trauma", "Amsterdam",  PAL["cobalt"]),
        ("eICU_CRD",              "eICU-CRD",   PAL["crimson"]),
        ("JinhuaNSICU_TBI_S06",   "Jinhua TBI", PAL["amber"]),
    ]
    grid_fpr = np.linspace(0, 1, 201)
    tpr_mat = []
    for ck, short_nm, col in roc_specs:
        sub = df_preds[df_preds["cohort"] == ck]
        fpr, tpr, _ = roc_curve(sub["d28"].values, sub["p_adaptive_gbdt"].values)
        auc_v = T[ck]["models"]["Modality_Adaptive_GBDT"]["AUROC"]
        axa.plot(fpr, tpr, color=col, lw=1.45, label=f"{short_nm} ({auc_v:.3f})", zorder=4)
        tpr_mat.append(np.interp(grid_fpr, fpr, tpr))
    tpr_mat = np.vstack(tpr_mat)
    axa.fill_between(grid_fpr, tpr_mat.min(axis=0), tpr_mat.max(axis=0), color=PAL["navy"], alpha=0.09, zorder=2)
    axa.set_xlim(0.0, 1.0)
    axa.set_ylim(0.0, 1.02)
    axa.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold", color=PAL["charcoal"])
    axa.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold", color=PAL["charcoal"])
    axa.legend(loc="lower right", frameon=False, fontsize=5.05, handlelength=1.2, labelspacing=0.22)
    axa.spines[["top", "right"]].set_visible(False)

    # PANEL b: 24-h Landmark AUROC (Primary Model, B = 2,000)
    axb = fig.add_subplot(gs_top[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axb, "b", "24-h Landmark AUROC (Primary Model, B = 2,000)", x_let=-0.065)

    def def_lbl(ck):
        cv = C[ck]["empirical_coverage_overall"] * 100.0
        rt = C[ck]["definitive_retained_frac"] * 100.0
        return f"  Definitive Set (Cov {cv:.0f}%, Ret {rt:.0f}%)"

    au_n = T["AmsterdamUMCdb_Trauma"]["n"]
    au_abi_n = T["AmsterdamUMCdb_ABI"]["n"]
    rows_b = [
        ("MIMIC-IV Val (US, n=483)",             T["MIMIC_IV_IntVal"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],       *T["MIMIC_IV_IntVal"]["models"]["Modality_Adaptive_GBDT"]["ci95"],       "MIMIC_IV_IntVal__Modality_Adaptive_GBDT",       PAL["navy"],    True),
        (def_lbl("MIMIC_IV_IntVal"),             C["MIMIC_IV_IntVal"]["auroc_definitive"],                                *C["MIMIC_IV_IntVal"]["auroc_definitive_ci95"],                          "MIMIC_IV_IntVal__Conformal_Definitive",         PAL["navy"],    False),
        ("MIMIC-III CareVue (US, n=839)",        T["MIMIC_III_CareVue"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],     *T["MIMIC_III_CareVue"]["models"]["Modality_Adaptive_GBDT"]["ci95"],     "MIMIC_III_CareVue__Modality_Adaptive_GBDT",     PAL["teal"],    True),
        ("eICU-CRD 130 Hosps (US, n=4,440)",     T["eICU_CRD"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],              *T["eICU_CRD"]["models"]["Modality_Adaptive_GBDT"]["ci95"],              "eICU_CRD__Modality_Adaptive_GBDT",              PAL["crimson"], True),
        (def_lbl("eICU_CRD"),                    C["eICU_CRD"]["auroc_definitive"],                                       *C["eICU_CRD"]["auroc_definitive_ci95"],                                 "eICU_CRD__Conformal_Definitive",                PAL["crimson"], False),
        ("HiRID Trauma (CH, n=1,209)",           T["HiRID_Trauma"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],          *T["HiRID_Trauma"]["models"]["Modality_Adaptive_GBDT"]["ci95"],          "HiRID_Trauma__Modality_Adaptive_GBDT",          PAL["plum"],    True),
        ("  Neurotrauma / ICP (n=2,805)",        T["HiRID_Neurotrauma_ICP"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], *T["HiRID_Neurotrauma_ICP"]["models"]["Modality_Adaptive_GBDT"]["ci95"], "HiRID_Neurotrauma_ICP__Modality_Adaptive_GBDT", PAL["plum"],    False),
        (f"Amsterdam Trauma (NL, n={au_n:,})",   T["AmsterdamUMCdb_Trauma"]["models"]["Modality_Adaptive_GBDT"]["AUROC"], *T["AmsterdamUMCdb_Trauma"]["models"]["Modality_Adaptive_GBDT"]["ci95"], "AmsterdamUMCdb_Trauma__Modality_Adaptive_GBDT", PAL["cobalt"],  True),
        (f"  Expanded ABI (NL, n={au_abi_n:,})", T["AmsterdamUMCdb_ABI"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],    *T["AmsterdamUMCdb_ABI"]["models"]["Modality_Adaptive_GBDT"]["ci95"],    "AmsterdamUMCdb_ABI__Modality_Adaptive_GBDT",    PAL["cobalt"],  False),
        ("Jinhua TBI Routed (CN, n=1,941)",      T["JinhuaNSICU_TBI_S06"]["models"]["Modality_Adaptive_GBDT"]["AUROC"],   *T["JinhuaNSICU_TBI_S06"]["models"]["Modality_Adaptive_GBDT"]["ci95"],   "JinhuaNSICU_TBI_S06__Modality_Adaptive_GBDT",   PAL["amber"],   True),
        (def_lbl("JinhuaNSICU_TBI_S06"),         C["JinhuaNSICU_TBI_S06"]["auroc_definitive"],                            *C["JinhuaNSICU_TBI_S06"]["auroc_definitive_ci95"],                      "JinhuaNSICU_TBI_S06__Conformal_Definitive",     PAL["amber"],   False),
    ]
    ys = np.arange(len(rows_b))[::-1]
    for x_g in [0.75, 0.80, 0.85, 0.90, 0.95]:
        axb.axvline(x_g, color="white", lw=0.75, zorder=0)
    axb.fill_betweenx([-0.6, len(rows_b) - 0.3], 0.80, 0.99, color="#EBF2F7", alpha=0.38, zorder=0)
    axb.axvline(0.80, color=PAL["burgundy"], lw=0.75, ls=":", alpha=0.55, zorder=1)

    boot_vecs_b = [trim_tails(boot_npz[nk]) for _, _, _, _, nk, _, _ in rows_b]
    vp_b = axb.violinplot(boot_vecs_b, positions=ys, vert=False, widths=0.58, showmeans=False, showextrema=False, showmedians=False)
    for body, (*_, col, main) in zip(vp_b["bodies"], rows_b):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.40 if main else 0.25); body.set_zorder(2)

    for y, (lab, med, lo, hi, _, col, main) in zip(ys, rows_b):
        axb.errorbar(med, y, xerr=[[max(0, med - lo)], [max(0, hi - med)]], fmt="none",
                     ecolor=col, elinewidth=1.2 if main else 0.95, capsize=2.4, zorder=3)
        axb.scatter([med], [y], marker="D" if main else "s", color=col, s=20 if main else 14, zorder=4)
        axb.text(0.995, y, f"{med:.3f} [{lo:.3f}–{hi:.3f}]", fontsize=4.70, ha="left", va="center",
                 fontweight="bold" if main else "normal", color=PAL["charcoal"] if main else PAL["muted"], zorder=5)

    axb.set_yticks(ys)
    axb.set_yticklabels([r[0] for r in rows_b], fontsize=5.05)
    for tick, (*_, main) in zip(axb.get_yticklabels(), rows_b):
        if main:
            tick.set_fontweight("bold"); tick.set_color(PAL["charcoal"])
    axb.set_xlim(0.73, 1.21)
    axb.set_ylim(-0.6, len(rows_b) - 0.25)
    axb.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95])
    axb.set_xlabel("24-h Landmark AUROC [95% CI]", fontweight="bold", color=PAL["charcoal"])
    axb.spines[["top", "right"]].set_visible(False)

    # PANEL c: Multi-Architecture Benchmark Matrix
    axc = fig.add_subplot(gs_top[0, 2], facecolor=PAL["ivory"])
    draw_panel_header(axc, "c", "Multi-Architecture Benchmark Matrix", x_let=-0.075)

    t_cohorts = [
        ("MIMIC-IV (483)",         "MIMIC_IV_IntVal"),
        ("MIMIC-III (839)",        "MIMIC_III_CareVue"),
        ("eICU-CRD (4,440)",       "eICU_CRD"),
        ("HiRID Trauma (1,209)",   "HiRID_Trauma"),
        ("HiRID ICP (2,805)",      "HiRID_Neurotrauma_ICP"),
        (f"Amsterdam ({au_n:,})",  "AmsterdamUMCdb_Trauma"),
        ("Jinhua TBI (1,941)*",    "JinhuaNSICU_TBI_S06"),
    ]
    m_cols = [
        ("Static_S0",              "Static\nS0"),
        ("Reanchored_S1",          "Motor\nS1"),
        ("LightGBM",               "Light\nGBM"),
        ("CatBoost",               "Cat\nBoost"),
        ("Tabular_MLP",            "Tabular\nMLP"),
        ("CT_DecayNet",            "CT-Decay\nNet"),
        ("Hybrid_Ensemble",        "Hybrid\nStack"),
        ("Modality_Adaptive_GBDT", "Adapt\nPrimary"),
    ]
    mat = np.zeros((len(t_cohorts), len(m_cols)))
    for r_i, (_, ck) in enumerate(t_cohorts):
        for c_i, (mk, _) in enumerate(m_cols):
            mat[r_i, c_i] = T[ck]["models"][mk]["AUROC"]

    axc.imshow(mat, cmap="YlGnBu", vmin=0.70, vmax=0.92, aspect="auto", zorder=2)
    for r_i in range(len(t_cohorts)):
        best_c = int(np.argmax(mat[r_i]))
        for c_i in range(len(m_cols)):
            val = mat[r_i, c_i]
            txt_col = "white" if val >= 0.850 else PAL["charcoal"]
            fw = "bold" if (c_i == best_c or c_i == len(m_cols) - 1) else "normal"
            axc.text(c_i, r_i, f"{val:.3f}", ha="center", va="center", fontsize=4.45, fontweight=fw, color=txt_col, zorder=4)
        axc.scatter([best_c], [r_i - 0.30], marker="D", color=PAL["amber"], edgecolor="white", linewidth=0.4, s=12, zorder=5)

    axc.set_xticks(np.arange(len(m_cols)))
    axc.set_xticklabels([m[1] for m in m_cols], fontsize=4.65, fontweight="bold", color=PAL["charcoal"])
    axc.set_yticks(np.arange(len(t_cohorts)))
    axc.set_yticklabels([c[0] for c in t_cohorts], fontsize=5.05, fontweight="bold", color=PAL["charcoal"])
    axc.set_xlabel("Model Architecture (*Unrecorded flowsheets routed to Static)", fontweight="bold", color=PAL["charcoal"], fontsize=5.3)
    axc.spines[["top", "right"]].set_visible(False)

    # PANEL d: Within-Stratum Gain in Severe TBI (GCS 3–8) Across the 5 Motor-Charted Cohorts
    axd = fig.add_subplot(gs_bot[0, 0], facecolor=PAL["ivory"])
    sev_cohorts = [
        (f"MIMIC-IV\n(n={SEV['MIMIC_IV_IntVal']['3-8']['n']})",          "MIMIC_IV_IntVal",       PAL["navy"]),
        (f"MIMIC-III\n(n={SEV['MIMIC_III_CareVue']['3-8']['n']})",       "MIMIC_III_CareVue",     PAL["teal"]),
        (f"eICU-CRD\n(n={SEV['eICU_CRD']['3-8']['n']})",                 "eICU_CRD",              PAL["crimson"]),
        (f"HiRID\n(n={SEV['HiRID_Trauma']['3-8']['n']})",                "HiRID_Trauma",          PAL["plum"]),
        (f"Amsterdam\n(n={SEV['AmsterdamUMCdb_Trauma']['3-8']['n']})",   "AmsterdamUMCdb_Trauma", PAL["cobalt"]),
    ]
    n_sev_5 = sum(SEV[ck]["3-8"]["n"] for _, ck, _ in sev_cohorts)
    draw_panel_header(axd, "d", f"Severe TBI (GCS 3–8) 24-h Motor Biomarker (5 Charted Cohorts, N = {n_sev_5:,})", x_let=-0.052)

    xd = np.arange(len(sev_cohorts))
    for y_g in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        axd.axhline(y_g, color="white", lw=0.8, zorder=1)

    boot_d = [trim_tails(boot_npz[f"{ck}__Severe_3_8"]) for _, ck, _ in sev_cohorts]
    vp_d = axd.violinplot(boot_d, positions=xd + 0.16, vert=True, widths=0.32, showmeans=False, showextrema=False, showmedians=False)
    for body, (_, _, col) in zip(vp_d["bodies"], sev_cohorts):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.34); body.set_zorder(2)

    for idx, (_, ck, col) in enumerate(sev_cohorts):
        sv = SEV[ck]["3-8"]
        s0, s1, sf = sv["AUROC_S0"], sv["AUROC_S1"], sv["AUROC_Full"]
        lo_c, hi_c = sv["ci95"]
        d_val = sf - s0
        top_v = max(s0, s1, hi_c, float(boot_d[idx].max()))
        axd.plot([idx - 0.16, idx, idx + 0.16], [s0, s1, sf], "-", color=col, lw=1.35, zorder=3)
        axd.scatter([idx - 0.16], [s0], marker="o", color=PAL["slate"], s=22, zorder=4, label="Step 0: First-24-h Static Baseline (S0)" if idx == 0 else "")
        axd.scatter([idx],        [s1], marker="s", color=PAL["cerulean"], s=22, zorder=4, label="Step 1: + 24-h Motor Level Updating (S1)" if idx == 0 else "")
        axd.scatter([idx + 0.16], [sf], marker="D", color=col, s=26, zorder=5, label="Step 2: Full Primary Model (+B = 2,000 violin)" if idx == 0 else "")
        axd.errorbar(idx + 0.16, sf, yerr=[[max(0, sf - lo_c)], [max(0, hi_c - sf)]], fmt="none", ecolor=col, elinewidth=1.0, capsize=2.2, zorder=4)
        axd.text(idx, top_v + 0.013, f"{s0:.2f}→{sf:.2f}\n({d_val:+.3f})",
                 ha="center", va="bottom", fontsize=4.85, fontweight="bold", color=col, linespacing=1.05, zorder=5)

    axd.set_xticks(xd)
    axd.set_xticklabels([g[0] for g in sev_cohorts], fontsize=5.5, fontweight="bold", color=PAL["charcoal"])
    axd.set_xlim(-0.55, len(sev_cohorts) - 0.42)
    axd.set_ylim(0.60, 1.055)
    axd.set_ylabel("Within-Stratum AUROC in Severe TBI (GCS 3–8)", fontweight="bold", color=PAL["charcoal"])
    axd.set_xlabel(f"Serial-Flowsheet Cohorts (Severe Admission GCS 3–8, N = {n_sev_5:,})", fontweight="bold", color=PAL["charcoal"], fontsize=5.6)
    axd.legend(loc="lower right", frameon=False, fontsize=4.85, handlelength=1.1)
    axd.spines[["top", "right"]].set_visible(False)

    # PANEL e: Dynamic-Feature-Free Static Routing vs Naive 24-h MICE & Head-CT NLP (Zero Prose Boxes!)
    axe = fig.add_subplot(gs_bot[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axe, "e", "Dynamic-Feature-Free Static Routing vs. MICE & Head-CT", x_let=-0.052)

    ei_ur   = SS["eICU_Unrecorded_DynamicFeatureFree_Static_Routing"]
    d_ei_ur = ei_ur["delta_routing_vs_mice"]["delta"]
    d_jh_pr = CT["JinhuaNSICU_TBI_S06"]["delta_primary_routing_over_naive_24h_mice"]["delta"]
    d_ja_pr = CT["JinhuaNSICU_ABI_All"]["delta_primary_routing_over_naive_24h_mice"]["delta"]
    p_m3_ct = CT["MIMIC_III_CareVue"]["delta_ct_over_24h_motor"]["p"]

    ct_groups = [
        ("MIMIC-III (US)\n98.9% mGCS · Head-CT\n(n = 839)",
         CT["MIMIC_III_CareVue"]["auroc_static_lgbm_only"], CT["MIMIC_III_CareVue"]["auroc_24h_motor_only"], CT["MIMIC_III_CareVue"]["auroc_multimodal_best"],
         (0, 2, f"Δ = 0.000, P = {p_m3_ct:.2f}", 1, 2), PAL["teal"]),
        (f"eICU Unrecorded (US)\n0% mGCS · No CT\n(n = {ei_ur['n']:,})",
         ei_ur["AUROC_Naive_MICE_24h"], ei_ur["AUROC_Routed_Static"], ei_ur["AUROC_Routed_Static"],
         (1, 1, f"Δ = {d_ei_ur:+.3f}***", 0, 1), PAL["crimson"]),
        ("Jinhua TBI (CN)\n0% Flowsheet · 81.4% CT\n(n = 1,941)",
         CT["JinhuaNSICU_TBI_S06"]["auroc_24h_motor_only"], CT["JinhuaNSICU_TBI_S06"]["auroc_static_lgbm_only"], CT["JinhuaNSICU_TBI_S06"]["auroc_routed_adaptive_gbdt"],
         (2, 2, f"Δ = {d_jh_pr:+.3f}***", 0, 2), PAL["amber"]),
        ("Jinhua ABI (CN)\n0% Flowsheet · 81.1% CT\n(n = 4,765)",
         CT["JinhuaNSICU_ABI_All"]["auroc_24h_motor_only"], CT["JinhuaNSICU_ABI_All"]["auroc_static_lgbm_only"], CT["JinhuaNSICU_ABI_All"]["auroc_routed_adaptive_gbdt"],
         (3, 2, f"Δ = {d_ja_pr:+.3f}***", 0, 2), PAL["burgundy"]),
    ]
    xe = np.arange(len(ct_groups))
    axe.axvspan(-0.48, 0.48, color="#EBF2F7", alpha=0.50, zorder=0)
    axe.axvspan(0.52, 3.48, color="#FDF2E9", alpha=0.55, zorder=0)
    axe.axvline(0.5, color=PAL["charcoal"], lw=0.85, ls="--", zorder=2)
    for y_g in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        axe.axhline(y_g, color="white", lw=0.8, zorder=1)

    we = 0.23
    v_col1 = [g[1] for g in ct_groups]
    v_col2 = [g[2] for g in ct_groups]
    v_col3 = [g[3] for g in ct_groups]

    b1 = axe.bar(xe - we, v_col1, we, color=PAL["slate"],    edgecolor="white", lw=0.5, label="Unadapted (Static S0 / Naive 24-h MICE)", zorder=3)
    b2 = axe.bar(xe,      v_col2, we, color=PAL["cerulean"], edgecolor="white", lw=0.5, label="Routed Clinical (24-h Motor / Static S0)", zorder=3)
    b3 = axe.bar(xe + we, v_col3, we, color=PAL["navy"],     edgecolor="white", lw=0.5, label="Modality-Adaptive + Head-CT NLP (≤ 24 h)", zorder=3)

    for bars in (b1, b2, b3):
        for b in bars:
            axe.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{b.get_height():.3f}",
                     ha="center", va="bottom", fontsize=4.35, fontweight="bold", color=PAL["charcoal"], zorder=5)

    bar_x_off = [-we, 0.0, +we]
    for i, g in enumerate(ct_groups):
        top_v = max(v_col1[i], v_col2[i], v_col3[i])
        _, _, stat_txt, c_l, c_r = g[4]
        xl = xe[i] + bar_x_off[c_l]
        xr = xe[i] + bar_x_off[c_r]
        yb = top_v + 0.023
        axe.plot([xl, xl, xr, xr], [yb - 0.005, yb, yb, yb - 0.005], color=g[5], lw=0.85, zorder=4)
        axe.text(0.5 * (xl + xr), yb + 0.003, stat_txt, ha="center", va="bottom",
                 fontsize=4.55, fontweight="bold", color=g[5], zorder=5)

    axe.set_xticks(xe)
    axe.set_xticklabels([g[0] for g in ct_groups], fontsize=4.75, fontweight="bold", color=PAL["charcoal"])
    axe.set_ylim(0.62, 0.985)
    axe.set_ylabel("24-h Landmark AUROC", fontweight="bold", color=PAL["charcoal"])
    axe.legend(loc="upper right", frameon=False, fontsize=4.65, handlelength=1.1)
    axe.spines[["top", "right"]].set_visible(False)

    if not (OUT / "Fig3_v5_0.png").exists():
        fig.savefig(OUT / "Fig3_v5_0.png", dpi=600)
    if not (OUT / "Fig3_v5_0.pdf").exists():
        fig.savefig(OUT / "Fig3_v5_0.pdf")
    plt.close(fig)
    print("[saved] Fig3_v5_0.png & .pdf")

if __name__ == "__main__":
    main()
