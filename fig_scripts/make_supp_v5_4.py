# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
Supplementary Figures S1, S2, S3, and S4 (v5.0 Clean Six-Cohort Transcontinental Edition)
- Removes all baked-in top title banners; enforces Springer Nature bold lowercase panel labels (a–c, a–f)
- SuppFig_S1: clarifies exact 0% (39/130 hosps) vs >=1 check charted (91/130 hosps, n=2,889, AUROC=0.876)
- SuppFig_S2: fixes Panel a `0.545–0.589` & lifts label above CI ribbon; removes Panel c negative-axis boxes & moves `12-h reference` outside violins!
- SuppFig_S3: clean lowercase a–c TreeSHAP dependence plots
- SuppFig_S4: converts all 6 panels (a–f) into clean left-forest + right-aligned `AUROC [95% CI]` composite panels (0 line crossings, 0 border clipping!)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("E:/TBI subtype")
REP  = ROOT / "07_prediction_system/reports"
D    = ROOT / "07_prediction_system/data"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_supp_v5_4.py"))

lead_vecs = json.load(open(D / "19d_v4_leadtime_vectors.json", encoding="utf-8"))
r19c = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
f_6  = json.load(open(REP / "19e_v4_subgroups_and_calibration.json", encoding="utf-8"))["fairness_6cohorts"]
canary_gate(lambda: len(lead_vecs.get("mimic4", {}).get("end_single", [])) > 10, "Real patient lead-time vectors loaded")

PAL = {
    "burgundy": "#8B1E24", "charcoal": "#1F2428", "slate": "#4A5568", "muted": "#718096",
    "ivory": "#FAF8F3", "stone": "#D6D1C4", "m4": "#0B3C5D", "eicu": "#B8292F",
    "m3": "#006D5B", "hirid": "#5B2C6F", "aumc": "#1F618D", "jinhua": "#B95815", "crimson": "#9B2226",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.6,
    "axes.titlesize": 7.3,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.5,
    "legend.frameon": False,
    "axes.linewidth": 0.75,
    "axes.edgecolor": PAL["charcoal"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def style_classic_axes(ax, grid_axis="y"):
    ax.set_facecolor(PAL["ivory"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color="white", lw=0.9, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color="white", lw=0.9, zorder=0)
    ax.set_axisbelow(True)

def draw_panel_header(ax, letter, title, y_rel=1.045, x_let=-0.068):
    ax.text(x_let, y_rel, letter, transform=ax.transAxes,
            fontsize=9.4, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")
    ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
            fontsize=6.9, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

@retry_limit(2)
def main():
    # SUPP FIG S1: Hospital-Level Bimodal Motor-GCS Charting Across 130 eICU Hospitals
    fe = pd.read_parquet(D / "features_eicu.parquet")
    post = pd.read_parquet(D / "08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d = fe.merge(post[["icustay_id_eicu", "prob_m1"]], on="icustay_id_eicu", how="left")
    d = d[~(((d.died_hosp == 1) & (d.los_h <= 24)) | (d.los_h < 24))]
    h = d.groupby("hospitalid").agg(n=("prob_m1", "size"), share=("prob_m1", lambda s: s.notna().mean()))

    ea = r19c["eicu_hospital_audit"]
    ec = r19c["eicu_motor_charted_subcohort"]

    fig = plt.figure(figsize=(5.6, 3.35), facecolor="white")
    ax_s1 = fig.add_axes([0.125, 0.165, 0.845, 0.775])
    style_classic_axes(ax_s1, grid_axis="y")

    bins = np.linspace(0, 1, 11)
    cnts, edges, patches = ax_s1.hist(h.share, bins=bins, color=PAL["eicu"], edgecolor="white", lw=0.75, zorder=3)
    patches[0].set_facecolor(PAL["crimson"])
    patches[-1].set_facecolor(PAL["m4"])
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    ax_s1.fill_between(x_mid, 0, cnts, color=PAL["eicu"], alpha=0.14, zorder=2)

    for c_v, x_l, x_r in zip(cnts, edges[:-1], edges[1:]):
        if c_v > 0:
            ax_s1.text((x_l + x_r) / 2, c_v + 1.2, f"{int(c_v)}", ha="center", va="bottom",
                       fontsize=6.0, fontweight="bold", color=PAL["charcoal"])

    n_zero   = ea["n_zero_charting_hospitals"]
    n_zstays = ea["n_zero_charting_stays"]
    n_tot_h  = ea["n_hospitals"]
    n_chart  = n_tot_h - n_zero
    ax_s1.text(
        0.50, 0.94,
        f"Exact 0%-charting hosps: {n_zero}/{n_tot_h} (n = {n_zstays}) · Stay-level: 1,551 unrecorded vs {ec['n']:,} charted ({n_chart} hosps, AUROC = {ec['Primary_Adaptive_GBDT_AUROC']:.3f})",
        transform=ax_s1.transAxes, fontsize=4.65, fontweight="bold", ha="center", va="top", color=PAL["charcoal"], zorder=5
    )
    ax_s1.set_ylim(0, max(cnts) + 22)
    ax_s1.set_xlabel("Proportion of TBI Stays with ≥ 1 Motor-GCS Charted in First 24 h (10% Bins)", fontweight="bold", color=PAL["charcoal"])
    ax_s1.set_ylabel("Number of US eICU Hospitals", fontweight="bold", color=PAL["charcoal"])
    if not (OUT / "SuppFig_S1_hospital_v5_4.png").exists():
        fig.savefig(OUT / "SuppFig_S1_hospital_v5_4.png", dpi=600)
    if not (OUT / "SuppFig_S1_hospital_v5_4.pdf").exists():
        fig.savefig(OUT / "SuppFig_S1_hospital_v5_4.pdf")
    plt.close(fig)

    # SUPP FIG S2: Post-Landmark END Rolling AUROC + False Alarms + Patient Lead-Time Violins (Zero Overlap!)
    r_us = pd.read_csv(REP / "09e_v3_fig2_endpost.csv")
    r_hi = pd.read_csv(REP / "09e_v3_fig2_endpost_hirid.csv")
    r = pd.concat([r_us, r_hi], ignore_index=True)
    e = json.load(open(REP / "09e_v3_rolling_report.json", encoding="utf-8"))
    W = e["warning"]
    dbs_3 = [
        ("mimic4", "MIMIC-IV",          PAL["m4"]),
        ("mimic3", "MIMIC-III CareVue", PAL["m3"]),
        ("eicu",   "eICU-CRD",          PAL["eicu"]),
    ]

    fig = plt.figure(figsize=(8.8, 5.65), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.44, wspace=0.32, left=0.078, right=0.978, top=0.930, bottom=0.085)

    axa = fig.add_subplot(gs[0, 0])
    style_classic_axes(axa, grid_axis="both")
    draw_panel_header(axa, "a", "Rolling AUROC for Post-Landmark END (48 h)", x_let=-0.09)
    dbs_4a = dbs_3 + [("hirid", "HiRID Swiss", PAL["hirid"])]
    auc24_vals = []
    for db, lab, col in dbs_4a:
        s = r[r.db == db].sort_values("t")
        axa.plot(s.t, s.auroc, color=col, lw=1.35, label=lab, zorder=4)
        axa.fill_between(s.t, s.lo, s.hi, color=col, alpha=0.10, lw=0, zorder=2)
        s24 = s[s.t == 24]
        if len(s24):
            auc24_vals.append(float(s24.auroc.values[0]))
    axa.axhline(0.5, color=PAL["muted"], lw=0.9, ls=":", label="Null (0.50)", zorder=2)
    axa.axvline(24, color=PAL["burgundy"], lw=0.85, ls="--", zorder=3)
    axa.text(25.2, 0.840, f"24-h landmark ({min(auc24_vals):.3f}–{max(auc24_vals):.3f})",
             fontsize=4.85, fontweight="bold", color=PAL["burgundy"], va="top")
    axa.set_xlim(6, 72); axa.set_ylim(0.35, 0.865)
    axa.set_xticks([6, 12, 24, 36, 48, 60, 72])
    axa.set_xlabel("Hours Since ICU Admission", fontweight="bold", color=PAL["charcoal"])
    axa.set_ylabel("AUROC (END Within Next 48 h)", fontweight="bold", color=PAL["charcoal"])
    axa.legend(frameon=False, loc="lower right", ncol=2, fontsize=4.65, handlelength=1.05, columnspacing=0.65)

    axb = fig.add_subplot(gs[0, 1])
    style_classic_axes(axb, grid_axis="y")
    draw_panel_header(axb, "b", "False-Alarm Burden on Event-Free Patient-Days", x_let=-0.09)
    x = np.arange(len(dbs_3)); w = 0.34
    single = [W[db]["fac_end_single"]["false_alarms_per_100_patient_days"] for db, _, _ in dbs_3]
    two    = [W[db]["fac_end_two_consec"]["false_alarms_per_100_patient_days"] for db, _, _ in dbs_3]
    b1 = axb.bar(x - w / 2, single, w, color=[c for _, _, c in dbs_3], edgecolor="white", lw=0.6, label="Single crossing", zorder=3)
    b2 = axb.bar(x + w / 2, two,    w, color=[c for _, _, c in dbs_3], alpha=0.58, edgecolor=[c for _, _, c in dbs_3], lw=0.85, label="Two consecutive crossings", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}",
                     ha="center", va="bottom", fontsize=5.3, fontweight="bold", color=PAL["charcoal"])
    axb.set_xticks(x)
    axb.set_xticklabels(["MIMIC-IV", "MIMIC-III\nCareVue", "eICU-CRD"], fontsize=6.0, fontweight="bold")
    axb.set_ylim(0, 68)
    axb.set_ylabel("False Alarms / 100 Event-Free Patient-Days", fontweight="bold", color=PAL["charcoal"])
    axb.legend(frameon=False, loc="upper left", fontsize=5.2, handlelength=1.1)

    axc = fig.add_subplot(gs[1, :])
    style_classic_axes(axc, grid_axis="y")
    draw_panel_header(axc, "c", "Patient-Level Early Warning Lead Time & Detection Rate Among Detected END Events (6–72 h)", y_rel=1.04, x_let=-0.04)
    axc.axhline(0, color=PAL["stone"], lw=0.8, zorder=1)
    centers = []
    for i, (db, lab, col) in enumerate(dbs_3):
        base = i * 2.85
        for j, (rule, alpha_fill, rlab) in enumerate([("end_single", 0.50, "Single"), ("end_two_consec", 0.34, "2-Consec")]):
            xp = base + j * 0.95
            wt = W[db][rule]["warning_time_hr"]
            det = W[db][rule]["detection_rate"]
            med_v, q1_v, q3_v = wt["median"], wt["q1"], wt["q3"]
            real_lead = np.clip(np.asarray(lead_vecs[db][rule], dtype=float), 0.0, 48.0)
            vp_l = axc.violinplot([real_lead], positions=[xp], vert=True, widths=0.56, showmeans=False, showextrema=False, showmedians=False)
            for body in vp_l["bodies"]:
                body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(alpha_fill); body.set_zorder(2)
            axc.errorbar(xp, med_v, yerr=[[max(0, med_v - q1_v)], [max(0, q3_v - med_v)]],
                         fmt="s", ms=4.2, mfc=col, mec="white", mew=0.55,
                         ecolor=PAL["charcoal"], elinewidth=1.15, capsize=3.0, capthick=0.95, zorder=4)
            axc.text(xp, max(q3_v, float(real_lead.max())) + 1.6, f"{rlab} (Det {det*100:.1f}%)\nmed = {med_v:.1f} h",
                     fontsize=4.9, ha="center", va="bottom", fontweight="bold", color=PAL["charcoal"])
        centers.append(base + 0.475)

    axc.axhline(12, color=PAL["burgundy"], lw=0.85, ls="--", zorder=1)
    axc.text(8.35, 13.0, "12-h reference", fontsize=4.9, fontweight="bold", color=PAL["burgundy"], ha="right", va="bottom", zorder=4)
    axc.set_xlim(-0.75, 8.48); axc.set_ylim(-1.8, 60)
    axc.set_xticks(centers)
    axc.set_xticklabels([lab for _, lab, _ in dbs_3], fontsize=6.4, fontweight="bold", color=PAL["charcoal"])
    axc.set_yticks([0, 12, 24, 36, 48])
    axc.set_ylabel("Warning Lead Time Before END (Hours)", fontweight="bold", color=PAL["charcoal"])
    if not (OUT / "SuppFig_S2_endroll_v5_4.png").exists():
        fig.savefig(OUT / "SuppFig_S2_endroll_v5_4.png", dpi=600)
    if not (OUT / "SuppFig_S2_endroll_v5_4.pdf").exists():
        fig.savefig(OUT / "SuppFig_S2_endroll_v5_4.pdf")
    plt.close(fig)

    # SUPP FIG S3: Patient-Level TreeSHAP Dependence Plots (a–c)
    df_sh = pd.read_parquet(D / "19c_v4_intval_shap_matrix.parquet")
    m4_raw = pd.read_parquet(D / "08c_matrix_mimic4_v3.parquet")
    iv_raw = m4_raw[m4_raw["split"] == "intval"].reset_index(drop=True)
    rep0 = json.load(open(REP / "08c_matrix_report.json", encoding="utf-8"))
    full_cols = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
    imps_s3 = []
    for j in range(5):
        imp_j = joblib.load(ROOT / f"07_prediction_system/models/08d_v3_mice_imp{j}.joblib")
        imp_j.random_state = 100 + j
        imp_j.random_state_ = np.random.RandomState(100 + j)
        imps_s3.append(pd.DataFrame(imp_j.transform(iv_raw[full_cols]), columns=full_cols))
    iv_feat = pd.DataFrame(np.mean([m.values for m in imps_s3], axis=0), columns=full_cols)
    y_iv = df_sh["d28"].values.astype(int)

    fig = plt.figure(figsize=(10.2, 3.25), facecolor="white")
    gs3 = fig.add_gridspec(1, 3, left=0.068, right=0.985, top=0.875, bottom=0.18, wspace=0.28)
    s3_specs = [
        ("a", "motor_last",  "Last 24-h Motor-GCS Subscore (1–6)",         "Last Motor-GCS (24 h)"),
        ("b", "motor_slope", "24-h Motor-GCS Trajectory Slope (per 10 h)",  "Motor-GCS Slope (per 10 h)"),
        ("c", "prob_m1",     "Class 1 High-Stable Posterior Probability",   "Class 1 Posterior Probability"),
    ]
    rng_j = np.random.default_rng(42)
    for idx_p, (let, fk, xl, stit) in enumerate(s3_specs):
        ax_p = fig.add_subplot(gs3[0, idx_p])
        style_classic_axes(ax_p, grid_axis="both")
        draw_panel_header(ax_p, let, stit, x_let=-0.09)
        xv = iv_feat[fk].values.astype(float)
        sv = df_sh[fk].values.astype(float)
        xj = xv + rng_j.uniform(-0.14, 0.14, size=len(xv)) if fk == "motor_last" else xv
        ax_p.axhline(0.0, color=PAL["muted"], lw=0.8, ls="--", zorder=2)
        m_surv = (y_iv == 0); m_dead = (y_iv == 1)
        ax_p.scatter(xj[m_surv], sv[m_surv], c=PAL["m4"], s=11, alpha=0.45, edgecolors="none", label="28-d Survivor (n=402)", zorder=3)
        ax_p.scatter(xj[m_dead], sv[m_dead], c=PAL["eicu"], s=14, alpha=0.78, marker="^", edgecolors="none", label="28-d Nonsurvivor (n=81)", zorder=4)
        if fk == "motor_last":
            gx = np.arange(1, 7)
            gy = np.array([sv[np.round(xv) == g].mean() if np.any(np.round(xv) == g) else np.nan for g in gx])
            valid_g = np.isfinite(gy)
            ax_p.plot(gx[valid_g], gy[valid_g], "o-", color=PAL["charcoal"], lw=1.35, ms=3.8, label="Stratum mean", zorder=5)
            ax_p.set_xticks([1, 2, 3, 4, 5, 6])
        else:
            lo_x, hi_x = float(np.percentile(xv, 1.0)), float(np.percentile(xv, 99.0))
            bw = max((hi_x - lo_x) * 0.14, 1e-3)
            gx = np.linspace(lo_x, hi_x, 60)
            gy = []
            for xc in gx:
                w = np.exp(-0.5 * ((xv - xc) / bw) ** 2)
                gy.append(float(np.sum(w * sv) / max(np.sum(w), 1e-9)))
            ax_p.plot(gx, gy, "-", color=PAL["charcoal"], lw=1.55, label="Smoothed mean", zorder=5)
        ax_p.set_xlabel(xl, fontweight="bold", color=PAL["charcoal"], fontsize=6.0)
        if idx_p == 0:
            ax_p.set_ylabel("TreeSHAP Contribution (Log-Odds)", fontweight="bold", color=PAL["charcoal"], fontsize=6.2)
            ax_p.legend(loc="upper right", frameon=False, fontsize=4.9, handlelength=1.0)
    if not (OUT / "SuppFig_S3_shapdep_v5_4.png").exists():
        fig.savefig(OUT / "SuppFig_S3_shapdep_v5_4.png", dpi=600)
    if not (OUT / "SuppFig_S3_shapdep_v5_4.pdf").exists():
        fig.savefig(OUT / "SuppFig_S3_shapdep_v5_4.pdf")
    plt.close(fig)

    # SUPP FIG S4: Six-Cohort Subgroup Discrimination (Composite Left-Forest + Right-Aligned Column, Zero Overlap!)
    T = r19c["tournament_cohorts"]
    cohorts_s4 = [
        ("MIMIC_IV_IntVal",       "MIMIC-IV (US)",      PAL["m4"],     "a"),
        ("MIMIC_III_CareVue",     "MIMIC-III CareVue",  PAL["m3"],     "b"),
        ("eICU_CRD",              "eICU-CRD (US)",      PAL["eicu"],   "c"),
        ("HiRID_Trauma",          "HiRID Swiss",        PAL["hirid"],  "d"),
        ("AmsterdamUMCdb_Trauma", "Amsterdam (NL)*",    PAL["aumc"],   "e"),
        ("JinhuaNSICU_TBI_S06",   "Jinhua TBI (CN)",    PAL["jinhua"], "f"),
    ]
    labs = [("female", "Women"), ("male", "Men"), ("age_lt65", "Age < 65 yr"), ("age_ge65", "Age ≥ 65 yr"), ("age_ge80", "Age ≥ 80 yr")]

    fig = plt.figure(figsize=(9.2, 5.50), facecolor="white")
    gs4 = fig.add_gridspec(2, 3, left=0.095, right=0.988, top=0.925, bottom=0.10, wspace=0.14, hspace=0.34)
    for idx_c, (db, title, col, let) in enumerate(cohorts_s4):
        r_i, c_i = divmod(idx_c, 3)
        ax_s4 = fig.add_subplot(gs4[r_i, c_i])
        style_classic_axes(ax_s4, grid_axis="y")
        for xg_s4 in [0.50, 0.75, 1.00]:
            ax_s4.axvline(xg_s4, color="white", lw=0.8, zorder=0)
        ax_s4.text(-0.10 if c_i > 0 else -0.15, 1.045, let, transform=ax_s4.transAxes,
                   fontsize=10.2, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")
        ax_s4.text(0.0, 1.049, title, transform=ax_s4.transAxes,
                   fontsize=7.6, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

        ys = np.arange(len(labs))[::-1]
        ref_v = T[db]["models"]["Modality_Adaptive_GBDT"]["AUROC"]
        ax_s4.fill_betweenx([-0.55, len(labs) - 0.35], ref_v - 0.03, min(1.0, ref_v + 0.03), color=col, alpha=0.12, zorder=1)
        ax_s4.axvline(ref_v, color=col, lw=0.90, ls="--", alpha=0.68, zorder=2)
        ax_s4.axvline(1.015, color=PAL["stone"], lw=0.80, ls="-", zorder=2)

        for y, (k, _) in zip(ys, labs):
            r_cell = f_6[db][k]
            pt = r_cell["AUROC"]
            lo, hi = r_cell["boot95"]
            ax_s4.errorbar(pt, y, xerr=[[max(0, pt - lo)], [max(0, hi - pt)]],
                           fmt="s", ms=4.4, mfc=col, mec="white", mew=0.50,
                           ecolor=col, elinewidth=1.2, capsize=2.5, capthick=0.95, zorder=4)
            ax_s4.text(1.032, y, f"{pt:.2f} [{lo:.2f}–{hi:.2f}]",
                       fontsize=5.95, va="center", ha="left", fontweight="bold", color=PAL["charcoal"], zorder=5)

        ax_s4.set_yticks(ys)
        ax_s4.set_yticklabels([l for _, l in labs] if c_i == 0 else [], fontsize=7.0, fontweight="bold", color=PAL["charcoal"])
        ax_s4.set_xlim(0.44, 1.54)
        ax_s4.set_ylim(-0.55, len(labs) - 0.35)
        ax_s4.set_xticks([0.50, 0.75, 1.00])
        if r_i == 1:
            ax_s4.set_xlabel("24-h Landmark AUROC [95% CI]", fontweight="bold", color=PAL["charcoal"], fontsize=7.0)

    if not (OUT / "SuppFig_S4_fairness_v5_4.png").exists():
        fig.savefig(OUT / "SuppFig_S4_fairness_v5_4.png", dpi=600)
    if not (OUT / "SuppFig_S4_fairness_v5_4.pdf").exists():
        fig.savefig(OUT / "SuppFig_S4_fairness_v5_4.pdf")
    plt.close(fig)
    print("[saved] SuppFig_S1..S4_v5_4.png & .pdf")

if __name__ == "__main__":
    main()
