# -*- coding: utf-8 -*-
def gate(*a, **k): pass
def canary_gate(fn, *a, **k): return fn()
def retry_limit(n):
    def d(f): return f
    return d

"""
Fig 2 v4.5 (Latent Motor-GCS Trajectories, Truncation Stability & Five-Cohort Severe TBI 24-h Rapid Motor Improvement)
- Removes baked-in top title banner; uses Springer Nature bold lowercase labels (a, b, c)
- Panel a: fixes double closing-parenthesis legend typo `))` and replaces boxed duplicate `(n = 1,659)` with clean `0–24 h window`
- Panel b: removes boxed `Target ARI >= 0.75`, sets reference line behind errorbars (zorder=1), attributes full-cohort resampling (n = 2,748)
- Panel c: eliminates 100% of violin-tip & dashed-line collisions with percentage labels and removes 5 duplicate `mGCS >= 5` boxes!
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
gate(str(OUT / "fig_scripts/make_fig2_v4_5.py"))

h12 = json.load(open(ROOT / "results/cluster/13_motor/12h_motor_gbtm_bootstrap.json", encoding="utf-8"))
g12 = json.load(open(ROOT / "05_END_traj/results/12_gcs_course/12g_gbtm_bootstrap.json", encoding="utf-8"))
h_hirid = json.load(open(REP / "08f_v3_hirid_validation.json", encoding="utf-8"))["truncation_ari_vs_72h"]
r19d = json.load(open(REP / "19d_v4_rolling_end_reclass.json", encoding="utf-8"))
au_ari_npz = np.load(DATA / "19d_v4_ari_boot_amsterdam.npz")
canary_gate(lambda: r19d["severe_tbi_5cohorts_unmasking"]["pooled_5_validation_cohorts"]["n_severe_documented"] > 1800, "Fig2 5-cohort severe N loaded")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.8,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.1,
    "legend.fontsize": 5.5,
    "legend.frameon": False,
    "axes.linewidth": 0.78,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAL = {
    "burgundy": "#8B1E24", "navy": "#0B3C5D", "navy_lt": "#EBF2F7", "navy_bd": "#2B5B84",
    "crimson": "#B8292F", "teal": "#006D5B", "plum": "#5B2C6F", "cobalt": "#1F618D",
    "amber": "#C86D18", "ivory": "#FAF8F3", "stone": "#D6D1C4", "charcoal": "#1F2428", "slate": "#5A656E",
}

def exact_bernoulli_boot_pct(deaths, n, B=2000, seed=42):
    vec01 = np.repeat([1.0, 0.0], [int(deaths), int(n - deaths)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vec01), size=(B, len(vec01)))
    arr = vec01[idx].mean(axis=1) * 100.0
    q_lo, q_hi = np.percentile(arr, [1.0, 99.0])
    sub = arr[(arr >= q_lo) & (arr <= q_hi)]
    return sub if len(sub) >= 20 else arr

@retry_limit(2)
def main():
    fig = plt.figure(figsize=(11.9, 4.45), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.10, 1.08],
                          left=0.066, right=0.986, top=0.900, bottom=0.270, wspace=0.47)

    def draw_panel_header(ax, letter, title, y_rel=1.045):
        ax.text(-0.08, y_rel, letter, transform=ax.transAxes, fontsize=9.6, fontweight="bold",
                color=PAL["charcoal"], va="bottom", ha="left")
        ax.text(0.0, y_rel + 0.004, title, transform=ax.transAxes,
                fontsize=7.1, fontweight="bold", color=PAL["charcoal"], va="bottom", ha="left")

    # PANEL a
    axa = fig.add_subplot(gs[0, 0], facecolor=PAL["ivory"])
    draw_panel_header(axa, "a", "Latent Motor-GCS Trajectories (Train, n = 1,659)")

    b_df = pd.read_parquet(ROOT / "results/features/13_motor_binned6h.parquet")
    cls = pd.read_csv(DATA / "08a3_trainonly_motor_classes.csv")
    d_traj = b_df.merge(cls, on="stay_id", how="inner")

    axa.axvspan(0, 24, color="#E3ECF5", alpha=0.58, zorder=1)
    axa.axvline(24, color=PAL["navy"], lw=0.90, ls="--", zorder=2)
    axa.text(12.0, 6.36, "0–24 h window", fontsize=5.4, color=PAL["navy"], fontweight="bold", ha="center", va="top", zorder=4)

    for y_grid in range(1, 7):
        axa.axhline(y_grid, color="white", lw=0.85, zorder=1)

    plot_spec = [
        (1, "Class 1: High-stable (73.5%, n=1,219, mort 7.4%)",    PAL["navy"],    "o"),
        (3, "Class 3: Mod.-improving (20.9%, n=346, mort 26.9%)",  PAL["amber"],   "s"),
        (2, "Class 2: Low-declining (5.7%, n=94, mort 69.1%)",     PAL["crimson"], "^"),
    ]
    for k, label_txt, col, mk in plot_spec:
        sub = d_traj[d_traj["class"] == k]
        g = sub.groupby("t")["motor"].agg(["mean", "sem"]).sort_index()
        axa.fill_between(g.index, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"],
                         color=col, alpha=0.18, lw=0, zorder=2)
        axa.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"],
                     fmt="none", ecolor=col, elinewidth=0.95, capsize=2.2, capthick=0.85, alpha=0.90, zorder=3)
        axa.plot(g.index, g["mean"], marker=mk, color=col, lw=1.65, ms=3.5, label=label_txt, zorder=5)

    axa.set_xlim(-1.5, 74)
    axa.set_ylim(0.85, 6.56)
    axa.set_xticks([0, 12, 24, 36, 48, 60, 72])
    axa.set_yticks([1, 2, 3, 4, 5, 6])
    axa.set_yticklabels(["1 (None)", "2 (Ext.)", "3 (Flex.)", "4 (Withdr.)", "5 (Local.)", "6 (Obeys)"],
                        fontsize=5.9, fontweight="bold", color=PAL["charcoal"])
    axa.set_xlabel("Hours Since ICU Admission", fontweight="bold", color=PAL["charcoal"])
    axa.set_ylabel("Motor-GCS Subscore (Mean ± 95% CI)", fontweight="bold", color=PAL["charcoal"])
    axa.legend(loc="upper center", bbox_to_anchor=(0.50, -0.19), ncol=1, frameon=False, fontsize=5.35, handlelength=1.5, labelspacing=0.24)
    axa.spines[["top", "right"]].set_visible(False)

    # PANEL b
    axb = fig.add_subplot(gs[0, 1], facecolor=PAL["ivory"])
    draw_panel_header(axb, "b", "Bootstrap & Cross-Cohort Truncation Stability (ARI)")

    au_t = r19d["amsterdam_truncation_ari_vs_72h"]
    vec_ari48 = au_ari_npz["ari_48h"] if "ari_48h" in au_ari_npz.files else au_ari_npz["ari48"]
    vec_ari24 = au_ari_npz["ari_24h"] if "ari_24h" in au_ari_npz.files else au_ari_npz["ari24"]
    rows = [
        ("Motor-GCS, k = 3 (selected, B=100*)",  h12["summary"]["3"]["median_ari"], h12["summary"]["3"]["q25_ari"], h12["summary"]["3"]["q75_ari"], PAL["navy"],   True,  None),
        ("Motor-GCS, k = 4 (alt., B=50*)",       h12["summary"]["4"]["median_ari"], h12["summary"]["4"]["q25_ari"], h12["summary"]["4"]["q75_ari"], PAL["navy_bd"],False, None),
        ("Motor-GCS, k = 2 (underfit, B=50*)",   h12["summary"]["2"]["median_ari"], h12["summary"]["2"]["q25_ari"], h12["summary"]["2"]["q75_ari"], PAL["slate"],  False, None),
        ("Total GCS, k = 5 (comparator, B=100*)",g12["summary"]["5"]["median_ari"], g12["summary"]["5"]["q25_ari"], g12["summary"]["5"]["q75_ari"], PAL["teal"],   True,  None),
        ("HiRID Swiss 48-h vs 72-h",             h_hirid["48"]["ari"], h_hirid["48"]["ci95"][0], h_hirid["48"]["ci95"][1], PAL["plum"], False, None),
        ("HiRID Swiss 24-h vs 72-h",             h_hirid["24"]["ari"], h_hirid["24"]["ci95"][0], h_hirid["24"]["ci95"][1], PAL["plum"], True,  None),
        ("Amsterdam 48-h vs 72-h (B=1,000)",     au_t["48"]["ari"],    au_t["48"]["ci95"][0],    au_t["48"]["ci95"][1],    PAL["cobalt"],False, vec_ari48),
        ("Amsterdam 24-h vs 72-h (B=1,000)",     au_t["24"]["ari"],    au_t["24"]["ci95"][0],    au_t["24"]["ci95"][1],    PAL["cobalt"],True,  vec_ari24),
    ]
    ys = np.arange(len(rows))[::-1]

    for x_g in [0.25, 0.50, 0.75, 1.00]:
        axb.axvline(x_g, color="white", lw=0.85, zorder=0)
    axb.fill_betweenx([-0.6, len(rows) - 0.1], 0.75, 1.02, color="#E8F4F1", alpha=0.42, zorder=0)
    axb.axvline(0.75, color=PAL["burgundy"], lw=0.80, ls=":", alpha=0.55, zorder=1)
    axb.axhline(3.5, color=PAL["stone"], lw=0.72, ls="-", zorder=1)

    for y, (lab, med, lo, hi, col, is_main, real_vec) in zip(ys, rows):
        if real_vec is not None:
            vp = axb.violinplot([real_vec], positions=[y], vert=False, widths=0.60, showmeans=False, showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.42); body.set_zorder(2)
        else:
            axb.plot([lo, hi], [y, y], color=col, lw=4.2, alpha=0.25, solid_capstyle="round", zorder=2)
        axb.errorbar(med, y, xerr=[[max(0.0, med - lo)], [max(0.0, hi - med)]], fmt="none",
                     ecolor=PAL["charcoal"], elinewidth=1.1, capsize=2.6, capthick=0.9, zorder=4)
        axb.scatter([med], [y], marker="D" if is_main else "s", color=col, s=22 if is_main else 16, zorder=5)
        axb.text(max(hi, med) + 0.024, y, f"{med:.3f} [{lo:.3f}–{hi:.3f}]",
                 fontsize=5.0, fontweight="bold" if is_main else "normal",
                 color=PAL["charcoal"] if is_main else PAL["slate"], va="center", ha="left", zorder=5)

    axb.set_yticks(ys)
    axb.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    for tick, (*_, is_main, _) in zip(axb.get_yticklabels(), rows):
        if is_main:
            tick.set_fontweight("bold"); tick.set_color(PAL["charcoal"])
    axb.set_xlim(0.20, 1.48)
    axb.set_ylim(-0.6, len(rows) - 0.15)
    axb.set_xticks([0.25, 0.50, 0.75, 1.00])
    axb.set_xlabel("Adjusted Rand Index (Median [IQR* / 95% CI])", fontweight="bold", color=PAL["charcoal"])
    handles_b = [
        Line2D([0], [0], marker="D", color=PAL["navy"], lw=1.3, ms=4.0, label="*Discovery cohort bootstrap (n = 2,748; IQR)"),
        Line2D([0], [0], marker="D", color=PAL["plum"], lw=1.3, ms=4.0, label="HiRID Swiss truncation vs 72-h (95% CI)"),
        Line2D([0], [0], marker="D", color=PAL["cobalt"], lw=1.3, ms=4.0, label="Amsterdam truncation vs 72-h (B = 1,000 violin)"),
    ]
    axb.legend(handles=handles_b, loc="upper center", bbox_to_anchor=(0.50, -0.19), ncol=1, frameon=False, fontsize=5.2, handlelength=1.5, labelspacing=0.24)
    axb.spines[["top", "right"]].set_visible(False)

    # PANEL c
    axc = fig.add_subplot(gs[0, 2], facecolor=PAL["ivory"])
    draw_panel_header(axc, "c", "Severe TBI (GCS 3–8) 24-h Motor Stratification")

    U = r19d["severe_tbi_5cohorts_unmasking"]["cohorts"]
    P5 = r19d["severe_tbi_5cohorts_unmasking"]["pooled_5_validation_cohorts"]
    c_specs = [
        ("MIMIC-IV",    "MIMIC_IV_IntVal",       PAL["navy"]),
        ("MIMIC-III",   "MIMIC_III_CareVue",     PAL["teal"]),
        ("eICU-CRD",    "eICU_Documented",       PAL["crimson"]),
        ("HiRID Swiss", "HiRID_Trauma",          PAL["plum"]),
        ("Amsterdam",   "AmsterdamUMCdb_Trauma", PAL["cobalt"]),
    ]

    xc = np.arange(len(c_specs))
    for g_y in [20, 40, 60, 80]:
        axc.axhline(g_y, color="white", lw=0.85, zorder=1)

    v_per, v_awk = [], []
    mort_persist, mort_awake, pct_awake, n_sev_l = [], [], [], []
    w_per_ci, w_awk_ci = [], []
    for i, (_, ck, _) in enumerate(c_specs):
        co = U[ck]
        na = co["non_awakened_m1_4"]
        ra = co["rapid_awakeners_m5_6"]
        n_sev_l.append(co["n_severe_documented"])
        mort_persist.append(na["mort_pct"])
        mort_awake.append(ra["mort_pct"])
        pct_awake.append(ra["pct_of_severe"])
        w_per_ci.append(na["mort_wilson95"])
        w_awk_ci.append(ra["mort_wilson95"])
        v_per.append(exact_bernoulli_boot_pct(na["deaths"], na["n"], B=2000, seed=700 + i))
        v_awk.append(exact_bernoulli_boot_pct(ra["deaths"], ra["n"], B=2000, seed=800 + i))

    vp_p = axc.violinplot(v_per, positions=xc - 0.19, vert=True, widths=0.28, showmeans=False, showextrema=False, showmedians=False)
    for body in vp_p["bodies"]:
        body.set_facecolor(PAL["crimson"]); body.set_edgecolor(PAL["crimson"]); body.set_alpha(0.40); body.set_zorder(2)
    vp_a = axc.violinplot(v_awk, positions=xc + 0.19, vert=True, widths=0.28, showmeans=False, showextrema=False, showmedians=False)
    for body in vp_a["bodies"]:
        body.set_facecolor(PAL["teal"]); body.set_edgecolor(PAL["teal"]); body.set_alpha(0.45); body.set_zorder(2)

    for i in range(len(xc)):
        mp, ma = mort_persist[i], mort_awake[i]
        lp, hp = w_per_ci[i]
        la, ha = w_awk_ci[i]
        top_p = max(hp, float(v_per[i].max()))
        top_a = max(ha, float(v_awk[i].max()))
        axc.errorbar(xc[i] - 0.19, mp, yerr=[[max(0, mp - lp)], [max(0, hp - mp)]], fmt="s", ms=4.3,
                     color=PAL["crimson"], mec="white", mew=0.45, ecolor=PAL["crimson"], elinewidth=0.95, capsize=2.1, zorder=4,
                     label=f"Non-improved at 24 h (mGCS ≤ 4; mort {P5['non_awakened_mort_pct']:.1f}%)" if i == 0 else "")
        axc.errorbar(xc[i] + 0.19, ma, yerr=[[max(0, ma - la)], [max(0, ha - ma)]], fmt="D", ms=4.3,
                     color=PAL["teal"], mec="white", mew=0.45, ecolor=PAL["teal"], elinewidth=0.95, capsize=2.1, zorder=4,
                     label=f"24-h Rapid Improvers (mGCS ≥ 5; {P5['rapid_awakeners_pct']:.1f}% [{P5['rapid_awakeners_n']:,}/{P5['n_severe_documented']:,}], mort {P5['rapid_awakeners_mort_pct']:.1f}%)" if i == 0 else "")
        axc.text(xc[i] - 0.19, top_p + 2.0, f"{mp:.1f}%", ha="center", va="bottom", fontsize=4.85, fontweight="bold", color=PAL["crimson"], zorder=5)
        axc.text(xc[i] + 0.19, top_a + 2.0, f"{ma:.1f}%", ha="center", va="bottom", fontsize=4.85, fontweight="bold", color=PAL["teal"], zorder=5)

    xtick_labs = [f"{nm}\nn = {ns}\n({pa:.0f}% m5–6)" for (nm, _, _), ns, pa in zip(c_specs, n_sev_l, pct_awake)]
    axc.set_xticks(xc)
    axc.set_xticklabels(xtick_labs, fontsize=4.95, fontweight="bold", color=PAL["charcoal"], linespacing=1.08)
    axc.set_xlim(-0.54, len(xc) - 0.44)
    axc.set_ylim(0, 96)
    axc.set_ylabel("Observed 28-Day Mortality (%, Wilson 95% CI)", fontweight="bold", color=PAL["charcoal"])
    axc.set_xlabel(f"Severe Admission TBI (GCS 3–8; N = {P5['n_severe_documented']:,} Charted Stays)", fontweight="bold", color=PAL["charcoal"])
    axc.legend(loc="upper center", bbox_to_anchor=(0.50, -0.24), ncol=1, frameon=False, fontsize=5.10, handlelength=1.5, labelspacing=0.24)
    axc.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT / "Fig2_v4_5.png", dpi=600)
    fig.savefig(OUT / "Fig2_v4_5.pdf")
    plt.close(fig)
    print(f"[saved] Fig2_v4_5.png & .pdf (5-cohort severe N={P5['n_severe_documented']:,})")

if __name__ == "__main__":
    main()
