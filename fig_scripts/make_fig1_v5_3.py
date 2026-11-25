# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
Fig 1 v5.0 (Six-Cohort Transcontinental Proactive Motor Biomarker & Conformal Routing Architecture)
- Springer Nature bold lowercase panel labels (a, b), zero baked-in top banner
- 100% US English & programmatic card-collision verification
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path("E:/TBI subtype")
REP  = ROOT / "07_prediction_system/reports"
OUT  = ROOT / "07_prediction_system/manuscript/figures"
gate(str(OUT / "fig_scripts/make_fig1_v5_3.py"))

s19a = json.load(open(REP / "19a_v4_extraction_summary.json", encoding="utf-8"))
s19c = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
s19d = json.load(open(REP / "19d_v4_rolling_end_reclass.json", encoding="utf-8"))
s19e = json.load(open(REP / "19e_v4_table1_and_ablation.json", encoding="utf-8"))
s19h = json.load(open(REP / "19h_v5_proactive_biomarker_voi.json", encoding="utf-8"))
canary_gate(lambda: s19e["cohort_totals"]["primary_strict_trauma_6cohorts_n"] == 12181, "Fig1 cohort sum == 12,181")

N_PRIM    = s19e["cohort_totals"]["primary_strict_trauma_6cohorts_n"]
N_EXT_P   = s19e["cohort_totals"]["external_strict_trauma_5cohorts_n"]
N_EXT_A   = s19e["cohort_totals"]["external_expanded_abi_5cohorts_n"]
N_CTR_EXT = s19e["cohort_totals"]["unique_centers_external_5dbs"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAL = {
    "burgundy": "#8B1E24", "navy": "#0B3C5D", "navy_lt": "#EBF2F7", "navy_bd": "#2B5B84",
    "crimson": "#B8292F", "crim_lt": "#FBE9EA", "teal": "#006D5B", "teal_lt": "#E8F4F1",
    "plum": "#5B2C6F", "plum_lt": "#F4ECF7", "cobalt": "#1F618D", "cob_lt": "#EAF2F8",
    "amber": "#B95815", "amb_lt": "#FDF2E9", "ivory": "#FAF8F3", "stone": "#D6D1C4",
    "charcoal": "#1F2428", "slate": "#5A656E",
}

@retry_limit(2)
def main():
    fig = plt.figure(figsize=(11.2, 4.45), facecolor="white")
    ax = fig.add_axes([0.010, 0.018, 0.980, 0.962])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(14.8, 66.0)

    def draw_header(x, y, letter, title):
        ax.text(x, y, letter, fontsize=9.8, fontweight="bold", color=PAL["charcoal"], va="center", ha="left", zorder=6)
        ax.text(x + 1.55, y, title, fontsize=7.1, fontweight="bold", color=PAL["charcoal"], va="center", ha="left", zorder=6)

    draw_header(0.4, 64.2, "a", "Phenotype Discovery & Disjoint Split (MIMIC-IV · n = 2,700)")
    draw_header(32.8, 64.2, "b", f"Proactive Motor Phenotyping & Conformal Routing Across {N_CTR_EXT} Hospital Sites (n = {N_EXT_P:,} Primary · {N_EXT_A:,} ABI)")

    ax.add_patch(FancyBboxPatch((0.4, 15.6), 31.2, 46.4, boxstyle="round,pad=0.30,rounding_size=1.1",
                                fc=PAL["ivory"], ec=PAL["stone"], lw=0.95, zorder=1))
    ax.add_patch(Rectangle((0.4, 58.3), 31.2, 3.7, fc=PAL["navy"], ec="none", zorder=2))
    ax.text(16.0, 60.15, "STEP 1A · PHENOTYPING DEVELOPMENT & SPLIT (BOSTON, USA · 2008–2022)",
            color="white", fontsize=5.7, fontweight="bold", ha="center", va="center", zorder=3)

    ax.add_patch(FancyBboxPatch((32.8, 15.6), 66.8, 46.4, boxstyle="round,pad=0.30,rounding_size=1.1",
                                fc=PAL["ivory"], ec=PAL["stone"], lw=0.95, zorder=1))
    ax.add_patch(Rectangle((32.8, 58.3), 66.8, 3.7, fc=PAL["teal"], ec="none", zorder=2))
    ax.text(66.2, 60.15, "STEPS 1B–4 · PROACTIVE MOTOR PHENOTYPING & CONFORMAL TRIAGE (USA · SWITZERLAND · NETHERLANDS · CHINA)",
            color="white", fontsize=5.55, fontweight="bold", ha="center", va="center", zorder=3)

    boxes = []
    def card(x, y, w, h, title, subtitle, fc="white", ec=PAL["charcoal"], header_fc=None, title_col=PAL["charcoal"], fs_t=5.6, fs_s=4.9, header_h=2.55):
        boxes.append((x - w/2, y - h/2, x + w/2, y + h/2))
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.12,rounding_size=0.65",
                                    fc=fc, ec=ec, lw=0.85, zorder=3))
        if header_fc:
            ax.add_patch(Rectangle((x - w/2, y + h/2 - header_h), w, header_h, fc=header_fc, ec="none", zorder=4))
            ax.text(x, y + h/2 - header_h/2, title, fontsize=fs_t, fontweight="bold", color="white", ha="center", va="center", zorder=5)
            ax.text(x, y - header_h * 0.48, subtitle, fontsize=fs_s, color=PAL["charcoal"], ha="center", va="center", linespacing=1.22, zorder=5)
        else:
            n_lines = subtitle.count("\n") + 1 if subtitle else 0
            t_off = h * 0.28 if n_lines >= 2 else (h * 0.22 if n_lines == 1 else 0)
            s_off = -h * 0.14 if n_lines >= 2 else -h * 0.18
            ax.text(x, y + t_off, title, fontsize=fs_t, fontweight="bold", color=title_col, ha="center", va="center", zorder=5)
            if subtitle:
                ax.text(x, y + s_off, subtitle, fontsize=fs_s, color=PAL["slate"], ha="center", va="center", linespacing=1.20, zorder=5)

    def arrow_v(x, y1, y2, col=PAL["charcoal"], label=None):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.85, color=col, mutation_scale=7.5, shrinkA=0, shrinkB=0), zorder=4)
        if label:
            ax.text(x, (y1 + y2)/2, label, fontsize=4.70, color=PAL["charcoal"], va="center", ha="center", linespacing=1.12,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=PAL["stone"], lw=0.55), zorder=5)

    m4_ct_n    = s19a["ct_coverage_strict_24h"]["MIMIC-IV"]["ct_24h_n"]
    m3_ct_pct  = s19a["ct_coverage_strict_24h"]["MIMIC-III"]["pct"]
    ei_aud     = s19c["eicu_hospital_audit"]
    au_sev_n   = s19d["severe_tbi_5cohorts_unmasking"]["cohorts"]["AmsterdamUMCdb_Trauma"]["n_severe_documented"]
    au_awk_p   = s19d["severe_tbi_5cohorts_unmasking"]["cohorts"]["AmsterdamUMCdb_Trauma"]["rapid_awakeners_m5_6"]["pct_of_severe"]
    jh_ct_n    = s19a["ct_coverage_strict_24h"]["JinhuaNSICU_TBI"]["ct_24h_n"]
    jh_ct_pct  = s19a["ct_coverage_strict_24h"]["JinhuaNSICU_TBI"]["pct"]
    jh_map_pct = 100.0 * s19a["jinhua_gcs_proxy_mapped_n"] / s19a["jinhua_tbi_s06_n"]

    card(16.0, 52.8, 29.2, 7.2,
         "MIMIC-IV v3.1 Adult TBI Admissions (n = 2,751)",
         f"First ICU stay (2,700 unique subjects) · 82,527 hourly GCS checks\nEnglish Head-CT ≤ 24 h: n = {m4_ct_n:,} ({100*m4_ct_n/2700:.1f}%) · IV sedative logs",
         fc="white", ec=PAL["navy_bd"], title_col=PAL["navy"], fs_t=5.70, fs_s=4.95)
    arrow_v(16.0, 49.0, 42.8, col=PAL["navy_bd"], label="Excluded < 24 h (n = 51): ICU death or discharge < 24 h")
    card(16.0, 38.4, 29.2, 7.8,
         "24-h Landmark Cohort (n = 2,700 · 402 deaths [14.9%])",
         "Severe GCS 3–8: n = 684 · Mod 9–12: n = 555 · Mild 13–15: n = 1,456\nPost-washout motor phenotyping: n = 2,695 (99.8% · median 13 checks)",
         fc=PAL["navy_lt"], ec=PAL["navy"], title_col=PAL["navy"], fs_t=5.75, fs_s=5.0)

    ax.plot([16.0, 16.0, 16.0], [34.3, 33.0, 31.8], color=PAL["navy_bd"], lw=0.85, zorder=3)
    ax.plot([5.8, 16.0, 26.2], [31.8, 31.8, 31.8], color=PAL["navy_bd"], lw=0.85, zorder=3)
    for x_sp in (5.8, 16.0, 26.2):
        ax.annotate("", xy=(x_sp, 29.0), xytext=(x_sp, 31.8),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PAL["navy_bd"], mutation_scale=7.0, shrinkA=0, shrinkB=0), zorder=4)

    card(5.8, 23.0, 9.4, 11.0, "Train (2008–17)",
         "n = 1,661 stays\n248 deaths (14.9%)\nLCGM fit: n = 1,659\nMotor & CT fits",
         fc="white", ec=PAL["navy"], header_fc=PAL["navy"], fs_t=5.15, fs_s=4.68)
    card(16.0, 23.0, 9.4, 11.0, "Tune (2018–19)",
         "n = 556 (73 deaths)\nSelect: n = 278\nCalib: n = 278\nUntouched 90% C(x)",
         fc="white", ec=PAL["navy_bd"], header_fc=PAL["navy_bd"], fs_t=5.15, fs_s=4.68)
    card(26.2, 23.0, 9.4, 11.0, "Step 1a · Temp Val",
         "n = 483 (2020–22)\n81 deaths (16.8%)\nSevere 3–8: n = 114\nVoI NNR = 2.58",
         fc="white", ec=PAL["burgundy"], header_fc=PAL["burgundy"], fs_t=5.15, fs_s=4.68)

    cols_b = [
        (39.8, PAL["teal"], PAL["teal_lt"],
         "Step 1b · MIMIC-III (US)", "1,950 TBI ICU admissions\n1,425 adult stays ≥ 24 h\nBoston (2001–2008)",
         "Exclude MetaVision\noverlap (n = 586)", "Cross-Era CareVue Set",
         "n = 839 · 155 deaths (18.5%)\nLegacy CareVue EHR\nZero MIMIC-IV overlap",
         "Proactive Motor + CT", f"• Motor charted: 98.9%\n• Head-CT ≤ 24 h: {m3_ct_pct:.1f}%\n• Severe 3–8: n = 321\n• VoI NNR = 2.29"),

        (53.0, PAL["crimson"], PAL["crim_lt"],
         "Step 2 · eICU-CRD (US)", "200,859 total ICU stays\n6,677 adult TBI stays\n208 US hospitals (2014–15)",
         "Exclude < 24 h\n(n = 2,237 stays)", "US Multicenter Set",
         "n = 4,440 · 520 deaths (11.7%)\n130 US hospitals\nSevere GCS 3–8: n = 821",
         "Hospital-Adaptive Triage", f"• Charted stays: 2,889 ({ei_aud['n_hospitals'] - ei_aud['n_zero_charting_hospitals']} hosps)\n• Unrecorded: 1,551 ({ei_aud['n_zero_charting_hospitals']} 0%-hosps)\n• Hospital-split conformal\n• Charted NNR = 2.99"),

        (66.2, PAL["plum"], PAL["plum_lt"],
         "Step 3a · HiRID (Swiss)", "33,905 ICU admissions\nBern Univ. Hosp (2008–16)\n2-min vitals · 1-h GCS",
         "APACHE trauma / ICP\n& ABI stays ≥ 24 h", "Swiss Academic Set",
         "Trauma: n = 1,209 (101 d)\nNeuro/ICP: n = 2,805 (285 d)\nAll ABI: n = 6,349 (495 d)",
         "Sedation & ICP Strata", "• ABI ICP: n = 1,736\n• Deep RASS ≤ -3: n = 854\n• Severe 3–8: n = 379\n• VoI NNR = 1.96"),

        (79.4, PAL["cobalt"], PAL["cob_lt"],
         "Step 3b · Amsterdam (NL)", "23,106 ICU admissions\nAmsterdam UMC (2003–16)\nFirst ICU stay · 18 GCS IDs",
         "Exclude neoplasms;\nTrauma / ABI ≥ 24 h", "Dutch Academic Set",
         f"Trauma: n = {s19a['amsterdam_trauma_n']:,} ({s19a['amsterdam_trauma_d28']} d)\nExpanded ABI: n = {s19a['amsterdam_abi_n']:,} ({s19a['amsterdam_abi_d28']} d)\n99.9% observed 3-part GCS",
         "Awakening & IV Sedation", f"• Sedated 3–8: {s19h['sedation_decoupled_awakening']['AmsterdamUMCdb_Trauma_Severe']['sedated_or_deep_rass']['n']}/{au_sev_n} ({100.0*s19h['sedation_decoupled_awakening']['AmsterdamUMCdb_Trauma_Severe']['sedated_or_deep_rass']['n']/au_sev_n:.1f}%)\n• Trauma GCS 3–8: n = {au_sev_n}\n• Awakeners: {au_awk_p:.1f}%\n• VoI NNR = 2.10"),

        (92.6, PAL["amber"], PAL["amb_lt"],
         "Step 4 · Jinhua (China)", "5,151 NSICU admissions\nJinhua Hosp (2018–2024)\nSpecialist Neuro-ICU",
         "Adult TBI / ABI ≥ 24 h\n(strict ≤ 24 h window)", "Asian Specialist Set",
         f"S06/TBI: n = {s19a['jinhua_tbi_s06_n']:,} ({s19a['jinhua_tbi_s06_d28']} d)\nAll ABI: n = {s19a['jinhua_abi_n']:,} ({s19a['jinhua_abi_d28']} d)\nIn-hospital ≤ 28-d mortality",
         "Proactive CT Escalation", f"• Head-CT ≤ 24 h: {jh_ct_pct:.1f}% ({jh_ct_n:,})\n• Unexported hourly GCS\n• Admit GCS proxy: {jh_map_pct:.1f}%\n• CT VoI NNR = 3.88"),
    ]

    for xc, col, lfc, t_tit, t_sub, arr_l, m_tit, m_sub, b_tit, b_sub in cols_b:
        card(xc, 52.8, 12.3, 7.2, t_tit, t_sub, fc="white", ec=col, title_col=col, fs_t=5.25, fs_s=4.68)
        arrow_v(xc, 49.0, 42.8, col=col, label=arr_l)
        card(xc, 38.4, 12.3, 7.8, m_tit, m_sub, fc=lfc, ec=col, title_col=col, fs_t=5.25, fs_s=4.68)
        arrow_v(xc, 34.3, 29.0, col=col)
        card(xc, 23.0, 12.3, 11.0, b_tit, b_sub, fc="white", ec=col, header_fc=col, fs_t=5.15, fs_s=4.65)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            overlap = (ax1 < bx2 - 0.05) and (ax2 > bx1 + 0.05) and (ay1 < by2 - 0.05) and (ay2 > by1 + 0.05)
            assert not overlap, f"Card collision between {i} and {j}!"

    if not (OUT / "Fig1_v5_3.png").exists():
        fig.savefig(OUT / "Fig1_v5_3.png", dpi=600)
    if not (OUT / "Fig1_v5_3.pdf").exists():
        fig.savefig(OUT / "Fig1_v5_3.pdf")
    plt.close(fig)
    print(f"[saved] Fig1_v5_3.png & .pdf (N_PRIM={N_PRIM:,}, 0 collisions)")

if __name__ == "__main__":
    main()
