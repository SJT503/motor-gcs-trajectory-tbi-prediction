# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19h_v5_proactive_biomarker_voi.py — Post-Resuscitation Motor Awakening Digital Biomarker
& Conformal Value-of-Information (VoI) Proactive Clinical Decision Analysis (v5.0)

Computes 100% real patient-level empirical metrics across the 6 transcontinental cohorts:
  1. Three-Country Sedation-Stratified Severe TBI Motor Awakening Analysis:
     - USA MIMIC-IV (active 0-24h IV sedative-hypnotic infusions: propofol/midazolam/dexmedetomidine [itemids 222168, 221668, 225150, 229420] vs none)
     - Swiss HiRID (deep sedation RASS <= -3 vs RASS > -3 / unrecorded in 0-24h)
     - Dutch AmsterdamUMCdb (systemic 0-24h IV sedatives/opioids: propofol/midazolam/fentanyl/remifentanil/sufentanil in drugitems.csv vs none)
     Quantifies 24-h motor awakening rate (mGCS >= 5), 28-d mortality fold-reduction, and
     Anti-Premature-WLST Conformal Protection (surviving awakeners rescued from Static {1}/{0,1} into {0}).
  2. Six-Cohort Conformal Value-of-Information (VoI) & Proactive Sensing Cascade:
     - Out-of-fold / Tune-Calib 90% Mondrian split-conformal sets on Static S0 vs +24h Motor Biomarker vs Modality-Adaptive Primary
     - Ambiguity resolution rate (% of Static S0 {0,1} uncertain stays converted into definitive {0}/{1} singletons)
     - Net singleton expansion, binary Shannon entropy reduction (bits & %), and Number Needed to Re-check (NNR)
     - Stepwise modality VoI in JinhuaNSICU (Static S0 -> +Head-CT NLP) vs MIMIC-III (Motor-first vs CT-first)
"""
import json, math
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path("E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP  = ROOT / "07_prediction_system/reports"
gate(str(ROOT / "07_prediction_system/scripts/19h_v5_proactive_biomarker_voi.py"))

df_preds = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
m4_mat   = pd.read_parquet(DATA / "08c_matrix_mimic4_v3.parquet")
hi_mat   = pd.read_parquet(DATA / "08c_matrix_hirid_v3.parquet")
au_mat   = pd.read_parquet(DATA / "19a_matrix_amsterdam_v4.parquet")
jh_mat   = pd.read_parquet(DATA / "19a_matrix_jinhua_v4.parquet")
ct_df    = pd.read_parquet(DATA / "19a_ct_nlp_3cohorts_v4.parquet")

canary_gate(lambda: len(df_preds) >= 9964 and "p_static" in df_preds.columns, "Preds loaded")


def exact_mondrian_thresholds(y_cal, p_cal, alpha=0.10):
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    s1 = np.sort(1.0 - p_cal[y_cal == 1])
    s0 = np.sort(p_cal[y_cal == 0])
    n1, n0 = len(s1), len(s0)
    k1 = min(max(1, math.ceil((1.0 - alpha) * (n1 + 1))), n1)
    k0 = min(max(1, math.ceil((1.0 - alpha) * (n0 + 1))), n0)
    return float(1.0 - s1[k1 - 1]), float(s0[k0 - 1])


def shannon_bits(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def eval_oof_conformal(y, p_raw, abstain_mask=None, alpha=0.10, n_splits=20, seed=42):
    """20-repeat 50/50 stratified split Platt + Mondrian OOF consensus (matching 19c/19g)."""
    y = np.asarray(y, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)
    n = len(y)
    if abstain_mask is None:
        abstain_mask = np.zeros(n, dtype=bool)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=seed)
    c0_votes = np.zeros(n, dtype=float)
    c1_votes = np.zeros(n, dtype=float)
    cnts     = np.zeros(n, dtype=float)
    p_cal_sum = np.zeros(n, dtype=float)

    for cal_i, te_i in sss.split(np.zeros(n), y):
        for ci, ti in [(cal_i, te_i), (te_i, cal_i)]:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(
                logit(np.clip(p_raw[ci], 1e-5, 1 - 1e-5)).reshape(-1, 1), y[ci]
            )
            p_ci = lr.predict_proba(logit(np.clip(p_raw[ci], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            p_ti = lr.predict_proba(logit(np.clip(p_raw[ti], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            t_lo, t_hi = exact_mondrian_thresholds(y[ci], p_ci, alpha=alpha)
            in0 = (p_ti <= t_hi) | abstain_mask[ti]
            in1 = (p_ti >= t_lo) | abstain_mask[ti]
            c0_votes[ti] += in0.astype(float)
            c1_votes[ti] += in1.astype(float)
            p_cal_sum[ti] += p_ti
            cnts[ti] += 1.0

    in_c0 = (c0_votes / cnts) >= 0.5
    in_c1 = (c1_votes / cnts) >= 0.5
    # Guarantee non-empty set if thresholds cross in small strata: assign argmax probability
    empty = (~in_c0) & (~in_c1)
    p_oof = p_cal_sum / cnts
    in_c0[empty] = p_oof[empty] < 0.5
    in_c1[empty] = p_oof[empty] >= 0.5
    is_sing0 = in_c0 & (~in_c1) & (~abstain_mask)
    is_sing1 = in_c1 & (~in_c0) & (~abstain_mask)
    is_def   = is_sing0 | is_sing1
    is_amb   = (~is_def)
    cov = float(np.mean(np.where(y == 0, in_c0, in_c1)))
    auc_def = float(roc_auc_score(y[is_def], p_oof[is_def])) if (is_def.sum() > 20 and len(np.unique(y[is_def])) == 2) else None
    return {
        "p_oof": p_oof,
        "in_c0": in_c0,
        "in_c1": in_c1,
        "is_sing0": is_sing0,
        "is_sing1": is_sing1,
        "is_def": is_def,
        "is_amb": is_amb,
        "coverage": round(cov, 4),
        "retained_frac": round(float(is_def.mean()), 4),
        "ambiguous_frac": round(float(is_amb.mean()), 4),
        "mean_entropy_bits": round(float(shannon_bits(p_oof).mean()), 4),
        "def_auroc": round(auc_def, 4) if auc_def is not None else None,
    }


@retry_limit(2)
def main():
    out = {}

    # =========================================================================
    # PART 1: Three-Country Sedation-Stratified Severe TBI (GCS 3-8) Awakening
    # =========================================================================
    con = duckdb.connect()
    inf_m4 = con.execute(f"""
        SELECT DISTINCT e.stay_id
        FROM read_csv_auto('{ROOT}/data/mimic-iv-3.1/icu/inputevents.csv.gz') e
        JOIN read_parquet('{ROOT}/results/cohort_audit/02_tbi_cohort.parquet') co ON e.stay_id = co.stay_id
        WHERE e.itemid IN (222168, 221668, 229420, 225150)
          AND e.starttime IS NOT NULL AND e.endtime IS NOT NULL
          AND e.statusdescription != 'Rewritten'
          AND e.starttime <= co.intime + INTERVAL 24 HOUR
          AND e.endtime >= co.intime
    """).df()
    sed_ids_m4 = set(inf_m4["stay_id"].astype(int))

    # 1a) MIMIC-IV full cohort (all splits N=2,700 for phenotype epidemiology, plus validation n=483)
    m4_sev = m4_mat[(m4_mat["gcs_total_first"] <= 8) & (m4_mat["motor_last"].notna())].copy()
    m4_sev["sed24"] = m4_sev["stay_id"].astype(int).isin(sed_ids_m4)

    # 1b) HiRID APACHE Trauma (n=1,209) & Expanded ABI (n=6,349)
    hi_tr = hi_mat[hi_mat["is_trauma"] == True].copy().reset_index(drop=True)
    hi_sev = hi_tr[(hi_tr["gcs_total_first"] <= 8) & (hi_tr["motor_last"].notna())].copy()
    hi_sev["sed24"] = hi_sev["rass_min_24h"].fillna(0) <= -3.0

    # 1c) AmsterdamUMCdb Trauma (n=1,052): exact 0-24h systemic IV sedative/opioid infusions from drugitems.csv
    au_tr = au_mat[au_mat["is_trauma"] == True].copy().reset_index(drop=True)
    au_tr_ids = set(au_tr["admissionid"].astype(int))
    au_sed_ids = set()
    for ch in pd.read_csv(
        "F:/AmsterdamUMCdb/drugitems.csv",
        encoding="latin1",
        usecols=["admissionid", "item", "start", "stop"],
        chunksize=500000,
    ):
        sub_c = ch[ch["admissionid"].isin(au_tr_ids)]
        if len(sub_c) == 0:
            continue
        it_s = sub_c["item"].astype(str)
        is_iv_sed = it_s.isin([
            "Propofol (Diprivan)",
            "Midazolam (Dormicum)",
            "Fentanyl",
            "Remifentanyl",
            "Sufentanil (Sufenta)",
        ]) & (sub_c["start"] <= 24 * 3600 * 1000) & (sub_c["stop"] >= 0)
        au_sed_ids.update(sub_c.loc[is_iv_sed, "admissionid"].astype(int).tolist())
    au_sev = au_tr[(au_tr["gcs_total_first"] <= 8) & (au_tr["motor_last"].notna())].copy()
    au_sev["sed24"] = au_sev["admissionid"].astype(int).isin(au_sed_ids)

    def summarize_awakening_strata(df_s, y_col="d28"):
        res = {}
        for label, mask in [
            ("all_severe", np.ones(len(df_s), dtype=bool)),
            ("sedated_or_deep_rass", df_s["sed24"].values.astype(bool)),
            ("unsedated_or_light", ~df_s["sed24"].values.astype(bool)),
        ]:
            sub = df_s[mask]
            if len(sub) == 0:
                continue
            awk = sub["motor_last"].values >= 5.0
            non = ~awk
            y_sub = sub[y_col].values.astype(int)
            m_awk = float(y_sub[awk].mean()) if awk.sum() > 0 else None
            m_non = float(y_sub[non].mean()) if non.sum() > 0 else None
            fold  = round(m_non / max(m_awk, 1e-6), 2) if (m_awk is not None and m_awk > 0 and m_non is not None) else None
            res[label] = {
                "n": int(len(sub)),
                "n_awakeners_m56": int(awk.sum()),
                "awakening_pct": round(100.0 * float(awk.mean()), 1),
                "mort_awakeners_pct": round(100.0 * m_awk, 1) if m_awk is not None else None,
                "mort_nonawakeners_pct": round(100.0 * m_non, 1) if m_non is not None else None,
                "mortality_fold_reduction": fold,
            }
        return res

    sed_awakening = {
        "MIMIC_IV_All_TBI_Severe": summarize_awakening_strata(m4_sev, "d28"),
        "HiRID_Trauma_Severe": summarize_awakening_strata(hi_sev, "d28"),
        "AmsterdamUMCdb_Trauma_Severe": summarize_awakening_strata(au_sev, "d28"),
    }
    # Pooled 3-country sedated severe TBI summary
    pooled_3c = pd.concat([
        m4_sev[["motor_last", "d28", "sed24"]],
        hi_sev[["motor_last", "d28", "sed24"]],
        au_sev[["motor_last", "d28", "sed24"]],
    ], ignore_index=True)
    sed_awakening["Pooled_3Country_Severe"] = summarize_awakening_strata(pooled_3c, "d28")
    out["sedation_decoupled_awakening"] = sed_awakening

    # =========================================================================
    # PART 2: Conformal Value-of-Information (VoI) Proactive Sensing Cascade
    # =========================================================================
    # Reconstruct exact miss_motor_no_ct masks matching 19g
    fe_ei = pd.read_parquet(DATA / "features_eicu.parquet")
    post_ei = pd.read_parquet(DATA / "08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d_ei = fe_ei.merge(post_ei[["icustay_id_eicu", "prob_m1"]], on="icustay_id_eicu", how="left")
    d_ei = d_ei[~(((d_ei.died_hosp == 1) & (d_ei.los_h <= 24)) | (d_ei.los_h < 24))].copy().reset_index(drop=True)
    ei_miss_m = d_ei["prob_m1"].isna().values

    d_jh_tbi = jh_mat[jh_mat["is_tbi_s06"] == True].copy().reset_index(drop=True)
    ct_jh = ct_df[ct_df["cohort"] == "JinhuaNSICU"].set_index("id")
    jh_id_col = "hadm_id" if d_jh_tbi["hadm_id"].isin(ct_jh.index).sum() > 100 else "subject_id"
    jh_has_ct = ct_jh.reindex(d_jh_tbi[jh_id_col].values)["ct_any_24h"].fillna(0).values > 0.5
    jh_miss_no_ct = ~jh_has_ct

    fe_m3 = pd.read_parquet(DATA / "features_mimic3.parquet")
    post_m3 = pd.read_parquet(DATA / "08b_posterior24_mimic3_trainonly.parquet").rename(columns={"id": "icustay_id"})
    d_m3 = fe_m3.merge(post_m3[["icustay_id", "prob_m1"]], on="icustay_id", how="left")
    cv_ids = con.execute(
        f"SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'"
    ).df()
    d_m3 = d_m3[d_m3.icustay_id.isin(cv_ids.icustay_id)].copy()
    d_m3 = d_m3[~(((d_m3.d28 == 1) & (d_m3.days_to_death < 1.0)) | (d_m3.los_h < 24))].copy().reset_index(drop=True)
    ct_m3 = ct_df[ct_df["cohort"] == "MIMIC-III"].set_index("id")
    m3_has_ct = ct_m3.reindex(d_m3["icustay_id"].values)["ct_any_24h"].fillna(0).values > 0.5
    m3_miss_m_no_ct = d_m3["prob_m1"].isna().values & (~m3_has_ct)

    voi_results = {}
    cohort_specs = [
        ("MIMIC_IV_IntVal", None),
        ("MIMIC_III_CareVue", m3_miss_m_no_ct),
        ("eICU_CRD", ei_miss_m),
        ("HiRID_Trauma", None),
        ("AmsterdamUMCdb_Trauma", None),
        ("JinhuaNSICU_TBI_S06", jh_miss_no_ct),
    ]

    total_s0_amb = 0
    total_s0_amb_resolved = 0
    flowsheet_s0_amb = 0
    flowsheet_s0_amb_resolved = 0

    for ck, abst_mask in cohort_specs:
        sub = df_preds[df_preds["cohort"] == ck].copy().reset_index(drop=True)
        y = sub["d28"].values.astype(int)
        # Step A: Baseline Static S0 conformal triage (using identical 20x 50/50 protocol; no flowsheet needed for Static S0)
        res_s0  = eval_oof_conformal(y, sub["p_static"].values, abstain_mask=None, alpha=0.10)
        # Step B: +24h Motor Biomarker (Step 1 reanchored) where charted
        res_m1  = eval_oof_conformal(y, sub["p_reanchored"].values, abstain_mask=abst_mask, alpha=0.10)
        # Step C: Full Modality-Adaptive Primary (Motor when charted, Head-CT when unrecorded)
        res_pri = eval_oof_conformal(y, sub["p_adaptive_gbdt"].values, abstain_mask=abst_mask, alpha=0.10)

        s0_amb = res_s0["is_amb"]
        n_s0_amb = int(s0_amb.sum())
        # Among patients uncertain under Static S0 ({0,1}), how many become definitive ({0} or {1}) after acquiring primary modality?
        resolved_pri = s0_amb & res_pri["is_def"]
        n_res_pri = int(resolved_pri.sum())
        conv_rate_pri = round(100.0 * n_res_pri / max(n_s0_amb, 1), 1)
        nnr_selective = round(n_s0_amb / max(n_res_pri, 1), 2)

        # Split converted singletons into rescued low-risk {0} vs escalated high-risk {1}
        n_to_low0 = int((s0_amb & res_pri["is_sing0"]).sum())
        n_to_high1 = int((s0_amb & res_pri["is_sing1"]).sum())
        acc_converted = float(
            ((y == 0) & s0_amb & res_pri["is_sing0"]).sum() + ((y == 1) & s0_amb & res_pri["is_sing1"]).sum()
        ) / max(n_res_pri, 1)

        ent_s0  = res_s0["mean_entropy_bits"]
        ent_pri = res_pri["mean_entropy_bits"]
        rel_ent_drop = round(100.0 * (ent_s0 - ent_pri) / max(ent_s0, 1e-6), 1)

        # Anti-Premature-WLST shield in Severe TBI (admission GCS 3-8 survivors falsely in Static {1} or {0,1} -> moved to {0})
        sev_surv = (sub["gcs_total_first_raw"].fillna(15).values <= 8.0) & (y == 0)
        n_sev_surv = int(sev_surv.sum())
        s0_non_low = sev_surv & (~res_s0["is_sing0"])
        rescued_to_low0 = s0_non_low & res_pri["is_sing0"]

        total_s0_amb += n_s0_amb
        total_s0_amb_resolved += n_res_pri
        if ck in ("MIMIC_IV_IntVal", "MIMIC_III_CareVue", "HiRID_Trauma", "AmsterdamUMCdb_Trauma"):
            flowsheet_s0_amb += n_s0_amb
            flowsheet_s0_amb_resolved += n_res_pri

        voi_results[ck] = {
            "n": int(len(y)),
            "events": int(y.sum()),
            "static_s0_coverage": res_s0["coverage"],
            "static_s0_retained_pct": round(100.0 * res_s0["retained_frac"], 1),
            "static_s0_ambiguous_n": n_s0_amb,
            "static_s0_ambiguous_pct": round(100.0 * res_s0["ambiguous_frac"], 1),
            "motor_s1_retained_pct": round(100.0 * res_m1["retained_frac"], 1),
            "primary_coverage": res_pri["coverage"],
            "primary_retained_pct": round(100.0 * res_pri["retained_frac"], 1),
            "net_singleton_gain_pp": round(100.0 * (res_pri["retained_frac"] - res_s0["retained_frac"]), 1),
            "s0_ambiguous_resolved_n": n_res_pri,
            "s0_ambiguous_conversion_pct": conv_rate_pri,
            "converted_to_low_risk_0_n": n_to_low0,
            "converted_to_high_risk_1_n": n_to_high1,
            "converted_singleton_concordance_pct": round(100.0 * acc_converted, 1),
            "selective_nnr_in_ambiguous_zone": nnr_selective,
            "entropy_s0_bits": ent_s0,
            "entropy_primary_bits": ent_pri,
            "relative_entropy_reduction_pct": rel_ent_drop,
            "severe_survivors_n": n_sev_surv,
            "severe_survivors_s0_uncertain_or_high_n": int(s0_non_low.sum()),
            "severe_survivors_proactively_shielded_low0_n": int(rescued_to_low0.sum()),
            "severe_survivors_shielded_pct_of_flagged": round(100.0 * float(rescued_to_low0.sum()) / max(int(s0_non_low.sum()), 1), 1),
        }

    # Also evaluate eICU motor-charted subset (n=2,889) specifically for pure motor-check VoI!
    sub_ei_ch = df_preds[df_preds["cohort"] == "eICU_CRD"].iloc[~ei_miss_m].copy().reset_index(drop=True)
    y_eich = sub_ei_ch["d28"].values.astype(int)
    r_eich_s0  = eval_oof_conformal(y_eich, sub_ei_ch["p_static"].values, abstain_mask=None, alpha=0.10)
    r_eich_pri = eval_oof_conformal(y_eich, sub_ei_ch["p_adaptive_gbdt"].values, abstain_mask=None, alpha=0.10)
    n_eich_amb = int(r_eich_s0["is_amb"].sum())
    n_eich_res = int((r_eich_s0["is_amb"] & r_eich_pri["is_def"]).sum())
    flowsheet_s0_amb += n_eich_amb
    flowsheet_s0_amb_resolved += n_eich_res
    voi_results["eICU_CRD_Motor_Charted"] = {
        "n": int(len(y_eich)),
        "static_s0_retained_pct": round(100.0 * r_eich_s0["retained_frac"], 1),
        "primary_retained_pct": round(100.0 * r_eich_pri["retained_frac"], 1),
        "net_singleton_gain_pp": round(100.0 * (r_eich_pri["retained_frac"] - r_eich_s0["retained_frac"]), 1),
        "static_s0_ambiguous_n": n_eich_amb,
        "s0_ambiguous_resolved_n": n_eich_res,
        "s0_ambiguous_conversion_pct": round(100.0 * n_eich_res / max(n_eich_amb, 1), 1),
        "selective_nnr_in_ambiguous_zone": round(n_eich_amb / max(n_eich_res, 1), 2),
        "relative_entropy_reduction_pct": round(
            100.0 * (r_eich_s0["mean_entropy_bits"] - r_eich_pri["mean_entropy_bits"]) / max(r_eich_s0["mean_entropy_bits"], 1e-6), 1
        ),
    }

    # Also evaluate JinhuaNSICU CT-imaged subset (n=1,580) specifically for pure Head-CT NLP VoI when motor flowsheets are absent!
    sub_jh_ct = df_preds[df_preds["cohort"] == "JinhuaNSICU_TBI_S06"].iloc[jh_has_ct].copy().reset_index(drop=True)
    y_jhct = sub_jh_ct["d28"].values.astype(int)
    r_jhct_s0  = eval_oof_conformal(y_jhct, sub_jh_ct["p_static"].values, abstain_mask=None, alpha=0.10)
    r_jhct_pri = eval_oof_conformal(y_jhct, sub_jh_ct["p_adaptive_gbdt"].values, abstain_mask=None, alpha=0.10)
    n_jhct_amb = int(r_jhct_s0["is_amb"].sum())
    n_jhct_res = int((r_jhct_s0["is_amb"] & r_jhct_pri["is_def"]).sum())
    voi_results["JinhuaNSICU_CT_Imaged"] = {
        "n": int(len(y_jhct)),
        "static_s0_retained_pct": round(100.0 * r_jhct_s0["retained_frac"], 1),
        "primary_retained_pct": round(100.0 * r_jhct_pri["retained_frac"], 1),
        "net_singleton_gain_pp": round(100.0 * (r_jhct_pri["retained_frac"] - r_jhct_s0["retained_frac"]), 1),
        "static_s0_ambiguous_n": n_jhct_amb,
        "s0_ambiguous_resolved_n": n_jhct_res,
        "s0_ambiguous_conversion_pct": round(100.0 * n_jhct_res / max(n_jhct_amb, 1), 1),
        "selective_nnr_in_ambiguous_zone": round(n_jhct_amb / max(n_jhct_res, 1), 2),
        "relative_entropy_reduction_pct": round(
            100.0 * (r_jhct_s0["mean_entropy_bits"] - r_jhct_pri["mean_entropy_bits"]) / max(r_jhct_s0["mean_entropy_bits"], 1e-6), 1
        ),
    }

    out["proactive_conformal_voi"] = voi_results
    out["pooled_proactive_summary"] = {
        "five_charted_cohorts_s0_ambiguous_n": flowsheet_s0_amb,
        "five_charted_cohorts_resolved_by_motor_n": flowsheet_s0_amb_resolved,
        "five_charted_cohorts_conversion_pct": round(100.0 * flowsheet_s0_amb_resolved / max(flowsheet_s0_amb, 1), 1),
        "five_charted_cohorts_nnr": round(flowsheet_s0_amb / max(flowsheet_s0_amb_resolved, 1), 2),
        "all_six_cohorts_s0_ambiguous_n": total_s0_amb,
        "all_six_cohorts_resolved_n": total_s0_amb_resolved,
        "all_six_cohorts_conversion_pct": round(100.0 * total_s0_amb_resolved / max(total_s0_amb, 1), 1),
    }

    out_path = REP / "19h_v5_proactive_biomarker_voi.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(out["sedation_decoupled_awakening"]["Pooled_3Country_Severe"], indent=2))
    print(json.dumps(out["pooled_proactive_summary"], indent=2))
    for k, v in voi_results.items():
        print(f"  {k}: S0_ret={v['static_s0_retained_pct']}% -> Pri_ret={v['primary_retained_pct']}% | AmbConv={v['s0_ambiguous_conversion_pct']}% (NNR={v['selective_nnr_in_ambiguous_zone']}) | dEnt=-{v['relative_entropy_reduction_pct']}%")


if __name__ == "__main__":
    main()
