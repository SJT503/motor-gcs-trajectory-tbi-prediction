# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19i_v5_2_sequential_conformal_corridor.py — True Chronological 0h -> 24h -> 48h -> 72h
Sequential Conformal Proactive Triage Corridor & Unsedated-Coma Severity Audit (v5.2)

Resolves Top-Reviewer Methodological Critiques:
  1. True t = 0 h Arrival Baseline (Static-3: admission_age, gcs_total_first, gcs_motor_first;
     zero 0-24h labs/vitals) -> t = 24 h Post-Resuscitation Motor/CT Triage:
     Quantifies true chronological 0h -> 24h uncertainty resolution alongside 24h marginal
     Static-S0 -> Primary modality VoI.
  2. Longitudinal Closure of Residual 24-h {0, 1} Uncertain Stays at t = 48 h and t = 72 h
     across all 5 serial-flowsheet cohorts (MIMIC-IV, MIMIC-III, eICU-Charted, HiRID, AmsterdamUMCdb):
     Tracks clinical ICU exit (24-72h) and new conformal singleton conversions at 48h and 72h
     under leak-free rolling features and >=90% Mondrian coverage.
  3. Three-Country Unsedated vs Sedated Severe TBI (GCS 3-8) Arrival Coma Profile:
     Quantifies arrival GCS = 3 / mGCS = 1 prevalence and age to verify confounding by profound
     organic coma (obtunded patients tolerating intubation without sedatives).
"""
import json, math, re, warnings
from pathlib import Path
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

ROOT   = Path("E:/TBI subtype")
SYS    = ROOT / "07_prediction_system"
DATA   = SYS / "data"
REP    = SYS / "reports"
MODELS = SYS / "models"
gate(str(SYS / "scripts/19i_v5_2_sequential_conformal_corridor.py"))

df_preds = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
art_calc = json.loads((MODELS / "19c_v4_web_calculator_artifact.json").read_text(encoding="utf-8"))
card     = json.loads((MODELS / "08d_v3_model_card.json").read_text(encoding="utf-8"))
rep0     = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
SEL      = card["features"]
FULL     = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

imputers = [joblib.load(MODELS / f"08d_v3_mice_imp{j}.joblib") for j in range(5)]
frozen_lgb_boosters = [lgb.Booster(model_file=str(MODELS / f"08d_v3_lgbm_sel_imp{j}.txt")) for j in range(5)]

canary_gate(lambda: len(df_preds) == 9964 and len(art_calc["static3_5boosters"]["boosters_trees"]) == 5, "v5.2 inputs loaded")


def eval_tree(node, x_vec):
    if "leaf_value" in node:
        return float(node["leaf_value"])
    return eval_tree(node["left_child"], x_vec) if x_vec[int(node["split_feature"])] <= float(node["threshold"]) else eval_tree(node["right_child"], x_vec)


def predict_static3_arrival(df_mat):
    """Exact 1,000-tree Static-3 (admission_age, gcs_total_first, gcs_motor_first) at t = 0 h."""
    s3_trees_5 = art_calc["static3_5boosters"]["boosters_trees"]
    s3_cols = ["admission_age", "gcs_total_first", "gcs_motor_first"]
    df_tmp = df_mat.copy()
    for c in FULL:
        if c not in df_tmp.columns:
            df_tmp[c] = np.nan
    # Strictly mask ALL post-arrival (0-24h) lab/vital/dynamic columns to NaN so MICE uses ONLY t=0h arrival info!
    keep_t0 = {"admission_age", "sex_female", "gcs_total_first", "gcs_motor_first", "gcs_eye_first", "gcs_verbal_first"}
    for c in FULL:
        if c not in keep_t0:
            df_tmp[c] = np.nan
    p_sum = np.zeros(len(df_tmp), dtype=np.float64)
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        imp_arr = pd.DataFrame(imputers[j].transform(df_tmp[FULL]), columns=FULL)[s3_cols].values
        trees_j = [t["tree_structure"] for t in s3_trees_5[j]]
        raw_j = np.array([expit(sum(eval_tree(tr, row) for tr in trees_j)) for row in imp_arr], dtype=np.float64)
        p_sum += raw_j / 5.0
    return p_sum


def exact_mondrian_thresholds(y_cal, p_cal, alpha=0.10):
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    s1 = np.sort(1.0 - p_cal[y_cal == 1])
    s0 = np.sort(p_cal[y_cal == 0])
    n1, n0 = len(s1), len(s0)
    k1 = min(max(1, math.ceil((1.0 - alpha) * (n1 + 1))), n1)
    k0 = min(max(1, math.ceil((1.0 - alpha) * (n0 + 1))), n0)
    return float(1.0 - s1[k1 - 1]), float(s0[k0 - 1])


def eval_oof_conformal(y, p_raw, abstain_mask=None, alpha=0.10, n_splits=20, seed=42):
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
        "is_sing0": is_sing0,
        "is_sing1": is_sing1,
        "is_def": is_def,
        "is_amb": is_amb,
        "coverage": round(cov, 4),
        "retained_pct": round(100.0 * float(is_def.mean()), 1),
        "ambiguous_pct": round(100.0 * float(is_amb.mean()), 1),
        "def_auroc": round(auc_def, 4) if auc_def is not None else None,
        "raw_auroc": round(float(roc_auc_score(y, p_raw)), 4),
    }


# Leak-free LCGM + LightGBM helper for HiRID & Amsterdam at t=48h & t=72h
raw_lcgm = json.loads((REP / "08a3_gbtm_coefs_trainonly.json").read_text(encoding="utf-8"))
entries = [(str(e["name"]).strip(), float(e["value"])) for e in raw_lcgm["motor"]["fit"]["best_vec"]]
def _fc(pat):
    return {int(re.fullmatch(pat, n).group(1)): v for n, v in entries if re.fullmatch(pat, n)}
B0 = _fc(r"intercept\s*class\s*(\d+)")
B1 = _fc(r"t10\s*class\s*(\d+)")
B2 = _fc(r"I\(t10\^2\)\s*class\s*(\d+)")
TAU2 = next((v for n, v in entries if re.fullmatch(r"var\s*1", n)), 0.0)
SIG2 = next((v for n, v in entries if re.fullmatch(r"std\.?err", n)), 1.0) ** 2
SH = np.asarray(raw_lcgm["motor"]["class_share"], dtype=float)
PI_PRIOR = SH / SH.sum()
V7 = ["hr", "sbp", "dbp", "mbp", "rr", "temp", "spo2"]
LABS = ["hematocrit", "hemoglobin", "platelet", "wbc", "aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine", "glucose_lab", "sodium", "potassium", "inr", "ptt"]


def compute_posteriors(df_gcs, id_col="id", max_hr=48.0):
    d_m = df_gcs[(df_gcs["offset_hr"] <= max_hr) & df_gcs["gcs_motor"].notna()].copy()
    d_m["t10"] = d_m["offset_hr"] / 10.0
    ids_order = np.array(sorted(d_m[id_col].unique()))
    idx_map = {g: i for i, g in enumerate(ids_order)}
    d_m["row"] = d_m[id_col].map(idx_map)
    logp = np.full((len(ids_order), 3), -np.inf)
    for g in range(1, 4):
        m_val = B0[g] + B1[g] * d_m.t10 + B2[g] * d_m.t10 ** 2
        r = d_m.gcs_motor.values - m_val.values
        agg = pd.DataFrame({"row": d_m.row.values, "r": r}).groupby("row").agg(
            K=("r", "size"), S1=("r", "sum"), S2=("r", lambda z: float((z ** 2).sum()))
        )
        K = agg.K.values.astype(float); S1 = agg.S1.values; S2 = agg.S2.values
        logdet = K * np.log(SIG2) + np.log1p(K * TAU2 / SIG2)
        q = (S2 - (TAU2 / (SIG2 + K * TAU2)) * S1 ** 2) / SIG2
        lp = -0.5 * (K * np.log(2 * np.pi) + logdet + q) + np.log(PI_PRIOR[g - 1])
        logp[agg.index.values, g - 1] = lp
    mx = logp.max(axis=1, keepdims=True)
    p_post = np.exp(logp - mx); p_post /= p_post.sum(axis=1, keepdims=True)
    return pd.DataFrame({id_col: ids_order, "prob_m1": p_post[:, 0], "prob_m2": p_post[:, 1], "prob_m3": p_post[:, 2]})


def predict_eu_at_hr(meta_df, id_col, bins_df, gcs_df, hr, is_hirid=False):
    if "hours_to_death" in meta_df.columns:
        alive_m = (meta_df["los_h"] >= hr) & (meta_df["hours_to_death"].isna() | (meta_df["hours_to_death"] > hr))
    else:
        alive_m = (meta_df["los_h"] >= hr)
    sub_meta = meta_df.loc[alive_m].copy()
    g_sub = gcs_df[(gcs_df["id"].isin(set(sub_meta[id_col]))) & (gcs_df["offset_hr"] <= hr)].sort_values(["id", "offset_hr"])
    g_mot = g_sub[g_sub["gcs_motor"].notna()]
    valid_ids = set(g_mot["id"].unique())
    sub_meta = sub_meta[sub_meta[id_col].isin(valid_ids)].copy().reset_index(drop=True)
    g_sub = g_sub[g_sub["id"].isin(valid_ids)]
    g_mot = g_mot[g_mot["id"].isin(valid_ids)]

    def first_nn(s):
        v = s.dropna()
        return float(v.iloc[0]) if len(v) else np.nan

    ga_h = g_sub.groupby("id").agg(
        gcs_total_min=("gcs_total", "min"), gcs_total_first=("gcs_total", first_nn),
        gcs_motor_min=("gcs_motor", "min"), gcs_motor_first=("gcs_motor", first_nn),
        gcs_eye_min=("gcs_eye", "min"), gcs_eye_first=("gcs_eye", first_nn),
        gcs_verbal_min=("gcs_verbal", "min"), gcs_verbal_first=("gcs_verbal", first_nn),
    ).reset_index().rename(columns={"id": id_col})
    grp_m = g_mot.groupby("id")
    dyn_h = grp_m["gcs_motor"].agg(
        motor_first="first", motor_last="last", motor_min="min", motor_max="max",
        motor_sd=lambda s: float(np.std(s.values, ddof=1)) if len(s) >= 2 else 0.0
    ).reset_index().rename(columns={"id": id_col})
    sl_h = pd.DataFrame([
        (pid_k, float(np.polyfit(gk["offset_hr"].values, gk["gcs_motor"].values, 1)[0]) if (len(gk) >= 2 and gk["offset_hr"].max() > gk["offset_hr"].min()) else 0.0)
        for pid_k, gk in grp_m
    ], columns=[id_col, "motor_slope"])
    post_h = compute_posteriors(g_mot, id_col="id", max_hr=float(hr)).rename(columns={"id": id_col})

    b_h = bins_df[(bins_df[id_col].isin(valid_ids)) & (bins_df["hr_bin"] <= hr)]
    agg_b = b_h.groupby([id_col, "feat"], as_index=False).agg(mn=("mn", "min"), mx=("mx", "max"), sm=("sm", "sum"), cnt=("cnt", "sum"))
    agg_b["mean_v"] = agg_b["sm"] / np.maximum(agg_b["cnt"], 1)
    piv_mn = agg_b.pivot(index=id_col, columns="feat", values="mn")
    piv_mx = agg_b.pivot(index=id_col, columns="feat", values="mx")
    piv_me = agg_b.pivot(index=id_col, columns="feat", values="mean_v")

    X_hr = sub_meta[[id_col, "admission_age", "sex_female", "d28"]].merge(ga_h, on=id_col, how="left") \
                   .merge(dyn_h, on=id_col, how="left").merge(sl_h, on=id_col, how="left") \
                   .merge(post_h, on=id_col, how="left")
    idx_ids = X_hr[id_col].values
    for v in V7:
        X_hr[f"{v}_min"] = piv_mn[v].reindex(idx_ids).values if v in piv_mn.columns else np.nan
        X_hr[f"{v}_mean"] = piv_me[v].reindex(idx_ids).values if v in piv_me.columns else np.nan
        X_hr[f"{v}_max"] = piv_mx[v].reindex(idx_ids).values if v in piv_mx.columns else np.nan
    for lb in LABS:
        X_hr[f"{lb}_min"] = piv_mn[lb].reindex(idx_ids).values if lb in piv_mn.columns else np.nan
        X_hr[f"{lb}_max"] = piv_mx[lb].reindex(idx_ids).values if lb in piv_mx.columns else np.nan
    if is_hirid:
        X_hr["hematocrit_min"] = X_hr["hemoglobin_min"] * 3.0
        X_hr["hematocrit_max"] = X_hr["hemoglobin_max"] * 3.0
        X_hr["pt_min"] = X_hr["inr_min"] * 11.5
        X_hr["pt_max"] = X_hr["inr_max"] * 11.5
    else:
        pt_d_mn = piv_mn["pt_direct"].reindex(idx_ids).values if "pt_direct" in piv_mn.columns else np.full(len(idx_ids), np.nan)
        pt_i_mn = piv_mn["pt_from_inr"].reindex(idx_ids).values if "pt_from_inr" in piv_mn.columns else np.full(len(idx_ids), np.nan)
        pt_d_mx = piv_mx["pt_direct"].reindex(idx_ids).values if "pt_direct" in piv_mx.columns else np.full(len(idx_ids), np.nan)
        pt_i_mx = piv_mx["pt_from_inr"].reindex(idx_ids).values if "pt_from_inr" in piv_mx.columns else np.full(len(idx_ids), np.nan)
        X_hr["pt_min"] = pd.Series(pt_d_mn).fillna(pd.Series(pt_i_mn)).values
        X_hr["pt_max"] = pd.Series(pt_d_mx).fillna(pd.Series(pt_i_mx)).values
    X_hr["mbp_min"] = X_hr["mbp_min"].fillna((X_hr["sbp_min"] + 2.0 * X_hr["dbp_min"]) / 3.0)
    X_hr["mbp_mean"] = X_hr["mbp_mean"].fillna((X_hr["sbp_mean"] + 2.0 * X_hr["dbp_mean"]) / 3.0)
    X_hr["mbp_max"] = X_hr["mbp_max"].fillna((X_hr["sbp_max"] + 2.0 * X_hr["dbp_max"]) / 3.0)
    X_hr["aniongap_min"] = X_hr["aniongap_min"].fillna(X_hr["sodium_min"] - (X_hr["chloride_max"] + X_hr["bicarbonate_max"]))
    X_hr["aniongap_max"] = X_hr["aniongap_max"].fillna(X_hr["sodium_max"] - (X_hr["chloride_min"] + X_hr["bicarbonate_min"]))
    X_hr["motor_delta"] = X_hr["motor_last"] - X_hr["motor_first"]
    X_hr["charlson_comorbidity_score"] = np.nan
    for c in FULL:
        if c not in X_hr.columns:
            X_hr[c] = np.nan
    p_sum = np.zeros(len(X_hr), dtype=np.float64)
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        imp_df = pd.DataFrame(imputers[j].transform(X_hr[FULL]), columns=FULL, index=X_hr.index)
        p_sum += frozen_lgb_boosters[j].predict(imp_df[SEL]) / 5.0
    return pd.DataFrame({"id": X_hr[id_col].astype(int).values, "d28": X_hr["d28"].astype(int).values, "p": p_sum})


@retry_limit(2)
def main():
    con = duckdb.connect()
    # Load exact 6-cohort baseline matrices in exact row order matching df_preds
    m4_all = pd.read_parquet(DATA / "08c_matrix_mimic4_v3.parquet")
    d_m4 = m4_all[m4_all["split"] == "intval"].copy().reset_index(drop=True)

    fe_m3 = pd.read_parquet(DATA / "features_mimic3.parquet").rename(columns={"age": "admission_age"})
    post_m3 = pd.read_parquet(DATA / "08b_posterior24_mimic3_trainonly.parquet").rename(columns={"id": "icustay_id"})
    d_m3 = fe_m3.merge(post_m3[["icustay_id", "prob_m1"]], on="icustay_id", how="left")
    cv_ids = con.execute(f"SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'").df()
    d_m3 = d_m3[d_m3.icustay_id.isin(cv_ids.icustay_id)].copy()
    d_m3 = d_m3[~(((d_m3.d28 == 1) & (d_m3.days_to_death < 1.0)) | (d_m3.los_h < 24))].copy().reset_index(drop=True)

    fe_ei = pd.read_parquet(DATA / "features_eicu.parquet").rename(columns={"age": "admission_age"})
    post_ei = pd.read_parquet(DATA / "08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d_ei = fe_ei.merge(post_ei[["icustay_id_eicu", "prob_m1"]], on="icustay_id_eicu", how="left")
    d_ei = d_ei[~(((d_ei.died_hosp == 1) & (d_ei.los_h <= 24)) | (d_ei.los_h < 24))].copy().reset_index(drop=True)
    d_ei["d28"] = d_ei["died_hosp_28d"].astype(int)
    ei_charted_m = d_ei["prob_m1"].notna().values

    hi_all = pd.read_parquet(DATA / "08c_matrix_hirid_v3.parquet")
    d_hi = hi_all[hi_all["is_trauma"] == True].copy().reset_index(drop=True)

    au_all = pd.read_parquet(DATA / "19a_matrix_amsterdam_v4.parquet")
    d_au = au_all[au_all["is_trauma"] == True].copy().reset_index(drop=True)

    jh_all = pd.read_parquet(DATA / "19a_matrix_jinhua_v4.parquet")
    d_jh = jh_all[jh_all["is_tbi_s06"] == True].copy().reset_index(drop=True)
    ct_df = pd.read_parquet(DATA / "19a_ct_nlp_3cohorts_v4.parquet")
    ct_jh = ct_df[ct_df["cohort"] == "JinhuaNSICU"].set_index("id")
    jh_id_col = "hadm_id" if d_jh["hadm_id"].isin(ct_jh.index).sum() > 100 else "subject_id"
    jh_has_ct = ct_jh.reindex(d_jh[jh_id_col].values)["ct_any_24h"].fillna(0).values > 0.5

    # Load 48h & 72h rolling predictions across all 5 serial-flowsheet cohorts
    pr_m4 = pd.read_parquet(DATA / "09d_rolling_preds_mimic4_v3.parquet")
    pr_m3 = pd.read_parquet(DATA / "09d_rolling_preds_mimic3_v3.parquet")
    pr_ei = pd.read_parquet(DATA / "09d_rolling_preds_eicu_v3.parquet")

    hi_meta = pd.read_parquet(DATA / "19a_roll72_hirid_meta_v4.parquet")
    hi_bins = pd.read_parquet(DATA / "19a_roll72_hirid_v4.parquet")
    hi_gcs  = pd.read_parquet(DATA / "19a_gcs72_hirid_roll_v4.parquet")
    hi_48 = predict_eu_at_hr(hi_meta, "patientid", hi_bins, hi_gcs, 48, is_hirid=True)
    hi_72 = predict_eu_at_hr(hi_meta, "patientid", hi_bins, hi_gcs, 72, is_hirid=True)

    au_meta = pd.read_parquet(DATA / "19a_roll72_amsterdam_meta_v4.parquet")
    au_bins = pd.read_parquet(DATA / "19a_roll72_amsterdam_v4.parquet")
    au_gcs  = pd.read_parquet(DATA / "19a_gcs72_amsterdam_roll_v4.parquet")
    au_48 = predict_eu_at_hr(au_meta, "admissionid", au_bins, au_gcs, 48, is_hirid=False)
    au_72 = predict_eu_at_hr(au_meta, "admissionid", au_bins, au_gcs, 72, is_hirid=False)

    corridor = {}
    serial_specs = [
        ("MIMIC_IV_IntVal", d_m4, "stay_id", df_preds[df_preds["cohort"] == "MIMIC_IV_IntVal"].reset_index(drop=True),
         pr_m4[pr_m4["t"] == 48][["id", "d28", "p"]], pr_m4[pr_m4["t"] == 72][["id", "d28", "p"]], None),
        ("MIMIC_III_CareVue", d_m3, "icustay_id", df_preds[df_preds["cohort"] == "MIMIC_III_CareVue"].reset_index(drop=True),
         pr_m3[pr_m3["t"] == 48][["id", "d28", "p"]], pr_m3[pr_m3["t"] == 72][["id", "d28", "p"]], None),
        ("eICU_CRD_Motor_Charted", d_ei[ei_charted_m].reset_index(drop=True), "icustay_id_eicu",
         df_preds[df_preds["cohort"] == "eICU_CRD"].iloc[np.where(ei_charted_m)[0]].reset_index(drop=True),
         pr_ei[(pr_ei["t"] == 48) & (pr_ei["id"].astype(int).isin(set(d_ei.loc[ei_charted_m, "icustay_id_eicu"].astype(int))))][["id", "d28", "p"]],
         pr_ei[(pr_ei["t"] == 72) & (pr_ei["id"].astype(int).isin(set(d_ei.loc[ei_charted_m, "icustay_id_eicu"].astype(int))))][["id", "d28", "p"]], None),
        ("HiRID_Trauma", d_hi, "patientid", df_preds[df_preds["cohort"] == "HiRID_Trauma"].reset_index(drop=True),
         hi_48, hi_72, None),
        ("AmsterdamUMCdb_Trauma", d_au, "admissionid", df_preds[df_preds["cohort"] == "AmsterdamUMCdb_Trauma"].reset_index(drop=True),
         au_48, au_72, None),
    ]

    pooled_n = 0
    pooled_t0_amb = 0
    pooled_t0_res_by_24h = 0
    pooled_t0_res_by_b5 = 0
    pooled_t24_amb = 0
    pooled_t24_amb_active48 = 0
    pooled_t24_amb_res48 = 0
    pooled_t24_amb_active72 = 0
    pooled_t24_amb_res48_or_72 = 0
    pooled_t24_amb_exited_icu_by_72 = 0
    pooled_persistent_72h_amb = 0

    for cname, d_coh, id_c, sub_p, df48, df72, abst_m in serial_specs:
        ids = d_coh[id_c].astype(int).values
        y = sub_p["d28"].values.astype(int)
        assert np.array_equal(y, d_coh["d28"].values.astype(int)), f"Row order mismatch in {cname}"

        # Stage 0 (t = 0 h): True Arrival Static-3 (zero 24-h labs/vitals!)
        p_t0 = predict_static3_arrival(d_coh)
        c_t0 = eval_oof_conformal(y, p_t0, alpha=0.10)

        # Stage 1 (t = 24 h): Bedside-5 (zero-lab) & Full Modality-Adaptive Primary
        c_b5  = eval_oof_conformal(y, sub_p["p_bedside5"].values, alpha=0.10)
        c_t24 = eval_oof_conformal(y, sub_p["p_adaptive_gbdt"].values, alpha=0.10)

        t0_amb = c_t0["is_amb"]
        n_t0_amb = int(t0_amb.sum())
        n_t0_res24 = int((t0_amb & c_t24["is_def"]).sum())
        n_t0_res_b5 = int((t0_amb & c_b5["is_def"]).sum())

        # Stage 2 (t = 48 h & t = 72 h): Evaluate conformal sets on stays remaining in ICU at 48h & 72h
        df48_u = df48.drop_duplicates(subset=["id"]).copy()
        df48_u["id"] = df48_u["id"].astype(int)
        df48_u = df48_u[df48_u["id"].isin(set(ids))].reset_index(drop=True)
        c_48 = eval_oof_conformal(df48_u["d28"].values, df48_u["p"].values, alpha=0.10)
        def48_ids = set(df48_u.loc[c_48["is_def"], "id"].values)
        act48_ids = set(df48_u["id"].values)

        df72_u = df72.drop_duplicates(subset=["id"]).copy()
        df72_u["id"] = df72_u["id"].astype(int)
        df72_u = df72_u[df72_u["id"].isin(set(ids))].reset_index(drop=True)
        c_72 = eval_oof_conformal(df72_u["d28"].values, df72_u["p"].values, alpha=0.10)
        def72_ids = set(df72_u.loc[c_72["is_def"], "id"].values)
        act72_ids = set(df72_u["id"].values)

        t24_amb = c_t24["is_amb"]
        amb24_ids = set(ids[t24_amb])
        n_t24_amb = len(amb24_ids)

        amb24_act48 = amb24_ids & act48_ids
        amb24_res48 = amb24_ids & def48_ids
        amb24_act72 = amb24_ids & act72_ids
        amb24_res48_or_72 = amb24_ids & (def48_ids | def72_ids)
        amb24_exited_by_72 = amb24_ids - act72_ids
        # Persistent 72h uncertain = still in ICU at 72h AND resolved at neither 24h, 48h, nor 72h!
        persistent_72 = amb24_act72 - (def48_ids | def72_ids)

        corridor[cname] = {
            "n_total_24h": int(len(ids)),
            "t0_arrival_static3": {
                "auroc": c_t0["raw_auroc"],
                "coverage": c_t0["coverage"],
                "definitive_pct": c_t0["retained_pct"],
                "uncertain_n": n_t0_amb,
                "uncertain_pct": c_t0["ambiguous_pct"],
            },
            "t0_to_t24_chronological_voi": {
                "resolved_by_primary_n": n_t0_res24,
                "resolved_by_primary_pct": round(100.0 * n_t0_res24 / max(n_t0_amb, 1), 1),
                "selective_nnr_primary": round(n_t0_amb / max(n_t0_res24, 1), 2),
                "resolved_by_bedside5_n": n_t0_res_b5,
                "resolved_by_bedside5_pct": round(100.0 * n_t0_res_b5 / max(n_t0_amb, 1), 1),
                "selective_nnr_bedside5": round(n_t0_amb / max(n_t0_res_b5, 1), 2),
            },
            "t24_residual_to_48_72h_closure": {
                "t24_uncertain_n": n_t24_amb,
                "t24_uncertain_pct": c_t24["ambiguous_pct"],
                "active_in_icu_48h_n": len(amb24_act48),
                "newly_resolved_at_48h_n": len(amb24_res48),
                "newly_resolved_at_48h_pct_of_active48": round(100.0 * len(amb24_res48) / max(len(amb24_act48), 1), 1),
                "coverage_48h": c_48["coverage"],
                "active_in_icu_72h_n": len(amb24_act72),
                "newly_resolved_by_48_or_72h_n": len(amb24_res48_or_72),
                "newly_resolved_by_48_or_72h_pct_of_t24_uncertain": round(100.0 * len(amb24_res48_or_72) / max(n_t24_amb, 1), 1),
                "exited_icu_clinically_24_72h_n": len(amb24_exited_by_72),
                "coverage_72h": c_72["coverage"],
                "persistent_72h_uncertain_n": len(persistent_72),
                "persistent_72h_uncertain_pct_of_cohort": round(100.0 * len(persistent_72) / len(ids), 1),
                "cumulative_conformal_or_clinical_resolved_by_72h_pct": round(100.0 * (len(ids) - len(persistent_72)) / len(ids), 1),
            },
        }
        pooled_n += len(ids)
        pooled_t0_amb += n_t0_amb
        pooled_t0_res_by_24h += n_t0_res24
        pooled_t0_res_by_b5 += n_t0_res_b5
        pooled_t24_amb += n_t24_amb
        pooled_t24_amb_active48 += len(amb24_act48)
        pooled_t24_amb_res48 += len(amb24_res48)
        pooled_t24_amb_active72 += len(amb24_act72)
        pooled_t24_amb_res48_or_72 += len(amb24_res48_or_72)
        pooled_t24_amb_exited_icu_by_72 += len(amb24_exited_by_72)
        pooled_persistent_72h_amb += len(persistent_72)

    # Also compute t=0h Static-3 -> t=24h Head-CT chronological VoI in Jinhua CT-Imaged (n=1,580)
    d_jh_ct = d_jh[jh_has_ct].reset_index(drop=True)
    sub_jh_ct = df_preds[df_preds["cohort"] == "JinhuaNSICU_TBI_S06"].iloc[np.where(jh_has_ct)[0]].reset_index(drop=True)
    y_jhc = sub_jh_ct["d28"].values.astype(int)
    p_t0_jhc = predict_static3_arrival(d_jh_ct)
    c_t0_jhc = eval_oof_conformal(y_jhc, p_t0_jhc, alpha=0.10)
    c_24_jhc = eval_oof_conformal(y_jhc, sub_jh_ct["p_adaptive_gbdt"].values, alpha=0.10)
    n_t0_amb_jh = int(c_t0_jhc["is_amb"].sum())
    n_t0_res_jh = int((c_t0_jhc["is_amb"] & c_24_jhc["is_def"]).sum())
    corridor["JinhuaNSICU_CT_Imaged_0h_to_24h"] = {
        "n_ct_imaged": int(len(y_jhc)),
        "t0_arrival_static3_auroc": c_t0_jhc["raw_auroc"],
        "t0_arrival_definitive_pct": c_t0_jhc["retained_pct"],
        "t0_arrival_uncertain_n": n_t0_amb_jh,
        "resolved_at_24h_by_labs_and_ct_n": n_t0_res_jh,
        "resolved_at_24h_pct": round(100.0 * n_t0_res_jh / max(n_t0_amb_jh, 1), 1),
        "selective_nnr": round(n_t0_amb_jh / max(n_t0_res_jh, 1), 2),
    }

    pooled_summary = {
        "n_pooled_5_serial_cohorts": pooled_n,
        "t0_arrival_uncertain_n": pooled_t0_amb,
        "t0_arrival_uncertain_pct_of_all": round(100.0 * pooled_t0_amb / pooled_n, 1),
        "t0_uncertain_resolved_at_24h_primary_n": pooled_t0_res_by_24h,
        "t0_uncertain_resolved_at_24h_primary_pct": round(100.0 * pooled_t0_res_by_24h / pooled_t0_amb, 1),
        "t0_to_24h_primary_nnr": round(pooled_t0_amb / pooled_t0_res_by_24h, 2),
        "t0_uncertain_resolved_at_24h_bedside5_n": pooled_t0_res_by_b5,
        "t0_uncertain_resolved_at_24h_bedside5_pct": round(100.0 * pooled_t0_res_by_b5 / pooled_t0_amb, 1),
        "t0_to_24h_bedside5_nnr": round(pooled_t0_amb / pooled_t0_res_by_b5, 2),
        "t24_residual_uncertain_n": pooled_t24_amb,
        "t24_residual_uncertain_pct_of_all": round(100.0 * pooled_t24_amb / pooled_n, 1),
        "t24_uncertain_active_48h_n": pooled_t24_amb_active48,
        "t24_uncertain_newly_resolved_at_48h_n": pooled_t24_amb_res48,
        "t24_uncertain_newly_resolved_at_48h_pct_of_active48": round(100.0 * pooled_t24_amb_res48 / pooled_t24_amb_active48, 1),
        "t24_uncertain_newly_resolved_by_48_or_72h_n": pooled_t24_amb_res48_or_72,
        "t24_uncertain_newly_resolved_by_48_or_72h_pct": round(100.0 * pooled_t24_amb_res48_or_72 / pooled_t24_amb, 1),
        "t24_uncertain_exited_icu_clinically_24_72h_n": pooled_t24_amb_exited_icu_by_72,
        "t24_uncertain_exited_icu_clinically_24_72h_pct": round(100.0 * pooled_t24_amb_exited_icu_by_72 / pooled_t24_amb, 1),
        "persistent_72h_uncertain_n": pooled_persistent_72h_amb,
        "persistent_72h_uncertain_pct_of_all_stays": round(100.0 * pooled_persistent_72h_amb / pooled_n, 1),
        "cumulative_0_to_72h_resolved_pct_of_all_stays": round(100.0 * (pooled_n - pooled_persistent_72h_amb) / pooled_n, 1),
    }

    # Part 3: Unsedated vs Sedated GCS 3-8 Arrival Coma Depth (explaining why unsedated mortality is higher!)
    sed_m4_ids = set(con.execute(f"""
        SELECT DISTINCT CAST(e.stay_id AS BIGINT) AS stay_id
        FROM read_parquet('{DATA}/08c_matrix_mimic4_v3.parquet') m
        JOIN read_csv('{ROOT}/data/mimic-iv-3.1/icu/icustays.csv.gz', all_varchar=true) icu ON CAST(m.stay_id AS BIGINT) = CAST(icu.stay_id AS BIGINT)
        JOIN read_csv('{ROOT}/data/mimic-iv-3.1/icu/inputevents.csv.gz', all_varchar=true) e ON CAST(m.stay_id AS BIGINT) = CAST(e.stay_id AS BIGINT)
        WHERE e.itemid IN ('222168', '221668', '229420', '225150')
          AND TRY_CAST(e.starttime AS TIMESTAMP) < TRY_CAST(icu.intime AS TIMESTAMP) + INTERVAL 24 HOUR
          AND TRY_CAST(e.endtime AS TIMESTAMP)   > TRY_CAST(icu.intime AS TIMESTAMP)
    """).df()["stay_id"].tolist())
    sed_au_ids = set(con.execute(f"""
        SELECT DISTINCT CAST(admissionid AS BIGINT) AS admissionid
        FROM read_csv('F:/AmsterdamUMCdb/drugitems.csv', encoding='latin-1', all_varchar=true, strict_mode=false, ignore_errors=true)
        WHERE item IN ('Propofol (Diprivan)', 'Midazolam (Dormicum)', 'Fentanyl', 'Remifentanyl', 'Sufentanil (Sufenta)')
          AND TRY_CAST(start AS DOUBLE) < 86400000 AND TRY_CAST(stop AS DOUBLE) > 0
    """).df()["admissionid"].tolist())

    m4_sev = m4_all[(m4_all["gcs_total_first"] <= 8) & m4_all["motor_last"].notna()].copy()
    m4_sev["sed24"] = m4_sev["stay_id"].astype(int).isin(sed_m4_ids)
    hi_sev = d_hi[(d_hi["gcs_total_first"] <= 8) & d_hi["motor_last"].notna()].copy()
    hi_sev["sed24"] = hi_sev["rass_min_24h"].notna() & (hi_sev["rass_min_24h"] <= -3.0)
    au_sev = d_au[(d_au["gcs_total_first"] <= 8) & d_au["motor_last"].notna()].copy()
    au_sev["sed24"] = au_sev["admissionid"].astype(int).isin(sed_au_ids)

    p3 = pd.concat([
        m4_sev[["admission_age", "gcs_total_first", "gcs_motor_first", "motor_last", "d28", "sed24"]],
        hi_sev[["admission_age", "gcs_total_first", "gcs_motor_first", "motor_last", "d28", "sed24"]],
        au_sev[["admission_age", "gcs_total_first", "gcs_motor_first", "motor_last", "d28", "sed24"]],
    ], ignore_index=True)

    sed_sub = p3[p3["sed24"] == True]
    uns_sub = p3[p3["sed24"] == False]
    coma_profile = {
        "sedated_n": int(len(sed_sub)),
        "unsedated_n": int(len(uns_sub)),
        "sedated_median_age": round(float(sed_sub["admission_age"].median()), 1),
        "unsedated_median_age": round(float(uns_sub["admission_age"].median()), 1),
        "sedated_nonawakener_m1_pct": round(100.0 * float((sed_sub.loc[sed_sub["motor_last"] <= 4, "motor_last"] == 1).mean()), 1),
        "unsedated_nonawakener_m1_pct": round(100.0 * float((uns_sub.loc[uns_sub["motor_last"] <= 4, "motor_last"] == 1).mean()), 1),
        "sedated_age_ge70_pct": round(100.0 * float((sed_sub["admission_age"] >= 70).mean()), 1),
        "unsedated_age_ge70_pct": round(100.0 * float((uns_sub["admission_age"] >= 70).mean()), 1),
    }

    out = {
        "sequential_0_24_48_72h_corridor_by_cohort": corridor,
        "pooled_5_serial_cohorts_corridor": pooled_summary,
        "unsedated_vs_sedated_coma_profile": coma_profile,
    }
    out_path = REP / "19i_v5_2_sequential_conformal_corridor.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", out_path)
    print(json.dumps({"pooled": pooled_summary, "coma_profile": coma_profile}, indent=2))


if __name__ == "__main__":
    main()
