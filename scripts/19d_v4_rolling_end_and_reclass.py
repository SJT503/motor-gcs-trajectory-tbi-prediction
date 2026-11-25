# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19d_v4_rolling_end_and_reclass.py — Leak-Free 67-Checkpoint Rolling Evaluation, END Lead-Times & 5-Cohort Rapid-Awakening (v4.2)
Remediates Opus 5.5 Max R3, Y4, Y5, Y6, Y14:
  1. R3 Fix: Computes 100% leak-free [-6h/0, t] rolling static vitals/labs + gcs_*_first/min + motor dynamic + LCGM posteriors
     on the TRUE alive-at-t risk set (los_h >= t & alive at t & >=1 mGCS in [0, t]) for BOTH Dutch AmsterdamUMCdb AND Swiss HiRID
     (zero 24-h static matrix reuse, zero fallback to 24-h values!).
  2. Y4 Fix: Supplements Mann-Kendall trend statistics with patient-level paired bootstrap Delta-AUROC (24h vs 6h) on common
     24h-surviving cohorts to account for serial correlation, reporting all 5 serial-flowsheet cohorts transparently.
  3. Y5 Fix: Exports END event detection sensitivity & false-alarm burden per 100 patient-days alongside warning lead times.
  4. Y6 Fix: Deterministic MICE (random_state = 100 + j before every transform).
"""
import os, sys, json, time, re, warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from scipy.stats import kendalltau, norm

warnings.filterwarnings("ignore")
gate(__file__)

ROOT = r"E:\TBI subtype"
DATA = f"{ROOT}/07_prediction_system/data"
REP  = f"{ROOT}/07_prediction_system/reports"
MODELS = f"{ROOT}/07_prediction_system/models"

t0 = time.time()
def lg(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

card = json.load(open(f"{MODELS}/08d_v3_model_card.json", encoding="utf-8"))
SEL = card["features"]
THR_RAW = float(card["threshold_youden_train_oof"])
rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

imputers = [joblib.load(f"{MODELS}/08d_v3_mice_imp{j}.joblib") for j in range(5)]
frozen_lgb_boosters = [lgb.Booster(model_file=f"{MODELS}/08d_v3_lgbm_sel_imp{j}.txt") for j in range(5)]

def deterministic_predict_lgb(df_mat):
    p_sum = np.zeros(len(df_mat), dtype=np.float64)
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        imp_df = pd.DataFrame(imputers[j].transform(df_mat[FULL]), columns=FULL, index=df_mat.index)
        p_sum += frozen_lgb_boosters[j].predict(imp_df[SEL]) / 5.0
    return p_sum

raw_lcgm = json.load(open(f"{REP}/08a3_gbtm_coefs_trainonly.json", encoding="utf-8"))
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

def compute_posteriors(df_gcs, id_col="admissionid", max_hr=24.0):
    d_m = df_gcs[(df_gcs["offset_hr"] <= max_hr) & df_gcs["gcs_motor"].notna()].copy()
    if len(d_m) == 0:
        return pd.DataFrame(columns=[id_col, "prob_m1", "prob_m2", "prob_m3", "cls"])
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
    cls = p_post.argmax(axis=1) + 1
    return pd.DataFrame({id_col: ids_order, "prob_m1": p_post[:, 0], "prob_m2": p_post[:, 1], "prob_m3": p_post[:, 2], "cls": cls})

def wilson_ci(k, n, z=1.959964):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    den = 1.0 + z*z / n
    cen = (p + z*z / (2*n)) / den
    rad = (z / den) * np.sqrt(max(0.0, p*(1.0 - p)/n + z*z / (4.0 * n * n)))
    return [round(max(0.0, (cen - rad) * 100.0), 2), round(min(100.0, (cen + rad) * 100.0), 2)]

def boot_auc_ci(y, p, B=300, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = [roc_auc_score(y[idx], p[idx]) for idx in [np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]) for _ in range(B)]]
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))

canary_gate(lambda: wilson_ci(50, 100), "Wilson CI & LCGM posterior canary")

V7 = ["hr", "sbp", "dbp", "mbp", "rr", "temp", "spo2"]
LABS = ["hematocrit", "hemoglobin", "platelet", "wbc", "aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine", "glucose_lab", "sodium", "potassium", "inr", "ptt"]

def run_leakfree_rolling_67(meta_df, id_col, bins_df, gcs_df, db_tag, is_hirid=False):
    """
    R3 Fix: At EVERY hour hr in 6..72, filters strictly to stays alive in ICU at hr (los_h >= hr & alive at hr)
    with >= 1 motor-GCS in [0, hr], and aggregates ONLY observations with hr_bin <= hr (vitals [0, hr], labs [-6h, hr])
    and GCS offset_hr <= hr — ZERO reuse of 24-h static matrices and ZERO fallback to 24-h values!
    """
    rows = []
    preds_t6, preds_t24 = {}, {}

    def first_nn(s):
        v = s.dropna()
        return float(v.iloc[0]) if len(v) else np.nan

    for hr in range(6, 73):
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

        # 1. Static + dynamic GCS strictly in [0, hr]
        ga_h = g_sub.groupby("id").agg(
            gcs_total_min=("gcs_total", "min"),   gcs_total_first=("gcs_total", first_nn),
            gcs_motor_min=("gcs_motor", "min"),   gcs_motor_first=("gcs_motor", first_nn),
            gcs_eye_min=("gcs_eye", "min"),       gcs_eye_first=("gcs_eye", first_nn),
            gcs_verbal_min=("gcs_verbal", "min"), gcs_verbal_first=("gcs_verbal", first_nn),
        ).reset_index().rename(columns={"id": id_col})

        grp_m = g_mot.groupby("id")
        dyn_h = grp_m["gcs_motor"].agg(
            motor_first="first", motor_last="last", motor_min="min", motor_max="max",
            motor_sd=lambda s: float(np.std(s.values, ddof=1)) if len(s) >= 2 else 0.0
        ).reset_index().rename(columns={"id": id_col})
        sl_list = []
        for pid_k, gk in grp_m:
            if len(gk) >= 2 and (gk["offset_hr"].max() - gk["offset_hr"].min()) > 0:
                sl_list.append((pid_k, float(np.polyfit(gk["offset_hr"].values, gk["gcs_motor"].values, 1)[0])))
            else:
                sl_list.append((pid_k, 0.0))
        sl_h = pd.DataFrame(sl_list, columns=[id_col, "motor_slope"])
        post_h = compute_posteriors(g_mot, id_col="id", max_hr=float(hr)).rename(columns={"id": id_col})

        # 2. Vitals [0, hr] & Labs [-6h, hr] strictly up to hr_bin <= hr
        b_h = bins_df[(bins_df[id_col].isin(valid_ids)) & (bins_df["hr_bin"] <= hr)]
        agg_b = b_h.groupby([id_col, "feat"], as_index=False).agg(mn=("mn", "min"), mx=("mx", "max"), sm=("sm", "sum"), cnt=("cnt", "sum"))
        agg_b["mean_v"] = agg_b["sm"] / np.maximum(agg_b["cnt"], 1)

        piv_mn = agg_b.pivot(index=id_col, columns="feat", values="mn")
        piv_mx = agg_b.pivot(index=id_col, columns="feat", values="mx")
        piv_me = agg_b.pivot(index=id_col, columns="feat", values="mean_v")

        X_hr = sub_meta[[id_col, "admission_age", "sex_female", "d28"]].merge(ga_h, on=id_col, how="left") \
                       .merge(dyn_h, on=id_col, how="left").merge(sl_h, on=id_col, how="left") \
                       .merge(post_h[[id_col, "prob_m1", "prob_m2", "prob_m3"]], on=id_col, how="left")
        idx_ids = X_hr[id_col].values
        for v in V7:
            X_hr[f"{v}_min"]  = piv_mn[v].reindex(idx_ids).values if v in piv_mn.columns else np.nan
            X_hr[f"{v}_mean"] = piv_me[v].reindex(idx_ids).values if v in piv_me.columns else np.nan
            X_hr[f"{v}_max"]  = piv_mx[v].reindex(idx_ids).values if v in piv_mx.columns else np.nan
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

        X_hr["mbp_min"]  = X_hr["mbp_min"].fillna((X_hr["sbp_min"] + 2.0 * X_hr["dbp_min"]) / 3.0)
        X_hr["mbp_mean"] = X_hr["mbp_mean"].fillna((X_hr["sbp_mean"] + 2.0 * X_hr["dbp_mean"]) / 3.0)
        X_hr["mbp_max"]  = X_hr["mbp_max"].fillna((X_hr["sbp_max"] + 2.0 * X_hr["dbp_max"]) / 3.0)
        X_hr["aniongap_min"] = X_hr["aniongap_min"].fillna(X_hr["sodium_min"] - (X_hr["chloride_max"] + X_hr["bicarbonate_max"]))
        X_hr["aniongap_max"] = X_hr["aniongap_max"].fillna(X_hr["sodium_max"] - (X_hr["chloride_min"] + X_hr["bicarbonate_min"]))
        X_hr["motor_delta"] = X_hr["motor_last"] - X_hr["motor_first"]
        X_hr["charlson_comorbidity_score"] = np.nan

        for c in FULL:
            if c not in X_hr.columns:
                X_hr[c] = np.nan

        p_hr = deterministic_predict_lgb(X_hr)
        yv = X_hr["d28"].values.astype(int)
        auc_v = float(roc_auc_score(yv, p_hr))
        lo_v, hi_v = boot_auc_ci(yv, p_hr, B=300, seed=42 + hr)
        rows.append({"db": db_tag, "t": hr, "n": int(len(yv)), "events": int(yv.sum()), "auroc": round(auc_v, 4), "lo": round(lo_v, 4), "hi": round(hi_v, 4)})
        if hr == 6:
            preds_t6 = dict(zip(X_hr[id_col].values, zip(yv, p_hr)))
        elif hr == 24:
            preds_t24 = dict(zip(X_hr[id_col].values, zip(yv, p_hr)))

    # Patient-level paired bootstrap Delta-AUROC (24h vs 6h) on stays present at both 6h and 24h (accounting for serial correlation!)
    common_ids = sorted(set(preds_t6.keys()) & set(preds_t24.keys()))
    yc = np.array([preds_t24[k][0] for k in common_ids], int)
    p6c = np.array([preds_t6[k][1] for k in common_ids], float)
    p24c = np.array([preds_t24[k][1] for k in common_ids], float)
    rng_p = np.random.default_rng(77)
    pos_c, neg_c = np.where(yc == 1)[0], np.where(yc == 0)[0]
    diffs_p = [roc_auc_score(yc[idx], p24c[idx]) - roc_auc_score(yc[idx], p6c[idx]) for idx in [np.concatenate([rng_p.choice(pos_c, len(pos_c), True), rng_p.choice(neg_c, len(neg_c), True)]) for _ in range(1000)]]
    d_24_6 = float(roc_auc_score(yc, p24c) - roc_auc_score(yc, p6c))
    p_24_6 = float(2 * (1 - norm.cdf(abs(d_24_6) / max(float(np.std(diffs_p, ddof=1)), 1e-9))))
    paired_gain = {"n_common_6_24h": len(common_ids), "delta_24h_vs_6h": round(d_24_6, 4), "ci95": [round(float(np.percentile(diffs_p, 2.5)), 4), round(float(np.percentile(diffs_p, 97.5)), 4)], "p_paired_boot": float(f"{p_24_6:.6g}")}
    return pd.DataFrame(rows), paired_gain


def paired_24_vs_6_from_parquet(df_roll, id_col="id"):
    r6 = df_roll[df_roll["t"] == 6].set_index(id_col)
    r24 = df_roll[df_roll["t"] == 24].set_index(id_col)
    common = sorted(set(r6.index) & set(r24.index))
    yc = r24.loc[common, "d28"].values.astype(int)
    p6 = r6.loc[common, "p"].values.astype(float)
    p24 = r24.loc[common, "p"].values.astype(float)
    rng_p = np.random.default_rng(77)
    pos_c, neg_c = np.where(yc == 1)[0], np.where(yc == 0)[0]
    diffs = [roc_auc_score(yc[idx], p24[idx]) - roc_auc_score(yc[idx], p6[idx])
             for idx in [np.concatenate([rng_p.choice(pos_c, len(pos_c), True), rng_p.choice(neg_c, len(neg_c), True)]) for _ in range(1000)]]
    d_obs = float(roc_auc_score(yc, p24) - roc_auc_score(yc, p6))
    p_val = float(2 * (1 - norm.cdf(abs(d_obs) / max(float(np.std(diffs, ddof=1)), 1e-9))))
    return {
        "n_common_6_24h": int(len(common)),
        "delta_24h_vs_6h": round(d_obs, 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "p_paired_boot": float(f"{p_val:.6g}")
    }

def summarize_curve(df_c, paired_info):
    sub36 = df_c[(df_c["t"] >= 6) & (df_c["t"] <= 36)].sort_values("t")
    sub72 = df_c[(df_c["t"] >= 6) & (df_c["t"] <= 72)].sort_values("t")
    tau36, p36 = kendalltau(sub36["t"].values, sub36["auroc"].values)
    tau72, p72 = kendalltau(sub72["t"].values, sub72["auroc"].values)
    by_t = df_c.set_index("t")
    return {
        "n_6h": int(by_t.loc[6, "n"]),
        "n_24h": int(by_t.loc[24, "n"]),
        "n_72h": int(by_t.loc[72, "n"]),
        "auroc_6h": round(float(by_t.loc[6, "auroc"]), 4),
        "auroc_12h": round(float(by_t.loc[12, "auroc"]), 4),
        "auroc_24h": round(float(by_t.loc[24, "auroc"]), 4),
        "auroc_36h": round(float(by_t.loc[36, "auroc"]), 4),
        "auroc_48h": round(float(by_t.loc[48, "auroc"]), 4),
        "auroc_72h": round(float(by_t.loc[72, "auroc"]), 4),
        "paired_24h_vs_6h": paired_info,
        "mann_kendall_tau_6_36h": round(float(tau36), 4),
        "mann_kendall_p_6_36h": float(f"{p36:.6g}"),
        "mann_kendall_tau_6_72h": round(float(tau72), 4),
        "mann_kendall_p_6_72h": float(f"{p72:.6g}"),
    }

def compute_cnri_idi(y, p_new, p_old):
    y = np.asarray(y, int); p_new = np.asarray(p_new, float); p_old = np.asarray(p_old, float)
    ev = (y == 1); nev = (y == 0)
    nri_e = float(np.mean(p_new[ev] > p_old[ev]) - np.mean(p_new[ev] < p_old[ev]))
    nri_ne = float(np.mean(p_new[nev] < p_old[nev]) - np.mean(p_new[nev] > p_old[nev]))
    idi = float((np.mean(p_new[ev]) - np.mean(p_old[ev])) - (np.mean(p_new[nev]) - np.mean(p_old[nev])))
    return {
        "cNRI_events": round(nri_e, 3),
        "cNRI_nonevents": round(nri_ne, 3),
        "cNRI_total": round(nri_e + nri_ne, 3),
        "IDI": round(idi, 3)
    }

def build_severe_profile(label, g0, ml, y, p_s, p_dyn):
    sev = pd.notna(g0) & (g0 <= 8) & pd.notna(ml)
    ml_s = ml[sev]; y_s = y[sev]; ps_s = p_s[sev]; pd_s = p_dyn[sev]
    n_tot = int(sev.sum())
    d_tot = int(y_s.sum())

    m56 = (ml_s >= 5)
    m4  = (ml_s == 4)
    m13 = (ml_s <= 3)
    m14 = (ml_s <= 4)

    def grp_stat(msk, with_risk=False):
        n_g = int(msk.sum())
        d_g = int(y_s[msk].sum())
        mp = round(100.0 * d_g / max(n_g, 1), 1)
        d_out = {
            "n": n_g,
            "pct_of_severe": round(100.0 * n_g / max(n_tot, 1), 1),
            "deaths": d_g,
            "mort_pct": mp,
            "mort_wilson95": wilson_ci(d_g, n_g),
        }
        if with_risk and n_g > 0:
            surv = msk & (y_s == 0)
            n_sv = int(surv.sum())
            d_out["mean_p_S"] = round(float(np.mean(ps_s[msk])), 3)
            d_out["mean_p_SLDP"] = round(float(np.mean(pd_s[msk])), 3)
            d_out["survivors_n"] = n_sv
            d_out["survivors_risk_lowered_pct"] = round(100.0 * float(np.mean(pd_s[surv] < ps_s[surv])), 1) if n_sv > 0 else 0.0
            d_out["survivors_flagged_high_S"] = int(np.sum(ps_s[surv] >= THR_RAW))
            d_out["survivors_flagged_high_SLDP"] = int(np.sum(pd_s[surv] >= THR_RAW))
        return d_out

    r_awk = grp_stat(m56, with_risk=True)
    r_m4  = grp_stat(m4)
    r_m13 = grp_stat(m13)
    r_m14 = grp_stat(m14)
    ratio = round((r_m13["deaths"] / max(r_m13["n"], 1)) / max(r_awk["deaths"] / max(r_awk["n"], 1), 1e-6), 2)
    return {
        "cohort": label,
        "n_severe_documented": n_tot,
        "deaths_severe": d_tot,
        "mort_severe_pct": round(100.0 * d_tot / max(n_tot, 1), 1),
        "rapid_awakeners_m5_6": r_awk,
        "withdraws_m4": r_m4,
        "depressed_m1_3": r_m13,
        "non_awakened_m1_4": r_m14,
        "mort_ratio_m1_3_vs_m5_6": ratio,
        "reclassification_SLDP_vs_S": compute_cnri_idi(y_s, pd_s, ps_s)
    }

@retry_limit(2)
def main():
    lg("1. Running 100% leak-free 67-checkpoint [-6h/0, t] rolling evaluation on Dutch AmsterdamUMCdb & Swiss HiRID...")
    au_meta = pd.read_parquet(f"{DATA}/19a_roll72_amsterdam_meta_v4.parquet")
    au_bins = pd.read_parquet(f"{DATA}/19a_roll72_amsterdam_v4.parquet")
    au_gcs_r = pd.read_parquet(f"{DATA}/19a_gcs72_amsterdam_roll_v4.parquet")
    df_roll_au, paired_au = run_leakfree_rolling_67(au_meta, "admissionid", au_bins, au_gcs_r, "amsterdam_trauma", is_hirid=False)
    df_roll_au.to_csv(f"{REP}/19d_v4_rolling_amsterdam.csv", index=False)
    lg(f"  Amsterdam leak-free rolling: n@6h={df_roll_au.loc[df_roll_au.t==6,'n'].values[0]}, n@24h={df_roll_au.loc[df_roll_au.t==24,'n'].values[0]}, AUC@6h={df_roll_au.loc[df_roll_au.t==6,'auroc'].values[0]:.4f} -> AUC@24h={df_roll_au.loc[df_roll_au.t==24,'auroc'].values[0]:.4f} (paired Δ={paired_au['delta_24h_vs_6h']:+.4f}, p={paired_au['p_paired_boot']})")

    hi_meta = pd.read_parquet(f"{DATA}/19a_roll72_hirid_meta_v4.parquet")
    hi_bins = pd.read_parquet(f"{DATA}/19a_roll72_hirid_v4.parquet")
    hi_gcs_r = pd.read_parquet(f"{DATA}/19a_gcs72_hirid_roll_v4.parquet")
    df_roll_hi, paired_hi = run_leakfree_rolling_67(hi_meta, "patientid", hi_bins, hi_gcs_r, "hirid", is_hirid=True)
    df_roll_hi.to_csv(f"{REP}/09e_v3_fig2_d28_hirid.csv", index=False)
    lg(f"  HiRID leak-free rolling:     n@6h={df_roll_hi.loc[df_roll_hi.t==6,'n'].values[0]}, n@24h={df_roll_hi.loc[df_roll_hi.t==24,'n'].values[0]}, AUC@6h={df_roll_hi.loc[df_roll_hi.t==6,'auroc'].values[0]:.4f} -> AUC@24h={df_roll_hi.loc[df_roll_hi.t==24,'auroc'].values[0]:.4f} (paired Δ={paired_hi['delta_24h_vs_6h']:+.4f}, p={paired_hi['p_paired_boot']})")

    lg("2. Computing paired 24h vs 6h rolling gains & unified eICU Charted (prob_m1.notna, n=2,889) curve...")
    d09_csv = pd.read_csv(f"{REP}/09e_v3_fig2_d28.csv")
    pr_m4 = pd.read_parquet(f"{DATA}/09d_rolling_preds_mimic4_v3.parquet")
    pr_m3 = pd.read_parquet(f"{DATA}/09d_rolling_preds_mimic3_v3.parquet")
    pr_ei = pd.read_parquet(f"{DATA}/09d_rolling_preds_eicu_v3.parquet")

    # Y22 Fix: Unify eICU_CRD_Charted strictly with Table 1 / 19c / 19e definition (08b_posterior24_eicu_trainonly prob_m1.notna())
    post_ei = pd.read_parquet(f"{DATA}/08b_posterior24_eicu_trainonly.parquet")
    charted_ei_ids = set(post_ei.loc[post_ei["prob_m1"].notna(), "id"].astype(int))
    pr_ei_ch = pr_ei[pr_ei["id"].astype(int).isin(charted_ei_ids)].copy()
    ei_ch_rows = []
    for hr in sorted(pr_ei_ch["t"].unique()):
        sub_e = pr_ei_ch[pr_ei_ch["t"] == hr]
        yv_e = sub_e["d28"].values.astype(int); pv_e = sub_e["p"].values.astype(float)
        lo_e, hi_e = boot_auc_ci(yv_e, pv_e, B=300, seed=42 + int(hr))
        ei_ch_rows.append({"db": "eicu_charted", "t": int(hr), "n": int(len(yv_e)), "events": int(yv_e.sum()), "auroc": round(float(roc_auc_score(yv_e, pv_e)), 4), "lo": round(lo_e, 4), "hi": round(hi_e, 4)})
    df_roll_ei_ch = pd.DataFrame(ei_ch_rows)
    df_roll_ei_ch.to_csv(f"{REP}/19d_v4_rolling_eicu_charted.csv", index=False)

    rolling_cohorts = {
        "MIMIC_IV": summarize_curve(d09_csv[d09_csv["db"] == "mimic4"], paired_24_vs_6_from_parquet(pr_m4)),
        "MIMIC_III": summarize_curve(d09_csv[d09_csv["db"] == "mimic3"], paired_24_vs_6_from_parquet(pr_m3)),
        "eICU_CRD_All": summarize_curve(d09_csv[d09_csv["db"] == "eicu"], paired_24_vs_6_from_parquet(pr_ei)),
        "eICU_CRD_Charted": summarize_curve(df_roll_ei_ch, paired_24_vs_6_from_parquet(pr_ei_ch)),
        "HiRID": summarize_curve(df_roll_hi, paired_hi),
        "AmsterdamUMCdb_Trauma": summarize_curve(df_roll_au, paired_au),
    }

    lg("3. Computing AmsterdamUMCdb first-stay Trauma LCGM truncation ARI (24h & 48h vs 72h)...")
    d_au_abi = pd.read_parquet(f"{DATA}/19a_matrix_amsterdam_v4.parquet")
    d_au_tr  = d_au_abi[d_au_abi["is_trauma"] == True].copy().reset_index(drop=True)
    g72_au   = pd.read_parquet(f"{DATA}/19a_gcs72_amsterdam_v4.parquet")
    g72_tr   = g72_au[g72_au["admissionid"].isin(set(d_au_tr["admissionid"]))].copy()
    c72 = compute_posteriors(g72_tr, "admissionid", 72.0).set_index("admissionid")["cls"]
    c48 = compute_posteriors(g72_tr, "admissionid", 48.0).set_index("admissionid").reindex(c72.index)["cls"]
    c24 = compute_posteriors(g72_tr, "admissionid", 24.0).set_index("admissionid").reindex(c72.index)["cls"]
    valid_ari = c72.notna() & c48.notna() & c24.notna()
    v72, v48, v24 = c72[valid_ari].values, c48[valid_ari].values, c24[valid_ari].values
    ari24 = float(adjusted_rand_score(v72, v24))
    ari48 = float(adjusted_rand_score(v72, v48))
    rng_a = np.random.default_rng(88)
    b_ari24 = np.array([adjusted_rand_score(v72[idx], v24[idx]) for idx in [rng_a.choice(len(v72), len(v72), True) for _ in range(1000)]], dtype=np.float32)
    b_ari48 = np.array([adjusted_rand_score(v72[idx], v48[idx]) for idx in [rng_a.choice(len(v72), len(v72), True) for _ in range(1000)]], dtype=np.float32)
    np.savez_compressed(f"{DATA}/19d_v4_ari_boot_amsterdam.npz", ari_24h=b_ari24, ari_48h=b_ari48)
    ari_out = {
        "n": int(len(v72)),
        "24": {"ari": round(ari24, 4), "ci95": [round(float(np.percentile(b_ari24, 2.5)), 4), round(float(np.percentile(b_ari24, 97.5)), 4)]},
        "48": {"ari": round(ari48, 4), "ci95": [round(float(np.percentile(b_ari48, 2.5)), 4), round(float(np.percentile(b_ari48, 97.5)), 4)]},
    }

    lg("4. Computing Secondary Endpoints (Composite END+Mortality, Invasive ICP ABI & Trauma scopes, END Detection & FAC)...")
    all_preds = pd.read_parquet(f"{DATA}/19c_v4_all_cohort_preds.parquet")
    m4 = pd.read_parquet(f"{DATA}/08c_matrix_mimic4_v3.parquet")
    iv_m4 = m4[m4.split == "intval"].copy().reset_index(drop=True)
    end_m4 = pd.read_parquet(f"{DATA}/09b_v3_endpost24_mimic4.parquet").rename(columns={"id": "stay_id"})
    end_m4["stay_id"] = end_m4["stay_id"].astype(int)
    iv_m4 = iv_m4.merge(end_m4[["stay_id", "endpost"]], on="stay_id", how="left")
    p_iv_hy = all_preds[all_preds["cohort"] == "MIMIC_IV_IntVal"]["p_adaptive_gbdt"].values
    iv_comp_m = iv_m4["endpost"].notna()
    iv_comp_y = ((iv_m4.loc[iv_comp_m, "endpost"] == 1) | (iv_m4.loc[iv_comp_m, "d28"] == 1)).astype(int).values

    import duckdb as _dd
    fe_m3 = pd.read_parquet(f"{DATA}/features_mimic3.parquet").rename(columns={"age": "admission_age"})
    _cv = _dd.connect().execute(f"SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'").df()
    d_m3 = fe_m3[fe_m3.icustay_id.isin(_cv.icustay_id)].copy()
    d_m3 = d_m3[~(((d_m3.d28 == 1) & (d_m3.days_to_death < 1.0)) | (d_m3.los_h < 24))].copy().reset_index(drop=True)
    end_m3 = pd.read_parquet(f"{DATA}/09b_v3_endpost24_mimic3.parquet").rename(columns={"id": "icustay_id"})
    end_m3["icustay_id"] = end_m3["icustay_id"].astype(int)
    d_m3 = d_m3.merge(end_m3[["icustay_id", "endpost"]], on="icustay_id", how="left")
    p_m3_hy = all_preds[all_preds["cohort"] == "MIMIC_III_CareVue"]["p_adaptive_gbdt"].values
    m3_comp_m = d_m3["endpost"].notna()
    m3_comp_y = ((d_m3.loc[m3_comp_m, "endpost"] == 1) | (d_m3.loc[m3_comp_m, "d28"] == 1)).astype(int).values

    fe_ei = pd.read_parquet(f"{DATA}/features_eicu.parquet").rename(columns={"age": "admission_age"})
    d_ei = fe_ei[~(((fe_ei.died_hosp == 1) & (fe_ei.los_h <= 24)) | (fe_ei.los_h < 24))].copy().reset_index(drop=True)
    d_ei["d28"] = d_ei["died_hosp_28d"].astype(int)
    end_ei = pd.read_parquet(f"{DATA}/09b_v3_endpost24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
    end_ei["icustay_id_eicu"] = end_ei["icustay_id_eicu"].astype(int)
    d_ei = d_ei.merge(end_ei[["icustay_id_eicu", "endpost"]], on="icustay_id_eicu", how="left")
    p_ei_hy = all_preds[all_preds["cohort"] == "eICU_CRD"]["p_adaptive_gbdt"].values
    ei_comp_m = d_ei["endpost"].notna()
    ei_comp_y = ((d_ei.loc[ei_comp_m, "endpost"] == 1) | (d_ei.loc[ei_comp_m, "d28"] == 1)).astype(int).values

    p_au_abi_lgb = deterministic_predict_lgb(d_au_abi)
    au_comp_y = ((d_au_abi["end_24_72h"] == 1) | (d_au_abi["d28"] == 1)).astype(int).values

    # Y14 Fix: Report Invasive ICP subgroups transparently for BOTH All-ABI AND Primary Trauma scopes!
    d_hi_all = pd.read_parquet(f"{DATA}/08c_matrix_hirid_v3.parquet")
    p_hi_all_lgb = deterministic_predict_lgb(d_hi_all)
    p_hi_tr_hy = all_preds[all_preds["cohort"] == "HiRID_Trauma"]["p_adaptive_gbdt"].values
    d_hi_tr = d_hi_all[d_hi_all["is_trauma"] == True].reset_index(drop=True)

    hi_icp_abi = (d_hi_all["has_icp_24h"] == 1)
    hi_gt20_abi = (d_hi_all["icp_max_24h"] > 20.0)
    hi_icp_tr  = (d_hi_tr["has_icp_24h"] == 1)

    p_au_tr_hy = all_preds[all_preds["cohort"] == "AmsterdamUMCdb_Trauma"]["p_adaptive_gbdt"].values
    au_icp_abi = (d_au_abi["has_icp_24h"] == 1)
    au_gt20_abi = (d_au_abi["icp_max_24h"] > 20.0)
    au_icp_tr  = (d_au_tr["has_icp_24h"] == 1)

    sec_out = {
        "MIMIC_IV_Composite_END_Mortality": {
            "n": int(iv_comp_m.sum()), "end_events": int(iv_m4.loc[iv_comp_m, "endpost"].sum()),
            "d28_events": int(iv_m4.loc[iv_comp_m, "d28"].sum()), "composite_events": int(iv_comp_y.sum()),
            "auroc_composite": round(float(roc_auc_score(iv_comp_y, p_iv_hy[iv_comp_m.values])), 4)
        },
        "MIMIC_III_Composite_END_Mortality": {
            "n": int(m3_comp_m.sum()), "end_events": int(d_m3.loc[m3_comp_m, "endpost"].sum()),
            "d28_events": int(d_m3.loc[m3_comp_m, "d28"].sum()), "composite_events": int(m3_comp_y.sum()),
            "auroc_composite": round(float(roc_auc_score(m3_comp_y, p_m3_hy[m3_comp_m.values])), 4)
        },
        "eICU_Composite_END_Mortality": {
            "n": int(ei_comp_m.sum()), "end_events": int(d_ei.loc[ei_comp_m, "endpost"].sum()),
            "d28_events": int(d_ei.loc[ei_comp_m, "d28"].sum()), "composite_events": int(ei_comp_y.sum()),
            "auroc_composite": round(float(roc_auc_score(ei_comp_y, p_ei_hy[ei_comp_m.values])), 4)
        },
        "AmsterdamUMCdb_Composite_END_Mortality": {
            "n": int(len(d_au_abi)), "end_events": int(d_au_abi["end_24_72h"].sum()),
            "d28_events": int(d_au_abi["d28"].sum()), "composite_events": int(au_comp_y.sum()),
            "auroc_composite": round(float(roc_auc_score(au_comp_y, p_au_abi_lgb)), 4)
        },
        "HiRID_Invasive_ICP_Subgroup": {
            "scope": "Expanded ABI (trauma subset reported alongside)",
            "n_icp_monitored": int(hi_icp_abi.sum()),
            "n_icp_hypertension_gt20": int(hi_gt20_abi.sum()),
            "auroc_icp_monitored": round(float(roc_auc_score(d_hi_all.loc[hi_icp_abi, "d28"], p_hi_all_lgb[hi_icp_abi.values])), 4),
            "auroc_icp_gt20": round(float(roc_auc_score(d_hi_all.loc[hi_gt20_abi, "d28"], p_hi_all_lgb[hi_gt20_abi.values])), 4),
            "trauma_only_icp_n": int(hi_icp_tr.sum()),
            "trauma_only_icp_events": int(d_hi_tr.loc[hi_icp_tr, "d28"].sum()),
            "trauma_only_auroc_hybrid": round(float(roc_auc_score(d_hi_tr.loc[hi_icp_tr, "d28"], p_hi_tr_hy[hi_icp_tr.values])), 4),
        },
        "AmsterdamUMCdb_Invasive_ICP_Subgroup": {
            "scope": "Expanded ABI (trauma subset reported alongside)",
            "n_icp_monitored": int(au_icp_abi.sum()),
            "n_icp_hypertension_gt20": int(au_gt20_abi.sum()),
            "auroc_icp_monitored": round(float(roc_auc_score(d_au_abi.loc[au_icp_abi, "d28"], p_au_abi_lgb[au_icp_abi.values])), 4),
            "auroc_icp_gt20": round(float(roc_auc_score(d_au_abi.loc[au_gt20_abi, "d28"], p_au_abi_lgb[au_gt20_abi.values])), 4),
            "trauma_only_icp_n": int(au_icp_tr.sum()),
            "trauma_only_icp_events": int(d_au_tr.loc[au_icp_tr, "d28"].sum()),
            "trauma_only_auroc_hybrid": round(float(roc_auc_score(d_au_tr.loc[au_icp_tr, "d28"], p_au_tr_hy[au_icp_tr.values])), 4),
        }
    }

    # Y5 Fix: Attach END detection rates, 12-h sensitivity, no-trigger specificity, & false alarms per 100 patient-days!
    r09e = json.load(open(f"{REP}/09e_v3_rolling_report.json", encoding="utf-8"))["warning"]
    end_warn_summary = {}
    for db_k, lbl_k in [("mimic4", "MIMIC_IV"), ("mimic3", "MIMIC_III"), ("eicu", "eICU_CRD")]:
        w = r09e[db_k]
        end_warn_summary[lbl_k] = {
            "single_trigger": {
                "n_end_events": w["end_single"]["n_events"],
                "detected_before_event": w["end_single"]["detected_before_event"],
                "detection_rate": w["end_single"]["detection_rate"],
                "median_lead_hr": w["end_single"]["warning_time_hr"]["median"],
                "iqr_lead_hr": [w["end_single"]["warning_time_hr"]["q1"], w["end_single"]["warning_time_hr"]["q3"]],
                "sens_12h_window": w["end_single"]["sens_12h_window"],
                "spec_no_trigger": w["end_single"]["spec_no_trigger"],
                "false_alarms_per_100_patient_days": w["fac_end_single"]["false_alarms_per_100_patient_days"],
            },
            "two_consecutive_triggers": {
                "n_end_events": w["end_two_consec"]["n_events"],
                "detected_before_event": w["end_two_consec"]["detected_before_event"],
                "detection_rate": w["end_two_consec"]["detection_rate"],
                "median_lead_hr": w["end_two_consec"]["warning_time_hr"]["median"],
                "iqr_lead_hr": [w["end_two_consec"]["warning_time_hr"]["q1"], w["end_two_consec"]["warning_time_hr"]["q3"]],
                "sens_12h_window": w["end_two_consec"]["sens_12h_window"],
                "spec_no_trigger": w["end_two_consec"]["spec_no_trigger"],
                "false_alarms_per_100_patient_days": w["fac_end_two_consec"]["false_alarms_per_100_patient_days"],
            }
        }

    lg("5. Recomputing 5-cohort severe TBI rapid motor improvement ('rapid-awakening') unmasking...")
    def sub_preds(cname):
        sub = all_preds[all_preds["cohort"] == cname]
        return sub["p_static"].values, sub["p_adaptive_gbdt"].values

    ps_iv, pd_iv = sub_preds("MIMIC_IV_IntVal")
    ps_m3, pd_m3 = sub_preds("MIMIC_III_CareVue")
    ps_ei, pd_ei = sub_preds("eICU_CRD")
    ps_hi, pd_hi = sub_preds("HiRID_Trauma")
    ps_au, pd_au = sub_preds("AmsterdamUMCdb_Trauma")

    m3_ml = d_m3["dyn_motor_last"].values if "dyn_motor_last" in d_m3.columns else d_m3["motor_last"].values
    ei_ml = d_ei["dyn_motor_last"].values if "dyn_motor_last" in d_ei.columns else d_ei["motor_last"].values
    ei_doc_m = d_ei["icustay_id_eicu"].astype(int).isin(charted_ei_ids).values

    sev_cohorts = {
        "MIMIC_IV_IntVal": build_severe_profile("MIMIC-IV temporal validation", iv_m4["gcs_total_first"].values, iv_m4["motor_last"].values, iv_m4["d28"].values.astype(int), ps_iv, pd_iv),
        "MIMIC_III_CareVue": build_severe_profile("MIMIC-III CareVue", d_m3["gcs_total_first"].values, m3_ml, d_m3["d28"].values.astype(int), ps_m3, pd_m3),
        "eICU_Documented": build_severe_profile("eICU-CRD motor-documented", d_ei.loc[ei_doc_m, "gcs_total_first"].values, ei_ml[ei_doc_m], d_ei.loc[ei_doc_m, "d28"].values.astype(int), ps_ei[ei_doc_m], pd_ei[ei_doc_m]),
        "HiRID_Trauma": build_severe_profile("HiRID Swiss external", d_hi_tr["gcs_total_first"].values, d_hi_tr["motor_last"].values, d_hi_tr["d28"].values.astype(int), ps_hi, pd_hi),
        "AmsterdamUMCdb_Trauma": build_severe_profile("AmsterdamUMCdb Trauma", d_au_tr["gcs_total_first"].values, d_au_tr["motor_last"].values, d_au_tr["d28"].values.astype(int), ps_au, pd_au),
    }

    # Expanded ABI sensitivity for Amsterdam
    S_COLS = [c for c in SEL if not (c.startswith("motor_") or c.startswith("prob_m"))]
    tr_m4 = m4[m4.split == "train"].reset_index(drop=True)
    m_s0_lgb = []
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        tr_imp = pd.DataFrame(imputers[j].transform(tr_m4[FULL]), columns=FULL)
        m_s0_lgb.append(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=25, scale_pos_weight=float(card["scale_pos_weight"]), random_state=42, n_jobs=4, verbose=-1).fit(tr_imp[S_COLS], tr_m4["d28"].values.astype(int)))
    ps_au_abi = np.zeros(len(d_au_abi))
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        au_imp = pd.DataFrame(imputers[j].transform(d_au_abi[FULL]), columns=FULL)
        ps_au_abi += m_s0_lgb[j].predict_proba(au_imp[S_COLS])[:, 1] / 5.0
    au_abi_sev = build_severe_profile("AmsterdamUMCdb Expanded ABI", d_au_abi["gcs_total_first"].values, d_au_abi["motor_last"].values, d_au_abi["d28"].values.astype(int), ps_au_abi, p_au_abi_lgb)

    tot_sev = sum(v["n_severe_documented"] for v in sev_cohorts.values())
    tot_dth = sum(v["deaths_severe"] for v in sev_cohorts.values())
    awk_n   = sum(v["rapid_awakeners_m5_6"]["n"] for v in sev_cohorts.values())
    awk_d   = sum(v["rapid_awakeners_m5_6"]["deaths"] for v in sev_cohorts.values())
    non_n   = sum(v["non_awakened_m1_4"]["n"] for v in sev_cohorts.values())
    non_d   = sum(v["non_awakened_m1_4"]["deaths"] for v in sev_cohorts.values())
    awk_mp  = round(100.0 * awk_d / max(awk_n, 1), 1)
    non_mp  = round(100.0 * non_d / max(non_n, 1), 1)
    fold_r  = round((non_d / max(non_n, 1)) / max(awk_d / max(awk_n, 1), 1e-6), 2)

    pooled_5 = {
        "n_severe_documented": int(tot_sev),
        "deaths_severe": int(tot_dth),
        "rapid_awakeners_n": int(awk_n),
        "rapid_awakeners_pct": round(100.0 * awk_n / max(tot_sev, 1), 1),
        "rapid_awakeners_deaths": int(awk_d),
        "rapid_awakeners_mort_pct": awk_mp,
        "rapid_awakeners_wilson95": wilson_ci(awk_d, awk_n),
        "non_awakened_n": int(non_n),
        "non_awakened_deaths": int(non_d),
        "non_awakened_mort_pct": non_mp,
        "non_awakened_wilson95": wilson_ci(non_d, non_n),
        "mortality_fold_reduction": fold_r
    }

    out = {
        "rolling_cohorts": rolling_cohorts,
        "amsterdam_truncation_ari_vs_72h": ari_out,
        "secondary_endpoints_end_icp": sec_out,
        "end_warning_and_false_alarm_summary": end_warn_summary,
        "severe_tbi_5cohorts_unmasking": {
            "cohorts": sev_cohorts,
            "amsterdam_abi_sensitivity": au_abi_sev,
            "pooled_5_validation_cohorts": pooled_5
        }
    }
    json.dump(out, open(f"{REP}/19d_v4_rolling_end_reclass.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    lg(f"DONE 19d! Pooled 5-cohort severe TBI rapid motor improvers: {awk_n}/{tot_sev} ({pooled_5['rapid_awakeners_pct']}%), mortality {awk_mp}% vs {non_mp}% ({fold_r}x)")

if __name__ == "__main__":
    main()
