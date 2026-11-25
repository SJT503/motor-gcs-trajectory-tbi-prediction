# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19g_v4_5_benchmark_closure.py — Empirical Benchmark Closure & Methodological Harmonization (v4.5)
Computes 100% real patient-level results for:
  1. Conformal confidence sweep (1 - alpha in 0.80..0.95): validity (empirical coverage) vs efficiency (retained singleton %)
  2. External local calibration sample-size learning curves (n_cal = 50, 100, 150, 200, 300 stays; 50 draws each)
  3. eICU-CRD 130-hospital leave-hospitals-out (GroupShuffleSplit by hospitalid, 20x 50/50) split-conformal & Platt calibration
  4. v4 deterministic MICE MIMIC-IV sensitivity checks with B=2,000 bootstrap CIs (excluding early DNR/DNI/CMO; sedative-infusion censoring)
  5. Two real anonymized MIMIC-IV validation longitudinal patient vignettes (6-72 h motor-GCS & rolling risk / conformal state)
"""
import json, math
from pathlib import Path
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

ROOT   = Path("E:/TBI subtype")
DATA   = ROOT / "07_prediction_system/data"
REP    = ROOT / "07_prediction_system/reports"
MODELS = ROOT / "07_prediction_system/models"
gate(str(ROOT / "07_prediction_system/scripts/19g_v4_5_benchmark_closure.py"))

df_preds = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
m4_mat   = pd.read_parquet(DATA / "08c_matrix_mimic4_v3.parquet")
r19c     = json.load(open(REP / "19c_v4_flagship_results.json", encoding="utf-8"))
mc_v3    = json.load(open(MODELS / "08d_v3_model_card.json", encoding="utf-8"))
rep_mat  = json.load(open(REP / "08c_matrix_report.json", encoding="utf-8"))

canary_gate(lambda: len(df_preds) >= 9964 and "p_adaptive_gbdt" in df_preds.columns, "v4 all-cohort predictions loaded")


def exact_mondrian_thresholds(y_cal, p_cal, alpha=0.10):
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    s1 = np.sort(1.0 - p_cal[y_cal == 1])
    s0 = np.sort(p_cal[y_cal == 0])
    n1, n0 = len(s1), len(s0)
    k1 = min(max(1, math.ceil((1.0 - alpha) * (n1 + 1))), n1)
    k0 = min(max(1, math.ceil((1.0 - alpha) * (n0 + 1))), n0)
    q1 = float(s1[k1 - 1])
    q0 = float(s0[k0 - 1])
    return 1.0 - q1, q0


def calc_slope_citl(y, p):
    y = np.asarray(y, dtype=int)
    lp = logit(np.clip(p, 1e-5, 1.0 - 1e-5))
    lr_s = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    slope = float(lr_s.coef_[0, 0])
    p_bar = float(np.clip(np.mean(p), 1e-5, 1.0 - 1e-5))
    y_bar = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    citl = float(logit(y_bar) - logit(p_bar))
    return slope, citl


def boot_auc(y, p, B=2000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        vals[b] = roc_auc_score(y[idx], p[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(roc_auc_score(y, p)), [round(float(lo), 4), round(float(hi), 4)], vals.astype(np.float32)


@retry_limit(2)
def main():
    out = {}
    npz_save = {}

    # ---------------- 0. Reconstruct Tune-Calib (n=278) predictions for MIMIC-IV ----------------
    FULL = rep_mat["blockA_cols"] + rep_mat["blockB_cols"] + rep_mat["blockC_cols"]
    SEL  = mc_v3["features"]
    imps = [joblib.load(MODELS / f"08d_v3_mice_imp{j}.joblib") for j in range(5)]
    boosts = [lgb.Booster(model_file=str(MODELS / f"08d_v3_lgbm_sel_imp{j}.txt")) for j in range(5)]

    def pred_frozen25_det(raw_df):
        pr = np.zeros(len(raw_df), dtype=np.float64)
        for j in range(5):
            imps[j].random_state = 100 + j
            imps[j].random_state_ = np.random.RandomState(100 + j)
            xm = pd.DataFrame(imps[j].transform(raw_df[FULL]), columns=FULL, index=raw_df.index)[SEL]
            pr += boosts[j].predict(xm) / 5.0
        return pr

    tu = m4_mat[m4_mat["split"] == "tune"].copy().reset_index(drop=True)
    iv = m4_mat[m4_mat["split"] == "intval"].copy().reset_index(drop=True)
    sss_tu = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    tu_sel_i, tu_cal_i = next(sss_tu.split(tu, tu["d28"].values.astype(int)))
    tu_sel = tu.iloc[tu_sel_i].copy().reset_index(drop=True)
    tu_cal = tu.iloc[tu_cal_i].copy().reset_index(drop=True)

    # Tune-Select Platt calibrator (exact match to 19c)
    p_tu_sel_raw = pred_frozen25_det(tu_sel)
    p_tu_cal_raw = pred_frozen25_det(tu_cal)
    platt_sel = LogisticRegression(C=1e6, max_iter=1000).fit(
        logit(np.clip(p_tu_sel_raw, 1e-5, 1 - 1e-5)).reshape(-1, 1), tu_sel["d28"].values.astype(int)
    )
    p_tu_cal = platt_sel.predict_proba(logit(np.clip(p_tu_cal_raw, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
    y_tu_cal = tu_cal["d28"].values.astype(int)

    # Identify miss_motor_no_ct per cohort in df_preds
    # In MIMIC-IV IntVal (100% motor charted), HiRID (100%), Amsterdam (100%): all False.
    # In MIMIC-III, eICU, Jinhua: reconstruct exact miss_motor_no_ct boolean vector!
    ct_df = pd.read_parquet(DATA / "19a_ct_nlp_3cohorts_v4.parquet")
    fe_ei = pd.read_parquet(DATA / "features_eicu.parquet")
    post_ei = pd.read_parquet(DATA / "08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d_ei = fe_ei.merge(post_ei[["icustay_id_eicu", "prob_m1"]], on="icustay_id_eicu", how="left")
    d_ei = d_ei[~(((d_ei.died_hosp == 1) & (d_ei.los_h <= 24)) | (d_ei.los_h < 24))].copy().reset_index(drop=True)
    ei_miss_m = d_ei["prob_m1"].isna().values  # eICU has no CT notes -> miss_motor_no_ct == prob_m1.isna()
    ei_hosp   = d_ei["hospitalid"].values

    d_jh_abi = pd.read_parquet(DATA / "19a_matrix_jinhua_v4.parquet").reset_index(drop=True)
    d_jh_tbi = d_jh_abi[d_jh_abi["is_tbi_s06"] == True].copy().reset_index(drop=True)
    ct_jh = ct_df[ct_df["cohort"] == "JinhuaNSICU"].set_index("id")
    jh_id_col = "hadm_id" if d_jh_tbi["hadm_id"].isin(ct_jh.index).sum() > 100 else "subject_id"
    jh_has_ct = ct_jh.reindex(d_jh_tbi[jh_id_col].values)["ct_any_24h"].fillna(0).values > 0.5
    assert int(jh_has_ct.sum()) == 1580, f"Expected 1,580 Jinhua CT-imaged TBI stays, got {jh_has_ct.sum()}"
    jh_miss_m_no_ct = ~jh_has_ct  # Jinhua has 0% flowsheet motor -> abstains when unimaged!

    fe_m3 = pd.read_parquet(DATA / "features_mimic3.parquet")
    post_m3 = pd.read_parquet(DATA / "08b_posterior24_mimic3_trainonly.parquet").rename(columns={"id": "icustay_id"})
    d_m3 = fe_m3.merge(post_m3[["icustay_id", "prob_m1"]], on="icustay_id", how="left")
    cv_ids = duckdb.connect().execute(
        f"SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id FROM read_csv('{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'"
    ).df()
    d_m3 = d_m3[d_m3.icustay_id.isin(cv_ids.icustay_id)].copy()
    d_m3 = d_m3[~(((d_m3.d28 == 1) & (d_m3.days_to_death < 1.0)) | (d_m3.los_h < 24))].copy().reset_index(drop=True)
    ct_m3 = ct_df[ct_df["cohort"] == "MIMIC-III"].set_index("id")
    m3_has_ct = ct_m3.reindex(d_m3["icustay_id"].values)["ct_any_24h"].fillna(0).values > 0.5
    m3_miss_m_no_ct = d_m3["prob_m1"].isna().values & (~m3_has_ct)

    miss_map = {
        "MIMIC_IV_IntVal":       np.zeros(483, dtype=bool),
        "MIMIC_III_CareVue":     m3_miss_m_no_ct,
        "eICU_CRD":              ei_miss_m,
        "HiRID_Trauma":          np.zeros(1209, dtype=bool),
        "AmsterdamUMCdb_Trauma": np.zeros(1052, dtype=bool),
        "JinhuaNSICU_TBI_S06":   jh_miss_m_no_ct,
    }

    # ---------------- 1. Conformal Confidence-Level Sweep (1 - alpha in 0.80..0.95) ----------------
    conf_levels = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    alpha_sweep = {}
    for ck, mm_vec in miss_map.items():
        sub = df_preds[df_preds["cohort"] == ck].reset_index(drop=True)
        y_v = sub["d28"].values.astype(int)
        p_v = sub["p_adaptive_gbdt"].values.astype(float)
        alpha_sweep[ck] = {}
        for cl in conf_levels:
            alp = round(1.0 - cl, 4)
            if ck == "MIMIC_IV_IntVal":
                t_lo, t_hi = exact_mondrian_thresholds(y_tu_cal, p_tu_cal, alpha=alp)
                ic1 = (p_v >= t_lo) | mm_vec
                ic0 = (p_v <= t_hi) | mm_vec
                cov = float(np.mean(((y_v == 1) & ic1) | ((y_v == 0) & ic0)))
                ret = float(np.mean(~((ic0 & ic1) | mm_vec)))
                dfn = ~((ic0 & ic1) | mm_vec)
                d_auc = float(roc_auc_score(y_v[dfn], p_v[dfn])) if dfn.sum() >= 20 else None
            else:
                sss = StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=42)
                covs, rets, daucs = [], [], []
                for ia, ib in sss.split(p_v, y_v):
                    for fi, ei in [(ia, ib), (ib, ia)]:
                        t_lo, t_hi = exact_mondrian_thresholds(y_v[fi], p_v[fi], alpha=alp)
                        pe, ye, me = p_v[ei], y_v[ei], mm_vec[ei]
                        ic1 = (pe >= t_lo) | me
                        ic0 = (pe <= t_hi) | me
                        covs.append(float(np.mean(((ye == 1) & ic1) | ((ye == 0) & ic0))))
                        dfn = ~((ic0 & ic1) | me)
                        rets.append(float(np.mean(dfn)))
                        if dfn.sum() >= 20 and ye[dfn].sum() >= 2 and (dfn.sum() - ye[dfn].sum()) >= 2:
                            daucs.append(float(roc_auc_score(ye[dfn], pe[dfn])))
                cov = float(np.mean(covs))
                ret = float(np.mean(rets))
                d_auc = float(np.mean(daucs)) if daucs else None
            alpha_sweep[ck][f"{int(round(cl*100))}"] = {
                "target_coverage": cl,
                "empirical_coverage": round(cov, 4),
                "empirical_error": round(1.0 - cov, 4),
                "retained_singleton_frac": round(ret, 4),
                "auroc_definitive": round(d_auc, 4) if d_auc is not None else None,
            }
    out["conformal_alpha_sweep"] = alpha_sweep

    # ---------------- 2. Local Calibration Sample-Size Learning Curve (n_cal = 50..300) ----------------
    n_cal_grid = [50, 100, 150, 200, 300]
    sample_curve = {}
    for ck in ["MIMIC_III_CareVue", "eICU_CRD", "HiRID_Trauma", "AmsterdamUMCdb_Trauma", "JinhuaNSICU_TBI_S06"]:
        sub = df_preds[df_preds["cohort"] == ck].reset_index(drop=True)
        y_v = sub["d28"].values.astype(int)
        p_v = sub["p_adaptive_gbdt"].values.astype(float)
        lp_v = logit(np.clip(p_v, 1e-5, 1 - 1e-5)).reshape(-1, 1)
        mm_vec = miss_map[ck]
        sample_curve[ck] = {}
        for nc in n_cal_grid:
            if nc >= len(y_v) - 100:
                continue
            sss_n = StratifiedShuffleSplit(n_splits=50, train_size=nc, random_state=100 + nc)
            covs_n, slopes_n, briers_n, rets_n = [], [], [], []
            for cal_i, ev_i in sss_n.split(p_v, y_v):
                if y_v[cal_i].sum() < 4:
                    continue
                cal_lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp_v[cal_i], y_v[cal_i])
                pr_ev = cal_lr.predict_proba(lp_v[ev_i])[:, 1]
                sl_v, _ = calc_slope_citl(y_v[ev_i], pr_ev)
                slopes_n.append(sl_v)
                briers_n.append(float(brier_score_loss(y_v[ev_i], pr_ev)))
                t_lo, t_hi = exact_mondrian_thresholds(y_v[cal_i], p_v[cal_i], alpha=0.10)
                pe, ye, me = p_v[ev_i], y_v[ev_i], mm_vec[ev_i]
                ic1 = (pe >= t_lo) | me
                ic0 = (pe <= t_hi) | me
                covs_n.append(float(np.mean(((ye == 1) & ic1) | ((ye == 0) & ic0))))
                rets_n.append(float(np.mean(~((ic0 & ic1) | me))))
            sample_curve[ck][str(nc)] = {
                "n_cal": nc,
                "approx_deaths_in_cal": round(float(nc * y_v.mean()), 1),
                "coverage_mean": round(float(np.mean(covs_n)), 4),
                "coverage_iqr": [round(float(x), 4) for x in np.percentile(covs_n, [25, 75])],
                "platt_slope_mean": round(float(np.mean(slopes_n)), 4),
                "brier_mean": round(float(np.mean(briers_n)), 4),
                "retained_mean": round(float(np.mean(rets_n)), 4),
            }
    out["local_calibration_sample_size_curve"] = sample_curve

    # ---------------- 3. eICU-CRD 130-Hospital Leave-Hospitals-Out Split-Conformal & Platt ----------------
    sub_ei = df_preds[df_preds["cohort"] == "eICU_CRD"].reset_index(drop=True)
    y_ei = sub_ei["d28"].values.astype(int)
    p_ei = sub_ei["p_adaptive_gbdt"].values.astype(float)
    lp_ei = logit(np.clip(p_ei, 1e-5, 1 - 1e-5)).reshape(-1, 1)
    gss = GroupShuffleSplit(n_splits=20, test_size=0.5, random_state=42)
    h_cov_all, h_cov_pos, h_cov_neg, h_rets, h_daucs, h_slopes, h_briers = [], [], [], [], [], [], []
    for ha, hb in gss.split(p_ei, y_ei, groups=ei_hosp):
        for fi, ei in [(ha, hb), (hb, ha)]:
            assert len(set(ei_hosp[fi]) & set(ei_hosp[ei])) == 0, "Hospital leak!"
            cal_lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp_ei[fi], y_ei[fi])
            pr_ev = cal_lr.predict_proba(lp_ei[ei])[:, 1]
            sl_v, _ = calc_slope_citl(y_ei[ei], pr_ev)
            h_slopes.append(sl_v)
            h_briers.append(float(brier_score_loss(y_ei[ei], pr_ev)))
            t_lo, t_hi = exact_mondrian_thresholds(y_ei[fi], p_ei[fi], alpha=0.10)
            pe, ye, me = p_ei[ei], y_ei[ei], ei_miss_m[ei]
            ic1 = (pe >= t_lo) | me
            ic0 = (pe <= t_hi) | me
            h_cov_all.append(float(np.mean(((ye == 1) & ic1) | ((ye == 0) & ic0))))
            h_cov_pos.append(float(np.mean(ic1[ye == 1])))
            h_cov_neg.append(float(np.mean(ic0[ye == 0])))
            dfn = ~((ic0 & ic1) | me)
            h_rets.append(float(np.mean(dfn)))
            h_daucs.append(float(roc_auc_score(ye[dfn], pe[dfn])))
    out["eicu_leave_hospitals_out_conformal"] = {
        "n_hospitals": 130,
        "splits": "20x 50/50 GroupShuffleSplit by hospitalid (zero hospital overlap)",
        "empirical_coverage_overall": round(float(np.mean(h_cov_all)), 4),
        "empirical_coverage_deaths": round(float(np.mean(h_cov_pos)), 4),
        "empirical_coverage_survivors": round(float(np.mean(h_cov_neg)), 4),
        "retained_singleton_frac": round(float(np.mean(h_rets)), 4),
        "auroc_definitive_mean": round(float(np.mean(h_daucs)), 4),
        "platt_slope_mean": round(float(np.mean(h_slopes)), 4),
        "platt_brier_mean": round(float(np.mean(h_briers)), 4),
    }

    # ---------------- 4. Harmonized v4 Deterministic MICE MIMIC-IV Sensitivity Checks + B=2,000 CIs ----------------
    sub_iv = df_preds[df_preds["cohort"] == "MIMIC_IV_IntVal"].reset_index(drop=True)
    y_iv = sub_iv["d28"].values.astype(int)
    p_iv_v4 = sub_iv["p_adaptive_gbdt"].values.astype(float)

    # 4a) Exclude early DNR/DNI/CMO code-status limitations on exact v4 primary predictions
    con = duckdb.connect()
    cs = con.execute(f"""
        SELECT DISTINCT c.stay_id
        FROM read_csv_auto('{ROOT}/data/mimic-iv-3.1/icu/chartevents.csv.gz', types={{'value':'VARCHAR'}}) c
        JOIN read_parquet('{ROOT}/results/cohort_audit/02_tbi_cohort.parquet') co ON c.stay_id = co.stay_id
        WHERE c.itemid = 223758 AND c.value IS NOT NULL
          AND lower(trim(c.value)) != 'full code'
          AND c.charttime <= co.intime + INTERVAL 24 HOUR
    """).df()
    lim24_ids = set(cs.stay_id.astype(int))
    cs_any = con.execute(f"""
        SELECT DISTINCT c.stay_id
        FROM read_csv_auto('{ROOT}/data/mimic-iv-3.1/icu/chartevents.csv.gz', types={{'value':'VARCHAR'}}) c
        JOIN read_parquet('{ROOT}/results/cohort_audit/02_tbi_cohort.parquet') co ON c.stay_id = co.stay_id
        WHERE c.itemid = 223758 AND c.value IS NOT NULL
          AND lower(trim(c.value)) != 'full code'
    """).df()
    lim_any_ids = set(cs_any.stay_id.astype(int))

    mask_no_lim24  = ~iv["stay_id"].isin(lim24_ids).values
    mask_no_limany = ~iv["stay_id"].isin(lim_any_ids).values
    auc_nolim24, ci_nolim24, vec_nolim24   = boot_auc(y_iv[mask_no_lim24], p_iv_v4[mask_no_lim24], B=2000, seed=501)
    auc_nolimany, ci_nolimany, vec_nolimany = boot_auc(y_iv[mask_no_limany], p_iv_v4[mask_no_limany], B=2000, seed=502)
    npz_save["MIMIC_IV_Excl_DNR24"]  = vec_nolim24
    npz_save["MIMIC_IV_Excl_DNRAny"] = vec_nolimany

    # 4b) Sedative-infusion motor censoring on v4 deterministic MICE (seed=100+j)
    inf = con.execute(f"""
        SELECT e.stay_id, e.starttime, e.endtime, co.intime
        FROM read_csv_auto('{ROOT}/data/mimic-iv-3.1/icu/inputevents.csv.gz') e
        JOIN read_parquet('{ROOT}/results/cohort_audit/02_tbi_cohort.parquet') co ON e.stay_id = co.stay_id
        WHERE e.itemid IN (222168, 221668, 229420, 225150)
          AND e.starttime IS NOT NULL AND e.endtime IS NOT NULL AND e.statusdescription != 'Rewritten'
    """).df()
    inf["starttime"] = pd.to_datetime(inf.starttime); inf["endtime"] = pd.to_datetime(inf.endtime)
    inf = inf[inf.endtime > inf.starttime]
    motor = pd.read_parquet(ROOT / "results/features/13_motor_long.parquet")
    coh_t = pd.read_parquet(ROOT / "results/cohort_audit/02_tbi_cohort.parquet")[["stay_id", "intime"]]
    motor = motor.merge(coh_t, on="stay_id")
    motor["charttime"] = pd.to_datetime(motor.intime) + pd.to_timedelta(motor.hr, unit="h")
    iv_ids_set = set(iv["stay_id"].astype(int))
    m_iv24 = motor[(motor.stay_id.isin(iv_ids_set)) & (motor.hr >= 0) & (motor.hr <= 24)].copy()
    inf_iv = inf[inf.stay_id.isin(iv_ids_set)]
    iv_map = {sid: list(g[["starttime", "endtime"]].itertuples(index=False, name=None)) for sid, g in inf_iv.groupby("stay_id")}
    keep_rows = []
    for row in m_iv24.itertuples(index=False):
        wins = iv_map.get(int(row.stay_id), [])
        under = any(st <= row.charttime <= en for st, en in wins)
        if not under:
            keep_rows.append((int(row.stay_id), float(row.hr), float(row.motor)))
    m_clean = pd.DataFrame(keep_rows, columns=["stay_id", "hr", "motor"])

    # Recompute motor_last, motor_max, motor_slope, motor_sd & LCGM posteriors from non-infusion obs
    g_coefs = json.load(open(REP / "08a3_gbtm_coefs_trainonly.json", encoding="utf-8"))
    iv_cens = iv.copy()
    for col_m in ["motor_last", "motor_max", "motor_min", "motor_mean", "motor_sd", "motor_slope", "prob_m1", "prob_m2", "prob_m3"]:
        if col_m in iv_cens.columns:
            iv_cens[col_m] = np.nan

    # Project LCGM posterior helper from 08a3_gbtm_coefs_trainonly.json (random = ~ -1, exact best_vec)
    bv = [float(x["value"]) for x in g_coefs["motor"]["fit"]["best_vec"]]
    lp_vec = [bv[0], bv[1], 0.0]
    ex_pi = [math.exp(x) for x in lp_vec]
    priors = {str(k + 1): ex_pi[k] / sum(ex_pi) for k in range(3)}
    betas = {
        "1": [bv[2], bv[5], bv[8]],
        "2": [bv[3], bv[6], bv[9]],
        "3": [bv[4], bv[7], bv[10]],
    }
    sigma_e = bv[11]
    G_cov = np.zeros((3, 3), dtype=float)

    for sid, grp in m_clean.groupby("stay_id"):
        grp = grp.sort_values("hr")
        idx_r = iv_cens.index[iv_cens["stay_id"] == sid]
        if len(idx_r) == 0:
            continue
        vals = grp["motor"].values; hrs = grp["hr"].values
        iv_cens.loc[idx_r, "motor_last"] = float(vals[-1])
        iv_cens.loc[idx_r, "motor_max"]  = float(vals.max())
        iv_cens.loc[idx_r, "motor_min"]  = float(vals.min())
        iv_cens.loc[idx_r, "motor_mean"] = float(vals.mean())
        if len(vals) >= 2:
            iv_cens.loc[idx_r, "motor_sd"] = float(vals.std(ddof=1))
            if hrs.max() > hrs.min():
                slp = np.polyfit(hrs / 10.0, vals, 1)[0]
                iv_cens.loc[idx_r, "motor_slope"] = float(slp)
        # Bayesian LCGM projection on 6h bins <= 24h
        grp["bin6"] = (np.floor(grp["hr"] / 6.0) * 6.0).clip(0, 24)
        b6 = grp.groupby("bin6")["motor"].mean()
        t_vec = b6.index.values / 10.0
        y_obs = b6.values
        Z = np.column_stack([np.ones_like(t_vec), t_vec, t_vec ** 2])
        V = Z @ G_cov @ Z.T + (sigma_e ** 2) * np.eye(len(t_vec))
        invV = np.linalg.inv(V)
        sign, logdet = np.linalg.slogdet(V)
        logliks = []
        for k_str in ["1", "2", "3"]:
            mu_k = Z @ np.array(betas[k_str], dtype=float)
            diff = y_obs - mu_k
            ll = math.log(priors[k_str]) - 0.5 * (logdet + float(diff.T @ invV @ diff))
            logliks.append(ll)
        m_ll = max(logliks)
        exps = [math.exp(l - m_ll) for l in logliks]
        tot_e = sum(exps)
        iv_cens.loc[idx_r, "prob_m1"] = exps[0] / tot_e
        iv_cens.loc[idx_r, "prob_m2"] = exps[1] / tot_e
        iv_cens.loc[idx_r, "prob_m3"] = exps[2] / tot_e

    p_iv_cens_raw = pred_frozen25_det(iv_cens)
    p_iv_cens = platt_sel.predict_proba(logit(np.clip(p_iv_cens_raw, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
    # For the 10 stays with 0 non-infusion motor obs, route via Modality-Adaptive Static/CT just like primary model!
    has_noninf_m = iv_cens["motor_last"].notna().values
    p_iv_cens_adapt = np.where(has_noninf_m, p_iv_cens, sub_iv["p_static"].values)
    auc_cens_f25, ci_cens_f25, vec_cens_f25     = boot_auc(y_iv, p_iv_cens, B=2000, seed=503)
    auc_cens_adp, ci_cens_adp, vec_cens_adp     = boot_auc(y_iv, p_iv_cens_adapt, B=2000, seed=504)
    npz_save["MIMIC_IV_Infusion_Censored"] = vec_cens_adp

    out["mimic4_v4_harmonized_sensitivities"] = {
        "primary_baseline_v4": {"n": 483, "events": int(y_iv.sum()), "auroc": round(float(roc_auc_score(y_iv, p_iv_v4)), 4)},
        "excl_early_code_limit_24h": {"n": int(mask_no_lim24.sum()), "events": int(y_iv[mask_no_lim24].sum()), "auroc": round(auc_nolim24, 4), "ci95": ci_nolim24},
        "excl_any_code_limit": {"n": int(mask_no_limany.sum()), "events": int(y_iv[mask_no_limany].sum()), "auroc": round(auc_nolimany, 4), "ci95": ci_nolimany},
        "sedative_infusion_censored_frozen25": {"n": 483, "n_with_noninf_motor": int(has_noninf_m.sum()), "auroc": round(auc_cens_f25, 4), "ci95": ci_cens_f25},
        "sedative_infusion_censored_adaptive": {"n": 483, "n_with_noninf_motor": int(has_noninf_m.sum()), "auroc": round(auc_cens_adp, 4), "ci95": ci_cens_adp},
    }

    # ---------------- 5. Two Real Anonymized MIMIC-IV Validation Patient Longitudinal Vignettes (6-72h) ----------------
    # Find 1 severe-admission rapid improver (gcs_total_first <= 6, motor_last >= 5, d28 == 0) & 1 persistent non-improver (gcs_total_first <= 6, motor_last <= 2, d28 == 1)
    iv_full = iv.copy()
    iv_full["p24"] = p_iv_v4
    iv_full["p_s0"] = sub_iv["p_static"].values
    cand_imp = iv_full[(iv_full["gcs_total_first"] <= 6) & (iv_full["gcs_motor_first"] <= 3) & (iv_full["motor_last"] >= 5) & (iv_full["d28"] == 0)].copy()
    cand_dep = iv_full[(iv_full["gcs_total_first"] <= 6) & (iv_full["gcs_motor_first"] <= 3) & (iv_full["motor_last"] <= 2) & (iv_full["d28"] == 1)].copy()
    # Pick stays with rich hourly motor-GCS checks across 0-72h
    m_iv72 = motor[(motor.stay_id.isin(iv_ids_set)) & (motor.hr >= 0) & (motor.hr <= 72)]
    obs_cnt = m_iv72.groupby("stay_id").size()
    cand_imp["n_obs72"] = cand_imp["stay_id"].map(obs_cnt).fillna(0)
    cand_dep["n_obs72"] = cand_dep["stay_id"].map(obs_cnt).fillna(0)
    cand_imp = cand_imp[(cand_imp["n_obs72"] >= 12) & (cand_imp["p_s0"] >= 0.15) & (cand_imp["p24"] < 0.0247)].sort_values("p_s0", ascending=False)
    if len(cand_imp) == 0:
        cand_imp = iv_full[(iv_full["gcs_total_first"] <= 8) & (iv_full["motor_last"] >= 5) & (iv_full["d28"] == 0) & (iv_full["p24"] < 0.0247)].sort_values("p_s0", ascending=False)
    cand_dep = cand_dep[(cand_dep["n_obs72"] >= 12) & (cand_dep["p24"] > 0.3063)].sort_values("p24", ascending=False)

    vignettes = {}
    for vkey, row_sel in [("rapid_improver_survivor", cand_imp.iloc[0]), ("persistent_depressed_nonsurvivor", cand_dep.iloc[0])]:
        sid = int(row_sel["stay_id"])
        pts_m = m_iv72[m_iv72["stay_id"] == sid].sort_values("hr")
        # Compute hourly rolling p_adaptive_gbdt across t = 6..72h for this patient!
        hrs_grid = list(range(6, 73, 3))
        roll_curve = []
        for t_h in hrs_grid:
            sub_t = pts_m[pts_m["hr"] <= t_h]
            if len(sub_t) == 0:
                continue
            r_tmp = iv[iv["stay_id"] == sid].copy()
            vals = sub_t["motor"].values; hs = sub_t["hr"].values
            r_tmp["motor_last"] = float(vals[-1])
            r_tmp["motor_max"]  = float(vals.max())
            r_tmp["motor_min"]  = float(vals.min())
            r_tmp["motor_mean"] = float(vals.mean())
            r_tmp["motor_sd"]   = float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0
            r_tmp["motor_slope"]= float(np.polyfit(hs / 10.0, vals, 1)[0]) if (len(vals) >= 2 and hs.max() > hs.min()) else 0.0
            sub_t = sub_t.copy()
            sub_t["bin6"] = (np.floor(sub_t["hr"] / 6.0) * 6.0).clip(0, 72)
            b6 = sub_t.groupby("bin6")["motor"].mean()
            tv = b6.index.values / 10.0; yo = b6.values
            Z = np.column_stack([np.ones_like(tv), tv, tv ** 2])
            V = Z @ G_cov @ Z.T + (sigma_e ** 2) * np.eye(len(tv))
            invV = np.linalg.inv(V); _, ldet = np.linalg.slogdet(V)
            lls = [math.log(priors[ks]) - 0.5 * (ldet + float((yo - Z @ np.array(betas[ks])).T @ invV @ (yo - Z @ np.array(betas[ks])))) for ks in ["1", "2", "3"]]
            mll = max(lls); ex = [math.exp(l - mll) for l in lls]; se = sum(ex)
            r_tmp["prob_m1"], r_tmp["prob_m2"], r_tmp["prob_m3"] = ex[0]/se, ex[1]/se, ex[2]/se
            p_raw_t = pred_frozen25_det(r_tmp)[0]
            p_cal_t = float(platt_sel.predict_proba(logit(np.clip([[p_raw_t]], 1e-5, 1 - 1e-5)))[0, 1])
            c_state = "{0} Low-risk" if p_cal_t < 0.0247 else ("{1} High-risk" if p_cal_t > 0.3063 else "{0,1} Defer")
            roll_curve.append({"hr": t_h, "motor_last": float(vals[-1]), "p_risk": round(p_cal_t, 4), "conformal_set": c_state})
        vignettes[vkey] = {
            "age_decade": f"{int(row_sel['admission_age'] // 10 * 10)}s",
            "gcs_total_first": int(row_sel["gcs_total_first"]),
            "gcs_motor_first": int(row_sel["gcs_motor_first"]),
            "motor_24h_last": int(row_sel["motor_last"]),
            "p_static_s0": round(float(row_sel["p_s0"]), 4),
            "p_24h_primary": round(float(row_sel["p24"]), 4),
            "d28_outcome": int(row_sel["d28"]),
            "motor_checks_hr_val": [[round(float(r.hr), 1), int(r.motor)] for r in pts_m.itertuples()],
            "rolling_trajectory": roll_curve,
        }
    out["patient_vignettes_mimic4_val"] = vignettes

    np.savez_compressed(DATA / "19g_v4_5_conformal_curves.npz", **npz_save)
    (REP / "19g_v4_5_benchmark_closure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[SUCCESS] Saved reports/19g_v4_5_benchmark_closure.json!")
    print("  eICU leave-hospitals-out:", out["eicu_leave_hospitals_out_conformal"])
    print("  MIMIC-IV v4 sensitivities:", out["mimic4_v4_harmonized_sensitivities"])
    for vk, vd in vignettes.items():
        print(f"  Vignette [{vk}]: age={vd['age_decade']}, GCS0={vd['gcs_total_first']} (m={vd['gcs_motor_first']}) -> m24={vd['motor_24h_last']}, p_S0={vd['p_static_s0']} -> p24={vd['p_24h_primary']}")


if __name__ == "__main__":
    main()
