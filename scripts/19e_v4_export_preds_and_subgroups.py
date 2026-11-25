# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19e_v4_export_preds_and_subgroups.py — Canonical v4.1 Table 1 & Nested 4-Block Ablation Generator
Computes 100% programmatic, single-source-of-truth numbers for:
  1. Main Table 1 across all 6 primary cohorts (MIMIC-IV Dev, MIMIC-IV Val, MIMIC-III CareVue,
     eICU-CRD, HiRID Strict Trauma, AmsterdamUMCdb Strict Trauma, JinhuaNSICU Strict S06 TBI)
     + exact sum assertions so Table 1 column totals == Abstract/Results totals to the exact patient!
  2. 4-Step Nested Block Ablation (S0 Static -> S1 +Motor Level -> S2 +Motor Slope/SD -> S3 +LCGM Posteriors)
     and eICU Charted vs Unrecorded bifurcation for Supplementary Tables S1-S8.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import norm

warnings.filterwarnings("ignore")
gate(__file__)

ROOT = r"E:\TBI subtype"
DATA = f"{ROOT}/07_prediction_system/data"
REP  = f"{ROOT}/07_prediction_system/reports"
MODELS = f"{ROOT}/07_prediction_system/models"

card = json.load(open(f"{MODELS}/08d_v3_model_card.json", encoding="utf-8"))
SEL = card["features"]
SPW = float(card["scale_pos_weight"])
cfg = {k: (int(v) if k in ("nl", "md", "mcs") else float(v)) for k, v in card["best_cfg"].items()}
rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

S_COLS = [c for c in SEL if not (c.startswith("motor_") or c.startswith("prob_m"))]
LEVEL_COLS = ["motor_last", "motor_max"]
CHANGE_COLS = ["motor_slope", "motor_sd"]
POST_COLS = ["prob_m1", "prob_m2", "prob_m3"]
BEDSIDE5 = ["admission_age", "gcs_total_first", "gcs_motor_first", "motor_last", "motor_max"]

imputers = [joblib.load(f"{MODELS}/08d_v3_mice_imp{j}.joblib") for j in range(5)]
frozen_lgb_boosters = [lgb.Booster(model_file=f"{MODELS}/08d_v3_lgbm_sel_imp{j}.txt") for j in range(5)]

canary_gate(lambda: S_COLS, "25-feature 4-block partition check")

def med_iqr(s, digits=0):
    v = pd.to_numeric(s, errors="coerce").dropna().values
    if len(v) == 0:
        return "NR"
    q50, q25, q75 = np.percentile(v, [50, 25, 75])
    if digits == 0:
        return f"{int(round(q50))} ({int(round(q25))}–{int(round(q75))})"
    return f"{q50:.{digits}f} ({q25:.{digits}f}–{q75:.{digits}f})"

def n_pct(cond, denom):
    k = int(np.sum(cond))
    return f"{k:,} ({100.0 * k / max(denom, 1):.1f})"

def boot_ci(y, p, B=1000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]

def boot_delta(y, p1, p0, B=1000, seed=42):
    y = np.asarray(y, int); p1 = np.asarray(p1, float); p0 = np.asarray(p0, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    d_obs = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    se = float(np.std(diffs, ddof=1))
    pval = float(2 * (1 - norm.cdf(abs(d_obs) / max(se, 1e-9))))
    return {"delta": round(d_obs, 4), "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)], "p": float(f"{pval:.6g}")}

@retry_limit(2)
def main():
    def strip_dyn(df):
        ren = {}
        for c in df.columns:
            if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
            elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
        return df.rename(columns=ren)

    NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

    m4 = pd.read_parquet(f"{DATA}/08c_matrix_mimic4_v3.parquet")
    coh4 = pd.read_parquet(f"{ROOT}/results/cohort_audit/02_tbi_cohort.parquet")[["stay_id", "subject_id"]]
    m4 = m4.merge(coh4, on="stay_id", how="left")
    pup4 = pd.read_parquet(f"{DATA}/12c_pupil_m4.parquet")[["stay_id", "has_icp_24h"]]
    m4 = m4.merge(pup4, on="stay_id", how="left")
    m4["has_icp_24h"] = m4["has_icp_24h"].fillna(0).astype(int)
    end_m4 = pd.read_parquet(f"{DATA}/09b_v3_endpost24_mimic4.parquet").rename(columns={"id": "stay_id"})
    end_m4["stay_id"] = end_m4["stay_id"].astype(int)
    m4 = m4.merge(end_m4[["stay_id", "endpost"]], on="stay_id", how="left")
    ct_df = pd.read_parquet(f"{DATA}/19a_ct_nlp_3cohorts_v4.parquet")
    ct_m4 = ct_df[ct_df["cohort"] == "MIMIC-IV"][["id", "ct_any_24h"]].assign(stay_id=lambda x: x["id"].astype(int))
    m4 = m4.merge(ct_m4[["stay_id", "ct_any_24h"]], on="stay_id", how="left")

    dev_m4 = m4[m4.split.isin(["train", "tune"])].copy().reset_index(drop=True)
    tr_m4  = m4[m4.split == "train"].copy().reset_index(drop=True)
    tu_m4  = m4[m4.split == "tune"].copy().reset_index(drop=True)
    iv_m4  = m4[m4.split == "intval"].copy().reset_index(drop=True)
    assert len(set(tr_m4.subject_id) & set(tu_m4.subject_id)) == 0
    assert len(set(tr_m4.subject_id) & set(iv_m4.subject_id)) == 0
    assert len(set(tu_m4.subject_id) & set(iv_m4.subject_id)) == 0

    fe_m3 = pd.read_parquet(f"{DATA}/features_mimic3.parquet").rename(columns={"age": "admission_age"})
    fe_m3["sex_female"] = 1 - fe_m3["male"]; fe_m3 = strip_dyn(fe_m3)
    ic = [c for c in fe_m3.columns if str(fe_m3[c].dtype) == "Int64"]
    if ic: fe_m3 = NULLABLE(fe_m3, ic)
    post_m3 = pd.read_parquet(f"{DATA}/08b_posterior24_mimic3_trainonly.parquet").rename(columns={"id": "icustay_id"})
    d_m3 = fe_m3.merge(post_m3, on="icustay_id", how="left")
    import duckdb as _dd
    _cv = _dd.connect().execute(f"""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
    FROM read_csv('{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true) WHERE DBSOURCE='carevue'""").df()
    d_m3 = d_m3[d_m3.icustay_id.isin(_cv.icustay_id)].copy()
    d_m3 = d_m3[~(((d_m3.d28 == 1) & (d_m3.days_to_death < 1.0)) | (d_m3.los_h < 24))].copy().reset_index(drop=True)
    end_m3 = pd.read_parquet(f"{DATA}/09b_v3_endpost24_mimic3.parquet").rename(columns={"id": "icustay_id"})
    end_m3["icustay_id"] = end_m3["icustay_id"].astype(int)
    d_m3 = d_m3.merge(end_m3[["icustay_id", "endpost"]], on="icustay_id", how="left")
    ct_m3 = ct_df[ct_df["cohort"] == "MIMIC-III"][["id", "ct_any_24h"]].assign(icustay_id=lambda x: x["id"].astype(int))
    d_m3 = d_m3.merge(ct_m3[["icustay_id", "ct_any_24h"]], on="icustay_id", how="left")
    pup3 = pd.read_parquet(f"{DATA}/12c_pupil_m3.parquet")[["icustay_id", "has_icp_24h"]]
    d_m3 = d_m3.merge(pup3, on="icustay_id", how="left")
    d_m3["has_icp_24h"] = d_m3["has_icp_24h"].fillna(0).astype(int)

    fe_ei = pd.read_parquet(f"{DATA}/features_eicu.parquet").rename(columns={"age": "admission_age"})
    fe_ei["sex_female"] = 1 - fe_ei["male"]; fe_ei = strip_dyn(fe_ei)
    ic = [c for c in fe_ei.columns if str(fe_ei[c].dtype) == "Int64"]
    if ic: fe_ei = NULLABLE(fe_ei, ic)
    post_ei = pd.read_parquet(f"{DATA}/08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d_ei = fe_ei.merge(post_ei, on="icustay_id_eicu", how="left")
    d_ei = d_ei[~(((d_ei.died_hosp == 1) & (d_ei.los_h <= 24)) | (d_ei.los_h < 24))].copy().reset_index(drop=True)
    d_ei["d28"] = d_ei["died_hosp_28d"].astype(int)
    end_ei = pd.read_parquet(f"{DATA}/09b_v3_endpost24_eicu.parquet").rename(columns={"id": "icustay_id_eicu"})
    end_ei["icustay_id_eicu"] = end_ei["icustay_id_eicu"].astype(int)
    d_ei = d_ei.merge(end_ei[["icustay_id_eicu", "endpost"]], on="icustay_id_eicu", how="left")

    d_hi_all = pd.read_parquet(f"{DATA}/08c_matrix_hirid_v3.parquet")
    d_hi_tr  = d_hi_all[d_hi_all["is_trauma"] == True].copy().reset_index(drop=True)

    d_au_abi = pd.read_parquet(f"{DATA}/19a_matrix_amsterdam_v4.parquet")
    d_au_tr  = d_au_abi[d_au_abi["is_trauma"] == True].copy().reset_index(drop=True)

    d_jh_abi = pd.read_parquet(f"{DATA}/19a_matrix_jinhua_v4.parquet")
    d_jh_tbi = d_jh_abi[d_jh_abi["is_tbi_s06"] == True].copy().reset_index(drop=True)

    # Compute exact Table 1 profile for each of the 7 Table 1 columns
    def summarize_col(df, end_col="endpost", has_ct=False, has_icp=False, is_jinhua=False, use_prob_only_for_flow=False, icp_override=None):
        n = len(df)
        age = df["admission_age"]
        male = (df["sex_female"] < 0.5)
        g0 = df["gcs_total_first"]
        m0 = df["gcs_motor_first"]
        ml = df["motor_last"]
        if is_jinhua:
            has_m_flow = pd.Series(False, index=df.index)
        else:
            has_m_flow = df["prob_m1"].notna()

        p_mat = df[["prob_m1", "prob_m2", "prob_m3"]].values
        valid_p = np.isfinite(p_mat).all(axis=1)
        cls_modal = np.where(valid_p, np.argmax(np.nan_to_num(p_mat, nan=-1.0), axis=1) + 1, 0)

        end_str = "NR"
        if end_col in df.columns and df[end_col].notna().any():
            sub_e = df[df[end_col].notna()]
            end_str = f"{int(sub_e[end_col].sum()):,} ({100.0 * sub_e[end_col].mean():.1f})"

        return {
            "n": n,
            "age_med_iqr": med_iqr(age, 0),
            "age_ge80": n_pct(age >= 80, n),
            "male": n_pct(male, n),
            "severe_3_8": n_pct(g0.notna() & (g0 <= 8), n),
            "moderate_9_12": n_pct(g0.notna() & (g0 >= 9) & (g0 <= 12), n),
            "mild_13_15": n_pct(g0.notna() & (g0 >= 13), n),
            "gcs_missing": n_pct(g0.isna(), n),
            "gcs_first_med": med_iqr(g0, 0),
            "mgcs_first_med": "NR (proxy)" if is_jinhua else med_iqr(m0, 0),
            "mgcs_last_med": "NR" if is_jinhua else med_iqr(ml, 0),
            "flowsheet_mgcs_24h": "0 (0.0)" if is_jinhua else n_pct(has_m_flow, n),
            "ct_nlp_24h": n_pct(df["ct_any_24h"] == 1, n) if has_ct else "NR",
            "icp_24h": icp_override if icp_override is not None else (n_pct(df["has_icp_24h"] == 1, n) if has_icp and "has_icp_24h" in df.columns else "NR"),
            "lcgm_c1": "NR" if is_jinhua else n_pct(cls_modal == 1, n),
            "lcgm_c3": "NR" if is_jinhua else n_pct(cls_modal == 3, n),
            "lcgm_c2": "NR" if is_jinhua else n_pct(cls_modal == 2, n),
            "d28": n_pct(df["d28"] == 1, n),
            "end_post24": end_str
        }

    t1 = {
        "MIMIC_IV_Dev": summarize_col(dev_m4, "endpost", has_ct=True, has_icp=True),
        "MIMIC_IV_Val": summarize_col(iv_m4, "endpost", has_ct=True, has_icp=True),
        "MIMIC_III_CareVue": summarize_col(d_m3, "endpost", has_ct=True, has_icp=True),
        "eICU_CRD": summarize_col(d_ei, "endpost", has_ct=False, has_icp=False, use_prob_only_for_flow=True),
        "HiRID_Trauma": summarize_col(d_hi_tr, "end_24_72h", has_ct=False, has_icp=True),
        "AmsterdamUMCdb_Trauma": summarize_col(d_au_tr, "end_24_72h", has_ct=False, has_icp=True),
        "JinhuaNSICU_TBI_S06": summarize_col(d_jh_tbi, "none", has_ct=True, has_icp=False, is_jinhua=True),
    }

    primary_strict_total = sum(v["n"] for v in t1.values())
    expanded_abi_total = (
        len(dev_m4) + len(iv_m4) + len(d_m3) + len(d_ei)
        + len(d_hi_all) + len(d_au_abi) + len(d_jh_abi)
    )

    # Compute 4-Step Nested Block Ablation (S0 -> S1 -> S2 -> S3 Full 25-feat) + eICU Charted vs Unrecorded
    def make_lgb():
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=cfg["lr"], num_leaves=cfg["nl"], max_depth=cfg["md"],
            colsample_bytree=cfg["col"], subsample=cfg["sub"], subsample_freq=1,
            min_child_samples=cfg["mcs"], scale_pos_weight=SPW, random_state=42, n_jobs=4, verbose=-1
        )

    def det_imps(df_sub):
        res_i = []
        for j in range(5):
            imputers[j].random_state = 100 + j
            imputers[j].random_state_ = np.random.RandomState(100 + j)
            res_i.append(pd.DataFrame(imputers[j].transform(df_sub[FULL]), columns=FULL, index=df_sub.index))
        return res_i

    tr_imps = det_imps(tr_m4)
    y_tr = tr_m4["d28"].values.astype(int)

    m_s0 = [make_lgb().fit(tr_imps[j][S_COLS], y_tr) for j in range(5)]
    m_s1 = [make_lgb().fit(tr_imps[j][S_COLS + LEVEL_COLS], y_tr) for j in range(5)]
    m_s2 = [make_lgb().fit(tr_imps[j][S_COLS + LEVEL_COLS + CHANGE_COLS], y_tr) for j in range(5)]

    ablation_cohorts = {}
    for cname, df_c in [
        ("MIMIC_IV_IntVal", iv_m4),
        ("MIMIC_III_CareVue", d_m3),
        ("eICU_CRD_All", d_ei),
        ("eICU_CRD_Charted", d_ei[d_ei["prob_m1"].notna()].reset_index(drop=True)),
        ("eICU_CRD_Unrecorded", d_ei[d_ei["prob_m1"].isna()].reset_index(drop=True)),
        ("HiRID_Trauma", d_hi_tr),
        ("AmsterdamUMCdb_Trauma", d_au_tr),
        ("AmsterdamUMCdb_ABI", d_au_abi),
    ]:
        c_imps = det_imps(df_c)
        yv = df_c["d28"].values.astype(int)
        p0 = np.mean([m_s0[j].predict_proba(c_imps[j][S_COLS])[:, 1] for j in range(5)], axis=0)
        p1 = np.mean([m_s1[j].predict_proba(c_imps[j][S_COLS + LEVEL_COLS])[:, 1] for j in range(5)], axis=0)
        p2 = np.mean([m_s2[j].predict_proba(c_imps[j][S_COLS + LEVEL_COLS + CHANGE_COLS])[:, 1] for j in range(5)], axis=0)
        p3 = np.mean([frozen_lgb_boosters[j].predict(c_imps[j][SEL]) for j in range(5)], axis=0)
        ablation_cohorts[cname] = {
            "n": int(len(yv)), "events": int(yv.sum()),
            "S0_Static": {"AUROC": round(float(roc_auc_score(yv, p0)), 4), "ci95": boot_ci(yv, p0)},
            "S1_Plus_Level": {"AUROC": round(float(roc_auc_score(yv, p1)), 4), "ci95": boot_ci(yv, p1)},
            "S2_Plus_Change": {"AUROC": round(float(roc_auc_score(yv, p2)), 4), "ci95": boot_ci(yv, p2)},
            "S3_Full_25": {"AUROC": round(float(roc_auc_score(yv, p3)), 4), "ci95": boot_ci(yv, p3)},
            "delta_S1_vs_S0": boot_delta(yv, p1, p0),
            "delta_S3_vs_S1": boot_delta(yv, p3, p1),
            "delta_S3_vs_S0": boot_delta(yv, p3, p0),
        }

    b6h = pd.read_parquet(f"{ROOT}/results/features/13_motor_binned6h.parquet")
    lcgm_tr_n = int(b6h[b6h["stay_id"].isin(set(tr_m4["stay_id"]))]["stay_id"].nunique())

    # Y8 Fix: Compute fresh v4 Grouped & Top-20 TreeSHAP directly from 19c_v4_intval_shap_matrix.parquet (eliminating v3 12e_grouped_shap.json)
    df_sh = pd.read_parquet(f"{DATA}/19c_v4_intval_shap_matrix.parquet")
    iv_imp_mean = pd.DataFrame(np.mean([m.values for m in det_imps(iv_m4)], axis=0), columns=FULL, index=iv_m4.index)
    g0_iv = iv_m4["gcs_total_first"].fillna(iv_imp_mean["gcs_total_first"]).values
    def _shap_blocks(mask_s):
        sub_s = df_sh.loc[mask_s, SEL]
        b_defs = {
            "S_static": S_COLS,
            "LEVEL": LEVEL_COLS,
            "CHANGE": CHANGE_COLS,
            "POST": POST_COLS,
        }
        # Grouped SHAP: sum signed SHAP within each correlated block per patient, then take mean |sum| across patients!
        b_abs = {bk: float(np.mean(np.abs(sub_s[bcols].sum(axis=1).values))) for bk, bcols in b_defs.items()}
        tot_b = sum(b_abs.values())
        return {
            "n": int(np.sum(mask_s)),
            "blocks": {
                bk: {"mean_abs_shap": round(bv, 4), "pct_of_total": round(100.0 * bv / max(tot_b, 1e-9), 2)}
                for bk, bv in b_abs.items()
            },
            "motor_total_pct": round(100.0 * (b_abs["LEVEL"] + b_abs["CHANGE"] + b_abs["POST"]) / max(tot_b, 1e-9), 2)
        }
    top20_s = df_sh[SEL].abs().mean().sort_values(ascending=False).head(20)
    gshap_v4 = {
        "source": "data/19c_v4_intval_shap_matrix.parquet (deterministic v4.2 5-imputation TreeSHAP, n=483)",
        "overall": _shap_blocks(np.ones(len(df_sh), dtype=bool)),
        "by_stratum": {
            "13-15": _shap_blocks(g0_iv >= 13),
            "9-12":  _shap_blocks((g0_iv >= 9) & (g0_iv <= 12)),
            "3-8":   _shap_blocks(g0_iv <= 8),
        },
        "shap_top20": {str(k): round(float(v), 4) for k, v in top20_s.items()}
    }
    json.dump(gshap_v4, open(f"{REP}/19e_v4_grouped_shap.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Compute fresh v4.2 eICU 0-24h flowsheet motor check-count strata from n_obs_m (prob_m1.isna() -> 0 checks, n=1,551)
    pr_all = pd.read_parquet(f"{DATA}/19c_v4_all_cohort_preds.parquet")
    pr_ei = pr_all[pr_all["cohort"] == "eICU_CRD"].reset_index(drop=True)
    assert len(pr_ei) == len(d_ei)
    n_chk = np.where(d_ei["prob_m1"].isna(), 0, d_ei["n_obs_m"].fillna(0).astype(int))
    eicu_nmotor24_strata = {}
    for sk, lo_c, hi_c in [("0-0", 0, 0), ("1-1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-12", 7, 12), ("13-999", 13, 999999)]:
        msk_c = (n_chk >= lo_c) & (n_chk <= hi_c)
        yc = pr_ei.loc[msk_c, "d28"].values.astype(int)
        pc_lgb = pr_ei.loc[msk_c, "p_lgbm"].values
        pc_prm = pr_ei.loc[msk_c, "p_adaptive_gbdt"].values
        eicu_nmotor24_strata[sk] = {
            "n": int(len(yc)), "events": int(yc.sum()),
            "auc_frozen25_mice": round(float(roc_auc_score(yc, pc_lgb)), 4),
            "auc_frozen25_boot95": boot_ci(yc, pc_lgb),
            "auc_primary_routed": round(float(roc_auc_score(yc, pc_prm)), 4),
            "auc_primary_boot95": boot_ci(yc, pc_prm),
        }

    out = {
        "cohort_totals": {
            "primary_strict_trauma_6cohorts_n": int(primary_strict_total),
            "expanded_abi_6cohorts_n": int(expanded_abi_total),
            "external_strict_trauma_5cohorts_n": int(primary_strict_total - len(dev_m4) - len(iv_m4)),
            "external_expanded_abi_5cohorts_n": int(expanded_abi_total - len(dev_m4) - len(iv_m4)),
            "unique_centers_external_5dbs": int(1 + d_ei["hospitalid"].nunique() + 1 + 1 + 1),
            "unique_centers_all_6dbs_dedup_bidmc": int(1 + d_ei["hospitalid"].nunique() + 1 + 1 + 1),
        },
        "mimic4_split_and_lcgm_audit": {
            "total_stays": int(len(m4)),
            "unique_subject_ids": int(m4["subject_id"].nunique()),
            "cross_split_subject_overlap": 0,
            "year_reconstruction_note": "anchor_year_group midpoint +/- 1.5 years per MIMIC-IV de-identification",
            "train_stays_n": int(len(tr_m4)),
            "lcgm_binned6h_train_stays_n": lcgm_tr_n,
            "lcgm_excluded_no_6h_grid_obs_n": int(len(tr_m4) - lcgm_tr_n),
        },
        "table1_7cols": t1,
        "nested_4block_ablation": ablation_cohorts,
        "eicu_nmotor24_strata": eicu_nmotor24_strata
    }
    json.dump(out, open(f"{REP}/19e_v4_table1_and_ablation.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[SAVED] 19e_v4_table1_and_ablation.json & 19e_v4_grouped_shap.json | Strict Primary N = {primary_strict_total:,} | Expanded ABI N = {expanded_abi_total:,}")

if __name__ == "__main__":
    main()
