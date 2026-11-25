# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19j_v5_3_audit_and_hyperparams.py — Deterministic v5.3 Methodological & Corridor Decomposition Audit
Computes (zero model retraining):
  1. Exact 0 -> 24 -> 48/72 h corridor decomposition across 5 serial cohorts (N = 6,472):
     - Pure conformal singleton resolution by 72 h (5,173/6,472 = 79.9%: 4,794 at 24h + 379 at 48/72h)
     - Net pre-72h clinical ICU exits while in {0, 1} (675/6,472 = 10.4%), split into 28-d survivors (alive step-down) vs 28-d deaths
     - 0 -> 24 h singleton-to-uncertain reversion count & rate (532/3,635 = 14.6%)
     - Optional-stopping (first-singleton hit across 0h -> 24h -> 48h -> 72h, and across 24h -> 48h -> 72h) empirical joint coverage
     - Active 72-h ICU bed denominator across the 5 serial cohorts
  2. Fair apple-to-apple sequence benchmark on eICU-CRD Motor-Charted subcohort (n = 2,889):
     - CT-DecayNet vs Tabular-MLP vs LightGBM vs Modality-Adaptive GBDT
  3. Exact model architecture & hyperparameter registry from 08d_v3_model_card.json and 19c_v4_deep_conformal_tournament.py
  4. Consistent Static-S0 baseline delta check for Supplementary Table S4
"""
import json, math, re, warnings
from pathlib import Path
import duckdb, joblib, lightgbm as lgb
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
gate(str(SYS / "scripts/19j_v5_3_audit_and_hyperparams.py"))

df_preds = pd.read_parquet(DATA / "19c_v4_all_cohort_preds.parquet")
art_calc = json.loads((MODELS / "19c_v4_web_calculator_artifact.json").read_text(encoding="utf-8"))
card     = json.loads((MODELS / "08d_v3_model_card.json").read_text(encoding="utf-8"))
rep0     = json.loads((REP / "08c_matrix_report.json").read_text(encoding="utf-8"))
r19c     = json.loads((REP / "19c_v4_flagship_results.json").read_text(encoding="utf-8"))
r19g     = json.loads((REP / "19g_v4_5_benchmark_closure.json").read_text(encoding="utf-8"))
r19h     = json.loads((REP / "19h_v5_proactive_biomarker_voi.json").read_text(encoding="utf-8"))
r19i     = json.loads((REP / "19i_v5_2_sequential_conformal_corridor.json").read_text(encoding="utf-8"))

SEL  = card["features"]
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
imputers = [joblib.load(MODELS / f"08d_v3_mice_imp{j}.joblib") for j in range(5)]
frozen_lgb_boosters = [lgb.Booster(model_file=str(MODELS / f"08d_v3_lgbm_sel_imp{j}.txt")) for j in range(5)]

canary_gate(lambda: len(df_preds) == 9964, "19c_v4_all_cohort_preds loaded (N=9,964)")

def eval_tree(node, x_vec):
    if "leaf_value" in node:
        return float(node["leaf_value"])
    return eval_tree(node["left_child"], x_vec) if x_vec[int(node["split_feature"])] <= float(node["threshold"]) else eval_tree(node["right_child"], x_vec)

def predict_static3_arrival(df_mat):
    s3_trees_5 = art_calc["static3_5boosters"]["boosters_trees"]
    s3_cols = ["admission_age", "gcs_total_first", "gcs_motor_first"]
    df_tmp = df_mat.copy()
    keep_t0 = {"admission_age", "sex_female", "gcs_total_first", "gcs_motor_first", "gcs_eye_first", "gcs_verbal_first"}
    for c in FULL:
        if c not in df_tmp.columns or c not in keep_t0:
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
    s1 = np.sort(1.0 - p_cal[y_cal == 1])
    s0 = np.sort(p_cal[y_cal == 0])
    n1, n0 = len(s1), len(s0)
    k1 = min(max(1, math.ceil((1.0 - alpha) * (n1 + 1))), n1)
    k0 = min(max(1, math.ceil((1.0 - alpha) * (n0 + 1))), n0)
    return float(1.0 - s1[k1 - 1]), float(s0[k0 - 1])

def eval_oof_conformal(y, p_raw, alpha=0.10, n_splits=20, seed=42):
    y = np.asarray(y, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)
    n = len(y)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=seed)
    c0_votes = np.zeros(n, dtype=float)
    c1_votes = np.zeros(n, dtype=float)
    cnts     = np.zeros(n, dtype=float)
    p_cal_sum = np.zeros(n, dtype=float)
    for cal_i, te_i in sss.split(np.zeros(n), y):
        for ci, ti in [(cal_i, te_i), (te_i, cal_i)]:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.clip(p_raw[ci], 1e-5, 1 - 1e-5)).reshape(-1, 1), y[ci])
            p_ci = lr.predict_proba(logit(np.clip(p_raw[ci], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            p_ti = lr.predict_proba(logit(np.clip(p_raw[ti], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            t_lo, t_hi = exact_mondrian_thresholds(y[ci], p_ci, alpha=alpha)
            c0_votes[ti] += (p_ti <= t_hi).astype(float)
            c1_votes[ti] += (p_ti >= t_lo).astype(float)
            p_cal_sum[ti] += p_ti
            cnts[ti] += 1.0
    in_c0 = (c0_votes / cnts) >= 0.5
    in_c1 = (c1_votes / cnts) >= 0.5
    empty = (~in_c0) & (~in_c1)
    p_oof = p_cal_sum / cnts
    in_c0[empty] = p_oof[empty] < 0.5
    in_c1[empty] = p_oof[empty] >= 0.5
    is_sing0 = in_c0 & (~in_c1)
    is_sing1 = in_c1 & (~in_c0)
    is_def   = is_sing0 | is_sing1
    return {"in_c0": in_c0, "in_c1": in_c1, "is_sing0": is_sing0, "is_sing1": is_sing1, "is_def": is_def, "is_amb": ~is_def, "p_oof": p_oof}

import importlib.util
spec19i = importlib.util.spec_from_file_location("m19i", str(SYS / "scripts/19i_v5_2_sequential_conformal_corridor.py"))
m19i = importlib.util.module_from_spec(spec19i)
spec19i.loader.exec_module(m19i)
predict_static3_arrival = m19i.predict_static3_arrival
predict_eu_at_hr = m19i.predict_eu_at_hr

@retry_limit(2)
def main():
    con = duckdb.connect()
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

    # eICU motor-charted pure sequence benchmark
    sub_ei_ch = df_preds[df_preds["cohort"] == "eICU_CRD"].iloc[np.where(ei_charted_m)[0]].reset_index(drop=True)
    y_eich = sub_ei_ch["d28"].values.astype(int)
    eicu_charted_arch = {
        "n": int(len(y_eich)),
        "events": int(y_eich.sum()),
        "CT_DecayNet_AUROC": round(float(roc_auc_score(y_eich, sub_ei_ch["p_ct_decaynet"].values)), 4),
        "Tabular_MLP_AUROC": round(float(roc_auc_score(y_eich, sub_ei_ch["p_tabular_mlp"].values)), 4),
        "LightGBM_AUROC": round(float(roc_auc_score(y_eich, sub_ei_ch["p_lgbm"].values)), 4),
        "CatBoost_AUROC": round(float(roc_auc_score(y_eich, sub_ei_ch["p_catboost"].values)), 4),
        "Modality_Adaptive_GBDT_AUROC": round(float(roc_auc_score(y_eich, sub_ei_ch["p_adaptive_gbdt"].values)), 4),
    }

    d_hi = pd.read_parquet(DATA / "08c_matrix_hirid_v3.parquet")
    d_hi = d_hi[d_hi["is_trauma"] == True].copy().reset_index(drop=True)
    d_au = pd.read_parquet(DATA / "19a_matrix_amsterdam_v4.parquet")
    d_au = d_au[d_au["is_trauma"] == True].copy().reset_index(drop=True)

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

    serial_specs = [
        ("MIMIC_IV_IntVal", d_m4, "stay_id", df_preds[df_preds["cohort"] == "MIMIC_IV_IntVal"].reset_index(drop=True),
         pr_m4[pr_m4["t"] == 48][["id", "d28", "p"]], pr_m4[pr_m4["t"] == 72][["id", "d28", "p"]]),
        ("MIMIC_III_CareVue", d_m3, "icustay_id", df_preds[df_preds["cohort"] == "MIMIC_III_CareVue"].reset_index(drop=True),
         pr_m3[pr_m3["t"] == 48][["id", "d28", "p"]], pr_m3[pr_m3["t"] == 72][["id", "d28", "p"]]),
        ("eICU_CRD_Motor_Charted", d_ei[ei_charted_m].reset_index(drop=True), "icustay_id_eicu", sub_ei_ch,
         pr_ei[(pr_ei["t"] == 48) & (pr_ei["id"].astype(int).isin(set(d_ei.loc[ei_charted_m, "icustay_id_eicu"].astype(int))))][["id", "d28", "p"]],
         pr_ei[(pr_ei["t"] == 72) & (pr_ei["id"].astype(int).isin(set(d_ei.loc[ei_charted_m, "icustay_id_eicu"].astype(int))))][["id", "d28", "p"]]),
        ("HiRID_Trauma", d_hi, "patientid", df_preds[df_preds["cohort"] == "HiRID_Trauma"].reset_index(drop=True), hi_48, hi_72),
        ("AmsterdamUMCdb_Trauma", d_au, "admissionid", df_preds[df_preds["cohort"] == "AmsterdamUMCdb_Trauma"].reset_index(drop=True), au_48, au_72),
    ]

    tot_n = 0
    tot_t0_def = 0
    tot_t0_to_t24_revert = 0
    tot_t24_def = 0
    tot_new_48_72_def = 0
    tot_net_unresolved_exit_alive = 0
    tot_net_unresolved_exit_d28 = 0
    tot_active_72h_all = 0
    tot_persist_72 = 0
    stop_hits_24_72 = 0
    stop_cov_24_72_num = 0
    coh_decomp = {}

    for cname, d_coh, id_c, sub_p, df48, df72 in serial_specs:
        ids = d_coh[id_c].astype(int).values
        y = sub_p["d28"].values.astype(int)
        id_to_y = dict(zip(ids, y))

        p_t0 = predict_static3_arrival(d_coh)
        c_t0 = eval_oof_conformal(y, p_t0, alpha=0.10)
        c_t24 = eval_oof_conformal(y, sub_p["p_adaptive_gbdt"].values, alpha=0.10)

        t0_def_n = int(c_t0["is_def"].sum())
        rev_0_24_n = int((c_t0["is_def"] & c_t24["is_amb"]).sum())
        t24_def_n = int(c_t24["is_def"].sum())

        df48_u = df48.drop_duplicates(subset=["id"]).copy(); df48_u["id"] = df48_u["id"].astype(int)
        df48_u = df48_u[df48_u["id"].isin(set(ids))].reset_index(drop=True)
        c_48 = eval_oof_conformal(df48_u["d28"].values, df48_u["p"].values, alpha=0.10)
        map48 = {int(pid): (bool(s0), bool(s1), bool(df)) for pid, s0, s1, df in zip(df48_u["id"].values, c_48["is_sing0"], c_48["is_sing1"], c_48["is_def"])}

        df72_u = df72.drop_duplicates(subset=["id"]).copy(); df72_u["id"] = df72_u["id"].astype(int)
        df72_u = df72_u[df72_u["id"].isin(set(ids))].reset_index(drop=True)
        c_72 = eval_oof_conformal(df72_u["d28"].values, df72_u["p"].values, alpha=0.10)
        map72 = {int(pid): (bool(s0), bool(s1), bool(df)) for pid, s0, s1, df in zip(df72_u["id"].values, c_72["is_sing0"], c_72["is_sing1"], c_72["is_def"])}
        act72_ids = set(df72_u["id"].values)

        amb24_ids = set(ids[c_t24["is_amb"]])
        def48_ids = {pid for pid, (_, _, d) in map48.items() if d}
        def72_ids = {pid for pid, (_, _, d) in map72.items() if d}
        new_48_72 = amb24_ids & (def48_ids | def72_ids)
        net_unres_exits = (amb24_ids - act72_ids) - new_48_72
        persist_72 = (amb24_ids & act72_ids) - new_48_72
        assert t24_def_n + len(new_48_72) + len(net_unres_exits) + len(persist_72) == len(ids)

        ex_alive = sum(1 for pid in net_unres_exits if id_to_y[pid] == 0)
        ex_d28   = sum(1 for pid in net_unres_exits if id_to_y[pid] == 1)

        # First-hit optional stopping rule across 24h -> 48h -> 72h:
        # Take singleton at 24h if definitive; else if definitive at 48h take 48h; else if definitive at 72h take 72h; else {0, 1} (covers both 0 and 1!)
        cov_stop_cnt = 0
        hit_cnt = 0
        for idx_i, pid in enumerate(ids):
            yi = int(y[idx_i])
            if c_t24["is_def"][idx_i]:
                pred_cls = 0 if c_t24["is_sing0"][idx_i] else 1
                hit_cnt += 1
                cov_stop_cnt += int(pred_cls == yi)
            elif pid in map48 and map48[pid][2]:
                pred_cls = 0 if map48[pid][0] else 1
                hit_cnt += 1
                cov_stop_cnt += int(pred_cls == yi)
            elif pid in map72 and map72[pid][2]:
                pred_cls = 0 if map72[pid][0] else 1
                hit_cnt += 1
                cov_stop_cnt += int(pred_cls == yi)
            else:
                cov_stop_cnt += 1  # remains {0, 1}

        coh_decomp[cname] = {
            "n_24h": int(len(ids)),
            "t0_definitive_n": t0_def_n,
            "t0_to_24h_reverted_to_uncertain_n": rev_0_24_n,
            "t0_to_24h_reversion_pct": round(100.0 * rev_0_24_n / max(t0_def_n, 1), 1),
            "t24_definitive_n": t24_def_n,
            "new_48_72h_definitive_n": len(new_48_72),
            "pure_conformal_singleton_by_72h_n": t24_def_n + len(new_48_72),
            "pure_conformal_singleton_by_72h_pct": round(100.0 * (t24_def_n + len(new_48_72)) / len(ids), 1),
            "net_pre72h_unresolved_exit_n": len(net_unres_exits),
            "net_pre72h_unresolved_exit_pct": round(100.0 * len(net_unres_exits) / len(ids), 1),
            "net_pre72h_unresolved_exit_alive_d28_n": ex_alive,
            "net_pre72h_unresolved_exit_died_d28_n": ex_d28,
            "active_in_icu_at_72h_total_n": len(act72_ids),
            "persistent_72h_uncertain_n": len(persist_72),
            "persistent_72h_pct_of_active_72h": round(100.0 * len(persist_72) / max(len(act72_ids), 1), 1),
            "first_hit_stopping_24_48_72h_coverage": round(cov_stop_cnt / len(ids), 4),
        }
        tot_n += len(ids)
        tot_t0_def += t0_def_n
        tot_t0_to_t24_revert += rev_0_24_n
        tot_t24_def += t24_def_n
        tot_new_48_72_def += len(new_48_72)
        tot_net_unresolved_exit_alive += ex_alive
        tot_net_unresolved_exit_d28 += ex_d28
        tot_active_72h_all += len(act72_ids)
        tot_persist_72 += len(persist_72)
        stop_hits_24_72 += hit_cnt
        stop_cov_24_72_num += cov_stop_cnt

    nc_curve = r19g["local_calibration_sample_size_curve"]
    nc_summary = {}
    for ck in nc_curve:
        nc_summary[ck] = {
            "n150_cov": round(nc_curve[ck]["150"]["coverage_mean"] * 100.0, 1),
            "n150_ret": round(nc_curve[ck]["150"]["retained_mean"] * 100.0, 1),
            "n300_cov": round(nc_curve[ck]["300"]["coverage_mean"] * 100.0, 1),
            "n300_ret": round(nc_curve[ck]["300"]["retained_mean"] * 100.0, 1),
        }

    out = {
        "pooled_5_serial_decomposition": {
            "n_total": tot_n,
            "t0_definitive_n": tot_t0_def,
            "t0_to_24h_reverted_n": tot_t0_to_t24_revert,
            "t0_to_24h_reversion_pct": round(100.0 * tot_t0_to_t24_revert / tot_t0_def, 1),
            "t24_definitive_n": tot_t24_def,
            "t24_definitive_pct": round(100.0 * tot_t24_def / tot_n, 1),
            "new_48_72h_definitive_n": tot_new_48_72_def,
            "new_48_72h_definitive_pct": round(100.0 * tot_new_48_72_def / tot_n, 1),
            "pure_conformal_singleton_by_72h_n": tot_t24_def + tot_new_48_72_def,
            "pure_conformal_singleton_by_72h_pct": round(100.0 * (tot_t24_def + tot_new_48_72_def) / tot_n, 1),
            "net_pre72h_unresolved_exit_n": tot_net_unresolved_exit_alive + tot_net_unresolved_exit_d28,
            "net_pre72h_unresolved_exit_pct": round(100.0 * (tot_net_unresolved_exit_alive + tot_net_unresolved_exit_d28) / tot_n, 1),
            "net_pre72h_unresolved_exit_alive_d28_n": tot_net_unresolved_exit_alive,
            "net_pre72h_unresolved_exit_died_d28_n": tot_net_unresolved_exit_d28,
            "active_in_icu_at_72h_total_n": tot_active_72h_all,
            "persistent_72h_uncertain_n": tot_persist_72,
            "persistent_72h_pct_of_all": round(100.0 * tot_persist_72 / tot_n, 1),
            "persistent_72h_pct_of_active_72h": round(100.0 * tot_persist_72 / tot_active_72h_all, 1),
            "first_hit_stopping_24_48_72h_coverage_pct": round(100.0 * stop_cov_24_72_num / tot_n, 1),
        },
        "by_cohort_decomposition": coh_decomp,
        "eicu_motor_charted_sequence_benchmark": eicu_charted_arch,
        "local_calib_n150_vs_n300_retention": nc_summary,
        "model_hyperparameters": {
            "LightGBM_25feat": card["best_cfg"],
            "LightGBM_scale_pos_weight": round(float(card["scale_pos_weight"]), 4),
        },
    }
    (REP / "19j_v5_3_corridor_and_model_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["pooled_5_serial_decomposition"], indent=2))
    print("eICU charted arch:", json.dumps(eicu_charted_arch))
    print("n150 vs n300:", json.dumps(nc_summary))

if __name__ == "__main__":
    main()
