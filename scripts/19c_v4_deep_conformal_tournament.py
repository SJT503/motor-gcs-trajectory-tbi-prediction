# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19c_v4_deep_conformal_tournament.py — Clean Leak-Free v4.2 Opus-5.5-Max Verified Modeling & Split-Conformal Pipeline
Remediates Opus 5.5 Max R4, R5, R6, R10, R11 & Y1, Y3, Y6, Y8, Y9, Y11, Y15, Y17, Y22:
  1. Y6 Fix: Deterministic MICE transform (resets random_state = 100 + j before every transform) for 100% cross-script reproducibility.
  2. R4/Y1/Y17 Fix: Splits MIMIC-IV Tune (n=556) into disjoint Tune-Select (n=278, for early stopping/stacking/calibrators)
     and Tune-Calib (n=278, strictly untouched, for exact k-th order-statistic np.sort(s)[k-1] 90% Mondrian thresholds);
     on external cohorts, reports BOTH zero-touch frozen transfer AND exact 20x 50/50 within-site out-of-fold Split-Conformal!
  3. R5 Fix: Designates a single prespecified primary model (Modality-Adaptive GBDT / Hybrid), adds Tabular-MLP (sequence-free
     neural ablation) and paired 2,000-bootstrap Delta-AUROC tests (CT-DecayNet vs LightGBM, CT-DecayNet vs Tabular-MLP).
  4. R6/Y8/Y15 Fix: Separates pure-static calibrator (cal_fb_s0) vs static+CT calibrator (cal_fb_s0_ct), attributes Jinhua/eICU
     rescue accurately to Modality-Adaptive non-imputation routing, and adds CT-observed-only (ct_any_24h==1) sensitivity.
  5. R10/Y11/R11 Fix: Adds eICU 130-hospital cluster bootstrap CI, hospital charting summary JSON, strict S06 / explicit trauma-dx
     & non-imputed GCS sensitivity, frozen calibration slope/CITL, and exports ALL 5x300 = 1,500 Bedside-5 trees + dedicated
     Bedside-5 Mondrian thresholds for the 100% heuristic-free web calculator!
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import norm
from scipy.special import logit, expit
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
gate(__file__)

torch.manual_seed(42)
np.random.seed(42)

ROOT = r"E:\TBI subtype"
DATA = f"{ROOT}/07_prediction_system/data"
REP  = f"{ROOT}/07_prediction_system/reports"
MODELS = f"{ROOT}/07_prediction_system/models"

t0 = time.time()
def lg(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

card = json.load(open(f"{MODELS}/08d_v3_model_card.json", encoding="utf-8"))
SEL = card["features"]
SPW = float(card["scale_pos_weight"])
THR_RAW = float(card["threshold_youden_train_oof"])
cfg = {k: (int(v) if k in ("nl", "md", "mcs") else float(v)) for k, v in card["best_cfg"].items()}
rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

S_COLS = [c for c in SEL if not (c.startswith("motor_") or c.startswith("prob_m"))]
LEVEL_COLS = ["motor_last", "motor_max"]
BEDSIDE5 = ["admission_age", "gcs_total_first", "gcs_motor_first", "motor_last", "motor_max"]
CT_COLS = [
    "ct_any_24h", "ct_midline_shift", "ct_cistern_compressed", "ct_herniation",
    "ct_sdh", "ct_edh", "ct_sah", "ct_contusion_ich", "ct_edema_mass",
    "ct_ivh", "ct_skull_fx", "ct_marshall_proxy"
]
CT_LESION_COLS = [c for c in CT_COLS if c != "ct_any_24h"]

imputers = [joblib.load(f"{MODELS}/08d_v3_mice_imp{j}.joblib") for j in range(5)]
frozen_lgb_boosters = [lgb.Booster(model_file=f"{MODELS}/08d_v3_lgbm_sel_imp{j}.txt") for j in range(5)]

def deterministic_mice_5(df):
    """Y6/Y1 Fix: Resets BOTH imputer.random_state and imputer.random_state_ before every transform for 100% call-order independence."""
    out = []
    for j in range(5):
        imputers[j].random_state = 100 + j
        imputers[j].random_state_ = np.random.RandomState(100 + j)
        out.append(pd.DataFrame(imputers[j].transform(df[FULL]), columns=FULL, index=df.index))
    return out

def fit_nonneg_stacker(Z, y, l2_reg=1.0):
    """Y2 Fix: Non-negative constrained logistic stacker (w_k >= 0) on Tune-Select so no base learner gets negative weight."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k_dim = Z.shape[1]
    def loss_fn(params):
        w, b = params[:k_dim], params[k_dim]
        z = np.clip(Z @ w + b, -30.0, 30.0)
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        return nll + 0.5 * l2_reg * float(np.sum(w ** 2))
    bounds = [(0.0, None)] * k_dim + [(None, None)]
    res = minimize(loss_fn, np.array([1.0 / k_dim] * k_dim + [0.0]), method="L-BFGS-B", bounds=bounds)
    w_hat, b_hat = res.x[:k_dim], float(res.x[k_dim])
    return w_hat, b_hat

def _canary():
    assert len(SEL) == 25 and len(frozen_lgb_boosters) == 5
    return SEL

canary_gate(_canary, "Frozen 08d_v3 25-feature boosters & deterministic MICE loaded")

def boot_ci_with_dist(y, p, B=2000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan, np.array([]))
    aucs = np.empty(B, dtype=np.float64)
    for b in range(B):
        ip = rng.choice(pos, len(pos), replace=True)
        ineg = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        aucs[b] = roc_auc_score(y[idx], p[idx])
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), aucs)

def cluster_boot_ci(y, p, clusters, B=2000, seed=42):
    """Y11 Fix: Hospital-level cluster bootstrap 95% CI across 130 eICU hospitals."""
    y = np.asarray(y, int); p = np.asarray(p, float); clusters = np.asarray(clusters)
    uniq_c = np.unique(clusters)
    c_to_idx = [np.where(clusters == c)[0] for c in uniq_c]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(B):
        chosen = rng.integers(0, len(uniq_c), len(uniq_c))
        idx = np.concatenate([c_to_idx[k] for k in chosen])
        yb = y[idx]
        if yb.sum() >= 5 and (len(yb) - yb.sum()) >= 5:
            aucs.append(roc_auc_score(yb, p[idx]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]

def boot_delta_auc(y, p1, p0, B=2000, seed=42):
    y = np.asarray(y, int); p1 = np.asarray(p1, float); p0 = np.asarray(p0, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        ip = rng.choice(pos, len(pos), replace=True)
        ineg = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        diffs[b] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx])
    d_obs = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    se = float(np.std(diffs, ddof=1))
    pval = float(2 * (1 - norm.cdf(abs(d_obs) / max(se, 1e-9))))
    return {
        "delta": round(d_obs, 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "p": float(f"{pval:.6g}")
    }

def calc_slope_citl(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lg_p = logit(p).reshape(-1, 1)
    sl_m = LogisticRegression(C=1e6, max_iter=1500).fit(lg_p, y)
    citl = float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6)) - logit(np.clip(p.mean(), 1e-6, 1 - 1e-6)))
    return float(sl_m.coef_[0][0]), citl

def exact_mondrian_thresholds(y_cal, p_cal, alpha=0.10):
    """Y1 Fix: Exact k-th order statistic np.sort(s)[k-1] with k = ceil((n_y + 1)*(1 - alpha))."""
    s1 = np.sort(1.0 - p_cal[y_cal == 1])
    s0 = np.sort(p_cal[y_cal == 0])
    n1, n0 = len(s1), len(s0)
    k1 = min(n1, int(np.ceil((n1 + 1) * (1.0 - alpha))))
    k0 = min(n0, int(np.ceil((n0 + 1) * (1.0 - alpha))))
    q1_hat = float(s1[k1 - 1])
    q0_hat = float(s0[k0 - 1])
    return round(1.0 - q1_hat, 6), round(q0_hat, 6), k1, n1, k0, n0

MAX_SEQ_LEN = 24

def build_seq24_tensors(ids_vec, long_df):
    N = len(ids_vec)
    seq_v = np.zeros((N, MAX_SEQ_LEN, 5), dtype=np.float32)
    seq_dt = np.zeros((N, MAX_SEQ_LEN), dtype=np.float32)
    seq_mask = np.zeros((N, MAX_SEQ_LEN), dtype=np.float32)
    if long_df is None or len(long_df) == 0:
        return seq_v, seq_dt, seq_mask

    RAW2S = {"motor": "gcs_motor", "eye": "gcs_eye", "verbal": "gcs_verbal"}
    d = long_df.rename(columns=RAW2S)
    d = d[(d["offset_hr"] >= 0.0) & (d["offset_hr"] <= 24.0) & d["gcs_motor"].notna()].copy()
    if len(d) == 0:
        return seq_v, seq_dt, seq_mask

    id_to_idx = {int(k): i for i, k in enumerate(ids_vec)}
    d["id_int"] = pd.to_numeric(d["id"], errors="coerce")
    d = d[d["id_int"].isin(id_to_idx)].sort_values(["id_int", "offset_hr"])

    for gid, grp in d.groupby("id_int", sort=False):
        r_idx = id_to_idx[int(gid)]
        if len(grp) > MAX_SEQ_LEN:
            grp = grp.iloc[-MAX_SEQ_LEN:]
        K = len(grp)
        t_arr = grp["offset_hr"].values.astype(np.float32)
        m_arr = grp["gcs_motor"].values.astype(np.float32) / 6.0
        # Zero constant E/V fabrication: pass 0 with explicit availability indicators via observed subscores if present
        e_obs = grp["gcs_eye"].ffill().bfill()
        v_obs = grp["gcs_verbal"].ffill().bfill()
        e_arr = e_obs.fillna(0.0).values.astype(np.float32) / 4.0
        v_arr = v_obs.fillna(0.0).values.astype(np.float32) / 5.0
        has_ev = (e_obs.notna() & v_obs.notna()).values
        tot_arr = np.where(has_ev, np.clip(m_arr * 6.0 + e_arr * 4.0 + v_arr * 5.0, 3.0, 15.0) / 15.0, m_arr)

        dt_arr = np.zeros(K, dtype=np.float32)
        if K > 1:
            dt_arr[1:] = np.maximum(0.0, t_arr[1:] - t_arr[:-1])

        seq_v[r_idx, :K, 0] = m_arr
        seq_v[r_idx, :K, 1] = e_arr
        seq_v[r_idx, :K, 2] = v_arr
        seq_v[r_idx, :K, 3] = tot_arr
        seq_v[r_idx, :K, 4] = t_arr / 24.0
        seq_dt[r_idx, :K] = dt_arr
        seq_mask[r_idx, :K] = 1.0

    return seq_v, seq_dt, seq_mask

class TrueCTDecayAttentionNet(nn.Module):
    def __init__(self, d_tab, d_seq=5, d_hid=32, use_seq=True):
        super().__init__()
        self.d_hid = d_hid
        self.use_seq = use_seq
        if use_seq:
            self.dt_linear = nn.Linear(1, d_hid)
            self.softplus = nn.Softplus()
            self.gru_cell = nn.GRUCell(d_seq, d_hid)
            self.seq_attn = nn.MultiheadAttention(embed_dim=d_hid, num_heads=4, dropout=0.10, batch_first=True)
            self.seq_norm = nn.LayerNorm(d_hid)
            self.fusion_gate = nn.Sequential(nn.Linear(d_hid * 2 + 1, d_hid), nn.Sigmoid())

        self.tab_enc = nn.Sequential(
            nn.Linear(d_tab, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(64, d_hid), nn.LayerNorm(d_hid), nn.SiLU()
        )
        in_head = d_hid * 2 if use_seq else d_hid
        self.head = nn.Sequential(
            nn.Linear(in_head, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(0.15), nn.Linear(32, 1)
        )

    def forward(self, seq_v, seq_dt, seq_mask, tab_x):
        tab_ctx = self.tab_enc(tab_x)
        if not self.use_seq:
            return self.head(tab_ctx).squeeze(-1)
        B, L, _ = seq_v.shape
        h = torch.zeros(B, self.d_hid, device=seq_v.device, dtype=seq_v.dtype)
        h_list = []
        for k in range(L):
            dt_k = seq_dt[:, k:k+1]
            m_k = seq_mask[:, k:k+1]
            gamma_k = torch.exp(-self.softplus(self.dt_linear(dt_k)))
            h_decayed = h * gamma_k
            h_cand = self.gru_cell(seq_v[:, k, :], h_decayed)
            h = m_k * h_cand + (1.0 - m_k) * h
            h_list.append(h)

        H_seq = torch.stack(h_list, dim=1)
        has_any_seq = (seq_mask.sum(dim=1, keepdim=True) > 0.5).float()
        kp_mask = (seq_mask < 0.5)
        kp_mask[:, 0] = False
        attn_out, _ = self.seq_attn(H_seq, H_seq, H_seq, key_padding_mask=kp_mask)
        H_attn = self.seq_norm(H_seq + attn_out)
        denom = torch.clamp(seq_mask.sum(dim=1, keepdim=True), min=1.0)
        seq_ctx = (H_attn * seq_mask.unsqueeze(-1)).sum(dim=1) / denom * has_any_seq
        g = self.fusion_gate(torch.cat([tab_ctx, seq_ctx, has_any_seq], dim=-1))
        fused = torch.cat([tab_ctx, g * seq_ctx], dim=-1)
        return self.head(fused).squeeze(-1)

@retry_limit(2)
def main():
    lg("1. Loading all 6 transcontinental cohorts & deterministic 5-imputation feature tensors...")

    def strip_dyn(df):
        ren = {}
        for c in df.columns:
            if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
            elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
        return df.rename(columns=ren)

    NULLABLE = lambda d, cols: d.assign(**{c: d[c].to_numpy(dtype="float64", na_value=np.nan) for c in cols})

    m4 = pd.read_parquet(f"{DATA}/08c_matrix_mimic4_v3.parquet")
    tr = m4[m4.split == "train"].copy().reset_index(drop=True)
    tu = m4[m4.split == "tune"].copy().reset_index(drop=True)
    iv = m4[m4.split == "intval"].copy().reset_index(drop=True)
    lng_m4 = pd.read_parquet(f"{DATA}/gcs_long_mimic4.parquet")

    # R4 Fix: Stratified 50/50 split of Tune (n=556) into Tune-Select (n=278) & untouched Tune-Calib (n=278)!
    sss_tu = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    tu_sel_idx, tu_cal_idx = next(sss_tu.split(tu, tu["d28"].values.astype(int)))
    tu_sel = tu.iloc[tu_sel_idx].copy().reset_index(drop=True)
    tu_cal = tu.iloc[tu_cal_idx].copy().reset_index(drop=True)
    lg(f"  Disjoint Tune split: Tune-Select n={len(tu_sel)} (ev={int(tu_sel.d28.sum())}) | Untouched Tune-Calib n={len(tu_cal)} (ev={int(tu_cal.d28.sum())})")

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
    lng_m3 = pd.read_parquet(f"{DATA}/gcs_long_mimic3.parquet")

    fe_ei = pd.read_parquet(f"{DATA}/features_eicu.parquet").rename(columns={"age": "admission_age"})
    fe_ei["sex_female"] = 1 - fe_ei["male"]; fe_ei = strip_dyn(fe_ei)
    ic = [c for c in fe_ei.columns if str(fe_ei[c].dtype) == "Int64"]
    if ic: fe_ei = NULLABLE(fe_ei, ic)
    post_ei = pd.read_parquet(f"{DATA}/08b_posterior24_eicu_trainonly.parquet").rename(columns={"id": "icustay_id_eicu"})
    d_ei = fe_ei.merge(post_ei, on="icustay_id_eicu", how="left")
    d_ei = d_ei[~(((d_ei.died_hosp == 1) & (d_ei.los_h <= 24)) | (d_ei.los_h < 24))].copy().reset_index(drop=True)
    d_ei["d28"] = d_ei["died_hosp_28d"].astype(int)
    lng_ei = pd.read_parquet(f"{DATA}/gcs_long_eicu.parquet")

    # Export exact eICU 130-hospital charting distribution for Supp Fig S1 & text (resolving Y7 & ? #3)
    h_ei = d_ei.groupby("hospitalid").agg(n=("prob_m1", "size"), share=("prob_m1", lambda s: s.notna().mean()))
    eicu_hosp_audit = {
        "n_hospitals": int(len(h_ei)),
        "n_zero_charting_hospitals": int((h_ei["share"] == 0.0).sum()),
        "pct_zero_charting_hospitals": round(float((h_ei["share"] == 0.0).mean() * 100.0), 1),
        "n_zero_charting_stays": int(h_ei.loc[h_ei["share"] == 0.0, "n"].sum()),
        "n_full_100pct_hospitals": int((h_ei["share"] == 1.0).sum()),
        "pct_full_100pct_hospitals": round(float((h_ei["share"] == 1.0).mean() * 100.0), 1),
    }

    d_hi_all = pd.read_parquet(f"{DATA}/08c_matrix_hirid_v3.parquet")
    d_hi_tr = d_hi_all[d_hi_all["is_trauma"] == True].copy().reset_index(drop=True)
    d_hi_nt = d_hi_all[d_hi_all["is_neurotrauma_or_icp"] == True].copy().reset_index(drop=True)
    lng_hi = pd.read_parquet(f"{DATA}/19a_gcs24_hirid_v4.parquet")

    d_au_abi = pd.read_parquet(f"{DATA}/19a_matrix_amsterdam_v4.parquet").reset_index(drop=True)
    d_au_tr  = d_au_abi[d_au_abi["is_trauma"] == True].copy().reset_index(drop=True)
    lng_au = pd.read_parquet(f"{DATA}/19a_gcs72_amsterdam_v4.parquet")

    d_jh_abi = pd.read_parquet(f"{DATA}/19a_matrix_jinhua_v4.parquet").reset_index(drop=True)
    d_jh_tbi = d_jh_abi[d_jh_abi["is_tbi_s06"] == True].copy().reset_index(drop=True)

    def build_cohort_pack(raw_df, id_col, long_df):
        imps = deterministic_mice_5(raw_df)
        imp_mean = pd.DataFrame(np.mean([m.values for m in imps], axis=0), columns=FULL, index=raw_df.index)
        tab_df = imp_mean[SEL].copy()
        # Y7 Fix: Unified 0-24h flowsheet motor-GCS availability indicator (prob_m1.notna(); eICU = 2,889 charted / 1,551 unrecorded everywhere!)
        has_m = raw_df.get("prob_m1", pd.Series(np.nan, index=raw_df.index)).notna().astype(float)
        has_l = raw_df["bun_min"].notna().astype(float)
        tab_df["mask_motor_charted"] = has_m.values
        tab_df["mask_labs_charted"] = has_l.values
        ids_v = raw_df[id_col].values if id_col is not None else np.arange(len(raw_df))
        sv, sdt, smk = build_seq24_tensors(ids_v, long_df)
        y_v = raw_df["d28"].values.astype(int)
        return {"raw": raw_df, "imps": imps, "imp_mean": imp_mean, "tab": tab_df, "sv": sv, "sdt": sdt, "smk": smk, "y": y_v}

    # Verify Y1/Y6 call-order independence of deterministic_mice_5
    _chk1 = deterministic_mice_5(iv.head(25))[0].values
    _chk2 = deterministic_mice_5(iv.head(25))[0].values
    assert np.max(np.abs(_chk1 - _chk2)) == 0.0, "MICE random_state_ repeat diff != 0!"

    P_tr     = build_cohort_pack(tr, "stay_id", lng_m4)
    P_tu_sel = build_cohort_pack(tu_sel, "stay_id", lng_m4)
    P_tu_cal = build_cohort_pack(tu_cal, "stay_id", lng_m4)
    P_iv     = build_cohort_pack(iv, "stay_id", lng_m4)
    P_m3     = build_cohort_pack(d_m3, "icustay_id", lng_m3)
    P_ei     = build_cohort_pack(d_ei, "icustay_id_eicu", lng_ei)
    P_hi_tr  = build_cohort_pack(d_hi_tr, "patientid", lng_hi)
    P_hi_nt  = build_cohort_pack(d_hi_nt, "patientid", lng_hi)
    P_hi_abi = build_cohort_pack(d_hi_all, "patientid", lng_hi)
    P_au_tr  = build_cohort_pack(d_au_tr, "admissionid", lng_au)
    P_au_abi = build_cohort_pack(d_au_abi, "admissionid", lng_au)
    P_jh_tbi = build_cohort_pack(d_jh_tbi, None, None)
    P_jh_abi = build_cohort_pack(d_jh_abi, None, None)

    lg("2. Fitting 5-imputation GBDTs, Tabular-MLP ablation & CT-DecayNet strictly on MIMIC-IV train (early stopping on Tune-Select n=278)...")

    def make_lgb(n_est=400, seed=42):
        return lgb.LGBMClassifier(
            n_estimators=n_est, learning_rate=cfg["lr"], num_leaves=cfg["nl"], max_depth=cfg["md"],
            colsample_bytree=cfg["col"], subsample=cfg["sub"], subsample_freq=1,
            min_child_samples=cfg["mcs"], scale_pos_weight=SPW, random_state=seed, n_jobs=4, verbose=-1
        )

    lgb_s0_list = [make_lgb().fit(P_tr["imps"][j][S_COLS], P_tr["y"]) for j in range(5)]
    lgb_s1_list = [make_lgb().fit(P_tr["imps"][j][S_COLS + LEVEL_COLS], P_tr["y"]) for j in range(5)]
    lgb_b5_list = [
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=30, scale_pos_weight=SPW, random_state=42+j, n_jobs=4, verbose=-1)
        .fit(P_tr["imps"][j][BEDSIDE5], P_tr["y"]) for j in range(5)
    ]
    cb_list = [
        CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, auto_class_weights="Balanced", random_seed=42+j, verbose=0)
        .fit(P_tr["imps"][j][SEL], P_tr["y"]) for j in range(5)
    ]
    xgb_list = [
        xgb.XGBClassifier(n_estimators=450, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SPW, random_state=42+j, eval_metric="logloss", n_jobs=4)
        .fit(P_tr["imps"][j][SEL], P_tr["y"]) for j in range(5)
    ]
    cb_s0_list = [
        CatBoostClassifier(iterations=450, learning_rate=0.03, depth=4, auto_class_weights="Balanced", random_seed=42+j, verbose=0)
        .fit(P_tr["imps"][j][S_COLS], P_tr["y"]) for j in range(5)
    ]

    tab_scaler = StandardScaler()
    Xtr_tab    = tab_scaler.fit_transform(P_tr["tab"].values).astype(np.float32)
    Xtu_s_tab  = tab_scaler.transform(P_tu_sel["tab"].values).astype(np.float32)

    def train_neural(seed=42, epochs=45, use_seq=True):
        torch.manual_seed(seed)
        rng_d = np.random.default_rng(seed)
        model = TrueCTDecayAttentionNet(d_tab=Xtr_tab.shape[1], d_seq=5, d_hid=32, use_seq=use_seq)
        pos_w = torch.tensor([(len(P_tr["y"]) - P_tr["y"].sum()) / max(P_tr["y"].sum(), 1.0)], dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-2)
        ds = TensorDataset(
            torch.tensor(P_tr["sv"]), torch.tensor(P_tr["sdt"]),
            torch.tensor(P_tr["smk"]), torch.tensor(Xtr_tab),
            torch.tensor(P_tr["y"], dtype=torch.float32)
        )
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        sv_tu = torch.tensor(P_tu_sel["sv"]); sdt_tu = torch.tensor(P_tu_sel["sdt"])
        smk_tu = torch.tensor(P_tu_sel["smk"]); x_tu = torch.tensor(Xtu_s_tab)
        best_auc, best_state = -1.0, None
        for _ in range(epochs):
            model.train()
            for b_sv, b_sdt, b_smk, b_tab, b_y in dl:
                if use_seq and rng_d.random() < 0.20:
                    drop = torch.tensor((rng_d.random(len(b_y)) < 0.25), dtype=torch.float32)
                    b_smk = b_smk * (1.0 - drop.unsqueeze(1))
                opt.zero_grad()
                logits = model(b_sv, b_sdt, b_smk, b_tab)
                loss = crit(logits, b_y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(sv_tu, sdt_tu, smk_tu, x_tu)).numpy()
                auc = roc_auc_score(P_tu_sel["y"], pv)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        return model

    ct_models  = [train_neural(seed=42 + s, epochs=45, use_seq=True) for s in range(3)]
    mlp_models = [train_neural(seed=42 + s, epochs=45, use_seq=False) for s in range(3)]

    def predict_net_ens(pack, models_list):
        x_t = torch.tensor(tab_scaler.transform(pack["tab"].values).astype(np.float32))
        sv_t = torch.tensor(pack["sv"]); sdt_t = torch.tensor(pack["sdt"]); smk_t = torch.tensor(pack["smk"])
        with torch.no_grad():
            preds = [torch.sigmoid(m(sv_t, sdt_t, smk_t, x_t)).numpy() for m in models_list]
        return np.mean(preds, axis=0)

    ct_df = pd.read_parquet(f"{DATA}/19a_ct_nlp_3cohorts_v4.parquet")
    ct_m4 = ct_df[ct_df["cohort"] == "MIMIC-IV"].copy(); ct_m4["stay_id"] = ct_m4["id"].astype(int)
    ct_m3 = ct_df[ct_df["cohort"] == "MIMIC-III"].copy(); ct_m3["icustay_id"] = ct_m3["id"].astype(int)

    tr_ct_part     = tr[["stay_id"]].merge(ct_m4[["stay_id"] + CT_COLS], on="stay_id", how="left")
    tu_sel_ct_part = tu_sel[["stay_id"]].merge(ct_m4[["stay_id"] + CT_COLS], on="stay_id", how="left")
    tu_cal_ct_part = tu_cal[["stay_id"]].merge(ct_m4[["stay_id"] + CT_COLS], on="stay_id", how="left")
    iv_ct_part     = iv[["stay_id"]].merge(ct_m4[["stay_id"] + CT_COLS], on="stay_id", how="left")
    m3_ct_part     = d_m3[["icustay_id"]].merge(ct_m3[["icustay_id"] + CT_COLS], on="icustay_id", how="left")

    tr_ct_mask = (tr_ct_part["ct_any_24h"].values == 1)
    lgb_s0_ct_list, lgb_full_ct_list, cb_s0_ct_list = [], [], []
    for j in range(5):
        Xtr_j_ct = pd.concat([P_tr["imps"][j].reset_index(drop=True), tr_ct_part[CT_LESION_COLS].reset_index(drop=True)], axis=1)
        lgb_s0_ct_list.append(make_lgb().fit(Xtr_j_ct.loc[tr_ct_mask, S_COLS + CT_LESION_COLS], P_tr["y"][tr_ct_mask]))
        lgb_full_ct_list.append(make_lgb().fit(Xtr_j_ct.loc[tr_ct_mask, SEL + CT_LESION_COLS], P_tr["y"][tr_ct_mask]))
        cb_s0_ct_list.append(CatBoostClassifier(iterations=450, learning_rate=0.03, depth=4, auto_class_weights="Balanced", random_seed=42+j, verbose=0).fit(Xtr_j_ct.loc[tr_ct_mask, S_COLS + CT_LESION_COLS], P_tr["y"][tr_ct_mask]))

    def predict_base_pack(pack, ct_part_df=None):
        p_s0     = np.mean([lgb_s0_list[j].predict_proba(pack["imps"][j][S_COLS])[:, 1] for j in range(5)], axis=0)
        p_s0_cb  = np.mean([cb_s0_list[j].predict_proba(pack["imps"][j][S_COLS])[:, 1] for j in range(5)], axis=0)
        p_s1     = np.mean([lgb_s1_list[j].predict_proba(pack["imps"][j][S_COLS + LEVEL_COLS])[:, 1] for j in range(5)], axis=0)
        p_lgb_25 = np.mean([frozen_lgb_boosters[j].predict(pack["imps"][j][SEL]) for j in range(5)], axis=0)
        p_cb_25  = np.mean([cb_list[j].predict_proba(pack["imps"][j][SEL])[:, 1] for j in range(5)], axis=0)
        p_xgb_25 = np.mean([xgb_list[j].predict_proba(pack["imps"][j][SEL])[:, 1] for j in range(5)], axis=0)
        p_b5     = np.mean([lgb_b5_list[j].predict_proba(pack["imps"][j][BEDSIDE5])[:, 1] for j in range(5)], axis=0)
        p_ct_net = predict_net_ens(pack, ct_models)
        p_mlp_25 = predict_net_ens(pack, mlp_models)

        has_motor = pack["tab"]["mask_motor_charted"].values > 0.5
        has_ct_24h = (ct_part_df["ct_any_24h"].values == 1) if ct_part_df is not None else np.zeros(len(pack["y"]), dtype=bool)
        p_s0_pure = 0.5 * p_s0 + 0.5 * p_s0_cb
        if ct_part_df is not None:
            p_s0_ct_lgb = np.mean([
                lgb_s0_ct_list[j].predict_proba(pd.concat([pack["imps"][j][S_COLS].reset_index(drop=True), ct_part_df[CT_LESION_COLS].reset_index(drop=True)], axis=1))[:, 1]
                for j in range(5)
            ], axis=0)
            p_s0_ct = np.mean([
                0.5 * lgb_s0_ct_list[j].predict_proba(pd.concat([pack["imps"][j][S_COLS].reset_index(drop=True), ct_part_df[CT_LESION_COLS].reset_index(drop=True)], axis=1))[:, 1] +
                0.5 * cb_s0_ct_list[j].predict_proba(pd.concat([pack["imps"][j][S_COLS].reset_index(drop=True), ct_part_df[CT_LESION_COLS].reset_index(drop=True)], axis=1))[:, 1]
                for j in range(5)
            ], axis=0)
        else:
            p_s0_ct_lgb, p_s0_ct = None, None

        return {
            "Static_S0": p_s0,
            "Reanchored_S1": np.where(has_motor, p_s1, p_s0),
            "LightGBM": p_lgb_25,
            "CatBoost": p_cb_25,
            "XGBoost": p_xgb_25,
            "Tabular_MLP": p_mlp_25,
            "CT_DecayNet": p_ct_net,
            "Bedside_5": p_b5,
            "Static_S0_CT_LGB": p_s0_ct_lgb,
            "Static_Pure_Ens": p_s0_pure,
            "Static_CT_Ens": p_s0_ct,
            "has_motor": has_motor,
            "has_ct_24h": has_ct_24h
        }

    # Fit Non-Negative Super-Learner & Individual-Branch Calibrators ONLY on Tune-Select (n=278)
    tu_s_base = predict_base_pack(P_tu_sel, tu_sel_ct_part)
    Z_tu_s = np.column_stack([
        logit(np.clip(tu_s_base["LightGBM"], 1e-5, 1 - 1e-5)),
        logit(np.clip(tu_s_base["CatBoost"], 1e-5, 1 - 1e-5)),
        logit(np.clip(tu_s_base["CT_DecayNet"], 1e-5, 1 - 1e-5))
    ])
    w_stack, b_stack = fit_nonneg_stacker(Z_tu_s, P_tu_sel["y"], l2_reg=1.0)
    tu_s_ct_m = tu_s_base["has_ct_24h"] if tu_s_base["has_ct_24h"].sum() >= 30 else np.ones(len(P_tu_sel["y"]), dtype=bool)
    cal_fb_s0     = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["Static_Pure_Ens"], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"])
    cal_fb_s0_ct  = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["Static_CT_Ens"][tu_s_ct_m], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"][tu_s_ct_m])
    cal_lgb25     = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["LightGBM"], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"])
    cal_lgb_s0    = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["Static_S0"], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"])
    cal_lgb_s0_ct = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["Static_S0_CT_LGB"][tu_s_ct_m], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"][tu_s_ct_m])
    cal_b5_5      = LogisticRegression(C=1.0).fit(logit(np.clip(tu_s_base["Bedside_5"], 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"])

    def predict_with_super(pack, ct_part_df=None):
        b = predict_base_pack(pack, ct_part_df)
        Z = np.column_stack([
            logit(np.clip(b["LightGBM"], 1e-5, 1 - 1e-5)),
            logit(np.clip(b["CatBoost"], 1e-5, 1 - 1e-5)),
            logit(np.clip(b["CT_DecayNet"], 1e-5, 1 - 1e-5))
        ])
        p_super_motor = expit(np.clip(Z @ w_stack + b_stack, -30.0, 30.0))
        p_fb_s0 = cal_fb_s0.predict_proba(logit(np.clip(b["Static_Pure_Ens"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        if ct_part_df is not None:
            p_fb_ct = cal_fb_s0_ct.predict_proba(logit(np.clip(b["Static_CT_Ens"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            p_fb = np.where(b["has_ct_24h"], p_fb_ct, p_fb_s0)
        else:
            p_fb = p_fb_s0
        b["Hybrid_Ensemble"] = np.where(b["has_motor"], p_super_motor, p_fb)

        # Plan 2A Primary Model (Modality_Adaptive_GBDT): true individual-level dynamic-feature-free static routing!
        # Branch 1 (has 24h motor): frozen 25-feat LightGBM; Branch 2 (no motor, has 24h CT): Static+CT LightGBM; Branch 3 (neither): pure Static S0 LightGBM!
        p_lgb_act_cal = cal_lgb25.predict_proba(logit(np.clip(b["LightGBM"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        p_lgb_s0_cal  = cal_lgb_s0.predict_proba(logit(np.clip(b["Static_S0"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        if ct_part_df is not None:
            p_lgb_s0_ct_cal = cal_lgb_s0_ct.predict_proba(logit(np.clip(b["Static_S0_CT_LGB"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
            p_lgb_routed_fb = np.where(b["has_ct_24h"], p_lgb_s0_ct_cal, p_lgb_s0_cal)
        else:
            p_lgb_routed_fb = p_lgb_s0_cal
        b["Modality_Adaptive_GBDT"] = np.where(b["has_motor"], p_lgb_act_cal, p_lgb_routed_fb)
        b["miss_motor_no_ct"] = (~b["has_motor"]) & (~b["has_ct_24h"])
        b["Bedside_5_Cal"] = cal_b5_5.predict_proba(logit(np.clip(b["Bedside_5"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        return b

    iv_check = predict_with_super(P_iv, iv_ct_part)
    auc_iv_lgb = roc_auc_score(P_iv["y"], iv_check["LightGBM"])
    lg(f"  [ANCHOR CHECK] MIMIC-IV IntVal Call-Order-Independent (seed 100+j) Frozen 08d_v3 LightGBM AUROC = {auc_iv_lgb:.4f} | Non-neg weights = {np.round(w_stack, 4)}")
    assert abs(auc_iv_lgb - 0.9111) < 0.0015, f"Anchor mismatch: got {auc_iv_lgb}"
    assert np.all(w_stack >= -1e-9), f"Stacking weights not non-negative: {w_stack}"

    # 3. Calibrate 90% Mondrian Split-Conformal STRICTLY on untouched Tune-Calib (n=278, exact order statistics on Primary Modality_Adaptive_GBDT!)
    lg("3. Calibrating 90% Mondrian Split-Conformal on UNTOUCHED Tune-Calib (n=278, exact order statistics)...")
    tu_c_preds = predict_with_super(P_tu_cal, tu_cal_ct_part)
    y_tcal = P_tu_cal["y"]
    tau_low, tau_high, k1_c, n1_c, k0_c, n0_c = exact_mondrian_thresholds(y_tcal, tu_c_preds["Modality_Adaptive_GBDT"], alpha=0.10)
    tau_low_b5, tau_high_b5, _, _, _, _       = exact_mondrian_thresholds(y_tcal, tu_c_preds["Bedside_5_Cal"], alpha=0.10)
    tau_low_s0, tau_high_s0, _, _, _, _       = exact_mondrian_thresholds(y_tcal, cal_lgb_s0.predict_proba(logit(np.clip(tu_c_preds["Static_S0"], 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1], alpha=0.10)

    fpr_t, tpr_t, thr_t = roc_curve(y_tcal, tu_c_preds["Modality_Adaptive_GBDT"])
    THR_CAL = round(float(thr_t[int(np.argmax(tpr_t - fpr_t))]), 6)
    lg(f"  Untouched Tune-Calib (n1={n1_c}, k1={k1_c}; n0={n0_c}, k0={k0_c}) 90% Mondrian: tau_low={tau_low:.4f}, tau_high={tau_high:.4f} | Bedside-5 tau=({tau_low_b5:.4f}, {tau_high_b5:.4f})")

    eval_cohorts = [
        ("MIMIC_IV_IntVal",        P_iv,     iv_ct_part),
        ("MIMIC_III_CareVue",      P_m3,     m3_ct_part),
        ("eICU_CRD",               P_ei,     None),
        ("HiRID_Trauma",           P_hi_tr,  None),
        ("HiRID_Neurotrauma_ICP",  P_hi_nt,  None),
        ("HiRID_ABI_All",          P_hi_abi, None),
        ("AmsterdamUMCdb_Trauma",  P_au_tr,  None),
        ("AmsterdamUMCdb_ABI",     P_au_abi, None),
        ("JinhuaNSICU_TBI_S06",    P_jh_tbi, d_jh_tbi[CT_COLS]),
        ("JinhuaNSICU_ABI_All",    P_jh_abi, d_jh_abi[CT_COLS]),
    ]

    tournament_res, conformal_res, recal_split_res, deploy_metrics = {}, {}, {}, {}
    paired_arch_res = {}
    boot_arrays_to_save, oof_cal_by_cohort = {}, {}

    for cname, pack, ct_part in eval_cohorts:
        preds = predict_with_super(pack, ct_part)
        y_d = pack["y"]
        c_out = {"n": int(len(y_d)), "events": int(y_d.sum()), "event_rate": round(float(y_d.mean()), 4), "models": {}}

        for mname in ["Static_S0", "Reanchored_S1", "LightGBM", "CatBoost", "XGBoost", "Tabular_MLP", "CT_DecayNet", "Modality_Adaptive_GBDT", "Hybrid_Ensemble", "Bedside_5"]:
            pr = preds[mname]
            auc = float(roc_auc_score(y_d, pr))
            lo, hi, dist_vec = boot_ci_with_dist(y_d, pr, B=2000, seed=101)
            ap = float(average_precision_score(y_d, pr))
            br = float(brier_score_loss(y_d, pr))
            entry_m = {"AUROC": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)], "AUPRC": round(ap, 4), "Brier": round(br, 4)}
            if cname == "eICU_CRD" and mname in ("Hybrid_Ensemble", "Modality_Adaptive_GBDT", "LightGBM"):
                entry_m["cluster_boot_130hosp_ci95"] = cluster_boot_ci(y_d, pr, d_ei["hospitalid"].values, B=2000, seed=101)
            c_out["models"][mname] = entry_m
            if mname in ("Hybrid_Ensemble", "Modality_Adaptive_GBDT", "CT_DecayNet", "LightGBM"):
                boot_arrays_to_save[f"{cname}__{mname}"] = dist_vec.astype(np.float32)

        c_out["delta_reanchor_vs_static"] = boot_delta_auc(y_d, preds["Reanchored_S1"], preds["Static_S0"], B=2000)
        c_out["delta_full_vs_static"]     = boot_delta_auc(y_d, preds["Modality_Adaptive_GBDT"], preds["Static_S0"], B=2000)
        c_out["delta_adaptive_gbdt_vs_naive_mice"] = boot_delta_auc(y_d, preds["Modality_Adaptive_GBDT"], preds["LightGBM"], B=2000)
        tournament_res[cname] = c_out

        # R5 Fix: Paired 2,000-bootstrap Delta-AUROC tests across architectures & sequence-branch ablation
        paired_arch_res[cname] = {
            "ct_decaynet_vs_lgbm":        boot_delta_auc(y_d, preds["CT_DecayNet"], preds["LightGBM"], B=2000, seed=201),
            "ct_decaynet_vs_tabular_mlp": boot_delta_auc(y_d, preds["CT_DecayNet"], preds["Tabular_MLP"], B=2000, seed=202),
            "hybrid_vs_adaptive_gbdt":    boot_delta_auc(y_d, preds["Hybrid_Ensemble"], preds["Modality_Adaptive_GBDT"], B=2000, seed=203),
        }

        p_prim = preds["Modality_Adaptive_GBDT"]
        frz_slope, frz_citl = calc_slope_citl(y_d, p_prim)
        lp_prim = logit(np.clip(p_prim, 1e-5, 1 - 1e-5)).reshape(-1, 1)

        # Out-of-fold 20x 50/50 split-sample Platt AND exact within-site Split-Conformal on Primary Modality_Adaptive_GBDT (R2, R4 & R5 Fix!)
        sss = StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=42)
        oof_recal_sum = np.zeros(len(y_d), dtype=np.float64)
        oof_recal_cnt = np.zeros(len(y_d), dtype=np.float64)
        oof_c1_cnt    = np.zeros(len(y_d), dtype=np.float64)
        oof_c0_cnt    = np.zeros(len(y_d), dtype=np.float64)
        oof_hi_cnt    = np.zeros(len(y_d), dtype=np.float64)
        slopes_20, citls_20, briers_20 = [], [], []
        loc_cov_all, loc_cov_pos, loc_cov_neg, loc_abstain, loc_hi, loc_lo, loc_def_aucs = [], [], [], [], [], [], []

        miss_motor_no_ct = preds["miss_motor_no_ct"]

        for idx_a, idx_b in sss.split(lp_prim, y_d):
            for fit_i, ev_i in [(idx_a, idx_b), (idx_b, idx_a)]:
                cal_m = LogisticRegression(C=1e6, max_iter=1000).fit(lp_prim[fit_i], y_d[fit_i])
                pr_ev = cal_m.predict_proba(lp_prim[ev_i])[:, 1]
                oof_recal_sum[ev_i] += pr_ev
                oof_recal_cnt[ev_i] += 1.0
                sl_v, ci_v = calc_slope_citl(y_d[ev_i], pr_ev)
                slopes_20.append(sl_v); citls_20.append(ci_v)
                briers_20.append(float(brier_score_loss(y_d[ev_i], pr_ev)))

                # Within-site Split-Conformal on raw p_prim with individual-level abstention on miss_motor_no_ct
                t_lo_loc, t_hi_loc, _, _, _, _ = exact_mondrian_thresholds(y_d[fit_i], p_prim[fit_i], alpha=0.10)
                pe_raw = p_prim[ev_i]; ye_raw = y_d[ev_i]; mm_ev = miss_motor_no_ct[ev_i]
                ic1_l = (pe_raw >= t_lo_loc) | mm_ev
                ic0_l = (pe_raw <= t_hi_loc) | mm_ev
                cov_l = ((ye_raw == 1) & ic1_l) | ((ye_raw == 0) & ic0_l)
                abs_l = (ic0_l & ic1_l) | mm_ev
                def_l = ~abs_l
                hi_sing_l = ic1_l & (~ic0_l) & def_l
                oof_c1_cnt[ev_i] += ic1_l.astype(np.float64)
                oof_c0_cnt[ev_i] += ic0_l.astype(np.float64)
                oof_hi_cnt[ev_i] += hi_sing_l.astype(np.float64)
                loc_cov_all.append(float(np.mean(cov_l)))
                loc_cov_pos.append(float(np.mean(ic1_l[ye_raw == 1])))
                loc_cov_neg.append(float(np.mean(ic0_l[ye_raw == 0])))
                loc_abstain.append(float(np.mean(abs_l)))
                loc_hi.append(float(np.mean(ic1_l & (~ic0_l) & def_l)))
                loc_lo.append(float(np.mean(ic0_l & (~ic1_l) & def_l)))
                if def_l.sum() >= 20 and ye_raw[def_l].sum() >= 2 and (def_l.sum() - ye_raw[def_l].sum()) >= 2:
                    loc_def_aucs.append(float(roc_auc_score(ye_raw[def_l], pe_raw[def_l])))

        p_oof_cal = oof_recal_sum / np.maximum(oof_recal_cnt, 1.0)
        oof_cal_by_cohort[cname] = p_oof_cal
        recal_split_res[cname] = {
            "frozen_brier": round(float(brier_score_loss(y_d, p_prim)), 4),
            "frozen_slope": round(frz_slope, 4),
            "frozen_citl": round(frz_citl, 4),
            "recal_brier_mean": round(float(np.mean(briers_20)), 4),
            "recal_slope_mean": round(float(np.mean(slopes_20)), 4),
            "recal_citl_mean": round(float(np.mean(citls_20)), 4)
        }

        # Zero-touch Tune-Calib frozen thresholds (with individual-level miss_motor_no_ct -> {0,1} abstention)
        in_c1_frz = (p_prim >= tau_low) | miss_motor_no_ct
        in_c0_frz = (p_prim <= tau_high) | miss_motor_no_ct
        cov_frozen     = float(np.mean(((y_d == 1) & in_c1_frz) | ((y_d == 0) & in_c0_frz)))
        cov_frozen_pos = float(np.mean(in_c1_frz[y_d == 1]))
        cov_frozen_neg = float(np.mean(in_c0_frz[y_d == 0]))

        p_conf_in = p_prim if cname == "MIMIC_IV_IntVal" else p_oof_cal
        in_c1_platt = (p_conf_in >= tau_low) | miss_motor_no_ct
        in_c0_platt = (p_conf_in <= tau_high) | miss_motor_no_ct
        covered_platt = ((y_d == 1) & in_c1_platt) | ((y_d == 0) & in_c0_platt)

        # R5 Fix: Point estimate, 95% CI, and B=2,000 bootstrap violin in Fig 4g ALL come from the EXACT SAME consensus OOF / untouched Tune-Calib set!
        if cname == "MIMIC_IV_IntVal":
            in_c1_eval, in_c0_eval = in_c1_frz, in_c0_frz
        else:
            in_c1_eval = ((oof_c1_cnt / np.maximum(oof_recal_cnt, 1.0)) >= 0.5) | miss_motor_no_ct
            in_c0_eval = ((oof_c0_cnt / np.maximum(oof_recal_cnt, 1.0)) >= 0.5) | miss_motor_no_ct
        abstain_eval = (in_c0_eval & in_c1_eval) | miss_motor_no_ct
        definitive_eval = ~abstain_eval
        covered_eval = ((y_d == 1) & in_c1_eval) | ((y_d == 0) & in_c0_eval)

        auc_all = float(roc_auc_score(y_d, p_prim))
        assert definitive_eval.sum() >= 30 and y_d[definitive_eval].sum() >= 3 and (definitive_eval.sum() - y_d[definitive_eval].sum()) >= 3, f"Too few definitive stays in {cname}"
        auc_def = float(roc_auc_score(y_d[definitive_eval], p_prim[definitive_eval]))
        lo_d, hi_d, dist_def = boot_ci_with_dist(y_d[definitive_eval], p_prim[definitive_eval], B=2000, seed=108)
        assert lo_d - 1e-6 <= auc_def <= hi_d + 1e-6, f"Definitive CI mismatch in {cname}: {auc_def} not in [{lo_d}, {hi_d}]"
        boot_arrays_to_save[f"{cname}__Conformal_Definitive"] = dist_def.astype(np.float32)

        primary_cov_all = round(float(np.mean(covered_eval)) if cname == "MIMIC_IV_IntVal" else float(np.mean(loc_cov_all)), 4)
        primary_cov_pos = round(float(np.mean(in_c1_eval[y_d == 1])) if cname == "MIMIC_IV_IntVal" else float(np.mean(loc_cov_pos)), 4)
        primary_cov_neg = round(float(np.mean(in_c0_eval[y_d == 0])) if cname == "MIMIC_IV_IntVal" else float(np.mean(loc_cov_neg)), 4)
        primary_hi_frac = round(float(np.mean(in_c1_eval & (~in_c0_eval) & definitive_eval)), 4)
        primary_lo_frac = round(float(np.mean(in_c0_eval & (~in_c1_eval) & definitive_eval)), 4)
        primary_ab_frac = round(float(np.mean(abstain_eval)), 4)
        primary_def_auc = round(auc_def, 4)
        primary_def_ci  = [round(float(lo_d), 4), round(float(hi_d), 4)]

        conformal_res[cname] = {
            "empirical_coverage_overall": primary_cov_all,
            "empirical_coverage_deaths": primary_cov_pos,
            "empirical_coverage_survivors": primary_cov_neg,
            "empirical_coverage_consensus_oof": round(float(np.mean(covered_eval)), 4),
            "empirical_coverage_frozen_unrecal": round(cov_frozen, 4),
            "empirical_coverage_frozen_deaths": round(cov_frozen_pos, 4),
            "empirical_coverage_frozen_survivors": round(cov_frozen_neg, 4),
            "empirical_coverage_platt_transfer": round(float(np.mean(covered_platt)), 4),
            "singleton_high_risk_frac": primary_hi_frac,
            "singleton_low_risk_frac": primary_lo_frac,
            "abstained_ambiguous_or_uncharted_frac": primary_ab_frac,
            "uncharted_no_motor_no_ct_frac": round(float(np.mean(miss_motor_no_ct)), 4),
            "definitive_retained_frac": round(1.0 - primary_ab_frac, 4),
            "definitive_n": int(definitive_eval.sum()),
            "auroc_overall": round(auc_all, 4),
            "auroc_definitive": primary_def_auc,
            "auroc_definitive_ci95": primary_def_ci,
            "auroc_definitive_40half_mean": round(float(np.mean(loc_def_aucs)), 4) if loc_def_aucs else primary_def_auc,
            "delta_auroc_conformal_gate": round(primary_def_auc - auc_all, 4)
        }

        # Y22 Fix: Escalation alert triggered strictly among NON-ABSTAINED definitive high-risk singleton {1}!
        eval_elig = ~miss_motor_no_ct
        alert = (in_c1_eval & (~in_c0_eval) & definitive_eval & eval_elig).astype(int)
        tp = int(((alert == 1) & (y_d == 1)).sum())
        fp = int(((alert == 1) & (y_d == 0)).sum())
        fn = int(((alert == 0) & (y_d == 1) & eval_elig).sum())
        tn = int(((alert == 0) & (y_d == 0) & eval_elig).sum())
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        nne = (tp + fp) / max(tp, 1)
        deploy_metrics[cname] = {
            "n_evaluated_non_uncharted": int(eval_elig.sum()),
            "sens": round(sens, 4), "spec": round(spec, 4),
            "ppv": round(ppv, 4), "npv": round(npv, 4), "NNE": round(nne, 2)
        }
        lg(f"  {cname:24s} (n={len(y_d):4d}, ev={y_d.sum():3d}): S0={c_out['models']['Static_S0']['AUROC']:.3f} | LGBM={c_out['models']['LightGBM']['AUROC']:.3f} | PrimaryAdaptGBDT={auc_all:.3f} | CT-Net={c_out['models']['CT_DecayNet']['AUROC']:.3f} | Hybrid={c_out['models']['Hybrid_Ensemble']['AUROC']:.3f} | ConfCov={primary_cov_all*100:.1f}% (Retained={(1-primary_ab_frac)*100:.1f}%, DefAUC={primary_def_auc:.3f} [{lo_d:.3f}-{hi_d:.3f}], NNE={nne:.2f})")

    lg("4. Evaluating Bilingual Head-CT NLP, Individual-Level CT Routing, Unimaged Subgroup, & Proxy Remapping Sensitivities...")
    ct_fusion_res = {}
    for label, pack, ct_part in [
        ("MIMIC_IV_IntVal",     P_iv,     iv_ct_part),
        ("MIMIC_III_CareVue",   P_m3,     m3_ct_part),
        ("JinhuaNSICU_TBI_S06", P_jh_tbi, d_jh_tbi[CT_COLS]),
        ("JinhuaNSICU_ABI_All", P_jh_abi, d_jh_abi[CT_COLS]),
    ]:
        y_vec = pack["y"]
        pr_pack = predict_with_super(pack, ct_part)
        p_s0_lgb = pr_pack["Static_S0"]
        p_s0_only = pr_pack["Static_Pure_Ens"]
        p_s0_ct_lgb = pr_pack["Static_S0_CT_LGB"]
        p_s0_plus_ct = np.where(pr_pack["has_ct_24h"], pr_pack["Static_CT_Ens"], p_s0_only)
        p_routed_gbdt = pr_pack["Modality_Adaptive_GBDT"]

        has_m = pack["tab"]["mask_motor_charted"].mean() > 0.5
        p_full_only = pr_pack["LightGBM"]
        if has_m:
            p_full_ct_raw = np.mean([
                lgb_full_ct_list[j].predict_proba(pd.concat([pack["imps"][j][SEL].reset_index(drop=True), ct_part[CT_LESION_COLS].reset_index(drop=True)], axis=1))[:, 1]
                for j in range(5)
            ], axis=0)
            p_full_plus_ct = np.where(pr_pack["has_ct_24h"], p_full_ct_raw, p_full_only)
        else:
            p_full_plus_ct = p_routed_gbdt

        # Evaluate strictly within CT-observed stays (ct_any_24h == 1) AND unimaged stays (ct_any_24h == 0) per R2 & Y4!
        ct_obs_mask = (ct_part["ct_any_24h"].values == 1)
        ct_unobs_mask = ~ct_obs_mask
        ct_obs_sub, ct_unobs_sub = None, None
        if ct_obs_mask.sum() >= 30 and y_vec[ct_obs_mask].sum() >= 5:
            ct_obs_sub = {
                "n_ct_observed": int(ct_obs_mask.sum()),
                "events_ct_observed": int(y_vec[ct_obs_mask].sum()),
                "mort_pct_ct_observed": round(100.0 * float(y_vec[ct_obs_mask].mean()), 1),
                "auroc_static_lgbm_in_ct_obs": round(float(roc_auc_score(y_vec[ct_obs_mask], p_s0_lgb[ct_obs_mask])), 4),
                "auroc_static_ct_lgbm_in_ct_obs": round(float(roc_auc_score(y_vec[ct_obs_mask], p_s0_ct_lgb[ct_obs_mask])), 4),
                "delta_ct_lgbm_over_static_in_ct_obs": boot_delta_auc(y_vec[ct_obs_mask], p_s0_ct_lgb[ct_obs_mask], p_s0_lgb[ct_obs_mask], B=2000),
                "auroc_static_only_in_ct_obs": round(float(roc_auc_score(y_vec[ct_obs_mask], p_s0_only[ct_obs_mask])), 4),
                "auroc_static_plus_ct_in_ct_obs": round(float(roc_auc_score(y_vec[ct_obs_mask], pr_pack["Static_CT_Ens"][ct_obs_mask])), 4),
                "delta_ct_over_static_in_ct_obs": boot_delta_auc(y_vec[ct_obs_mask], pr_pack["Static_CT_Ens"][ct_obs_mask], p_s0_only[ct_obs_mask], B=2000),
            }
        if ct_unobs_mask.sum() >= 20 and y_vec[ct_unobs_mask].sum() >= 3:
            ct_unobs_sub = {
                "n_ct_unobserved": int(ct_unobs_mask.sum()),
                "events_ct_unobserved": int(y_vec[ct_unobs_mask].sum()),
                "mort_pct_ct_unobserved": round(100.0 * float(y_vec[ct_unobs_mask].mean()), 1),
                "auroc_pure_static_lgbm": round(float(roc_auc_score(y_vec[ct_unobs_mask], p_s0_lgb[ct_unobs_mask])), 4),
                "auroc_routed_primary": round(float(roc_auc_score(y_vec[ct_unobs_mask], p_routed_gbdt[ct_unobs_mask])), 4),
            }

        ct_fusion_res[label] = {
            "n": int(len(y_vec)), "events": int(y_vec.sum()),
            "flowsheet_motor_charted": bool(has_m),
            "auroc_static_lgbm_only": round(float(roc_auc_score(y_vec, p_s0_lgb)), 4),
            "auroc_routed_adaptive_gbdt": round(float(roc_auc_score(y_vec, p_routed_gbdt)), 4),
            "delta_routed_ct_gbdt_over_static_lgbm": boot_delta_auc(y_vec, p_routed_gbdt, p_s0_lgb, B=2000),
            "auroc_static_only": round(float(roc_auc_score(y_vec, p_s0_only)), 4),
            "auroc_static_plus_ct": round(float(roc_auc_score(y_vec, p_s0_plus_ct)), 4),
            "delta_ct_over_static": boot_delta_auc(y_vec, p_s0_plus_ct, p_s0_only, B=2000),
            "auroc_24h_motor_only": round(float(roc_auc_score(y_vec, p_full_only)), 4),
            "auroc_multimodal_best": round(float(roc_auc_score(y_vec, p_full_plus_ct)), 4),
            "delta_static_routing_over_naive_24h_mice": boot_delta_auc(y_vec, p_s0_lgb, p_full_only, B=2000),
            "delta_primary_routing_over_naive_24h_mice": boot_delta_auc(y_vec, p_routed_gbdt, p_full_only, B=2000),
            "delta_ct_over_24h_motor": boot_delta_auc(y_vec, p_full_plus_ct, p_full_only, B=2000),
            "ct_observed_only_sensitivity": ct_obs_sub,
            "ct_unobserved_subgroup": ct_unobs_sub,
        }

    # R1, R7 & R10 Sensitivities: Jinhua strict S06, Jinhua non-imputed proxy, Jinhua 'impaired_unresponsive' remapping (->10 & ->NaN), Amsterdam explicit trauma dx, eICU unrecorded routing
    jh_pr_all = predict_with_super(P_jh_tbi, d_jh_tbi[CT_COLS])
    au_pr_all = predict_with_super(P_au_tr, None)
    ei_pr_all = predict_with_super(P_ei, None)

    m_jh_s06_strict = d_jh_tbi["is_strict_s06_code"].values == True
    m_jh_gcs_obs    = d_jh_tbi["gcs_total_first"].notna().values
    m_au_tr_dx      = d_au_tr["is_strict_trauma_dx"].values == True
    m_ei_unrec      = ~ei_pr_all["has_motor"]
    m_ei_chart      = ei_pr_all["has_motor"]

    # R7 Fix: Remap Jinhua 'impaired_unresponsive' (n=614) to Moderate (10, m=4, e=3, v=3) OR to NaN (MICE) OR strip synthetic subscores to NaN
    d_jh_remap10 = d_jh_tbi.copy()
    m_imp_unr = (d_jh_remap10["gcs_proxy_source"].values == "impaired_unresponsive")
    for col_k, v_k in [("gcs_total_first", 10.0), ("gcs_total_min", 10.0), ("gcs_motor_first", 4.0), ("gcs_motor_min", 4.0), ("gcs_eye_first", 3.0), ("gcs_eye_min", 3.0), ("gcs_verbal_first", 3.0), ("gcs_verbal_min", 3.0)]:
        d_jh_remap10.loc[m_imp_unr, col_k] = v_k
    P_jh_r10 = build_cohort_pack(d_jh_remap10, None, None)
    pr_jh_r10 = predict_with_super(P_jh_r10, d_jh_remap10[CT_COLS])

    d_jh_remap_nan = d_jh_tbi.copy()
    for col_k in ["gcs_total_first", "gcs_total_min", "gcs_motor_first", "gcs_motor_min", "gcs_eye_first", "gcs_eye_min", "gcs_verbal_first", "gcs_verbal_min"]:
        d_jh_remap_nan.loc[m_imp_unr, col_k] = np.nan
    P_jh_rnan = build_cohort_pack(d_jh_remap_nan, None, None)
    pr_jh_rnan = predict_with_super(P_jh_rnan, d_jh_remap_nan[CT_COLS])

    d_jh_nosub = d_jh_tbi.copy()
    for col_k in ["gcs_motor_first", "gcs_motor_min", "gcs_eye_first", "gcs_eye_min", "gcs_verbal_first", "gcs_verbal_min"]:
        d_jh_nosub[col_k] = np.nan
    P_jh_nosub = build_cohort_pack(d_jh_nosub, None, None)
    pr_jh_nosub = predict_with_super(P_jh_nosub, d_jh_nosub[CT_COLS])

    strict_sens = {
        "Jinhua_Strict_ICD10_S06_Only": {
            "n": int(m_jh_s06_strict.sum()), "events": int(P_jh_tbi["y"][m_jh_s06_strict].sum()),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_s06_strict], jh_pr_all["Modality_Adaptive_GBDT"][m_jh_s06_strict])), 4),
            "AUROC_Hybrid": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_s06_strict], jh_pr_all["Hybrid_Ensemble"][m_jh_s06_strict])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_s06_strict], jh_pr_all["Static_S0"][m_jh_s06_strict])), 4),
            "AUROC_Naive_MICE_24h": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_s06_strict], jh_pr_all["LightGBM"][m_jh_s06_strict])), 4),
        },
        "Jinhua_Mapped_GCS_Proxy_Only_Excl_NaN": {
            "n": int(m_jh_gcs_obs.sum()), "events": int(P_jh_tbi["y"][m_jh_gcs_obs].sum()),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_gcs_obs], jh_pr_all["Modality_Adaptive_GBDT"][m_jh_gcs_obs])), 4),
            "AUROC_Hybrid": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_gcs_obs], jh_pr_all["Hybrid_Ensemble"][m_jh_gcs_obs])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_jh_tbi["y"][m_jh_gcs_obs], jh_pr_all["Static_S0"][m_jh_gcs_obs])), 4),
        },
        "Jinhua_Proxy_Remap_ImpairedUnresponsive_to_Moderate10": {
            "n_remapped": int(m_imp_unr.sum()), "severe_3_8_n_after": int((d_jh_remap10["gcs_total_first"] <= 8).sum()),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_r10["Modality_Adaptive_GBDT"])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_r10["Static_S0"])), 4),
        },
        "Jinhua_Proxy_Remap_ImpairedUnresponsive_to_NaN_MICE": {
            "n_set_nan": int(m_imp_unr.sum()),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_rnan["Modality_Adaptive_GBDT"])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_rnan["Static_S0"])), 4),
        },
        "Jinhua_Proxy_Subscores_Set_to_NaN_TotalOnly": {
            "n": int(len(d_jh_nosub)),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_nosub["Modality_Adaptive_GBDT"])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_jh_tbi["y"], pr_jh_nosub["Static_S0"])), 4),
        },
        "Amsterdam_Explicit_Trauma_Dx_Only": {
            "n": int(m_au_tr_dx.sum()), "events": int(P_au_tr["y"][m_au_tr_dx].sum()),
            "AUROC_Primary_Adaptive_GBDT": round(float(roc_auc_score(P_au_tr["y"][m_au_tr_dx], au_pr_all["Modality_Adaptive_GBDT"][m_au_tr_dx])), 4),
            "AUROC_LightGBM": round(float(roc_auc_score(P_au_tr["y"][m_au_tr_dx], au_pr_all["LightGBM"][m_au_tr_dx])), 4),
            "AUROC_Hybrid": round(float(roc_auc_score(P_au_tr["y"][m_au_tr_dx], au_pr_all["Hybrid_Ensemble"][m_au_tr_dx])), 4),
            "AUROC_Static_S0": round(float(roc_auc_score(P_au_tr["y"][m_au_tr_dx], au_pr_all["Static_S0"][m_au_tr_dx])), 4),
        },
        "eICU_Unrecorded_DynamicFeatureFree_Static_Routing": {
            "n": int(m_ei_unrec.sum()), "events": int(P_ei["y"][m_ei_unrec].sum()),
            "AUROC_Naive_MICE_24h": round(float(roc_auc_score(P_ei["y"][m_ei_unrec], ei_pr_all["LightGBM"][m_ei_unrec])), 4),
            "AUROC_Routed_Static": round(float(roc_auc_score(P_ei["y"][m_ei_unrec], ei_pr_all["Modality_Adaptive_GBDT"][m_ei_unrec])), 4),
            "AUROC_Routed_Static_Hybrid": round(float(roc_auc_score(P_ei["y"][m_ei_unrec], ei_pr_all["Hybrid_Ensemble"][m_ei_unrec])), 4),
            "delta_routing_vs_mice": boot_delta_auc(P_ei["y"][m_ei_unrec], ei_pr_all["Modality_Adaptive_GBDT"][m_ei_unrec], ei_pr_all["LightGBM"][m_ei_unrec], B=2000),
        }
    }

    jh_sens = {}
    for s_lbl, raw_j, pack_j in [("Jinhua_S06_Sens_ComaAutoDisch", d_jh_tbi, P_jh_tbi), ("Jinhua_ABI_Sens_ComaAutoDisch", d_jh_abi, P_jh_abi)]:
        ys = raw_j["d28_sens_composite"].values.astype(int)
        pr_j_pack = predict_with_super(pack_j, raw_j[CT_COLS])
        pr_s0 = pr_j_pack["Static_S0"]
        pr_ct = pr_j_pack["Modality_Adaptive_GBDT"]
        jh_sens[s_lbl] = {
            "n": int(len(ys)), "events": int(ys.sum()),
            "auroc_static": round(float(roc_auc_score(ys, pr_s0)), 4),
            "auroc_with_ct": round(float(roc_auc_score(ys, pr_ct)), 4),
            "delta": boot_delta_auc(ys, pr_ct, pr_s0, B=1000)
        }

    lg("5. Exporting exact 483x25 TreeSHAP matrix, bootstrap distributions & 100% pure 5-booster Bedside-5 web calculator artifact...")
    shap_5 = np.mean([frozen_lgb_boosters[j].predict(P_iv["imps"][j][SEL], pred_contrib=True)[:, :-1] for j in range(5)], axis=0)
    df_shap_iv = pd.DataFrame(shap_5, columns=SEL)
    df_shap_iv["stay_id"] = iv["stay_id"].values
    df_shap_iv["d28"] = P_iv["y"]
    df_shap_iv.to_parquet(f"{DATA}/19c_v4_intval_shap_matrix.parquet", index=False)

    # R11 Fix: Export ALL 5 boosters x 300 trees = 1,500 LightGBM trees of the exact validated Bedside-5 ensemble
    # PLUS pure 5-booster Static-3 (age, gcs_total_first, gcs_motor_first) fallback when 24h mGCS is unrecorded!
    BEDSIDE3_STATIC = ["admission_age", "gcs_total_first", "gcs_motor_first"]
    lgb_b3_s0_list = [
        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=30, scale_pos_weight=SPW, random_state=42+j, n_jobs=4, verbose=-1)
        .fit(P_tr["imps"][j][BEDSIDE3_STATIC], P_tr["y"]) for j in range(5)
    ]
    p_tu_s_b3 = np.mean([lgb_b3_s0_list[j].predict_proba(P_tu_sel["imps"][j][BEDSIDE3_STATIC])[:, 1] for j in range(5)], axis=0)
    cal_b3_s0 = LogisticRegression(C=1.0).fit(logit(np.clip(p_tu_s_b3, 1e-5, 1 - 1e-5)).reshape(-1, 1), P_tu_sel["y"])
    p_tu_c_b3 = cal_b3_s0.predict_proba(logit(np.clip(np.mean([lgb_b3_s0_list[j].predict_proba(P_tu_cal["imps"][j][BEDSIDE3_STATIC])[:, 1] for j in range(5)], axis=0), 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
    tau_low_b3, tau_high_b3, _, _, _, _ = exact_mondrian_thresholds(y_tcal, p_tu_c_b3, alpha=0.10)

    # Compute exact Bedside-5 calculator validation metrics on MIMIC-IV IntVal (for Supp Table S1/S5 disclosure per R11!)
    p_iv_b5_cal = iv_check["Bedside_5_Cal"]
    b5_iv_cov = float(np.mean(((P_iv["y"] == 1) & (p_iv_b5_cal >= tau_low_b5)) | ((P_iv["y"] == 0) & (p_iv_b5_cal <= tau_high_b5))))
    web_artifact = {
        "version": "v4.2_pure_5booster_verified",
        "bedside5_conformal_90_thresholds": {"tau_low": tau_low_b5, "tau_high": tau_high_b5},
        "static3_conformal_90_thresholds": {"tau_low": tau_low_b3, "tau_high": tau_high_b3},
        "bedside5_5boosters": {
            "features": BEDSIDE5,
            "platt_slope": float(cal_b5_5.coef_[0][0]),
            "platt_intercept": float(cal_b5_5.intercept_[0]),
            "intval_auroc": round(float(roc_auc_score(P_iv["y"], p_iv_b5_cal)), 4),
            "intval_brier": round(float(brier_score_loss(P_iv["y"], p_iv_b5_cal)), 4),
            "intval_conformal_coverage": round(b5_iv_cov, 4),
            "boosters_trees": [lgb_b5_list[j].booster_.dump_model()["tree_info"] for j in range(5)]
        },
        "static3_5boosters": {
            "features": BEDSIDE3_STATIC,
            "platt_slope": float(cal_b3_s0.coef_[0][0]),
            "platt_intercept": float(cal_b3_s0.intercept_[0]),
            "boosters_trees": [lgb_b3_s0_list[j].booster_.dump_model()["tree_info"] for j in range(5)]
        },
        "cohort_prevalence_priors": {
            cname: {"event_rate": tournament_res[cname]["event_rate"], "NNE": deploy_metrics[cname]["NNE"]}
            for cname in tournament_res
        }
    }
    json.dump(web_artifact, open(f"{MODELS}/19c_v4_web_calculator_artifact.json", "w", encoding="utf-8"))

    y_ei_ch = P_ei["y"][m_ei_chart]
    auc_ei_b5_ch = float(roc_auc_score(y_ei_ch, ei_pr_all["Bedside_5"][m_ei_chart]))
    lo_ei_b5, hi_ei_b5, _ = boot_ci_with_dist(y_ei_ch, ei_pr_all["Bedside_5"][m_ei_chart], B=2000, seed=105)
    auc_ei_prim_ch = float(roc_auc_score(y_ei_ch, ei_pr_all["Modality_Adaptive_GBDT"][m_ei_chart]))
    auc_ei_hy_ch = float(roc_auc_score(y_ei_ch, ei_pr_all["Hybrid_Ensemble"][m_ei_chart]))
    auc_ei_lg_ch = float(roc_auc_score(y_ei_ch, ei_pr_all["LightGBM"][m_ei_chart]))
    eicu_ch_sub = {
        "n": int(m_ei_chart.sum()),
        "events": int(y_ei_ch.sum()),
        "Bedside_5": {"AUROC": round(auc_ei_b5_ch, 4), "ci95": [round(lo_ei_b5, 4), round(hi_ei_b5, 4)]},
        "Primary_Adaptive_GBDT_AUROC": round(auc_ei_prim_ch, 4),
        "Hybrid_Ensemble_AUROC": round(auc_ei_hy_ch, 4),
        "LightGBM_AUROC": round(auc_ei_lg_ch, 4),
        "bedside5_retention_vs_primary_pct": round(100.0 * auc_ei_b5_ch / auc_ei_prim_ch, 2),
        "bedside5_retention_vs_hybrid_pct": round(100.0 * auc_ei_b5_ch / auc_ei_hy_ch, 2),
        "bedside5_retention_vs_lgbm_pct": round(100.0 * auc_ei_b5_ch / auc_ei_lg_ch, 2),
    }

    out_json = {
        "primary_model_name": "Modality_Adaptive_GBDT",
        "conformal_tune_thresholds": {
            "alpha": 0.10, "tune_select_n": int(len(tu_sel)), "tune_calib_n": int(len(tu_cal)),
            "tune_calib_deaths_n1": int(n1_c), "tune_calib_k1": int(k1_c),
            "tune_calib_survivors_n0": int(n0_c), "tune_calib_k0": int(k0_c),
            "tau_low": tau_low, "tau_high": tau_high,
            "tau_low_bedside5": tau_low_b5, "tau_high_bedside5": tau_high_b5,
            "thr_youden_cal": THR_CAL, "thr_raw_v3": THR_RAW
        },
        "super_learner_tune_weights": {
            "constraint": "non_negative_L2_logistic (w_k >= 0)",
            "LightGBM": round(float(w_stack[0]), 4),
            "CatBoost": round(float(w_stack[1]), 4),
            "CT_DecayNet": round(float(w_stack[2]), 4),
            "intercept": round(float(b_stack), 4)
        },
        "eicu_hospital_audit": eicu_hosp_audit,
        "tournament_cohorts": tournament_res,
        "paired_architecture_tests": paired_arch_res,
        "mondrian_conformal_and_abstention": conformal_res,
        "split_half_recalibration_20x": recal_split_res,
        "bilingual_head_ct_multimodal_fusion": ct_fusion_res,
        "strict_diagnosis_and_gcs_sensitivity": strict_sens,
        "jinhua_sensitivity_coma_autodischarge": jh_sens,
        "deployment_nne_metrics": deploy_metrics,
        "eicu_motor_charted_subcohort": eicu_ch_sub,
    }
    json.dump(out_json, open(f"{REP}/19c_v4_flagship_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    PRIMARY_6 = [
        ("MIMIC_IV_IntVal",        P_iv,     iv_ct_part),
        ("MIMIC_III_CareVue",      P_m3,     m3_ct_part),
        ("eICU_CRD",               P_ei,     None),
        ("HiRID_Trauma",           P_hi_tr,  None),
        ("AmsterdamUMCdb_Trauma",  P_au_tr,  None),
        ("JinhuaNSICU_TBI_S06",    P_jh_tbi, d_jh_tbi[CT_COLS]),
    ]
    pred_rows, fairness_6, severity_6 = [], {}, {}

    for cname, pack, ct_part in PRIMARY_6:
        pr = predict_with_super(pack, ct_part)
        y_d = pack["y"]
        p_prim = pr["Modality_Adaptive_GBDT"]
        p_hy = pr["Hybrid_Ensemble"]
        p_cal_oof = oof_cal_by_cohort[cname]
        p_conf_in = p_prim if cname == "MIMIC_IV_IntVal" else p_cal_oof

        miss_no_ct = pr["miss_motor_no_ct"]
        in_c1 = (p_conf_in >= tau_low) | miss_no_ct
        in_c0 = (p_conf_in <= tau_high) | miss_no_ct
        ambig = in_c0 & in_c1
        abstain = (ambig | miss_no_ct).astype(int)

        raw_d = pack["raw"]
        age_v = pack["imp_mean"]["admission_age"].values
        fem_v = (pack["imp_mean"]["sex_female"].values > 0.5).astype(int)
        gcs0_raw = raw_d["gcs_total_first"].values

        df_c = pd.DataFrame({
            "cohort": cname, "d28": y_d,
            "p_static": pr["Static_S0"], "p_reanchored": pr["Reanchored_S1"],
            "p_lgbm": pr["LightGBM"], "p_catboost": pr["CatBoost"], "p_xgboost": pr["XGBoost"],
            "p_tabular_mlp": pr["Tabular_MLP"], "p_ct_decaynet": pr["CT_DecayNet"],
            "p_adaptive_gbdt": p_prim, "p_hybrid": p_hy,
            "p_bedside5": pr["Bedside_5"], "p_cal_oof": p_cal_oof,
            "in_c0": in_c0.astype(int), "in_c1": in_c1.astype(int), "abstain": abstain,
            "age": age_v, "female": fem_v, "gcs_total_first_raw": gcs0_raw,
        })
        pred_rows.append(df_c)

        # Y5 Fix: In AmsterdamUMCdb (decadal bins: 60-69 -> 65.0), separate 60-69 from <60 and >=70 so 60-69 is not lumped into >=65!
        if cname == "AmsterdamUMCdb_Trauma":
            age_strata = [
                ("female", fem_v == 1), ("male", fem_v == 0),
                ("age_lt65", age_v < 60), ("age_60_69", (age_v >= 60) & (age_v < 70)),
                ("age_ge65", age_v >= 70), ("age_ge80", age_v >= 80)
            ]
        else:
            age_strata = [
                ("female", fem_v == 1), ("male", fem_v == 0),
                ("age_lt65", age_v < 65), ("age_ge65", age_v >= 65), ("age_ge80", age_v >= 80)
            ]
        f_sub = {}
        for sname, smask in age_strata:
            if smask.sum() >= 15 and y_d[smask].sum() >= 2 and (smask.sum() - y_d[smask].sum()) >= 2:
                auc_s = round(float(roc_auc_score(y_d[smask], p_prim[smask])), 4)
                lo_s, hi_s, _ = boot_ci_with_dist(y_d[smask], p_prim[smask], B=1000, seed=55)
                ci_s = [round(lo_s, 4), round(hi_s, 4)]
            else:
                auc_s, ci_s = None, [None, None]
            f_sub[sname] = {"n": int(smask.sum()), "events": int(y_d[smask].sum()), "AUROC": auc_s, "boot95": ci_s}
        fairness_6[cname] = f_sub

        sev_sub = {}
        for sv_name, sv_mask in [
            ("3-8",   pd.notna(gcs0_raw) & (gcs0_raw <= 8)),
            ("9-12",  pd.notna(gcs0_raw) & (gcs0_raw >= 9) & (gcs0_raw <= 12)),
            ("13-15", pd.notna(gcs0_raw) & (gcs0_raw >= 13))
        ]:
            if sv_mask.sum() >= 20 and y_d[sv_mask].sum() >= 3 and (sv_mask.sum() - y_d[sv_mask].sum()) >= 3:
                a0 = round(float(roc_auc_score(y_d[sv_mask], pr["Static_S0"][sv_mask])), 4)
                a1 = round(float(roc_auc_score(y_d[sv_mask], pr["Reanchored_S1"][sv_mask])), 4)
                af = round(float(roc_auc_score(y_d[sv_mask], p_prim[sv_mask])), 4)
                lo_sv, hi_sv, dist_sv = boot_ci_with_dist(y_d[sv_mask], p_prim[sv_mask], B=2000, seed=77)
                if sv_name == "3-8":
                    boot_arrays_to_save[f"{cname}__Severe_3_8"] = dist_sv.astype(np.float32)
                sev_sub[sv_name] = {"n": int(sv_mask.sum()), "events": int(y_d[sv_mask].sum()), "AUROC_S0": a0, "AUROC_S1": a1, "AUROC_Full": af, "ci95": [round(lo_sv, 4), round(hi_sv, 4)]}
            else:
                sev_sub[sv_name] = {"n": int(sv_mask.sum()), "events": int(y_d[sv_mask].sum()), "AUROC_S0": None, "AUROC_S1": None, "AUROC_Full": None, "ci95": [None, None]}
        severity_6[cname] = sev_sub

    pd.concat(pred_rows, ignore_index=True).to_parquet(f"{DATA}/19c_v4_all_cohort_preds.parquet", index=False)
    np.savez_compressed(f"{DATA}/19c_v4_boot_distributions.npz", **boot_arrays_to_save)
    json.dump({"fairness_6cohorts": fairness_6, "severity_6cohorts": severity_6}, open(f"{REP}/19e_v4_subgroups_and_calibration.json", "w", encoding="utf-8"), indent=2)
    lg("DONE! Saved v4.2 tournament, untouched split-conformal, paired tests, 1,500-tree calculator & NPZ bootstrap arrays!")

if __name__ == "__main__":
    main()
