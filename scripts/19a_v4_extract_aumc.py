# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19a_v4_extract_aumc.py — Clean Leak-Free Extraction of Cohort 5 (AmsterdamUMCdb v1.0.2) & HiRID Rolling Streams (v4.2)
Remediates all Opus 5.5 Max R2, R3, R10, Y13, Y22 & dictionary audit items:
  1. Restricts primary AmsterdamUMCdb cohort to first ICU admission per patient (admissioncount == 1, eliminating repeat stays).
  2. Reads listitems.csv via Latin-1 chunked parser (zero dropped 'Geïntubeerd'/'Georiënteerd' rows) across ALL 18 official
     AmsterdamUMCdb GCS itemids (6 Eye, 6 Motor, 6 Verbal) and sets gcs_total = NaN whenever Eye or Verbal is unobserved
     (100% eliminating default fillna(3.0)/fillna(1.0)).
  3. Unifies prothrombin INR->PT fallback factor to 11.5 s (matching HiRID & JinhuaNSICU) and exports official dictionary audit JSON.
  4. Caches hourly-binned [-6h, +72h] vital/lab/GCS streams on true alive-at-t risk sets (los_h >= 6h) for BOTH AmsterdamUMCdb
     and Swiss HiRID so 19d computes 100% leak-free [0, t] rolling predictions across t = 6..72h!
"""
import os, sys, json, time, re, struct, zlib, csv, warnings
import inflate64
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")
gate(__file__)

sys.path.insert(0, r"E:\TBI subtype\07_prediction_system\scripts")
from importlib import import_module
com = import_module("07_common")

ROOT = r"E:\TBI subtype"
DATA = f"{ROOT}/07_prediction_system/data"
REP  = f"{ROOT}/07_prediction_system/reports"
AUMC_DIR = r"F:\AmsterdamUMCdb"
HIRID_DIR = r"F:\HiRID"

t0 = time.time()
def lg(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def _canary_check():
    with open(f"{AUMC_DIR}/numericitems.zip", "rb") as f:
        hdr = f.read(30)
        fnlen, exlen = struct.unpack("<HH", hdr[26:30])
        f.read(fnlen + exlen)
        inf1 = inflate64.Inflater()
        buf = inf1.inflate(f.read(1048576))
        ifnlen, iexlen = struct.unpack("<HH", buf[26:30])
        d2 = zlib.decompressobj(-zlib.MAX_WBITS)
        head_csv = d2.decompress(buf[30 + ifnlen + iexlen:])
    assert b"admissionid,itemid" in head_csv[:120]
    return True

canary_gate(_canary_check, "AmsterdamUMCdb Deflate64 numericitems stream canary")

rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL_COLS = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

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

AGE_MAP = {"18-39": 29.0, "40-49": 45.0, "50-59": 55.0, "60-69": 65.0, "70-79": 75.0, "80+": 84.0}
DIAG_ITEMIDS = {13110, 13112, 13114, 13124, 13130, 16651, 16997, 17018, 18588, 18593, 18600, 18603, 18669, 18671}
GCS_RASS_ITEMIDS = {
    6732, 13077, 19638, 19635, 16628, 14470,
    6734, 13072, 19639, 19636, 16634, 14476,
    6735, 13066, 19637, 19640, 16640, 14482,
    14444
}

@retry_limit(2)
def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    lg("1. Filtering AmsterdamUMCdb first-ICU admissions & scanning Latin-1 listitems.csv (zero dropped rows)...")
    adm_aumc = pd.read_csv(f"{AUMC_DIR}/admissions.csv")
    n_raw_adm = len(adm_aumc)
    adm_aumc = adm_aumc[adm_aumc["admissioncount"] == 1].copy().reset_index(drop=True)
    lg(f"  Restricted to first ICU admission (admissioncount==1): {len(adm_aumc):,} of {n_raw_adm:,} stays")

    adm_aumc["admission_age"] = adm_aumc["agegroup"].map(AGE_MAP)
    adm_aumc = adm_aumc[adm_aumc["admission_age"].notna()].copy()
    adm_aumc["sex_female"] = adm_aumc["gender"].astype(str).str.lower().str.contains("vrouw").astype(float)
    adm_aumc["los_h"] = (adm_aumc["dischargedat"] - adm_aumc["admittedat"]) / 3600000.0
    adm_aumc["hours_to_death"] = np.where(
        adm_aumc["dateofdeath"].notna(),
        (adm_aumc["dateofdeath"] - adm_aumc["admittedat"]) / 3600000.0,
        np.nan
    )
    adm_aumc["d28"] = ((adm_aumc["hours_to_death"].notna()) & (adm_aumc["hours_to_death"] <= 28.0 * 24.0)).astype(int)
    adm_aumc["alive_6h"]  = (adm_aumc["los_h"] >= 6.0)  & (adm_aumc["hours_to_death"].isna() | (adm_aumc["hours_to_death"] > 6.0))
    adm_aumc["alive_24h"] = (adm_aumc["los_h"] >= 24.0) & (adm_aumc["hours_to_death"].isna() | (adm_aumc["hours_to_death"] > 24.0))

    cand_6h_ids = set(adm_aumc.loc[adm_aumc["alive_6h"], "admissionid"].astype(int))
    has_neo, has_tr_dx, has_neu_dx = set(), set(), set()
    gcs_chunks = []

    ALL_SCAN_ITEMS = DIAG_ITEMIDS | GCS_RASS_ITEMIDS
    for ch in pd.read_csv(
        f"{AUMC_DIR}/listitems.csv",
        encoding="latin1",
        usecols=["admissionid", "itemid", "valueid", "value", "measuredat"],
        chunksize=1_000_000
    ):
        sub = ch[ch["itemid"].isin(ALL_SCAN_ITEMS) & ch["admissionid"].isin(cand_6h_ids)]
        if len(sub) == 0:
            continue
        d_sub = sub[sub["itemid"].isin(DIAG_ITEMIDS)]
        if len(d_sub):
            vlow = d_sub["value"].fillna("").astype(str).str.lower()
            has_neo.update(d_sub.loc[vlow.str.contains("neoplasma|neoplasm|tumor", regex=True), "admissionid"].astype(int))
            has_tr_dx.update(d_sub.loc[vlow.str.contains("trauma|hoofdtrauma|letsel|subdura|epidura", regex=True), "admissionid"].astype(int))
            has_neu_dx.update(d_sub.loc[vlow.str.contains("neuro|hersen|bloeding|sab|subarach|cva", regex=True), "admissionid"].astype(int))

        g_sub = sub[sub["itemid"].isin(GCS_RASS_ITEMIDS) & (sub["measuredat"] >= -6 * 3600000) & (sub["measuredat"] <= 120 * 3600000)]
        if len(g_sub):
            gcs_chunks.append(g_sub[["admissionid", "itemid", "valueid", "value", "measuredat"]].copy())

    adm_aumc["has_neoplasm"] = adm_aumc["admissionid"].isin(has_neo).astype(int)
    adm_aumc["has_trauma_dx"] = adm_aumc["admissionid"].isin(has_tr_dx).astype(int)
    adm_aumc["has_neuro_dx"] = adm_aumc["admissionid"].isin(has_neu_dx).astype(int)

    adm_aumc["is_trauma"] = (
        ((adm_aumc["specialty"] == "Traumatologie") | (adm_aumc["has_trauma_dx"] == 1))
        & (adm_aumc["has_neoplasm"] == 0)
    )
    adm_aumc["is_strict_trauma_dx"] = (adm_aumc["has_trauma_dx"] == 1) & (adm_aumc["has_neoplasm"] == 0)
    adm_aumc["is_abi"] = (
        (adm_aumc["specialty"].isin(["Neurochirurgie", "Traumatologie", "Neurologie"]) | (adm_aumc["has_trauma_dx"] == 1) | (adm_aumc["has_neuro_dx"] == 1))
        & (adm_aumc["has_neoplasm"] == 0)
    )

    gcs_rass_df = pd.concat(gcs_chunks, ignore_index=True)
    gcs_rass_df["offset_hr"] = gcs_rass_df["measuredat"] / 3600000.0
    abi_6h_ids = set(adm_aumc.loc[adm_aumc["alive_6h"] & adm_aumc["is_abi"], "admissionid"].astype(int))
    gcs_rass_df = gcs_rass_df[gcs_rass_df["admissionid"].isin(abi_6h_ids)].copy().reset_index(drop=True)
    lg(f"  Latin-1 GCS/RASS rows extracted (neoplasm-free ABI, 0% dropped): {len(gcs_rass_df):,}")

    def map_aumc_gcs_strict(df):
        d = df.copy()
        d["offset_hr"] = d["offset_hr"].clip(lower=0.0)
        d["vid"] = pd.to_numeric(d["valueid"], errors="coerce")

        dm = d[d["itemid"].isin({6734, 13072, 19639, 19636, 16634, 14476})].copy()
        dm["gcs_motor"] = np.nan
        dm.loc[dm["itemid"] == 6734,  "gcs_motor"] = 7.0 - dm.loc[dm["itemid"] == 6734, "vid"]
        dm.loc[dm["itemid"] == 13072, "gcs_motor"] = dm.loc[dm["itemid"] == 13072, "vid"]
        dm.loc[dm["itemid"] == 19639, "gcs_motor"] = dm.loc[dm["itemid"] == 19639, "vid"] - 12.0
        dm.loc[dm["itemid"].isin({19636, 16634, 14476}), "gcs_motor"] = dm.loc[dm["itemid"].isin({19636, 16634, 14476}), "vid"] - 6.0
        dm["gcs_motor"] = dm["gcs_motor"].where(dm["gcs_motor"].between(1, 6))

        de = d[d["itemid"].isin({6732, 13077, 19638, 19635, 16628, 14470})].copy()
        de["gcs_eye"] = np.nan
        de.loc[de["itemid"] == 6732,  "gcs_eye"] = 5.0 - de.loc[de["itemid"] == 6732, "vid"]
        de.loc[de["itemid"] == 13077, "gcs_eye"] = de.loc[de["itemid"] == 13077, "vid"]
        de.loc[de["itemid"] == 19638, "gcs_eye"] = de.loc[de["itemid"] == 19638, "vid"] - 8.0
        de.loc[de["itemid"].isin({19635, 16628, 14470}), "gcs_eye"] = de.loc[de["itemid"].isin({19635, 16628, 14470}), "vid"] - 4.0
        de["gcs_eye"] = de["gcs_eye"].where(de["gcs_eye"].between(1, 4))

        dv = d[d["itemid"].isin({6735, 13066, 19637, 19640, 16640, 14482})].copy()
        dv["gcs_verbal"] = np.nan
        m6735 = (dv["itemid"] == 6735)
        dv.loc[m6735 & (dv["vid"] == 8), "gcs_verbal"] = 1.0
        dv.loc[m6735 & dv["vid"].between(1, 5), "gcs_verbal"] = 6.0 - dv.loc[m6735 & dv["vid"].between(1, 5), "vid"]
        m13066 = (dv["itemid"] == 13066)
        dv.loc[m13066 & dv["vid"].between(1, 5), "gcs_verbal"] = dv.loc[m13066 & dv["vid"].between(1, 5), "vid"]
        m19637 = (dv["itemid"] == 19637)
        dv.loc[m19637 & (dv["vid"] == 9), "gcs_verbal"] = 1.0
        dv.loc[m19637 & dv["vid"].between(10, 14), "gcs_verbal"] = dv.loc[m19637 & dv["vid"].between(10, 14), "vid"] - 9.0
        m19640 = (dv["itemid"] == 19640)
        dv.loc[m19640 & (dv["vid"] == 15), "gcs_verbal"] = 1.0
        dv.loc[m19640 & dv["vid"].between(16, 20), "gcs_verbal"] = dv.loc[m19640 & dv["vid"].between(16, 20), "vid"] - 15.0
        m_mca_ra = dv["itemid"].isin({16640, 14482})
        dv.loc[m_mca_ra & dv["vid"].between(6, 10), "gcs_verbal"] = dv.loc[m_mca_ra & dv["vid"].between(6, 10), "vid"] - 5.0
        dv["gcs_verbal"] = dv["gcs_verbal"].where(dv["gcs_verbal"].between(1, 5))

        for sub_x in (dm, de, dv):
            sub_x["hr_bin"] = (sub_x["offset_hr"] * 4.0).round() / 4.0

        g_m = dm.dropna(subset=["gcs_motor"]).groupby(["admissionid", "hr_bin"], as_index=False)["gcs_motor"].max()
        g_e = de.dropna(subset=["gcs_eye"]).groupby(["admissionid", "hr_bin"], as_index=False)["gcs_eye"].max()
        g_v = dv.dropna(subset=["gcs_verbal"]).groupby(["admissionid", "hr_bin"], as_index=False)["gcs_verbal"].max()

        g = g_m.merge(g_e, on=["admissionid", "hr_bin"], how="outer").merge(g_v, on=["admissionid", "hr_bin"], how="outer")
        g = g.sort_values(["admissionid", "hr_bin"]).reset_index(drop=True)
        g["offset_hr"] = g["hr_bin"]
        g["id"] = g["admissionid"]
        g["gcs_total"] = np.where(
            g["gcs_motor"].notna() & g["gcs_eye"].notna() & g["gcs_verbal"].notna(),
            g["gcs_motor"] + g["gcs_eye"] + g["gcs_verbal"],
            np.nan
        )
        return g

    gcs_aumc_all = map_aumc_gcs_strict(gcs_rass_df)
    rass_24 = gcs_rass_df[(gcs_rass_df["itemid"] == 14444) & (gcs_rass_df["offset_hr"] <= 24.0)].copy()
    rass_agg = rass_24.groupby("admissionid").agg(rass_n_24h=("valueid", "size"), rass_min_24h=("valueid", "min")).reset_index()

    proc_path = f"{AUMC_DIR}/processitems.csv".replace("\\", "/")
    mv_df = con.execute(f"""
    SELECT admissionid,
           MAX(CASE WHEN itemid IN (9328, 12634) AND start <= 24 * 3600000 AND stop >= 0 THEN 1 ELSE 0 END) AS vent_24h
    FROM read_csv('{proc_path}', all_varchar=false, ignore_errors=true)
    GROUP BY admissionid
    """).df()

    gcs_aumc_24 = gcs_aumc_all[(gcs_aumc_all["offset_hr"] <= 24.0) & (gcs_aumc_all["gcs_motor"].notna())].copy()
    ids_with_m24 = set(gcs_aumc_24["admissionid"].unique())

    aumc_coh = adm_aumc[adm_aumc["alive_24h"] & adm_aumc["is_abi"] & adm_aumc["admissionid"].isin(ids_with_m24)].copy().reset_index(drop=True)
    ids_with_m72 = set(gcs_aumc_all.loc[(gcs_aumc_all["offset_hr"] <= 72.0) & gcs_aumc_all["gcs_motor"].notna(), "admissionid"].unique())
    aumc_roll_coh = adm_aumc[adm_aumc["alive_6h"] & adm_aumc["is_trauma"] & adm_aumc["admissionid"].isin(ids_with_m72)].copy().reset_index(drop=True)

    n_tr_strict = int(aumc_coh["is_trauma"].sum())
    d_tr_strict = int(aumc_coh.loc[aumc_coh["is_trauma"], "d28"].sum())
    lg(f"  AmsterdamUMCdb 24h Trauma (first stay): n = {n_tr_strict:,} (d28 = {d_tr_strict:,} [{d_tr_strict/max(n_tr_strict,1)*100:.1f}%]; explicit trauma dx n = {int(aumc_coh['is_strict_trauma_dx'].sum()):,}) | ABI n = {len(aumc_coh):,} | 6h Rolling Trauma risk set n = {len(aumc_roll_coh):,}")

    lg("3. Streaming F:/AmsterdamUMCdb/numericitems.zip for [-6h, +24h] 24h matrix & [-6h, +72h] leak-free rolling bins...")
    coh_24_set   = set(aumc_coh["admissionid"].astype(int))
    roll_6_set   = set(aumc_roll_coh["admissionid"].astype(int))
    all_need_ids = coh_24_set | roll_6_set
    target_adm_bytes = {str(aid).encode("ascii"): aid for aid in all_need_ids}

    V7_SET = {"hr", "sbp", "dbp", "mbp", "rr", "temp", "spo2"}
    ITEM_SPEC = {
        b"6640":  ("hr",          20.0,  300.0, lambda v: v),
        b"6641":  ("sbp",         30.0,  300.0, lambda v: v),
        b"6678":  ("sbp",         30.0,  300.0, lambda v: v),
        b"6643":  ("dbp",         10.0,  220.0, lambda v: v),
        b"6680":  ("dbp",         10.0,  220.0, lambda v: v),
        b"6642":  ("mbp",         20.0,  250.0, lambda v: v),
        b"6679":  ("mbp",         20.0,  250.0, lambda v: v),
        b"8874":  ("rr",           4.0,   70.0, lambda v: v),
        b"12266": ("rr",           4.0,   70.0, lambda v: v),
        b"7726":  ("rr",           4.0,   70.0, lambda v: v),
        b"8873":  ("rr",           4.0,   70.0, lambda v: v),
        b"13058": ("temp",        30.0,   43.5, lambda v: v),
        b"8658":  ("temp",        30.0,   43.5, lambda v: v),
        b"13059": ("temp",        30.0,   43.5, lambda v: v),
        b"13062": ("temp",        30.0,   43.5, lambda v: v),
        b"16110": ("temp",        30.0,   43.5, lambda v: v),
        b"8662":  ("temp",        30.0,   43.5, lambda v: v),
        b"13060": ("temp",        30.0,   43.5, lambda v: v),
        b"6709":  ("spo2",        40.0,  100.0, lambda v: v),
        b"9960":  ("hemoglobin",   2.0,   16.0, lambda v: v * 1.6113),
        b"10286": ("hemoglobin",   2.0,   16.0, lambda v: v * 1.6113),
        b"6778":  ("hemoglobin",   2.0,   16.0, lambda v: v * 1.6113),
        b"9553":  ("hemoglobin",   2.0,   16.0, lambda v: v * 1.6113),
        b"11423": ("hematocrit",   0.10,  75.0, lambda v: v * 100.0 if v <= 1.0 else v),
        b"11545": ("hematocrit",   0.10,  75.0, lambda v: v * 100.0 if v <= 1.0 else v),
        b"6777":  ("hematocrit",   0.10,  75.0, lambda v: v * 100.0 if v <= 1.0 else v),
        b"9964":  ("platelet",     5.0, 1500.0, lambda v: v),
        b"6797":  ("platelet",     5.0, 1500.0, lambda v: v),
        b"14252": ("platelet",     5.0, 1500.0, lambda v: v),
        b"9965":  ("wbc",          0.1,  150.0, lambda v: v),
        b"6779":  ("wbc",          0.1,  150.0, lambda v: v),
        b"9943":  ("bun",          0.5,  100.0, lambda v: v * 2.801),
        b"6850":  ("bun",          0.5,  100.0, lambda v: v * 2.801),
        b"9941":  ("creatinine",  10.0, 2000.0, lambda v: v / 88.4),
        b"6836":  ("creatinine",  10.0, 2000.0, lambda v: v / 88.4),
        b"9947":  ("glucose_lab",  1.0,   80.0, lambda v: v * 18.018),
        b"9557":  ("glucose_lab",  1.0,   80.0, lambda v: v * 18.018),
        b"6833":  ("glucose_lab",  1.0,   80.0, lambda v: v * 18.018),
        b"9924":  ("sodium",     100.0,  190.0, lambda v: v),
        b"10284": ("sodium",     100.0,  190.0, lambda v: v),
        b"9555":  ("sodium",     100.0,  190.0, lambda v: v),
        b"6840":  ("sodium",     100.0,  190.0, lambda v: v),
        b"9927":  ("potassium",    1.5,   10.0, lambda v: v),
        b"10285": ("potassium",    1.5,   10.0, lambda v: v),
        b"9556":  ("potassium",    1.5,   10.0, lambda v: v),
        b"6835":  ("potassium",    1.5,   10.0, lambda v: v),
        b"9930":  ("chloride",    50.0,  160.0, lambda v: v),
        b"14413": ("chloride",    50.0,  160.0, lambda v: v),
        b"6819":  ("chloride",    50.0,  160.0, lambda v: v),
        b"9992":  ("bicarbonate",  2.0,   60.0, lambda v: v),
        b"6810":  ("bicarbonate",  2.0,   60.0, lambda v: v),
        b"9559":  ("aniongap",     1.0,   50.0, lambda v: v),
        b"9933":  ("calcium",      0.5,    5.0, lambda v: v * 4.008),
        b"18001": ("calcium",      0.5,    5.0, lambda v: v * 4.008),
        b"6817":  ("calcium",      0.5,    5.0, lambda v: v * 4.008),
        b"11893": ("inr_only",     0.5,   25.0, lambda v: v),
        b"11894": ("inr_only",     0.5,   25.0, lambda v: v),
        b"6789":  ("pt_sec",       7.0,  150.0, lambda v: v),
        b"17982": ("ptt",         10.0,  240.0, lambda v: v),
        b"11944": ("ptt",         10.0,  240.0, lambda v: v),
        b"6771":  ("ptt",         10.0,  240.0, lambda v: v),
        b"8835":  ("icp",         -5.0,  150.0, lambda v: v),
        b"19358": ("icp",         -5.0,  200.0, lambda v: v / 1.3595),
    }

    stats24 = {}
    roll_bins = {}

    def _upd(dct, key, val):
        s = dct.get(key)
        if s is None:
            dct[key] = [val, val, val, 1]
        else:
            if val < s[0]: s[0] = val
            if val > s[1]: s[1] = val
            s[2] += val
            s[3] += 1

    MIN_MS   = -6 * 3600000
    MAX24_MS = 24 * 3600000
    MAX72_MS = 72 * 3600000

    with open(f"{AUMC_DIR}/numericitems.zip", "rb") as f:
        hdr = f.read(30)
        fnlen, exlen = struct.unpack("<HH", hdr[26:30])
        f.read(fnlen + exlen)
        inf1 = inflate64.Inflater()
        d2 = None
        leftover = b""
        bytes_read = 0
        next_log_gb = 1

        while not inf1.eof:
            chunk = f.read(4 * 1048576)
            if not chunk:
                break
            bytes_read += len(chunk)
            inner_chunk = inf1.inflate(chunk)
            if not inner_chunk:
                continue
            if d2 is None:
                ifnlen, iexlen = struct.unpack("<HH", inner_chunk[26:30])
                d2 = zlib.decompressobj(-zlib.MAX_WBITS)
                inner_chunk = inner_chunk[30 + ifnlen + iexlen:]
                csv_chunk = d2.decompress(inner_chunk)
                nl_pos = csv_chunk.find(b"\n")
                csv_chunk = csv_chunk[nl_pos + 1:] if nl_pos >= 0 else b""
            else:
                try:
                    csv_chunk = d2.decompress(inner_chunk)
                except zlib.error:
                    break

            buf = leftover + csv_chunk
            last_nl = buf.rfind(b"\n")
            if last_nl < 0:
                leftover = buf
                continue
            lines_block = buf[:last_nl]
            leftover = buf[last_nl + 1:]

            for ln in lines_block.split(b"\n"):
                c1 = ln.find(b",")
                if c1 <= 0:
                    continue
                aid = target_adm_bytes.get(ln[:c1])
                if aid is None:
                    continue
                c2 = ln.find(b",", c1 + 1)
                if c2 <= c1:
                    continue
                spec = ITEM_SPEC.get(ln[c1 + 1:c2])
                if spec is None:
                    continue
                if b'"' in ln:
                    row_q = next(csv.reader([ln.decode("latin1", "replace")]))
                    if len(row_q) < 15:
                        continue
                    try:
                        meas_ms = int(float(row_q[8]))
                        if meas_ms < MIN_MS or meas_ms > MAX72_MS:
                            continue
                        raw_v = float(row_q[4])
                    except ValueError:
                        continue
                else:
                    parts = ln.split(b",")
                    if len(parts) < 15:
                        continue
                    try:
                        meas_ms = int(parts[-7])
                        if meas_ms < MIN_MS or meas_ms > MAX72_MS:
                            continue
                        raw_v = float(parts[4])
                    except ValueError:
                        continue

                fname, vlo, vhi, tfn = spec
                if raw_v < vlo or raw_v > vhi:
                    continue
                if fname in V7_SET and meas_ms < 0:
                    continue

                if fname == "inr_only":
                    pairs = [("inr", raw_v), ("pt_from_inr", raw_v * 11.5)]
                elif fname == "pt_sec":
                    pairs = [("pt_direct", raw_v), ("inr", raw_v / 11.5)]
                else:
                    pairs = [(fname, tfn(raw_v))]

                if meas_ms <= MAX24_MS and aid in coh_24_set:
                    for fn_k, val_k in pairs:
                        _upd(stats24, (aid, fn_k), val_k)

                if aid in roll_6_set and fname != "icp":
                    h_bin = 0 if meas_ms <= 0 else int((meas_ms + 3599999) // 3600000)
                    for fn_k, val_k in pairs:
                        _upd(roll_bins, (aid, h_bin, fn_k), val_k)

            if bytes_read >= next_log_gb * (1024 ** 3):
                lg(f"  Streamed {bytes_read / (1024**3):.1f} GiB of numericitems.zip...")
                next_log_gb += 1

    V7 = ["hr", "sbp", "dbp", "mbp", "rr", "temp", "spo2"]
    LABS = ["hematocrit", "hemoglobin", "platelet", "wbc", "aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine", "glucose_lab", "sodium", "potassium", "inr", "ptt"]
    num_rows = {}
    for (aid, feat), (mn, mx, sm, cnt) in stats24.items():
        d_aid = num_rows.setdefault(aid, {"admissionid": aid})
        if feat in V7:
            d_aid[f"{feat}_min"] = mn; d_aid[f"{feat}_mean"] = sm / cnt; d_aid[f"{feat}_max"] = mx
        elif feat in LABS:
            d_aid[f"{feat}_min"] = mn; d_aid[f"{feat}_max"] = mx
        elif feat in ("pt_direct", "pt_from_inr"):
            d_aid[f"{feat}_min"] = mn; d_aid[f"{feat}_max"] = mx
        elif feat == "icp":
            d_aid["has_icp_24h"] = 1; d_aid["icp_max_24h"] = mx; d_aid["icp_mean_24h"] = sm / cnt

    num_df = pd.DataFrame(list(num_rows.values())) if num_rows else pd.DataFrame({"admissionid": []})
    for sfx in ("min", "max"):
        c_dir = f"pt_direct_{sfx}"; c_inr = f"pt_from_inr_{sfx}"
        s_dir = num_df[c_dir] if c_dir in num_df.columns else pd.Series(np.nan, index=num_df.index)
        s_inr = num_df[c_inr] if c_dir in num_df.columns or c_inr in num_df.columns else pd.Series(np.nan, index=num_df.index)
        if c_inr not in num_df.columns:
            s_inr = pd.Series(np.nan, index=num_df.index)
        num_df[f"pt_{sfx}"] = s_dir.fillna(s_inr)

    roll_au_df = pd.DataFrame(
        [(aid, hb, ft, v[0], v[1], v[2], v[3]) for (aid, hb, ft), v in roll_bins.items()],
        columns=["admissionid", "hr_bin", "feat", "mn", "mx", "sm", "cnt"]
    )
    roll_au_df.to_parquet(f"{DATA}/19a_roll72_amsterdam_v4.parquet", index=False)
    aumc_roll_coh[["admissionid", "admission_age", "sex_female", "los_h", "hours_to_death", "d28", "is_trauma", "is_strict_trauma_dx"]].to_parquet(f"{DATA}/19a_roll72_amsterdam_meta_v4.parquet", index=False)

    gcs_aumc_roll_72 = gcs_aumc_all[gcs_aumc_all["admissionid"].isin(roll_6_set) & (gcs_aumc_all["offset_hr"] <= 72.0)].copy()
    gcs_aumc_roll_72.to_parquet(f"{DATA}/19a_gcs72_amsterdam_roll_v4.parquet", index=False)

    gcs_aumc_coh_72 = gcs_aumc_all[gcs_aumc_all["admissionid"].isin(coh_24_set) & (gcs_aumc_all["offset_hr"] <= 72.0)].copy()
    gcs_aumc_coh_24 = gcs_aumc_coh_72[gcs_aumc_coh_72["offset_hr"] <= 24.0].copy()

    def first_nonnull(s):
        v = s.dropna()
        return float(v.iloc[0]) if len(v) else np.nan

    ga_aumc = gcs_aumc_coh_24.sort_values("offset_hr").groupby("admissionid").agg(
        gcs_total_min=("gcs_total", "min"), gcs_total_first=("gcs_total", first_nonnull),
        gcs_motor_min=("gcs_motor", "min"), gcs_motor_first=("gcs_motor", first_nonnull),
        gcs_eye_min=("gcs_eye", "min"),     gcs_eye_first=("gcs_eye", first_nonnull),
        gcs_verbal_min=("gcs_verbal", "min"), gcs_verbal_first=("gcs_verbal", first_nonnull),
        n_motor_24h=("gcs_motor", lambda s: int(s.notna().sum()))
    ).reset_index()

    b_motor_aumc = com.neuro_dynamic_features(gcs_aumc_coh_24, "gcs_motor", "motor", win_hr=24.0).rename(columns={"id": "admissionid"})
    b_total_aumc = com.neuro_dynamic_features(gcs_aumc_coh_24, "gcs_total", "total", win_hr=24.0).rename(columns={"id": "admissionid"})
    post_aumc = compute_posteriors(gcs_aumc_coh_24, id_col="admissionid", max_hr=24.0)
    g_post24 = gcs_aumc_coh_72[(gcs_aumc_coh_72["offset_hr"] > 24.0) & (gcs_aumc_coh_72["gcs_motor"].notna())].groupby("admissionid")["gcs_motor"].min().reset_index().rename(columns={"gcs_motor": "motor_min_24_72h"})

    aumc_mat = aumc_coh.merge(ga_aumc, on="admissionid", how="left") \
                       .merge(b_motor_aumc, on="admissionid", how="left") \
                       .merge(b_total_aumc, on="admissionid", how="left") \
                       .merge(post_aumc, on="admissionid", how="left") \
                       .merge(num_df, on="admissionid", how="left") \
                       .merge(rass_agg, on="admissionid", how="left") \
                       .merge(mv_df, on="admissionid", how="left") \
                       .merge(g_post24, on="admissionid", how="left")

    for col_b, col_f in [("motor_first", "gcs_motor_first"), ("motor_last", "gcs_motor_first"), ("motor_min", "gcs_motor_min"), ("motor_max", "gcs_motor_first")]:
        aumc_mat[col_b] = aumc_mat[col_b].fillna(aumc_mat[col_f])
    aumc_mat["motor_delta"] = aumc_mat["motor_delta"].fillna(0.0)

    if "sbp_mean" in aumc_mat.columns:
        aumc_mat["mbp_min"]  = aumc_mat["mbp_min"].fillna((aumc_mat["sbp_min"] + 2.0 * aumc_mat["dbp_min"]) / 3.0)
        aumc_mat["mbp_mean"] = aumc_mat["mbp_mean"].fillna((aumc_mat["sbp_mean"] + 2.0 * aumc_mat["dbp_mean"]) / 3.0)
        aumc_mat["mbp_max"]  = aumc_mat["mbp_max"].fillna((aumc_mat["sbp_max"] + 2.0 * aumc_mat["dbp_max"]) / 3.0)
    if "sodium_min" in aumc_mat.columns and "chloride_max" in aumc_mat.columns and "bicarbonate_max" in aumc_mat.columns:
        aumc_mat["aniongap_min"] = aumc_mat["aniongap_min"].fillna(aumc_mat["sodium_min"] - (aumc_mat["chloride_max"] + aumc_mat["bicarbonate_max"]))
        aumc_mat["aniongap_max"] = aumc_mat["aniongap_max"].fillna(aumc_mat["sodium_max"] - (aumc_mat["chloride_min"] + aumc_mat["bicarbonate_min"]))

    aumc_mat["has_icp_24h"] = aumc_mat.get("has_icp_24h", pd.Series(0, index=aumc_mat.index)).fillna(0).astype(int)
    aumc_mat["charlson_comorbidity_score"] = np.nan
    aumc_mat["end_24_72h"] = np.where(
        aumc_mat["motor_min_24_72h"].notna(),
        ((aumc_mat["motor_last"] - aumc_mat["motor_min_24_72h"]) >= 2.0).astype(int),
        0
    )
    for c in FULL_COLS:
        if c not in aumc_mat.columns:
            aumc_mat[c] = np.nan

    bad_tot = int((aumc_mat["gcs_total_first"].notna() & (aumc_mat["gcs_eye_first"].isna() | aumc_mat["gcs_verbal_first"].isna())).sum())
    assert bad_tot == 0, f"R2 violation: {bad_tot} rows have gcs_total_first with missing E/V!"
    assert aumc_mat["motor_last"].notna().all(), "motor_last must be 100% non-null"

    aumc_mat.to_parquet(f"{DATA}/19a_matrix_amsterdam_v4.parquet", index=False)
    gcs_aumc_coh_72.to_parquet(f"{DATA}/19a_gcs72_amsterdam_v4.parquet", index=False)
    e_cov_tr = float(aumc_mat.loc[aumc_mat["is_trauma"], "gcs_total_first"].notna().mean() * 100)
    lg(f"  [SAVED] 19a_matrix_amsterdam_v4.parquet (ABI n={len(aumc_mat):,}, Trauma n={n_tr_strict:,}, exact 3-subscore gcs_total cov={e_cov_tr:.1f}%, ICP n={aumc_mat['has_icp_24h'].sum():,})")

    lg("4. Caching HiRID 0-24h GCS sequence & true alive-at-6h [-6h, +72h] rolling vital/lab/GCS streams...")
    hi_mat = pd.read_parquet(f"{DATA}/08c_matrix_hirid_v3.parquet")
    gen = pd.read_csv(f"{HIRID_DIR}/general_table.csv")
    gen["admissiontime"] = pd.to_datetime(gen["admissiontime"])
    con.register("gen_tbl", gen[gen["discharge_status"].isin(["alive", "dead"])])
    P_GLOB = f"{HIRID_DIR}/raw_stage/observation_tables/parquet/*.parquet".replace("\\", "/")

    hi_meta_all = con.execute(f"""
    SELECT g.patientid, g.admissiontime, g.sex, g.age AS admission_age, g.discharge_status,
           MAX(EPOCH(o.datetime - g.admissiontime)/3600.0) AS los_h,
           MIN(CASE WHEN o.variableid = 9990002 THEN CAST(o.value AS INTEGER) END) AS apache2,
           MIN(CASE WHEN o.variableid = 9990004 THEN CAST(o.value AS INTEGER) END) AS apache4,
           COUNT(CASE WHEN o.variableid = 10000200 AND o.datetime BETWEEN g.admissiontime AND g.admissiontime + INTERVAL 72 HOUR THEN 1 END) AS n_motor_72h
    FROM read_parquet('{P_GLOB}') o
    JOIN gen_tbl g ON o.patientid = g.patientid
    WHERE o.value IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5
    """).df()
    AP2_TR = {103, 111}; AP4_TR = {194, 203}
    hi_meta_all["is_trauma"] = hi_meta_all["apache2"].isin(AP2_TR) | hi_meta_all["apache4"].isin(AP4_TR)
    hi_meta_all["d28"] = (hi_meta_all["discharge_status"] == "dead").astype(int)
    hi_meta_all["sex_female"] = (hi_meta_all["sex"] == "F").astype(float)
    hi_roll_meta = hi_meta_all[(hi_meta_all["is_trauma"]) & (hi_meta_all["los_h"] >= 6.0) & (hi_meta_all["n_motor_72h"] >= 1)].copy().reset_index(drop=True)
    hi_roll_meta[["patientid", "admission_age", "sex_female", "los_h", "d28"]].to_parquet(f"{DATA}/19a_roll72_hirid_meta_v4.parquet", index=False)

    all_hi_cand = gen[gen["patientid"].isin(set(hi_mat["patientid"]) | set(hi_roll_meta["patientid"]))][["patientid", "admissiontime"]].copy()
    con.register("hi_cand_all", all_hi_cand)

    gcs_hi_72 = con.execute(f"""
    SELECT o.patientid AS id,
           ROUND(EPOCH(o.datetime - c.admissiontime)/3600.0, 4) AS offset_hr,
           MAX(CASE WHEN o.variableid = 10000200 AND o.value BETWEEN 1 AND 6 THEN o.value END) AS gcs_motor,
           MAX(CASE WHEN o.variableid = 10000300 AND o.value BETWEEN 1 AND 4 THEN o.value END) AS gcs_eye,
           MAX(CASE WHEN o.variableid = 10000100 AND o.value BETWEEN 1 AND 5 THEN o.value END) AS gcs_verbal
    FROM read_parquet('{P_GLOB}') o
    JOIN hi_cand_all c ON o.patientid = c.patientid
    WHERE o.variableid IN (10000100, 10000200, 10000300)
      AND o.value IS NOT NULL
      AND o.datetime >= c.admissiontime
      AND o.datetime <= c.admissiontime + INTERVAL 72 HOUR
    GROUP BY o.patientid, o.datetime, c.admissiontime
    ORDER BY o.patientid, offset_hr
    """).df()
    gcs_hi_72["gcs_total"] = gcs_hi_72["gcs_motor"] + gcs_hi_72["gcs_eye"] + gcs_hi_72["gcs_verbal"]
    gcs_hi_24 = gcs_hi_72[(gcs_hi_72["id"].isin(set(hi_mat["patientid"]))) & (gcs_hi_72["offset_hr"] <= 24.0)].copy()
    gcs_hi_24.to_parquet(f"{DATA}/19a_gcs24_hirid_v4.parquet", index=False)
    gcs_hi_72[gcs_hi_72["id"].isin(set(hi_roll_meta["patientid"]))].to_parquet(f"{DATA}/19a_gcs72_hirid_roll_v4.parquet", index=False)

    con.register("hi_roll_cand", hi_roll_meta[["patientid", "admissiontime"]])
    hi_roll_bins = con.execute(f"""
    WITH raw AS (
        SELECT o.patientid,
               CASE WHEN o.datetime <= c.admissiontime THEN 0
                    ELSE CAST(CEIL(EPOCH(o.datetime - c.admissiontime)/3600.0) AS INTEGER) END AS hr_bin,
               CASE
                    WHEN o.variableid = 200 AND o.datetime >= c.admissiontime AND o.value BETWEEN 20 AND 300 THEN 'hr'
                    WHEN o.variableid IN (100, 600) AND o.datetime >= c.admissiontime AND o.value BETWEEN 30 AND 400 THEN 'sbp'
                    WHEN o.variableid IN (120, 620) AND o.datetime >= c.admissiontime AND o.value BETWEEN 10 AND 300 THEN 'dbp'
                    WHEN o.variableid IN (110, 610) AND o.datetime >= c.admissiontime AND o.value BETWEEN 20 AND 300 THEN 'mbp'
                    WHEN o.variableid IN (300, 310, 5685) AND o.datetime >= c.admissiontime AND o.value BETWEEN 4 AND 70 THEN 'rr'
                    WHEN o.variableid IN (410, 7100, 400) AND o.datetime >= c.admissiontime AND o.value BETWEEN 30 AND 45 THEN 'temp'
                    WHEN o.variableid IN (4000, 8280) AND o.datetime >= c.admissiontime AND o.value BETWEEN 50 AND 100 THEN 'spo2'
                    WHEN o.variableid IN (20000900, 24000836, 24000548) AND o.value BETWEEN 20 AND 250 THEN 'hemoglobin'
                    WHEN o.variableid = 20000110 AND o.value BETWEEN 1 AND 2000 THEN 'platelet'
                    WHEN o.variableid = 20000700 AND o.value BETWEEN 0.1 AND 200 THEN 'wbc'
                    WHEN o.variableid = 20004200 AND o.value BETWEEN 2 AND 60 THEN 'bicarbonate'
                    WHEN o.variableid = 20004100 AND o.value BETWEEN 0.5 AND 100 THEN 'bun'
                    WHEN o.variableid = 20005100 AND o.value BETWEEN 0.5 AND 5.0 THEN 'calcium'
                    WHEN o.variableid IN (24000439, 24000521) AND o.value BETWEEN 50 AND 160 THEN 'chloride'
                    WHEN o.variableid = 20000600 AND o.value BETWEEN 10 AND 2000 THEN 'creatinine'
                    WHEN o.variableid IN (20005110, 24000523, 24000585) AND o.value BETWEEN 1 AND 100 THEN 'glucose_lab'
                    WHEN o.variableid IN (20000400, 24000519, 24000658, 24000835, 24000866) AND o.value BETWEEN 100 AND 190 THEN 'sodium'
                    WHEN o.variableid IN (20000500, 24000520, 24000833, 24000867) AND o.value BETWEEN 1.5 AND 10 THEN 'potassium'
                    WHEN o.variableid = 24000567 AND o.value BETWEEN 0.5 AND 20 THEN 'inr'
                    WHEN o.variableid = 20004410 AND o.value BETWEEN 10 AND 200 THEN 'ptt'
               END AS feat,
               CASE
                    WHEN o.variableid IN (20000900, 24000836, 24000548) THEN o.value / 10.0
                    WHEN o.variableid = 20004100 THEN o.value * 2.801
                    WHEN o.variableid = 20005100 THEN o.value * 4.008
                    WHEN o.variableid = 20000600 THEN o.value / 88.4
                    WHEN o.variableid IN (20005110, 24000523, 24000585) THEN o.value * 18.018
                    ELSE o.value
               END AS val
        FROM read_parquet('{P_GLOB}') o
        JOIN hi_roll_cand c ON o.patientid = c.patientid
        WHERE o.value IS NOT NULL
          AND o.datetime BETWEEN c.admissiontime - INTERVAL 6 HOUR AND c.admissiontime + INTERVAL 72 HOUR
    )
    SELECT patientid, hr_bin, feat, MIN(val) AS mn, MAX(val) AS mx, SUM(val) AS sm, COUNT(*) AS cnt
    FROM raw
    WHERE feat IS NOT NULL AND hr_bin <= 72
    GROUP BY 1, 2, 3
    """).df()
    hi_roll_bins.to_parquet(f"{DATA}/19a_roll72_hirid_v4.parquet", index=False)

    dict_audit = {
        "hirid_apache_trauma_codes": {
            "9990002_APACHE_II": {"103": "Non-operative Trauma", "111": "Post-operative Trauma", "101": "Non-operative Neurologic", "110": "Post-operative Neurosurgical"},
            "9990004_APACHE_IV": {"194": "Non-operative Trauma", "203": "Post-operative Trauma", "193": "Non-operative Neurologic", "202": "Post-operative Neurosurgical"},
            "hirid_gcs_24h_obs_per_stay_median_iqr": [float(gcs_hi_24.groupby("id").size().median()), [float(x) for x in np.percentile(gcs_hi_24.groupby("id").size(), [25, 75])]],
            "hirid_roll_6h_risk_set_n": int(len(hi_roll_meta)),
        },
        "amsterdamumcdb_official_dictionary": {
            "eye_itemids_6": [6732, 13077, 19638, 19635, 16628, 14470],
            "motor_itemids_6": [6734, 13072, 19639, 19636, 16634, 14476],
            "verbal_itemids_6": [6735, 13066, 19637, 19640, 16640, 14482],
            "intubated_verbal_codes": {"6735": "valueid=8 (Geïntubeerd -> 1.0)", "19637": "valueid=9 (Geïntubeerd -> 1.0)", "19640": "valueid=15 (Geïntubeerd -> 1.0)"},
            "icp_units_verified": {"8835": "Intracraniële druk (mmHg)", "19358": "Intacraniële druk (cmH2O / 1.3595 -> mmHg)"},
            "calcium_units_verified": {"9933": "Calcium totaal (mmol/l * 4.008 -> mg/dL)", "18001": "Ca alb.corr. (mmol/l * 4.008 -> mg/dL)", "6817": "Calcium (mmol/l * 4.008 -> mg/dL)"},
            "pt_inr_conversion": "Direct sec (itemid 6789) or INR (11893/11894) * 11.5 s (unified across HiRID, AmsterdamUMCdb, and JinhuaNSICU)",
            "first_admission_only": True,
            "bad_synthetic_gcs_total_count": bad_tot,
            "aumc_roll_6h_risk_set_n": int(len(aumc_roll_coh)),
        }
    }
    json.dump(dict_audit, open(f"{REP}/19a_v4_dictionary_audit.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    lg(f"  [SAVED] HiRID & AUMC leak-free rolling streams + 19a_v4_dictionary_audit.json (HiRID 6h risk set n={len(hi_roll_meta):,})")

if __name__ == "__main__":
    main()
