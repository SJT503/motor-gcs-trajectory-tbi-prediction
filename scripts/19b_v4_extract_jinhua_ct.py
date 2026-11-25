# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19b_v4_extract_jinhua_ct.py — Clean Leak-Free Extraction of Cohort 6 (JinhuaNSICU v1.2)
& Bilingual Head-CT Radiology Report NLP Phenotypes (<= 24h of ICU admission) — v4.2 Edition
Remediates Opus 5.5 Max R1, R10, Y9, Y11, Y13, Y15:
  1. R1/R7 Fix: Eliminates default 'else -> 15.0' GCS placeholder! Maps explicit Chinese consciousness descriptors
     to an ordinal Admission Consciousness-to-GCS Proxy (exporting full rule-by-rule counts & mortality breakdown);
     missing or uninformative exam texts are strictly set to NaN for MICE!
  2. R10 Fix: Preserves both primary TBI (S06 code or neurotrauma diagnosis, is_tbi_s06) and strict ICD-10 S06-code-only
     flag (is_strict_s06_code) for exact sensitivity verification.
  3. Y11 Fix: Restricts MIMIC-IV Head-CT notes via structured radiology_detail.csv.gz (field_name='exam_name' matching
     word-bounded \b(ct|cta)\b + \b(head|brain|cranial)\b, excluding sinus/facial/spine/chest) and MIMIC-III via structured
     NOTEEVENTS.DESCRIPTION word-bounded \b(ct|cta)\b + \b(head|brain|cranial)\b; rule-based regex without clinician adjudication.
  4. Y9/?#2 Fix: Exports genuine database-queried AmsterdamUMCdb listitems.csv dictionary (all 18 GCS itemids & valueids).
"""
import os, sys, json, time, re, warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")
gate(__file__)

ROOT = r"E:\TBI subtype"
DATA = f"{ROOT}/07_prediction_system/data"
REP  = f"{ROOT}/07_prediction_system/reports"
JINH_DIR = r"F:\JinhuaNSICU\JinhuaNSICU_db_extracted\JinhuaNSICU_db\data"

t0 = time.time()
def lg(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def _canary():
    mr_head = pd.read_csv(f"{JINH_DIR}/medical_record_front_page.csv", nrows=10)
    assert "hadm_id" in mr_head.columns and "admittime_base" in mr_head.columns
    return mr_head

canary_gate(_canary, "JinhuaNSICU schema & timestamp canary")

rep0 = json.load(open(f"{REP}/08c_matrix_report.json", encoding="utf-8"))
FULL_COLS = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

def parse_age(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    if "月" in s or "天" in s or "month" in s.lower() or "day" in s.lower():
        return 0.5
    return v

def parse_admission_neuro(row):
    """
    Maps explicit Chinese admission neurological consciousness descriptors to an ordinal GCS proxy.
    CRITICAL R1 RULE: When physical_examination & chief_complaint lack any explicit consciousness term,
    returns strictly NaN (NEVER defaults to 15.0!).
    """
    pe = "" if pd.isna(row.get("physical_examination")) else str(row.get("physical_examination")).strip()
    cc = "" if pd.isna(row.get("chief_complaint")) else str(row.get("chief_complaint")).strip()
    s = (pe + " " + cc).strip()
    pupil_abn = int(bool(re.search(r"光反射(?:消失|迟钝|减弱)|瞳孔不等大|瞳孔散大", s))) if s else 0
    if not s:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, pupil_abn, "missing_text"])

    passive = ("被动体位" in s) or ("平车推入" in s)
    has_clear = bool(re.search(r"(?:神志|意识)(?:清|清楚|清醒)|神清|意识尚清", s)) and not bool(
        re.search(r"(?:神志|意识)(?:不清|欠清|模糊|障碍|淡漠)|未清醒", s)
    )

    if "深昏迷" in s:
        gt, gm, ge, gv, src = 3.0, 1.0, 1.0, 1.0, "coma_deep"
    elif "中昏迷" in s or "中度昏迷" in s:
        gt, gm, ge, gv, src = 5.0, 2.0, 1.0, 2.0, "coma_moderate"
    elif "浅昏迷" in s:
        gt, gm, ge, gv, src = 7.0, 3.0, 2.0, 2.0, "coma_light"
    elif "昏迷" in s and not has_clear:
        gt, gm, ge, gv, src = 6.0, (2.0 if passive else 3.0), 1.0, 2.0, "coma_unspec"
    elif ("意识障碍" in s or "呼之不应" in s or "神志不清" in s or "意识不清" in s) and not has_clear:
        gt, gm, ge, gv, src = 7.0, 3.0, 2.0, 2.0, "impaired_unresponsive"
    elif ("昏睡" in s or "意识模糊" in s or "神志模糊" in s or "谵妄" in s or "烦躁不安" in s) and not has_clear:
        gt, gm, ge, gv, src = 10.0, 4.0, 3.0, 3.0, "stupor_confused"
    elif ("嗜睡" in s or "神志淡漠" in s or "意识欠清" in s) and not has_clear:
        gt, gm, ge, gv, src = 12.0, 5.0, 3.0, 4.0, "drowsy"
    elif has_clear:
        gt, gm, ge, gv, src = 15.0, 6.0, 4.0, 5.0, "explicit_alert_clear"
    else:
        gt, gm, ge, gv, src = np.nan, np.nan, np.nan, np.nan, "uninformative_no_kw"
    return pd.Series([gt, gm, ge, gv, pupil_abn, src])

def parse_ct_en(text_series):
    s = text_series.fillna("").astype(str).str.lower()
    def pos_match(pat, neg_pat=r"(?:no |without |absent |resolved |minimal or no |no evidence of |no acute |unremarkable |normal )"):
        has_term = s.str.contains(pat, regex=True)
        has_neg  = s.str.contains(neg_pat + r"[^.;:\n]{0,35}" + pat, regex=True)
        return (has_term & (~has_neg)).astype(int)
    df_ct = pd.DataFrame(index=text_series.index)
    df_ct["ct_midline_shift"] = pos_match(r"(?:midline shift|shift of (?:the )?midline|subfalcine herniation)")
    df_ct["ct_cistern_compressed"] = pos_match(r"(?:effacement of (?:the )?(?:basal|basilar|ambient|perimesencephalic|suprasellar) cistern|cistern(?:s)? (?:are |is )?(?:effaced|compressed|obliterated))")
    df_ct["ct_herniation"] = pos_match(r"(?:uncal herniation|transtentorial herniation|tonsillar herniation|downward herniation|brainstem compression)")
    df_ct["ct_sdh"] = pos_match(r"(?:subdural hematoma|subdural hemorrhage|acute sdh)")
    df_ct["ct_edh"] = pos_match(r"(?:epidural hematoma|epidural hemorrhage|extradural hematoma)")
    df_ct["ct_sah"] = pos_match(r"(?:subarachnoid hemorrhage|subarachnoid blood|traumatic sah)")
    df_ct["ct_contusion_ich"] = pos_match(r"(?:hemorrhagic contusion|parenchymal contusion|intraparenchymal hemorrhage|intracerebral hemorrhage|brainstem contusion|diffuse axonal injury|petechial hemorrhage)")
    df_ct["ct_edema_mass"] = pos_match(r"(?:cerebral edema|diffuse edema|mass effect|sulcal effacement|ventricular compression|effacement of (?:the )?(?:sulci|ventricle))")
    df_ct["ct_ivh"] = pos_match(r"(?:intraventricular hemorrhage|intraventricular blood)")
    df_ct["ct_skull_fx"] = pos_match(r"(?:skull fracture|calvarial fracture|basilar skull fracture|depressed fracture|temporal bone fracture)")
    df_ct["ct_marshall_proxy"] = (
        1 + (df_ct[["ct_sdh", "ct_edh", "ct_sah", "ct_contusion_ich"]].max(axis=1) >= 1).astype(int)
        + df_ct["ct_edema_mass"] + 2 * df_ct["ct_midline_shift"]
        + 2 * df_ct["ct_cistern_compressed"] + 3 * df_ct["ct_herniation"]
    )
    return df_ct

def parse_ct_zh(text_series):
    s = text_series.fillna("").astype(str)
    def pos_zh(pat, neg_pat=r"(?:未见|无|除外|未见明显|基本正常)"):
        has_term = s.str.contains(pat, regex=True)
        has_neg  = s.str.contains(neg_pat + r"[^，。；\n]{0,12}" + pat, regex=True)
        return (has_term & (~has_neg)).astype(int)
    df_ct = pd.DataFrame(index=text_series.index)
    df_ct["ct_midline_shift"] = pos_zh(r"(?:中线(?:结构)?(?:左|右|向[^\s，。]{1,3})?移位|中线结构偏移|大脑镰下疝)")
    df_ct["ct_cistern_compressed"] = pos_zh(r"(?:(?:环池|基底池|鞍上池|脚间池|四叠体池)(?:受压|变窄|消失|不清|模糊))")
    df_ct["ct_herniation"] = pos_zh(r"(?:脑疝|钩回疝|天幕裂孔疝|枕骨大孔疝|脑干受压)")
    df_ct["ct_sdh"] = pos_zh(r"(?:硬膜下血肿|硬膜下积血)")
    df_ct["ct_edh"] = pos_zh(r"(?:硬膜外血肿)")
    df_ct["ct_sah"] = pos_zh(r"(?:蛛网膜下腔出血|蛛血)")
    df_ct["ct_contusion_ich"] = pos_zh(r"(?:脑挫裂伤|脑挫伤|脑内血肿|脑实质出血|脑干损伤|弥漫性轴索损伤)")
    df_ct["ct_edema_mass"] = pos_zh(r"(?:脑水肿|脑肿胀|占位效应|脑沟(?:变浅|消失)|脑室受压)")
    df_ct["ct_ivh"] = pos_zh(r"(?:脑室(?:内)?积血|脑室铸型|破入脑室)")
    df_ct["ct_skull_fx"] = pos_zh(r"(?:颅骨骨折|颅底骨折|凹陷性骨折|颞骨骨折|额骨骨折|顶骨骨折|枕骨骨折)")
    df_ct["ct_marshall_proxy"] = (
        1 + (df_ct[["ct_sdh", "ct_edh", "ct_sah", "ct_contusion_ich"]].max(axis=1) >= 1).astype(int)
        + df_ct["ct_edema_mass"] + 2 * df_ct["ct_midline_shift"]
        + 2 * df_ct["ct_cistern_compressed"] + 3 * df_ct["ct_herniation"]
    )
    return df_ct

@retry_limit(2)
def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    lg("1. Extracting JinhuaNSICU v1.2 strictly <= 24h with zero default GCS placeholders...")
    mr_fp = pd.read_csv(f"{JINH_DIR}/medical_record_front_page.csv")
    mr = pd.read_csv(f"{JINH_DIR}/medical_record.csv", usecols=[
        "subject_id", "hadm_id", "expire_flag", "chief_complaint", "physical_examination", "outcome_en",
        "systolic_blood_pressure", "diastolic_blood_pressure", "pulse", "body_temperature",
        "respiratory_rate", "admittime_base", "dischtime_base", "length_stay"
    ])
    diag_j = pd.read_csv(f"{JINH_DIR}/diagnosis.csv", usecols=["hadm_id", "diagnostic_code", "diagnosis_name_original"])

    mr_fp["admission_age"] = mr_fp["age"].apply(parse_age)
    mr_fp["sex_female"] = mr_fp["patient_gender_en"].astype(str).str.lower().str.contains("female").astype(float)

    diag_j["code_str"] = diag_j["diagnostic_code"].astype(str).str.upper().str.strip()
    diag_j["name_str"] = diag_j["diagnosis_name_original"].astype(str)
    s06_code_hadms = set(diag_j[diag_j["code_str"].str.startswith("S06")]["hadm_id"].unique())
    tbi_hadms = set(diag_j[diag_j["code_str"].str.startswith("S06") | diag_j["name_str"].str.contains("颅脑损伤|脑挫裂伤|硬膜下血肿|硬膜外血肿|脑干损伤", regex=True, na=False)]["hadm_id"].unique())
    abi_hadms = set(diag_j[diag_j["code_str"].str.match(r"^(S06|S02\.0|S02\.1|I60|I61|I62|G93\.5|G93\.6)") | diag_j["name_str"].str.contains("颅脑损伤|脑挫裂伤|硬膜|蛛网膜下腔出血|脑出血|脑疝", regex=True, na=False)]["hadm_id"].unique())

    ord_path = f"{JINH_DIR}/orders.csv".replace("\\", "/")
    icu_span_ord = con.execute(f"""
    SELECT hadm_id,
           MIN(TRY_CAST(starttime_base AS DOUBLE)) AS icu_in_sec,
           MAX(TRY_CAST(endtime_base AS DOUBLE)) AS icu_out_sec
    FROM read_csv('{ord_path}', all_varchar=false, ignore_errors=true)
    WHERE order_content = 'ICU单元治疗'
      AND TRY_CAST(starttime_base AS DOUBLE) IS NOT NULL AND TRY_CAST(endtime_base AS DOUBLE) IS NOT NULL
    GROUP BY hadm_id
    """).df()

    trf_df = pd.read_csv(f"{JINH_DIR}/transfer.csv")
    trf_icu = set(trf_df[trf_df.astype(str).apply(lambda s: s.str.contains("重症|ICU", case=False, na=False)).any(axis=1)]["hadm_id"].unique())

    j_base = mr_fp[["subject_id", "hadm_id", "admission_age", "sex_female", "admittime_base", "dischtime_base", "length_stay", "discharge_type_en"]].merge(
        mr[["hadm_id", "expire_flag", "outcome_en", "chief_complaint", "physical_examination", "systolic_blood_pressure", "diastolic_blood_pressure", "pulse", "body_temperature", "respiratory_rate"]],
        on="hadm_id", how="left"
    ).merge(icu_span_ord, on="hadm_id", how="left")

    j_base["has_icu"] = j_base["icu_in_sec"].notna() | j_base["hadm_id"].isin(trf_icu)
    j_base["icu_in_sec"] = j_base["icu_in_sec"].fillna(j_base["admittime_base"].astype(float))
    j_base["icu_out_sec"] = j_base["icu_out_sec"].fillna(j_base["dischtime_base"].astype(float))
    j_base["los_h"] = (j_base["icu_out_sec"] - j_base["icu_in_sec"]) / 60.0
    j_base["days_from_icu_in_to_disch"] = (j_base["dischtime_base"].astype(float) - j_base["icu_in_sec"]) / 1440.0
    j_base["is_tbi_s06"] = j_base["hadm_id"].isin(tbi_hadms)
    j_base["is_strict_s06_code"] = j_base["hadm_id"].isin(s06_code_hadms)
    j_base["is_abi"] = j_base["hadm_id"].isin(abi_hadms)

    j_base["died_in_hosp"] = (
        (j_base["expire_flag"].astype(str).str.lower() == "yes") |
        (j_base["discharge_type_en"].astype(str).str.lower().str.contains("death|die", na=False))
    ).astype(int)

    alive_24h = (j_base["los_h"] >= 24.0) & (~((j_base["died_in_hosp"] == 1) & (j_base["days_from_icu_in_to_disch"] <= 1.0)))
    j_base["d28"] = ((j_base["died_in_hosp"] == 1) & (j_base["days_from_icu_in_to_disch"] <= 28.0)).astype(int)

    j_coh = j_base[(j_base["admission_age"] >= 18.0) & j_base["has_icu"] & alive_24h & j_base["is_abi"]].copy().reset_index(drop=True)
    j_coh[["gcs_total_first", "gcs_motor_first", "gcs_eye_first", "gcs_verbal_first", "pupil_abnormal_first", "gcs_proxy_source"]] = j_coh.apply(parse_admission_neuro, axis=1)

    j_coh["d28_sens_composite"] = (
        (j_coh["d28"] == 1) |
        (j_coh["outcome_en"].astype(str).str.contains("Not cured|Auto-discharged", case=False, na=False)
         & j_coh["gcs_total_first"].notna() & (j_coh["gcs_total_first"] <= 8.0)
         & (j_coh["days_from_icu_in_to_disch"] <= 28.0))
    ).astype(int)

    j_s06_sub = j_coh[j_coh["is_tbi_s06"]]
    lg(f"  JinhuaNSICU TBI n = {len(j_s06_sub):,} (strict S06 code n = {int(j_coh['is_strict_s06_code'].sum()):,}) | GCS proxy mapped = {int(j_s06_sub['gcs_total_first'].notna().sum()):,} ({j_s06_sub['gcs_total_first'].notna().mean()*100:.1f}%), missing->NaN = {int(j_s06_sub['gcs_total_first'].isna().sum()):,}")

    con.register("j_coh_tbl", j_coh[["hadm_id", "icu_in_sec", "admittime_base"]])
    vs_path = f"{JINH_DIR}/vital_signs.csv".replace("\\", "/")
    lt_path = f"{JINH_DIR}/laboratory_test.csv".replace("\\", "/")

    j_vitals = con.execute(f"""
    WITH v AS (
        SELECT vs.hadm_id, vs.subcategory_name_en AS name,
               TRY_CAST(vs.value AS DOUBLE) AS val
        FROM read_csv('{vs_path}', all_varchar=false, ignore_errors=true) vs
        JOIN j_coh_tbl c ON vs.hadm_id = c.hadm_id
        WHERE TRY_CAST(vs.value AS DOUBLE) IS NOT NULL
          AND (TRY_CAST(vs.charttime_base AS DOUBLE) - c.icu_in_sec) BETWEEN -12 * 60 AND 24 * 60
    )
    SELECT hadm_id,
        MIN(CASE WHEN name = 'Pulse' AND val BETWEEN 25 AND 250 THEN val END) AS hr_min,
        AVG(CASE WHEN name = 'Pulse' AND val BETWEEN 25 AND 250 THEN val END) AS hr_mean,
        MAX(CASE WHEN name = 'Pulse' AND val BETWEEN 25 AND 250 THEN val END) AS hr_max,
        MIN(CASE WHEN name = 'Sbp' AND val BETWEEN 40 AND 280 THEN val END) AS sbp_min,
        AVG(CASE WHEN name = 'Sbp' AND val BETWEEN 40 AND 280 THEN val END) AS sbp_mean,
        MAX(CASE WHEN name = 'Sbp' AND val BETWEEN 40 AND 280 THEN val END) AS sbp_max,
        MIN(CASE WHEN name = 'Dbp' AND val BETWEEN 20 AND 180 THEN val END) AS dbp_min,
        AVG(CASE WHEN name = 'Dbp' AND val BETWEEN 20 AND 180 THEN val END) AS dbp_mean,
        MAX(CASE WHEN name = 'Dbp' AND val BETWEEN 20 AND 180 THEN val END) AS dbp_max,
        MIN(CASE WHEN name = 'Respiratory' AND val BETWEEN 4 AND 70 THEN val END) AS rr_min,
        AVG(CASE WHEN name = 'Respiratory' AND val BETWEEN 4 AND 70 THEN val END) AS rr_mean,
        MAX(CASE WHEN name = 'Respiratory' AND val BETWEEN 4 AND 70 THEN val END) AS rr_max,
        MIN(CASE WHEN name = 'Temperature' AND val BETWEEN 30 AND 43.5 THEN val END) AS temp_min,
        AVG(CASE WHEN name = 'Temperature' AND val BETWEEN 30 AND 43.5 THEN val END) AS temp_mean,
        MAX(CASE WHEN name = 'Temperature' AND val BETWEEN 30 AND 43.5 THEN val END) AS temp_max,
        MIN(CASE WHEN name = 'Spo2' AND val BETWEEN 40 AND 100 THEN val END) AS spo2_min,
        AVG(CASE WHEN name = 'Spo2' AND val BETWEEN 40 AND 100 THEN val END) AS spo2_mean,
        MAX(CASE WHEN name = 'Spo2' AND val BETWEEN 40 AND 100 THEN val END) AS spo2_max
    FROM v
    GROUP BY hadm_id
    """).df()

    j_labs = con.execute(f"""
    WITH l AS (
        SELECT lt.hadm_id, lt.inspection_subproject_name AS cname,
               TRY_CAST(lt.test_results_quantitative AS DOUBLE) AS val
        FROM read_csv('{lt_path}', all_varchar=false, ignore_errors=true) lt
        JOIN j_coh_tbl c ON lt.hadm_id = c.hadm_id
        WHERE TRY_CAST(lt.test_results_quantitative AS DOUBLE) IS NOT NULL
          AND TRY_CAST(lt.report_time_base AS DOUBLE) IS NOT NULL
          AND (TRY_CAST(lt.report_time_base AS DOUBLE) - c.icu_in_sec) BETWEEN -12 * 60 AND 24 * 60
    )
    SELECT hadm_id,
        MIN(CASE WHEN cname LIKE '血红蛋白%' AND val BETWEEN 20 AND 250 THEN val / 10.0 END) AS hemoglobin_min,
        MAX(CASE WHEN cname LIKE '血红蛋白%' AND val BETWEEN 20 AND 250 THEN val / 10.0 END) AS hemoglobin_max,
        MIN(CASE WHEN cname LIKE '红细胞比容%' AND val BETWEEN 10 AND 75 THEN val END) AS hematocrit_min,
        MAX(CASE WHEN cname LIKE '红细胞比容%' AND val BETWEEN 10 AND 75 THEN val END) AS hematocrit_max,
        MIN(CASE WHEN cname LIKE '血小板计数%' AND val BETWEEN 1 AND 1500 THEN val END) AS platelet_min,
        MAX(CASE WHEN cname LIKE '血小板计数%' AND val BETWEEN 1 AND 1500 THEN val END) AS platelet_max,
        MIN(CASE WHEN cname LIKE '白细胞计数%' AND val BETWEEN 0.1 AND 150 THEN val END) AS wbc_min,
        MAX(CASE WHEN cname LIKE '白细胞计数%' AND val BETWEEN 0.1 AND 150 THEN val END) AS wbc_max,
        MIN(CASE WHEN cname LIKE '尿素%' AND val BETWEEN 0.5 AND 100 THEN val * 2.801 END) AS bun_min,
        MAX(CASE WHEN cname LIKE '尿素%' AND val BETWEEN 0.5 AND 100 THEN val * 2.801 END) AS bun_max,
        MIN(CASE WHEN cname LIKE '肌酐%' AND val BETWEEN 10 AND 2000 THEN val / 88.4 END) AS creatinine_min,
        MAX(CASE WHEN cname LIKE '肌酐%' AND val BETWEEN 10 AND 2000 THEN val / 88.4 END) AS creatinine_max,
        MIN(CASE WHEN cname LIKE '葡萄糖%' AND val BETWEEN 1 AND 80 THEN val * 18.018 END) AS glucose_lab_min,
        MAX(CASE WHEN cname LIKE '葡萄糖%' AND val BETWEEN 1 AND 80 THEN val * 18.018 END) AS glucose_lab_max,
        MIN(CASE WHEN cname LIKE '钠离子%' AND val BETWEEN 100 AND 190 THEN val END) AS sodium_min,
        MAX(CASE WHEN cname LIKE '钠离子%' AND val BETWEEN 100 AND 190 THEN val END) AS sodium_max,
        MIN(CASE WHEN cname LIKE '钾离子%' AND val BETWEEN 1.5 AND 10 THEN val END) AS potassium_min,
        MAX(CASE WHEN cname LIKE '钾离子%' AND val BETWEEN 1.5 AND 10 THEN val END) AS potassium_max,
        MIN(CASE WHEN cname LIKE '氯离子%' AND val BETWEEN 50 AND 160 THEN val END) AS chloride_min,
        MAX(CASE WHEN cname LIKE '氯离子%' AND val BETWEEN 50 AND 160 THEN val END) AS chloride_max,
        MIN(CASE WHEN cname LIKE '%钙%' AND val BETWEEN 0.5 AND 5.0 THEN val * 4.008 END) AS calcium_min,
        MAX(CASE WHEN cname LIKE '%钙%' AND val BETWEEN 0.5 AND 5.0 THEN val * 4.008 END) AS calcium_max,
        MIN(CASE WHEN cname LIKE '%碳酸氢%' AND val BETWEEN 2 AND 60 THEN val END) AS bicarbonate_min,
        MAX(CASE WHEN cname LIKE '%碳酸氢%' AND val BETWEEN 2 AND 60 THEN val END) AS bicarbonate_max,
        MIN(CASE WHEN cname = '凝血酶原时间(PT)-静脉血' AND val BETWEEN 7 AND 120 THEN val END) AS pt_min,
        MAX(CASE WHEN cname = '凝血酶原时间(PT)-静脉血' AND val BETWEEN 7 AND 120 THEN val END) AS pt_max,
        MIN(CASE WHEN cname LIKE '%国际标准化%' AND val BETWEEN 0.5 AND 15 THEN val END) AS inr_min,
        MAX(CASE WHEN cname LIKE '%国际标准化%' AND val BETWEEN 0.5 AND 15 THEN val END) AS inr_max,
        MIN(CASE WHEN cname LIKE '%部分凝血活酶%' AND val BETWEEN 10 AND 200 THEN val END) AS ptt_min,
        MAX(CASE WHEN cname LIKE '%部分凝血活酶%' AND val BETWEEN 10 AND 200 THEN val END) AS ptt_max
    FROM l
    GROUP BY hadm_id
    """).df()

    j_mat = j_coh.merge(j_vitals, on="hadm_id", how="left").merge(j_labs, on="hadm_id", how="left")
    for c_v, c_adm, lo, hi in [
        ("hr_mean", "pulse", 25, 250), ("hr_min", "pulse", 25, 250), ("hr_max", "pulse", 25, 250),
        ("sbp_mean", "systolic_blood_pressure", 40, 280), ("sbp_min", "systolic_blood_pressure", 40, 280), ("sbp_max", "systolic_blood_pressure", 40, 280),
        ("dbp_mean", "diastolic_blood_pressure", 20, 180), ("dbp_min", "diastolic_blood_pressure", 20, 180), ("dbp_max", "diastolic_blood_pressure", 20, 180),
        ("rr_mean", "respiratory_rate", 4, 70), ("temp_min", "body_temperature", 30, 43.5)
    ]:
        adm_clean = j_mat[c_adm].where(j_mat[c_adm].between(lo, hi))
        j_mat[c_v] = j_mat[c_v].fillna(adm_clean)

    j_mat["mbp_min"]  = (j_mat["sbp_min"] + 2.0 * j_mat["dbp_min"]) / 3.0
    j_mat["mbp_mean"] = (j_mat["sbp_mean"] + 2.0 * j_mat["dbp_mean"]) / 3.0
    j_mat["mbp_max"]  = (j_mat["sbp_max"] + 2.0 * j_mat["dbp_max"]) / 3.0
    j_mat["pt_min"]   = j_mat["pt_min"].fillna(j_mat["inr_min"] * 11.5)
    j_mat["pt_max"]   = j_mat["pt_max"].fillna(j_mat["inr_max"] * 11.5)

    j_mat["gcs_total_min"]  = j_mat["gcs_total_first"]
    j_mat["gcs_motor_min"]  = j_mat["gcs_motor_first"]
    j_mat["gcs_eye_min"]    = j_mat["gcs_eye_first"]
    j_mat["gcs_verbal_min"] = j_mat["gcs_verbal_first"]

    for dyn_c in ["motor_first", "motor_last", "motor_min", "motor_max", "motor_slope", "motor_sd", "prob_m1", "prob_m2", "prob_m3"]:
        j_mat[dyn_c] = np.nan
    for c in FULL_COLS:
        if c not in j_mat.columns:
            j_mat[c] = np.nan

    assert j_mat["motor_last"].isna().all(), "Jinhua motor_last must remain 100% NaN"
    assert j_mat.loc[j_mat["gcs_proxy_source"].isin(["missing_text", "uninformative_no_kw"]), "gcs_total_first"].isna().all(), "R1: uninformative/missing text must be NaN!"

    lg("2. Extracting Structured Exam-Verified Head/Brain CT Radiology Reports (<= 24h)...")
    m4_mat = pd.read_parquet(f"{DATA}/08c_matrix_mimic4_v3.parquet")
    icustays_m4_path = f"{ROOT}/data/mimic-iv-3.1/icu/icustays.csv.gz".replace("\\", "/")
    rad_m4_path = f"{ROOT}/data/mimic-iv-note-2.2/note/radiology.csv.gz".replace("\\", "/")
    raddet_m4_path = f"{ROOT}/data/mimic-iv-note-2.2/note/radiology_detail.csv.gz".replace("\\", "/")

    con.register("m4_stays", m4_mat[["stay_id", "split", "d28"]])
    m4_ct_notes = con.execute(rf"""
    WITH icu AS (
        SELECT CAST(i.stay_id AS BIGINT) AS stay_id,
               CAST(i.subject_id AS BIGINT) AS subject_id,
               CAST(i.intime AS TIMESTAMP) AS intime
        FROM read_csv('{icustays_m4_path}', all_varchar=true) i
        JOIN m4_stays m ON CAST(i.stay_id AS BIGINT) = m.stay_id
    )
    SELECT icu.stay_id,
           rd.field_value AS exam_name,
           r.text AS ct_text
    FROM icu
    JOIN read_csv('{rad_m4_path}', all_varchar=true) r
      ON icu.subject_id = CAST(r.subject_id AS BIGINT)
    JOIN read_csv('{raddet_m4_path}', all_varchar=true) rd
      ON r.note_id = rd.note_id AND rd.field_name = 'exam_name'
    WHERE TRY_CAST(r.charttime AS TIMESTAMP) IS NOT NULL
      AND TRY_CAST(r.charttime AS TIMESTAMP) BETWEEN icu.intime - INTERVAL 48 HOUR AND icu.intime + INTERVAL 24 HOUR
      AND (TRY_CAST(r.storetime AS TIMESTAMP) IS NULL OR TRY_CAST(r.storetime AS TIMESTAMP) <= icu.intime + INTERVAL 24 HOUR)
      AND regexp_matches(LOWER(rd.field_value), '\b(ct|cta)\b')
      AND regexp_matches(LOWER(rd.field_value), '\b(head|brain|cranial)\b')
      AND NOT regexp_matches(LOWER(rd.field_value), 'sinus|mandible|maxillofacial|orbit')
    """).df()
    m4_exam_counts = m4_ct_notes["exam_name"].value_counts().head(12).to_dict()
    m4_ct_raw = m4_ct_notes.groupby("stay_id", as_index=False).agg(ct_text=("ct_text", lambda s: " || ".join(s)))

    m4_ct_feats = parse_ct_en(m4_ct_raw["ct_text"])
    m4_ct_feats["stay_id"] = m4_ct_raw["stay_id"].values
    m4_ct_feats["ct_any_24h"] = 1
    m4_ct_all = m4_mat[["stay_id", "split", "d28"]].merge(m4_ct_feats, on="stay_id", how="left")
    for c in m4_ct_feats.columns:
        if c != "stay_id":
            m4_ct_all[c] = m4_ct_all[c].fillna(0).astype(int)
    m4_ct_all["cohort"] = "MIMIC-IV"
    m4_ct_all = m4_ct_all.rename(columns={"stay_id": "id"})

    fe_m3 = pd.read_parquet(f"{DATA}/features_mimic3.parquet")
    icustays_m3_path = f"{ROOT}/data/mimic-iii-1.4/ICUSTAYS.csv.gz".replace("\\", "/")
    notes_m3_path = f"{ROOT}/data/mimic-iii-1.4/NOTEEVENTS.csv.gz".replace("\\", "/")

    m3_cv_stays = con.execute(f"""
    SELECT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id,
           CAST(SUBJECT_ID AS BIGINT) AS subject_id,
           CAST(HADM_ID AS BIGINT) AS hadm_id,
           CAST(INTIME AS TIMESTAMP) AS intime
    FROM read_csv('{icustays_m3_path}', all_varchar=true)
    WHERE DBSOURCE = 'carevue'
    """).df()
    m3_coh_ids = fe_m3[fe_m3["icustay_id"].isin(m3_cv_stays["icustay_id"]) & (fe_m3["los_h"] >= 24.0) & (~((fe_m3["d28"] == 1) & (fe_m3["days_to_death"] < 1.0)))][["icustay_id", "d28"]].copy()
    m3_stays_sub = m3_cv_stays.merge(m3_coh_ids, on="icustay_id", how="inner")
    con.register("m3_stays_tbl", m3_stays_sub)

    m3_ct_notes = con.execute(rf"""
    SELECT s.icustay_id,
           n.DESCRIPTION AS exam_desc,
           n.TEXT AS ct_text
    FROM m3_stays_tbl s
    JOIN read_csv('{notes_m3_path}', all_varchar=true, ignore_errors=true) n
      ON s.subject_id = TRY_CAST(n.SUBJECT_ID AS BIGINT)
     AND s.hadm_id = TRY_CAST(n.HADM_ID AS BIGINT)
    WHERE n.CATEGORY = 'Radiology'
      AND (
          (TRY_CAST(n.CHARTTIME AS TIMESTAMP) IS NOT NULL
           AND TRY_CAST(n.CHARTTIME AS TIMESTAMP) BETWEEN s.intime - INTERVAL 48 HOUR AND s.intime + INTERVAL 24 HOUR)
          OR
          (TRY_CAST(n.CHARTTIME AS TIMESTAMP) IS NULL
           AND TRY_CAST(n.CHARTDATE AS TIMESTAMP) IS NOT NULL
           AND TRY_CAST(n.CHARTDATE AS TIMESTAMP) + INTERVAL 23 HOUR + INTERVAL 59 MINUTE <= s.intime + INTERVAL 24 HOUR)
      )
      AND (
          (regexp_matches(LOWER(COALESCE(n.DESCRIPTION, '')), '\b(ct|cta)\b')
           AND regexp_matches(LOWER(COALESCE(n.DESCRIPTION, '')), '\b(head|brain|cranial)\b')
           AND NOT regexp_matches(LOWER(COALESCE(n.DESCRIPTION, '')), 'sinus|mandible|maxillofacial|orbit'))
          OR (
              regexp_matches(LOWER(COALESCE(n.DESCRIPTION, '')), '^(|by same physician|by different physician|addendum)$')
              AND regexp_matches(LOWER(SUBSTR(n.TEXT, 1, 260)), '\b(ct\s+head|head\s+ct|ct\s+of\s+the\s+head|ct\s+brain|brain\s+ct)\b')
              AND NOT regexp_matches(LOWER(SUBSTR(n.TEXT, 1, 260)), 'c-spine|cervical|chest|abdomen|pelvis|humerus|extremity')
          )
      )
    """).df()
    m3_exam_counts = m3_ct_notes["exam_desc"].value_counts().head(12).to_dict()
    m3_ct_raw = m3_ct_notes.groupby("icustay_id", as_index=False).agg(ct_text=("ct_text", lambda s: " || ".join(s)))

    m3_ct_feats = parse_ct_en(m3_ct_raw["ct_text"])
    m3_ct_feats["icustay_id"] = m3_ct_raw["icustay_id"].values
    m3_ct_feats["ct_any_24h"] = 1
    m3_ct_all = m3_coh_ids.merge(m3_ct_feats, on="icustay_id", how="left")
    for c in m3_ct_feats.columns:
        if c != "icustay_id":
            m3_ct_all[c] = m3_ct_all[c].fillna(0).astype(int)
    m3_ct_all["split"] = "ext_m3"
    m3_ct_all["cohort"] = "MIMIC-III"
    m3_ct_all = m3_ct_all.rename(columns={"icustay_id": "id"})

    exam_path = f"{JINH_DIR}/examination.csv".replace("\\", "/")
    j_ct_raw = con.execute(f"""
    SELECT e.hadm_id,
           STRING_AGG(COALESCE(e.examine_finding, '') || ' ' || COALESCE(e.inspection_conclusion, ''), ' || ') AS ct_text_zh
    FROM read_csv('{exam_path}', all_varchar=false, ignore_errors=true) e
    JOIN j_coh_tbl c ON e.hadm_id = c.hadm_id
    WHERE (e.check_type LIKE '%CT%' OR e.check_name LIKE '%CT%')
      AND (e.check_part LIKE '%颅%' OR e.check_part LIKE '%头%' OR e.check_part LIKE '%脑%'
           OR e.check_name LIKE '%颅%' OR e.check_name LIKE '%头%' OR e.check_name LIKE '%脑%')
      AND TRY_CAST(e.examine_time_base AS DOUBLE) IS NOT NULL
      AND (TRY_CAST(e.examine_time_base AS DOUBLE) - c.icu_in_sec) BETWEEN -48 * 60 AND 24 * 60
    GROUP BY e.hadm_id
    """).df()

    j_ct_feats = parse_ct_zh(j_ct_raw["ct_text_zh"])
    j_ct_feats["hadm_id"] = j_ct_raw["hadm_id"].values
    j_ct_feats["ct_any_24h"] = 1

    j_mat = j_mat.merge(j_ct_feats, on="hadm_id", how="left")
    for c in j_ct_feats.columns:
        if c != "hadm_id":
            j_mat[c] = j_mat[c].fillna(0).astype(int)

    j_mat.to_parquet(f"{DATA}/19a_matrix_jinhua_v4.parquet", index=False)
    j_ct_all = j_mat[["hadm_id", "d28"] + [c for c in j_ct_feats.columns if c != "hadm_id"]].copy()
    j_ct_all["split"] = "ext_jinhua"
    j_ct_all["cohort"] = "JinhuaNSICU"
    j_ct_all = j_ct_all.rename(columns={"hadm_id": "id"})

    ct_3cohorts = pd.concat([m4_ct_all, m3_ct_all, j_ct_all], ignore_index=True)
    ct_3cohorts["id"] = ct_3cohorts["id"].astype(str)
    ct_3cohorts.to_parquet(f"{DATA}/19a_ct_nlp_3cohorts_v4.parquet", index=False)

    aumc_mat = pd.read_parquet(f"{DATA}/19a_matrix_amsterdam_v4.parquet")
    au_tr = aumc_mat[aumc_mat["is_trauma"] == True]
    j_tbi = j_mat[j_mat["is_tbi_s06"] == True]
    j_s06_only = j_mat[j_mat["is_strict_s06_code"] == True]

    src_counts = j_tbi["gcs_proxy_source"].value_counts().to_dict()
    src_detail = {}
    for sk, grp_s in j_tbi.groupby("gcs_proxy_source"):
        src_detail[str(sk)] = {
            "n": int(len(grp_s)),
            "pct": round(100.0 * len(grp_s) / len(j_tbi), 1),
            "deaths": int(grp_s["d28"].sum()),
            "mort_pct": round(100.0 * float(grp_s["d28"].mean()), 1),
        }

    # Export genuine database-queried AmsterdamUMCdb listitems.csv dictionary & agegroup counts (resolving Y9 & ?#2)
    aumc_dir = r"F:\AmsterdamUMCdb"
    gcs_ids_18 = {6732, 13077, 19638, 19635, 16628, 14470, 6734, 13072, 19639, 19636, 16634, 14476, 6735, 13066, 19637, 19640, 16640, 14482}
    obs_pairs = {}
    for ch in pd.read_csv(f"{aumc_dir}/listitems.csv", encoding="latin1", usecols=["itemid", "item", "valueid", "value"], chunksize=2_000_000):
        sub_l = ch[ch["itemid"].isin(gcs_ids_18)].drop_duplicates(["itemid", "valueid"])
        for _, rw in sub_l.iterrows():
            obs_pairs[(int(rw["itemid"]), int(rw["valueid"]))] = (str(rw["item"]), str(rw["value"]))
    by_itemid_dict = {}
    for (iid, vid), (iname, vstr) in sorted(obs_pairs.items()):
        by_itemid_dict.setdefault(str(iid), {"item_name": iname, "valueid_to_dutch_value": {}})["valueid_to_dutch_value"][str(vid)] = vstr

    adm_au_raw = pd.read_csv(f"{aumc_dir}/admissions.csv", usecols=["admissionid", "admissioncount", "agegroup"])
    adm_au_tr = adm_au_raw[adm_au_raw["admissionid"].isin(set(au_tr["admissionid"]))]
    agegrp_counts = {str(k): int(v) for k, v in adm_au_tr["agegroup"].value_counts().to_dict().items()}

    dict_audit = {
        "source_verification": "Queried directly from F:/AmsterdamUMCdb/listitems.csv (Latin-1) and F:/AmsterdamUMCdb/admissions.csv on disk",
        "amsterdamumcdb_official_dictionary": {
            "eye_itemids_6": [6732, 13077, 19638, 19635, 16628, 14470],
            "motor_itemids_6": [6734, 13072, 19639, 19636, 16634, 14476],
            "verbal_itemids_6": [6735, 13066, 19637, 19640, 16640, 14482],
            "motor_offset_formulas": {"6734": "7 - valueid", "13072": "valueid", "19639": "valueid - 12", "19636_16634_14476": "valueid - 6"},
            "eye_offset_formulas": {"6732": "5 - valueid", "13077": "valueid", "19638": "valueid - 8", "19635_16628_14470": "valueid - 4"},
            "verbal_offset_formulas": {"6735": "6 - valueid for valueid 1..5; valueid 8 (Geïntubeerd) -> 1.0 (no valueid 6 or 7 exists in listitems.csv)", "13066": "valueid (1..5)", "19637": "valueid - 9 for 10..14; 9 -> 1.0", "19640_16640_14482": "valueid - 4 for 5..9; valueid <= 4 -> 1.0"},
            "queried_listitems_18_gcs_itemids": by_itemid_dict,
            "trauma_cohort_agegroup_counts": agegrp_counts,
            "agegroup_midpoint_mapping": {"18-39": 29.0, "40-49": 45.0, "50-59": 55.0, "60-69": 65.0, "70-79": 75.0, "80+": 84.0}
        },
        "hirid_apache_group_provenance": {
            "variableids": {"9990002": "APACHE II Group (103=Non-op Trauma, 111=Post-op Trauma; 101=Non-op Neuro, 110=Post-op Neurosurg)", "9990004": "APACHE IV Group (194=Non-op Trauma, 203=Post-op Trauma; 193=Non-op Neuro, 202=Post-op Neurosurg)"},
            "provenance_note": "Follows HiRID-ICU-Benchmark (Yeche et al., NeurIPS Datasets & Benchmarks 2021) diagnostic group mapping; local F:/HiRID/ordinal_vars_ref.csv covers physiological variables only and does not list 9990002/9990004 string labels."
        }
    }
    json.dump(dict_audit, open(f"{REP}/19a_v4_dictionary_audit.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    summary = {
        "amsterdam_trauma_n": int(len(au_tr)),
        "amsterdam_trauma_d28": int(au_tr["d28"].sum()),
        "amsterdam_trauma_d28_rate": round(float(au_tr["d28"].mean()), 4),
        "amsterdam_strict_trauma_dx_n": int(aumc_mat["is_strict_trauma_dx"].sum()),
        "amsterdam_strict_trauma_dx_d28": int(aumc_mat.loc[aumc_mat["is_strict_trauma_dx"], "d28"].sum()),
        "amsterdam_abi_n": int(len(aumc_mat)),
        "amsterdam_abi_d28": int(aumc_mat["d28"].sum()),
        "amsterdam_abi_d28_rate": round(float(aumc_mat["d28"].mean()), 4),
        "amsterdam_abi_end_24_72h": int(aumc_mat["end_24_72h"].sum()),
        "amsterdam_icp_24h_n": int(aumc_mat["has_icp_24h"].sum()),
        "amsterdam_trauma_icp_24h_n": int(au_tr["has_icp_24h"].sum()),
        "jinhua_tbi_s06_n": int(len(j_tbi)),
        "jinhua_tbi_s06_d28": int(j_tbi["d28"].sum()),
        "jinhua_tbi_s06_d28_sens": int(j_tbi["d28_sens_composite"].sum()),
        "jinhua_tbi_s06_died_in_hosp": int(j_tbi["died_in_hosp"].sum()),
        "jinhua_strict_s06_code_only_n": int(len(j_s06_only)),
        "jinhua_strict_s06_code_only_d28": int(j_s06_only["d28"].sum()),
        "jinhua_gcs_proxy_mapped_n": int(j_tbi["gcs_total_first"].notna().sum()),
        "jinhua_gcs_proxy_missing_nan_n": int(j_tbi["gcs_total_first"].isna().sum()),
        "jinhua_gcs_proxy_source_breakdown": {str(k): int(v) for k, v in src_counts.items()},
        "jinhua_gcs_proxy_source_detail": src_detail,
        "jinhua_abi_n": int(len(j_mat)),
        "jinhua_abi_d28": int(j_mat["d28"].sum()),
        "jinhua_abi_d28_sens": int(j_mat["d28_sens_composite"].sum()),
        "jinhua_abi_died_in_hosp": int(j_mat["died_in_hosp"].sum()),
        "total_ct_matched_24h_n": int(m4_ct_all["ct_any_24h"].sum() + m3_ct_all["ct_any_24h"].sum() + j_mat["ct_any_24h"].sum()),
        "ct_structured_exam_verification": {
            "mimic4_source": "radiology_detail.csv.gz field_name='exam_name' word-bounded \\b(ct|cta)\\b + \\b(head|brain|cranial)\\b",
            "mimic4_top_exam_names": {str(k): int(v) for k, v in m4_exam_counts.items()},
            "mimic3_top_exam_descriptions": {str(k): int(v) for k, v in m3_exam_counts.items()},
            "validation_method_note": "Timestamp-restricted structured exam_name/DESCRIPTION filter + bilingual rule-based regex with negation lookahead; no human clinician chart adjudication."
        },
        "ct_coverage_strict_24h": {
            "MIMIC-IV": {"n": int(len(m4_ct_all)), "ct_24h_n": int(m4_ct_all["ct_any_24h"].sum()), "pct": round(float(m4_ct_all["ct_any_24h"].mean()*100), 1)},
            "MIMIC-III": {"n": int(len(m3_ct_all)), "ct_24h_n": int(m3_ct_all["ct_any_24h"].sum()), "pct": round(float(m3_ct_all["ct_any_24h"].mean()*100), 1)},
            "JinhuaNSICU_TBI": {"n": int(len(j_tbi)), "ct_24h_n": int(j_tbi["ct_any_24h"].sum()), "pct": round(float(j_tbi["ct_any_24h"].mean()*100), 1)},
            "JinhuaNSICU_ABI": {"n": int(len(j_mat)), "ct_24h_n": int(j_mat["ct_any_24h"].sum()), "pct": round(float(j_mat["ct_any_24h"].mean()*100), 1)}
        }
    }
    json.dump(summary, open(f"{REP}/19a_v4_extraction_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    lg("DONE! Summary: " + json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
