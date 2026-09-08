# Phase1-③c: eICU + MIMIC-III Charlson 合并症 (SAP_prediction §3.1 Block A 人口学/合并症)
# 码表逐字复用 04_charlson_comorbidity.py (mimic-code charlson.sql, Quan 2005 编码) 的 ICD-9 分支
# 两库均为纯 ICD-9: MIMIC-III DIAGNOSES_ICD 全码; eICU diagnosis.icd9code (逗号分隔多码 → unnest 展开)
# score 不含 age_score (与 discovery 04 一致, 年龄独立入模型); GREATEST 处理互斥类
# 输出: 合并进 data/features_eicu.parquet / features_mimic3.parquet + reports/07f_charlson_qc.json
import sys, json, duckdb, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
M3 = ROOT/"data/mimic-iii-1.4"; EI = ROOT/"data/eicu-crd-2.0"
OUT = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
qc = {}

# ---- 17 类 Quan 编码 CASE 骨架 (icd9 分支, 逐字对照 04:33-51) ----
CASES = {
 "myocardial_infarct": "SUBSTR(c,1,3) IN('410','412')",
 "congestive_heart_failure": "SUBSTR(c,1,3)='428' OR SUBSTR(c,1,5) IN('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') OR SUBSTR(c,1,4) BETWEEN '4254' AND '4259'",
 "peripheral_vascular_disease": "SUBSTR(c,1,3) IN('440','441') OR SUBSTR(c,1,4) IN('0930','4373','4471','5571','5579','V434') OR SUBSTR(c,1,4) BETWEEN '4431' AND '4439'",
 "cerebrovascular_disease": "SUBSTR(c,1,3) BETWEEN '430' AND '438' OR SUBSTR(c,1,5)='36234'",
 "dementia": "SUBSTR(c,1,3)='290' OR SUBSTR(c,1,4) IN('2941','3312')",
 "chronic_pulmonary_disease": "SUBSTR(c,1,3) BETWEEN '490' AND '505' OR SUBSTR(c,1,4) IN('4168','4169','5064','5081','5088')",
 "rheumatic_disease": "SUBSTR(c,1,3)='725' OR SUBSTR(c,1,4) IN('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148')",
 "peptic_ulcer_disease": "SUBSTR(c,1,3) IN('531','532','533','534')",
 "mild_liver_disease": "SUBSTR(c,1,3) IN('570','571') OR SUBSTR(c,1,4) IN('0706','0709','5733','5734','5738','5739','V427') OR SUBSTR(c,1,5) IN('07022','07023','07032','07033','07044','07054')",
 "diabetes_without_cc": "SUBSTR(c,1,4) IN('2500','2501','2502','2503','2508','2509')",
 "diabetes_with_cc": "SUBSTR(c,1,4) IN('2504','2505','2506','2507')",
 "paraplegia": "SUBSTR(c,1,3) IN('342','343') OR SUBSTR(c,1,4) IN('3341','3440','3441','3442','3443','3444','3445','3446','3449')",
 "renal_disease": "SUBSTR(c,1,3) IN('582','585','586','V56') OR SUBSTR(c,1,4) IN('5880','V420','V451') OR SUBSTR(c,1,4) BETWEEN '5830' AND '5837' OR SUBSTR(c,1,5) IN('40301','40311','40391','40402','40403','40412','40413','40492','40493')",
 "malignant_cancer": "SUBSTR(c,1,3) BETWEEN '140' AND '172' OR SUBSTR(c,1,4) BETWEEN '1740' AND '1958' OR SUBSTR(c,1,3) BETWEEN '200' AND '208' OR SUBSTR(c,1,4)='2386'",
 "severe_liver_disease": "SUBSTR(c,1,4) IN('4560','4561','4562') OR SUBSTR(c,1,4) BETWEEN '5722' AND '5728'",
 "metastatic_solid_tumor": "SUBSTR(c,1,3) IN('196','197','198','199')",
 "aids": "SUBSTR(c,1,3) IN('042','043','044')"}
FLAGS = list(CASES)
# score 公式 = 04:65-71 (无 age_score, GREATEST 互斥)
SCORE = ("COALESCE(myocardial_infarct,0)+COALESCE(congestive_heart_failure,0)+COALESCE(peripheral_vascular_disease,0)"
 "+COALESCE(cerebrovascular_disease,0)+COALESCE(dementia,0)+COALESCE(chronic_pulmonary_disease,0)"
 "+COALESCE(rheumatic_disease,0)+COALESCE(peptic_ulcer_disease,0)"
 "+GREATEST(COALESCE(mild_liver_disease,0), 3*COALESCE(severe_liver_disease,0))"
 "+GREATEST(2*COALESCE(diabetes_with_cc,0), COALESCE(diabetes_without_cc,0))"
 "+GREATEST(2*COALESCE(malignant_cancer,0), 6*COALESCE(metastatic_solid_tumor,0))"
 "+2*COALESCE(paraplegia,0)+2*COALESCE(renal_disease,0)+6*COALESCE(aids,0)")
def agg_sql(group_col):
    flags = ",\n  ".join(f"MAX(CASE WHEN {cond} THEN 1 ELSE 0 END) {k}" for k, cond in CASES.items())
    coals = ",\n  ".join(f"COALESCE({k},0) {k}" for k in FLAGS)
    return f"SELECT {group_col},\n  {flags}\nFROM dx GROUP BY {group_col}", \
           f"SELECT {group_col},\n  {coals},\n  {SCORE} AS charlson_comorbidity_score FROM com"

def summarize(tag, df, n_cohort):
    prev = df[FLAGS].mean().sort_values(ascending=False)
    qc[tag] = dict(n_rows=int(len(df)), n_cohort=int(n_cohort), coverage_pct=round(100*len(df)/max(1,n_cohort),1),
        score_median=float(df.charlson_comorbidity_score.median()), score_mean=round(float(df.charlson_comorbidity_score.mean()),2),
        score_iqr=[float(df.charlson_comorbidity_score.quantile(.25)), float(df.charlson_comorbidity_score.quantile(.75))],
        score_zero_pct=round(100*float((df.charlson_comorbidity_score==0).mean()),1),
        top5_prevalence={k: round(100*float(v),1) for k, v in prev.head(5).items()})
    print(f"[{tag}] {len(df)}/{n_cohort} 覆盖 | score 中位 {qc[tag]['score_median']:.0f} "
          f"IQR {qc[tag]['score_iqr'][0]:.0f}-{qc[tag]['score_iqr'][1]:.0f} | score=0 占 {qc[tag]['score_zero_pct']}%")
    print("  患病率Top5: " + ", ".join(f"{k} {100*v:.1f}%" for k, v in prev.head(5).items()))  # v为0-1小数, 打印须×100

# ========== eICU (diagnosis.icd9code 逗号分隔 → unnest) ==========
print("[eICU] Charlson...")
dx_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{(EI/'diagnosis.csv.gz').as_posix()}')").df().column_name)
assert {"patientunitstayid","icd9code"} <= dx_cols, f"eICU diagnosis 缺列, 实测: {sorted(dx_cols)}"
probe = con.execute(f"""SELECT icd9code, count(*) n FROM read_csv_auto('{(EI/'diagnosis.csv.gz').as_posix()}')
WHERE icd9code IS NOT NULL AND icd9code != ''
  AND patientunitstayid IN (SELECT icustay_id_eicu FROM read_parquet('{(OUT/'cohort_eicu.parquet').as_posix()}'))
GROUP BY 1 ORDER BY n DESC LIMIT 10""").df()
print(probe.to_string(index=False))
qc["eicu_icd9_top10"] = probe.icd9code.tolist()
qc["eicu_icd9_dot_format"] = bool(probe.icd9code.str.contains(r"\d\.").any())
print(f"  [格式] icd9 带点={qc['eicu_icd9_dot_format']} (replace 已防御性去点, 不依赖格式假设)")
flags_sql, score_sql = agg_sql("patientunitstayid")
ei = con.execute(f"""
WITH dx AS (
  SELECT patientunitstayid, replace(trim(t.u),'.','') c
  FROM read_csv_auto('{(EI/'diagnosis.csv.gz').as_posix()}') d,
       UNNEST(string_split(COALESCE(d.icd9code,''), ',')) AS t(u)
  WHERE patientunitstayid IN (SELECT icustay_id_eicu FROM read_parquet('{(OUT/'cohort_eicu.parquet').as_posix()}'))),
com AS ({flags_sql})
SELECT s.* FROM ({score_sql}) s""").df().rename(columns={"patientunitstayid":"icustay_id_eicu"})
coh_ei = pd.read_parquet(OUT/"cohort_eicu.parquet")
summarize("eicu", ei.merge(coh_ei[["icustay_id_eicu"]], on="icustay_id_eicu", how="right").fillna(
    {**{k:0 for k in FLAGS}, "charlson_comorbidity_score":0}), len(coh_ei))

# ========== MIMIC-III (DIAGNOSES_ICD 全码 by hadm_id) ==========
print("[MIMIC-III] Charlson...")
flags_sql, score_sql = agg_sql("hadm_id")
m3 = con.execute(f"""
WITH dx AS (
  SELECT hadm_id AS hadm_id, ICD9_CODE c
  FROM read_csv_auto('{(M3/'DIAGNOSES_ICD.csv.gz').as_posix()}', types={{'ICD9_CODE':'VARCHAR'}})
  WHERE hadm_id IN (SELECT hadm_id FROM read_parquet('{(OUT/'cohort_mimic3.parquet').as_posix()}'))),
com AS ({flags_sql})
SELECT s.* FROM ({score_sql}) s""").df()
coh_m3 = pd.read_parquet(OUT/"cohort_mimic3.parquet")
m3f = m3.merge(coh_m3[["icustay_id","hadm_id"]], on="hadm_id", how="right").fillna(
    {**{k:0 for k in FLAGS}, "charlson_comorbidity_score":0})
summarize("mimic_iii", m3f, len(coh_m3))

# discovery 锚点对照 (MIMIC-IV 04 产物)
try:
    iv = pd.read_parquet(ROOT/"results/features/04_charlson.parquet")
    qc["mimic_iv_anchor"] = dict(score_median=float(iv.charlson_comorbidity_score.median()),
        score_mean=round(float(iv.charlson_comorbidity_score.mean()),2), n=int(len(iv)))
    print(f"[锚点] MIMIC-IV(04): score 中位 {qc['mimic_iv_anchor']['score_median']:.0f} 均值 {qc['mimic_iv_anchor']['score_mean']} n={len(iv)}")
except Exception as e:
    print(f"[锚点] MIMIC-IV 04 parquet 不可读({e}), 跳过对照")

# ========== 并入 features ==========
for feat_path, key, ch, tag in [(OUT/"features_eicu.parquet", "icustay_id_eicu", ei, "eicu"),
                                (OUT/"features_mimic3.parquet", "icustay_id", m3f.drop(columns=["hadm_id"]), "mimic3")]:
    f = pd.read_parquet(feat_path)
    drop = [c for c in FLAGS+["charlson_comorbidity_score"] if c in f.columns]
    if drop: f = f.drop(columns=drop)                     # 重跑幂等: 先删旧 Charlson 列
    f = f.merge(ch[[key]+FLAGS+["charlson_comorbidity_score"]], on=key, how="left")
    for c in FLAGS+["charlson_comorbidity_score"]: f[c] = f[c].fillna(0)   # 无诊断记录 → 0 (与 04 COALESCE 同义)
    assert len(f) == len(pd.read_parquet(feat_path)), f"{tag} 并入后行数变了!"
    f.to_parquet(feat_path, index=False)
    print(f"  并入 {feat_path.name}: {f.shape[0]} × {f.shape[1]}")

(REP/"07f_charlson_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nDONE 07f")
