# -*- coding: utf-8 -*-
# Fairness subgroups (sex / age) for the v2 frozen predictions, three databases, with n, events, bootstrap CI.
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "E:/TBI subtype/07_prediction_system/data/"
def boot(y, p, B=1000, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float); rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]); out.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(x), 3) for x in np.percentile(out, [2.5, 97.5])]
res = {}
m4 = pd.read_parquet(D + "08c_matrix_mimic4.parquet")[["stay_id", "admission_age", "sex_female"]]
p4 = pd.read_parquet(D + "08d_v2_preds_mimic4.parquet").merge(m4, on="stay_id"); p4["female"] = p4.sex_female.astype(int); p4["age"] = p4.admission_age
fe = pd.read_parquet(D + "features_eicu.parquet")[["icustay_id_eicu", "age", "male"]]
pe = pd.read_parquet(D + "08e_v2_preds_eicu.parquet").merge(fe, on="icustay_id_eicu"); pe["female"] = 1 - pe.male.astype(int)
f3 = pd.read_parquet(D + "features_mimic3.parquet")[["icustay_id", "age", "male"]]
p3 = pd.read_parquet(D + "08e_v2_preds_mimic3.parquet").merge(f3, on="icustay_id"); p3["female"] = 1 - p3.male.astype(int)
for name, df in (("mimic4_intval", p4), ("eicu", pe), ("mimic3_carevue", p3)):
    df["age"] = df["age"].astype(float); out = {}
    for lab, mask in (("female", df.female == 1), ("male", df.female == 0), ("age_lt65", df.age < 65), ("age_ge65", df.age >= 65), ("age_ge80", df.age >= 80)):
        s = df[mask]
        if len(s) < 20 or s.d28.nunique() < 2: out[lab] = {"n": int(len(s)), "events": int(s.d28.sum()), "AUROC": None}; continue
        out[lab] = {"n": int(len(s)), "events": int(s.d28.sum()), "AUROC": round(float(roc_auc_score(s.d28, s.p_full)), 3), "boot95": boot(s.d28, s.p_full),
                    "mean_pred": round(float(s.p_full.mean()), 3), "obs": round(float(s.d28.mean()), 3)}
    res[name] = out
    print(name, json.dumps(out))
json.dump(res, open("E:/TBI subtype/07_prediction_system/reports/11c_v2_fairness_subgroups.json", "w", encoding="utf-8"), indent=1)
print("WROTE 11c")
