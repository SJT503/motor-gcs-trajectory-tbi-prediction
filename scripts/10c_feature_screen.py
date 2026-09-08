# -*- coding: utf-8 -*-
# 10c — Zhang 式特征双重筛选 (STAT_FRAMEWORK_v2_ULTIMATE §4, SAP §12 2026-09-06 PI 指令)
#   三步: ①临床相关性(§3 已文档化) ②可得性(>20% 缺失, 全过) ③LASSO ∩ RF 双重统计筛选
#   LASSO 腿: L1-LR 嵌套 CV 选 C → 非零系数 (08k 现成: C=0.1 → 37 个非零)
#   RF 腿: train 拟合 RF → importance top-20
#   合成: 交集为终选; 后备(预注册): 交集<15 则并集按 RF importance 截 25
#   全程 train-only, 验证集零接触
# 产出: reports/10c_feature_screen.json + 选中特征列表
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")
ROOT = "E:/TBI subtype"
DATA, REP, MODELS = f"{ROOT}/07_prediction_system/data", f"{ROOT}/07_prediction_system/reports", f"{ROOT}/07_prediction_system/models"

rep0 = json.loads(open(f"{REP}/08c_matrix_report.json", encoding="utf-8").read())
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]

m = pd.read_parquet(f"{DATA}/08c_matrix_mimic4.parquet")
tr = m[m.split == "train"]
ytr = tr.d28.values.astype(int)

# 用 08k 已跑的 imp0 插补集 (train-only, 验证集零接触)
imp0 = pd.DataFrame(joblib.load(f"{MODELS}/08d_mice_imp0.joblib").transform(m[FULL]),
                    columns=FULL, index=m.index)
Xtr = imp0.loc[tr.index]

# ===== LASSO 腿 (08k 已有: C=0.1 cw=balanced, 非零 37 个) =====
# 直接从 08k JSON 读, 保证与已跑基准一致
k = json.loads(open(f"{REP}/08k_ml_benchmark.json", encoding="utf-8").read())
l1_nzero = k["l1_sparsity_check"]["n_zero"]  # 42
l1_total = k["l1_sparsity_check"]["n_coef"]  # 79
# 但 08k 没存具体哪些列为零——需要重跑一次 L1-LR 获取列名
pipe = Pipeline([("sc", StandardScaler()),
                 ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                           class_weight="balanced", max_iter=5000, random_state=42))])
pipe.fit(Xtr, ytr)
coefs = pipe.named_steps["lr"].coef_[0]
lasso_selected = [c for c, v in zip(FULL, coefs) if abs(v) > 1e-10]
lasso_zeroed = [c for c, v in zip(FULL, coefs) if abs(v) <= 1e-10]
print(f"[LASSO] C=0.1 → 非零 {len(lasso_selected)} / 零 {len(lasso_zeroed)}")

# ===== RF 腿 =====
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=4)
rf.fit(Xtr, ytr)
imp = pd.Series(rf.feature_importances_, index=FULL).sort_values(ascending=False)
rf_top20 = imp.head(20).index.tolist()
print(f"[RF] top-20:", rf_top20)

# ===== 合成 =====
intersect = [c for c in lasso_selected if c in rf_top20]
union = list(set(lasso_selected) | set(rf_top20))
union_by_rf = [c for c in imp.index if c in union]  # 按 RF importance 降序

if len(intersect) >= 15:
    final = intersect
    rule = f"交集 (≥15): LASSO {len(lasso_selected)} ∩ RF top-20 → {len(intersect)}"
else:
    final = union_by_rf[:25]
    rule = f"后备并集截25 (交集 {len(intersect)} < 15): LASSO {len(lasso_selected)} ∪ RF top-20 → {len(union)} 按 RF imp 截 {len(final)}"

print(f"[合成] 规则: {rule}")
print(f"[终选] {len(final)} 特征:")
# 按块分组显示
for block, cols in [("A", rep0["blockA_cols"]), ("B", rep0["blockB_cols"]), ("C", rep0["blockC_cols"])]:
    sel_b = [c for c in final if c in cols]
    print(f"  Block {block}: {len(sel_b)}/{len(cols)} → {sel_b}")

# 保存
out = {
    "design": "Zhang-style dual screening (LASSO ∩ RF top-20, fallback union@25); train-only; SAP §12 2026-09-06",
    "lasso_leg": {"C": 0.1, "n_nonzero": len(lasso_selected), "selected": lasso_selected},
    "rf_leg": {"top20": rf_top20},
    "intersection": intersect,
    "union": union_by_rf,
    "rule_applied": rule,
    "final_features": final,
    "n_final": len(final),
    "block_composition": {
        "A": len([c for c in final if c in rep0["blockA_cols"]]),
        "B": len([c for c in final if c in rep0["blockB_cols"]]),
        "C": len([c for c in final if c in rep0["blockC_cols"]]),
    },
}
json.dump(out, open(f"{REP}/10c_feature_screen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n[saved] {REP}/10c_feature_screen.json")
