# -*- coding: utf-8 -*-
# ============================================================
# 09b — END@24 标签 + END_post(24,72] 标签 + 冻结风险分数对 END_post 的 AUROC
# 判据唯一源 = 12c_extract_gcs_24h.py: END(W) = (W 内首测 total-GCS − W 内最小值) >= 2; 可计算 = W 内 >=2 次
# 数据源 = 07a 冻结 gcs_long_{db}.parquet (72h, 窗 (24,72] 完整在内, 无需 09a)
# 预测 = 冻结管线产物 08d_preds_mimic4 / 08e_preds_{eicu,mimic3} (zero-touch, 不重算)
# 产出: data/09b_endpost24_{db}.parquet + reports/09b_endpost24.json
# ============================================================
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
rng = np.random.default_rng(42)

def end_window(g, lo, hi):
    """12c 判据逐字: 窗 [lo,hi] 内 total 序列 v(按时间序) → (END, n, first_val, event_offset)
    END = (v[0]-v.min())>=2; 可计算 = n>=2; 事件时刻 = 首个 value <= v[0]-2 的 offset"""
    s = g[(g.offset_hr>=lo)&(g.offset_hr<=hi)&g.gcs_total.notna()].sort_values("offset_hr")
    s = s.drop_duplicates("offset_hr", keep="first")
    v = s.gcs_total.values
    if len(v) < 2:
        return np.nan, len(v), np.nan, np.nan
    end = int((v[0]-v.min())>=2)
    ev = np.nan
    if end == 1:
        cross = s.offset_hr.values[v <= v[0]-2]
        ev = float(cross[0]) if len(cross) else np.nan
    return end, len(v), float(v[0]), ev

def boot_ci(y, p, B=1000):
    y = np.asarray(y); p = np.asarray(p)
    if y.sum() in (0, len(y)):
        return [None, None]
    idx = rng.integers(0, len(y), (B, len(y)))
    aucs = [roc_auc_score(y[i], p[i]) for i in idx if len(set(y[i]))>1]
    return [round(float(np.percentile(aucs,2.5)),4), round(float(np.percentile(aucs,97.5)),4)]

CFG = {
  "mimic4": dict(long="gcs_long_mimic4.parquet", preds=("08d_preds_mimic4.parquet","stay_id"),
                 out="09b_endpost24_mimic4.parquet"),
  "eicu":   dict(long="gcs_long_eicu.parquet",   preds=("08e_preds_eicu.parquet","icustay_id_eicu"),
                 out="09b_endpost24_eicu.parquet"),
  "mimic3": dict(long="gcs_long_mimic3.parquet", preds=("08e_preds_mimic3.parquet","icustay_id"),
                 out="09b_endpost24_mimic3.parquet"),
}
res = {}
for db, c in CFG.items():
    long = pd.read_parquet(DATA/c["long"])
    pf, idcol = c["preds"]
    preds = pd.read_parquet(DATA/pf)
    pc = "p_full" if "p_full" in preds.columns else [x for x in preds.columns if x.startswith("p_")][0]
    preds = preds.rename(columns={idcol:"id", pc:"p_full"})
    preds["id"] = preds.id.astype(str)
    lab = long.groupby("id")[["offset_hr","gcs_total"]].apply(
        lambda g: pd.Series(end_window(g, 0, 24), index=["end24","n24","first24","ev24"]))
    labp = long.groupby("id")[["offset_hr","gcs_total"]].apply(
        lambda g: pd.Series(end_window(g, 24, 72), index=["endpost","npost","firstpost","evpost"]))
    lab = lab.join(labp).reset_index()
    m = lab.merge(preds[["id","p_full","d28"]], on="id", how="left")
    m["in_preds"] = m.p_full.notna()
    m.to_parquet(DATA/c["out"], index=False)

    comp24 = m.dropna(subset=["end24"]); lm = comp24[comp24.end24==0]
    lmp = lm.dropna(subset=["endpost"])   # landmark + END_post 可计算 + 有冻结分数
    lmp_ev = lmp[lmp.in_preds & lmp.endpost.notna()]
    r = dict(
        n_long=int(m.shape[0]), n_end24_computable=int(comp24.shape[0]),
        end24_events=int(comp24.end24.sum()),
        end24_prev=round(float(comp24.end24.mean()),4),
        n_landmark=int(lm.shape[0]),
        n_endpost_computable=int(lmp.shape[0]),
        endpost_events=int(lmp.endpost.sum()),
        endpost_prev=round(float(lmp.endpost.mean()),4),
        floor_first24_le4=round(float((comp24.first24<=4).mean()),4),
        floor_landmark_firstpost_le4=round(float((lmp.firstpost<=4).mean()),4))
    ev = lmp_ev[lmp_ev.endpost==1]; nev = lmp_ev[lmp_ev.endpost==0]
    if len(ev)>0 and ev.d28.nunique()>1 and len(nev)>0:
        r["auroc_endpost"] = round(float(roc_auc_score(lmp_ev.endpost, lmp_ev.p_full)),4)
        r["auroc_endpost_ci"] = boot_ci(lmp_ev.endpost, lmp_ev.p_full)
        # 敏感性: 地板剔除 (窗首测>4, SAP_v2:27)
        sub = lmp_ev[lmp_ev.firstpost>4]
        if sub.endpost.nunique()>1:
            r["auroc_endpost_floor_excl"] = round(float(roc_auc_score(sub.endpost, sub.p_full)),4)
            r["n_floor_excl"] = int(sub.shape[0]); r["ev_floor_excl"] = int(sub.endpost.sum())
    # 描述性: 冻结分数 vs END24 本身 (窗内恶化)
    c24 = comp24[comp24.in_preds]
    if c24.end24.nunique()>1:
        r["auroc_end24_desc"] = round(float(roc_auc_score(c24.end24, c24.p_full)),4)
        r["auroc_end24_ci"] = boot_ci(c24.end24, c24.p_full)
    res[db] = r
    print(f"[{db}] " + json.dumps(r, ensure_ascii=False))

(REP/"09b_endpost24.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("DONE 09b")
