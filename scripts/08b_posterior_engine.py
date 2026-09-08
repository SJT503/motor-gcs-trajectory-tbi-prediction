# -*- coding: utf-8 -*-
# ============================================================
# 08b — Block C 截断后验引擎 (SAP §3.3 冻结: 只用 landmark 前观测, 防泄漏红线)
# 数学: lcmm hlme(mixture=~t10+t10^2, random=~-1) 的类条件边缘似然 =
#       compound-symmetric MVN: Σ_g = σ²_g I_K + τ² 1 1' → 闭式 logpdf
#       (Woodbury/Sherman-Morrison 展开到充分统计量 S1,S2, 逐主体向量化)
# 输入: reports/08a_gbtm_coefs.json (motor ng=08a2 冻结规则选中值 3/4/5 + total ng=5 敏感性)
#       data/gcs_long_{mimic4,eicu,mimic3}.parquet (07a 统一 schema)
# 验证锚 (必须过才准产出): M4 全 72h 观测 argmax 后验 vs R classes CSV
#       motor vs 13b_motor_classes.csv ≥95% 且 total vs 12f_gbtm_classes.csv ≥95%
# 产出: data/08b_posterior24_*.parquet (id + prob_m1..N + prob_t1..5 + n_obs; N=motor 选中 ng, 动态列)
# ============================================================
import json, re, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT / "07_prediction_system/data"
REP = ROOT / "07_prediction_system/reports"

# ---------- 1. 解析 08a 冻结系数 ----------
raw = json.loads((REP / "08a_gbtm_coefs.json").read_text(encoding="utf-8"))

def parse_model(block):
    """从 08a2 数组式系数 (best_vec: [{name,value}...]) 解析: 每类 intercept/t10/t10^2 + τ² + σ² + π
    lcmm 命名实证 (08a 2026-09-03 dump): intercept classK 出现两套
    (前 ng-1 个 = classmb 隶属 logit, 后 ng 个 = 固定效应) → 重复名取后现者(last-wins)。
    τ² = var 1 (若实证无此项 → raise, 按日志打印的真实键名修)"""
    if "best_vec" not in block["fit"]:
        raise RuntimeError(f"系数文件缺 best_vec (旧格式 best 已废, 重跑 08a2). fit keys: {list(block['fit'])}")
    entries = [(str(e["name"]).strip(), float(e["value"])) for e in block["fit"]["best_vec"]]
    names = [n for n, _ in entries]
    def find_classes_last(pat):
        out = {}
        for n, v in entries:
            m = re.fullmatch(pat, n)
            if m: out[int(m.group(1))] = v      # 后现者覆盖 → 固定效应集胜出
        return out
    b0 = find_classes_last(r"intercept\s*class\s*(\d+)")
    b1 = find_classes_last(r"t10\s*class\s*(\d+)")
    b2 = find_classes_last(r"I\(t10\^2\)\s*class\s*(\d+)")
    tau2 = find_classes_last(r"var\s*1\s*class\s*(\d+)")       # 类特异随机截距方差(若有)
    tau2_common = next((v for n, v in entries if re.fullmatch(r"var\s*1", n)), None)
    sig2 = find_classes_last(r"std\.?err\s*class\s*(\d+)")     # 类特异残差(若有)
    sig2_common = next((v for n, v in entries if re.fullmatch(r"std\.?err", n)), None)
    pm = find_classes_last(r"pmclass\s*(\d+)")
    ng = int(block["ng"])
    missing = [f"b0[{i}]" for i in range(1, ng+1) if i not in b0] + \
              [f"b1[{i}]" for i in range(1, ng+1) if i not in b1] + \
              [f"b2[{i}]" for i in range(1, ng+1) if i not in b2]
    if missing:
        raise RuntimeError(f"系数解析失败, 缺 {missing}. best 全部键名: {names}")
    # π: pmclass 多项 logit (class1 为参照) 优先, 否则用 pprob 类占比
    if len(pm) == ng - 1:
        e = np.array([1.0] + [np.exp(pm[k]) for k in range(2, ng + 1)])
        pi = e / e.sum()
    else:
        sh = np.asarray(block["class_share"], dtype=float)
        pi = sh / sh.sum()
    # σ²: lcmm 输出若为标准差则平方; 常数项单位由验证锚兜底
    def sig2_of(g):
        if g in sig2: return sig2[g] ** 2
        if sig2_common is not None: return sig2_common ** 2
        return None
    s2 = {g: sig2_of(g) for g in range(1, ng + 1)}
    if any(v is None for v in s2.values()):
        raise RuntimeError(f"残差方差解析失败. best 全部键名: {names}")
    if len(tau2) == ng:
        t2 = tau2
    elif tau2_common is not None:
        t2 = {g: tau2_common for g in range(1, ng + 1)}
    else:
        # LCGM: random=~-1 实为无随机效应 (20 参数实证: 4 logit + 15 fixed + stderr, 无 var 1)
        # → τ²=0, compound-symmetric 公式自动退化为独立观测乘积似然, 无需另写公式
        t2 = {g: 0.0 for g in range(1, ng + 1)}
    print(f"  [parse] ng={ng} | b0={[round(b0[g],2) for g in sorted(b0)]} | "
          f"tau2={t2[1]:.4f}(common={len(tau2)==1 and tau2_common is not None}) | "
          f"sig2={ {g: round(s2[g],4) for g in sorted(s2)} } | pi={np.round(pi,3).tolist()}")
    return dict(ng=ng, b0=b0, b1=b1, b2=b2, tau2=t2, sig2=s2, pi=pi)

print("=== 08b 截断后验引擎 ===")
motor = parse_model(raw["motor"])
total = parse_model(raw["total"])

# ---------- 2. 后验计算 (compound-symmetric MVN 闭式) ----------
def posterior(long_df, col_y, model, t_max):
    """long_df: id, offset_hr, col_y; 只用 0<=offset_hr<=t_max 的观测 → 每 id 的 ng 维后验"""
    d = long_df[(long_df.offset_hr >= 0) & (long_df.offset_hr <= t_max)].dropna(subset=[col_y])
    # id 统一 int64 (parquet 存 object 与 cohort/CSV 的 int64 合并会炸; 三库 id 均为整数, 非整数即报错)
    d = d.assign(id=pd.to_numeric(d["id"]).astype("int64"))
    d = d.assign(t10=d.offset_hr / 10.0, y=d[col_y].astype(float))
    ng = model["ng"]
    res = {gid: None for gid in d["id"].unique()}
    logp = np.full((len(res), ng), -np.inf)
    ids_order = np.array(sorted(res))
    idx = {g: i for i, g in enumerate(ids_order)}
    d["row"] = d["id"].map(idx)
    d = d.sort_values(["row"])
    # 逐类向量化 (充分统计量 S1,S2 via groupby)
    for g in range(1, ng + 1):
        m = model["b0"][g] + model["b1"][g] * d.t10 + model["b2"][g] * d.t10 ** 2
        r = d.y.values - m.values
        df_g = pd.DataFrame({"row": d.row.values, "r": r})
        agg = df_g.groupby("row").agg(K=("r", "size"), S1=("r", "sum"), S2=("r", lambda z: float((z ** 2).sum())))
        tau2, s2 = model["tau2"][g], model["sig2"][g]
        K = agg.K.values.astype(float); S1 = agg.S1.values; S2 = agg.S2.values
        logdet = K * np.log(s2) + np.log1p(K * tau2 / s2)
        q = (S2 - (tau2 / (s2 + K * tau2)) * S1 ** 2) / s2
        lp = -0.5 * (K * np.log(2 * np.pi) + logdet + q) + np.log(model["pi"][g - 1])
        logp[agg.index.values, g - 1] = lp
    # logsumexp 归一
    mx = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - np.where(np.isfinite(mx), mx, 0))
    p /= p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(ids_order, columns=["id"])
    for g in range(1, ng + 1):
        out[f"prob_{g}"] = p[:, g - 1]
    out["n_obs"] = np.where(np.isfinite(logp[:, 0]), np.nan, np.nan)
    out["n_obs"] = d.groupby("row").size().reindex(range(len(ids_order))).values
    return out

def run_scale(long_df, model, tag, t_max=24.0):
    m = posterior(long_df, "gcs_motor" if tag == "m" else "gcs_total", model, t_max)
    m.columns = ["id"] + [f"prob_{tag}{g}" for g in range(1, model["ng"] + 1)] + [f"n_obs_{tag}"]
    return m

# ---------- 3. 验证锚: M4 全 72h vs R classes ----------
def anchor(long_df, model, classes_csv, tag):
    full = posterior(long_df, "gcs_motor" if tag == "m" else "gcs_total", model, t_max=72.0)
    full["argmax"] = full[[f"prob_{g}" for g in range(1, model["ng"] + 1)]].values.argmax(axis=1) + 1
    ref = pd.read_csv(classes_csv)
    idcol = ref.columns[0]
    mg = full.merge(ref, left_on="id", right_on=idcol)
    agree = (mg.argmax == mg["class"]).mean()
    print(f"  [anchor:{tag}] n={len(mg)} 与 R classes 一致率 = {agree:.1%} (门禁 ≥95%)")
    return agree

print("\n--- 验证锚 (全 72h 观测, 双模型) ---")
# 07a 两代命名并存: M4 件是原始名 (motor/eye/verbal), eICU/M3 件是统一名 (gcs_motor/...)
# 读取处统一 rename 到 SCHEMA (缺列时 rename 为 no-op, 对三库皆安全)
RAW2SCHEMA = {"motor": "gcs_motor", "eye": "gcs_eye", "verbal": "gcs_verbal"}
m4 = pd.read_parquet(DATA / "gcs_long_mimic4.parquet").rename(columns=RAW2SCHEMA)
print("[mimic4 gcs_long] 列:", list(m4.columns), "| 行数:", len(m4))
a_m = anchor(m4, motor, ROOT / "results/cluster/13_motor/13b_motor_classes.csv", "m")
a_t = anchor(m4, total, ROOT / "results/cluster/12_gcs_course/12f_gbtm_classes.csv", "t")
if a_m < 0.95 or a_t < 0.95:
    print("!! 验证锚未过 (解析或公式有误) — 停止, 不产出 landmark 后验")
    sys.exit(1)

# ---------- 4. landmark 24h 后验 × 三库 ----------
for db in ["mimic4", "eicu", "mimic3"]:
    df = m4 if db == "mimic4" else pd.read_parquet(DATA / f"gcs_long_{db}.parquet")
    pm = run_scale(df, motor, "m", 24.0)
    pt = run_scale(df, total, "t", 24.0)
    out = pm.merge(pt, on="id")
    # 描述: argmax 分布 (跨库漂移预览) + 无观测人数
    am_cols = [c for c in out.columns if c.startswith("prob_m")]   # motor ng 可为 3/4/5
    am = out[am_cols].values.argmax(axis=1) + 1
    dist = pd.Series(am).value_counts(normalize=True).sort_index()
    n0 = int((out.n_obs_m == 0).sum())
    print(f"[{db}] n={len(out)} | 无motor观测 {n0} | motor argmax 分布: "
          f"{ {int(k): f'{v:.1%}' for k, v in dist.items()} }")
    out.to_parquet(DATA / f"08b_posterior24_{db}.parquet", index=False)
    print(f"  → {DATA}/08b_posterior24_{db}.parquet")

print("\nDONE 08b")
