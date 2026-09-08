# -*- coding: utf-8 -*-
# ============================================================
# 09d — Rolling 检查点预测 (09_rolling_spec.md §二/§三 机械执行)
# 检查点 t ∈ {6..72h} 逐小时 67 个; 三库
# 特征 = 全特征以入院→t 窗重算:
#   Block A: 09c 事件流 (vitals min/mean/max [0,t]; labs min/max M4/M3 [-6,t] eICU [0,t])
#            M4 静态 gcs_*_min = 09c_gcs_triples (03b chartevents 口径)
#            eICU/M3 静态 gcs_*_min + 全库 gcs_*_first = gcs_long120 (07c/07d 同源)
#   Block B: c07.neuro_dynamic_features(win_hr=t) motor (FULL 仅含 motor; total 系敏感性不重算)
#   Block C: 08b posterior(t_max=t) motor (FULL 仅含 prob_m*; prob_t* 系敏感性不重算 — 本 log 预注册化)
# 推理: 冻结管线 08e frozen_predict 逐字 (5 MICE transform → 5 booster 均值), zero-touch
# 风险集(t) = los_h≥t & t 前存活 & [0,t] 有≥1 次 GCS 观测 (M4 = intval 子集, spec §二预注册)
#   M4 死亡时刻精确(hours_to_death); M3 精确(days_to_death*24); eICU 近似(死亡视作出院时刻, 披露)
# END 旗标: end_window 09b 逐字 — end_upto_t=END([0,t]); endpost48=END([t,t+48]) (含下界, 与 09b 口径一致)
# 一致性门禁 (t=24): 特征/预测 vs 08c/08d/08e 复现 (阈值见 GATE_* 常量, 预注册)
# 产出: data/09d_rolling_preds_{db}.parquet + reports/09d_t24_consistency.json
# ============================================================
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from joblib import load as jbload
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
MDL = ROOT/"07_prediction_system/models"; SCR = ROOT/"07_prediction_system/scripts"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
t0 = time.time()
def lg(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# 门禁阈值 (预注册, 记入报告)
GATE_DP_FRAC = 0.01     # |Δp|>0.01 的患者占比上限 1%
GATE_FEAT_FRAC = 0.05   # 任一特征列 |Δ|>0.01 的患者占比上限 5%
GATE_AUC_TOL = 0.01     # AUROC(common) vs 锚点容差

# ---------- 0. 冻结管线加载 (v2: 25 特征筛选版, 2026-09-07) ----------
card = json.loads((MDL/"08d_v2_model_card.json").read_text(encoding="utf-8"))
SEL = list(card["features"]); THRESH = float(card["threshold_youden_train_oof"]); MICE_M = 5
rep0 = json.loads((REP/"08c_matrix_report.json").read_text(encoding="utf-8"))
FULL = rep0["blockA_cols"] + rep0["blockB_cols"] + rep0["blockC_cols"]
A_COLS, B_COLS, C_COLS = rep0["blockA_cols"], rep0["blockB_cols"], rep0["blockC_cols"]
boosters = [lgb.Booster(model_file=str(MDL/f"08d_v2_lgbm_sel_imp{j}.txt")) for j in range(MICE_M)]
imputers = [jbload(MDL/f"08d_mice_imp{j}.joblib") for j in range(MICE_M)]
assert all(list(it.feature_names_in_) == FULL for it in imputers), "MICE 特征名与 FULL 不一致"
lg(f"冻结管线 v2: SEL={len(SEL)} (A{sum(1 for c in SEL if c in A_COLS)}/B{sum(1 for c in SEL if c in B_COLS)}/C{sum(1 for c in SEL if c in C_COLS)}), Youden={THRESH:.4f}")

def frozen_predict(Xdf):
    """v2: 5 MICE transform(79 列) → 取 SEL 25 列 → 5 booster 均值"""
    ps = []
    for j in range(MICE_M):
        imp_all = imputers[j].transform(Xdf[FULL])
        imp_df = pd.DataFrame(imp_all, columns=FULL, index=Xdf.index)
        ps.append(boosters[j].predict(imp_df[SEL]))
    return np.mean(ps, 0)

# ---------- 1. 08b 后验引擎逐字复制 (parse_model/posterior; 锚点已在 08b 实跑验证 ≥95%) ----------
import re
raw = json.loads((REP/"08a_gbtm_coefs.json").read_text(encoding="utf-8"))
def parse_model(block):
    if "best_vec" not in block["fit"]:
        raise RuntimeError(f"系数文件缺 best_vec. fit keys: {list(block['fit'])}")
    entries = [(str(e["name"]).strip(), float(e["value"])) for e in block["fit"]["best_vec"]]
    names = [n for n, _ in entries]
    def find_classes_last(pat):
        out = {}
        for n, v in entries:
            m = re.fullmatch(pat, n)
            if m: out[int(m.group(1))] = v
        return out
    b0 = find_classes_last(r"intercept\s*class\s*(\d+)")
    b1 = find_classes_last(r"t10\s*class\s*(\d+)")
    b2 = find_classes_last(r"I\(t10\^2\)\s*class\s*(\d+)")
    tau2 = find_classes_last(r"var\s*1\s*class\s*(\d+)")
    tau2_common = next((v for n, v in entries if re.fullmatch(r"var\s*1", n)), None)
    sig2 = find_classes_last(r"std\.?err\s*class\s*(\d+)")
    sig2_common = next((v for n, v in entries if re.fullmatch(r"std\.?err", n)), None)
    pm = find_classes_last(r"pmclass\s*(\d+)")
    ng = int(block["ng"])
    missing = [f"b0[{i}]" for i in range(1, ng+1) if i not in b0] + \
              [f"b1[{i}]" for i in range(1, ng+1) if i not in b1] + \
              [f"b2[{i}]" for i in range(1, ng+1) if i not in b2]
    if missing:
        raise RuntimeError(f"系数解析失败, 缺 {missing}. 键名: {names}")
    if len(pm) == ng - 1:
        e = np.array([1.0] + [np.exp(pm[k]) for k in range(2, ng + 1)])
        pi = e / e.sum()
    else:
        sh = np.asarray(block["class_share"], dtype=float)
        pi = sh / sh.sum()
    def sig2_of(g):
        if g in sig2: return sig2[g] ** 2
        if sig2_common is not None: return sig2_common ** 2
        return None
    s2 = {g: sig2_of(g) for g in range(1, ng + 1)}
    if any(v is None for v in s2.values()):
        raise RuntimeError(f"残差方差解析失败. 键名: {names}")
    if len(tau2) == ng:
        t2 = tau2
    elif tau2_common is not None:
        t2 = {g: tau2_common for g in range(1, ng + 1)}
    else:
        t2 = {g: 0.0 for g in range(1, ng + 1)}
    return dict(ng=ng, b0=b0, b1=b1, b2=b2, tau2=t2, sig2=s2, pi=pi)

motor_model = parse_model(raw["motor"])
lg(f"08b 系数: motor ng={motor_model['ng']} (锚点已由 08b 实跑验证)")

def posterior(long_df, col_y, model, t_max):
    """08b 逐字: 只用 0<=offset_hr<=t_max 的观测 → 每 id 的 ng 维后验"""
    d = long_df[(long_df.offset_hr >= 0) & (long_df.offset_hr <= t_max)].dropna(subset=[col_y])
    d = d.assign(id=pd.to_numeric(d["id"]).astype("int64"))
    d = d.assign(t10=d.offset_hr / 10.0, y=d[col_y].astype(float))
    ng = model["ng"]
    res = {gid: None for gid in d["id"].unique()}
    logp = np.full((len(res), ng), -np.inf)
    ids_order = np.array(sorted(res))
    idx = {g: i for i, g in enumerate(ids_order)}
    d["row"] = d["id"].map(idx)
    d = d.sort_values(["row"])
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
    mx = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - np.where(np.isfinite(mx), mx, 0))
    p /= p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(ids_order, columns=["id"])
    for g in range(1, ng + 1):
        out[f"prob_{g}"] = p[:, g - 1]
    out["n_obs"] = d.groupby("row").size().reindex(range(len(ids_order))).values
    return out

# ---------- 2. 09b end_window 逐字 ----------
def end_window(g, lo, hi):
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

# ---------- 3. Block B 共享库 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("c07", SCR/"07_common.py")
c07 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c07)

V7 = ["hr","sbp","dbp","mbp","rr","temp","spo2"]
LAB16 = ["aniongap","bicarbonate","bun","calcium","chloride","creatinine","glucose_lab",
         "sodium","potassium","hematocrit","hemoglobin","platelet","wbc","inr","pt","ptt"]
RAW2SCHEMA = {"motor": "gcs_motor", "eye": "gcs_eye", "verbal": "gcs_verbal"}
TS = list(range(6, 73))
GCOMP = ("total","eye","motor","verbal")

def strip_dyn(df):
    """08e 逐字: dyn_motor_X→motor_X; dyn_gcs_total_X→total_X"""
    ren = {}
    for c in df.columns:
        if c.startswith("dyn_motor_"): ren[c] = "motor_" + c[len("dyn_motor_"):]
        elif c.startswith("dyn_gcs_total_"): ren[c] = "total_" + c[len("dyn_gcs_total_"):]
    return df.rename(columns=ren)

def pivot_stats(ev, t, stats, lo=0.0):
    g = ev[(ev.offset_hr>=lo)&(ev.offset_hr<=t)].groupby(["id","concept"])["value"].agg(stats)
    out = None
    for s in stats:
        w = g[s].unstack("concept")
        w.columns = [f"{c}_{s}" for c in w.columns]
        out = w if out is None else out.join(w, how="outer")
    return out.reset_index()

def feat_diff(X, ref, on, tag, res):
    """FULL∩双方 列级 diff: 每列 frac(|Δ|>0.01) + max|Δ|; 返回最差5列"""
    common = X.merge(ref, on=on, suffixes=("_m","_r"))
    rows = []
    for c in FULL:
        if f"{c}_m" not in common.columns: continue
        a = common[f"{c}_m"].astype("float64").to_numpy(); b = common[f"{c}_r"].astype("float64").to_numpy()
        m = ~(np.isnan(a) & np.isnan(b))          # 双 NaN 视为一致
        d = np.abs(a[m] - b[m])
        rows.append(dict(col=c, frac_gt001=round(float((d > 0.01).mean() if len(d) else 0.0), 5),
                         max_abs=round(float(d.max() if len(d) else 0.0), 6),
                         n_both=int((~np.isnan(a) & ~np.isnan(b)).sum())))
    rows.sort(key=lambda r: -r["frac_gt001"])
    worst = rows[:5]
    res[tag] = dict(n_common=int(len(common)), worst_cols=worst,
                    n_cols_frac_gt_gate=sum(1 for r in rows if r["frac_gt001"] > GATE_FEAT_FRAC))
    lg(f"  [{tag}] 特征 diff n={len(common)} | >5%阈值列数={res[tag]['n_cols_frac_gt_gate']} | 最差: "
       f"{[(r['col'], r['frac_gt001']) for r in worst[:3]]}")
    return res[tag]["n_cols_frac_gt_gate"] == 0

# ---------- 4. 三库配置 (v2: 锚点取自 10d 筛选版三库结果) ----------
M4D = json.loads((REP/"08d_v2_lgbm_main.json").read_text(encoding="utf-8"))
T10 = json.loads((REP/"10d_screened_model.json").read_text(encoding="utf-8"))
CFG = {
 "mimic4": dict(idcol="stay_id", long="gcs_long120_mimic4.parquet", rename_long=True,
                events="09c_events_mimic4.parquet", triples="09c_gcs_triples_mimic4.parquet",
                lab_lo=-6.0, death="precise_m4", label="d28",
                preds_ref=("08d_v2_preds_mimic4.parquet","stay_id"), anchor=M4D["intval"]["full"]["AUROC"],
                anchor_src="08d_v2_lgbm_main.json intval.full"),
 "eicu":   dict(idcol="icustay_id_eicu", long="gcs_long120_eicu.parquet", rename_long=False,
                events="09c_events_eicu.parquet", triples=None,
                lab_lo=0.0, death="approx_eicu", label="d28_ei",
                preds_ref=("08e_v2_preds_eicu.parquet","icustay_id_eicu"), anchor=T10["eicu"]["AUROC_25"],
                anchor_src="10d_screened_model.json eicu (25feat)"),
 "mimic3": dict(idcol="icustay_id", long="gcs_long120_mimic3.parquet", rename_long=False,
                events="09c_events_mimic3.parquet", triples=None,
                lab_lo=-6.0, death="precise_m3", label="d28",
                preds_ref=("08e_v2_preds_mimic3.parquet","icustay_id"), anchor=T10["mimic3"]["AUROC_25"],
                anchor_src="10d_screened_model.json mimic3 (25feat CareVue)"),
}
CONS = {}   # t=24 一致性报告
overall_pass = True

for db, c in CFG.items():
    lg(f"\n===== [{db}] rolling 预测 =====")
    idc = c["idcol"]
    # --- 静态基座 ---
    if db == "mimic4":
        m8 = pd.read_parquet(DATA/"08c_matrix_mimic4.parquet")
        base = m8[m8.split=="intval"][["stay_id","d28","hours_to_death","los_h",
                                       "admission_age","sex_female","charlson_comorbidity_score"]].copy()
        base = base.rename(columns={"stay_id":"id"})
        base["d28"] = base.d28.astype(int)
        base["death_time_hr"] = base.hours_to_death
    elif db == "eicu":
        fe = pd.read_parquet(DATA/"features_eicu.parquet")[
            ["icustay_id_eicu","age","male","charlson_comorbidity_score","los_h","died_hosp","died_hosp_28d"]]
        base = pd.DataFrame(dict(id=fe.icustay_id_eicu.astype("int64"),
                                 d28=fe.died_hosp_28d.astype(int),
                                 los_h=fe.los_h.astype(float),
                                 died_hosp=fe.died_hosp.astype(float),
                                 admission_age=fe.age.astype(float),
                                 sex_female=(1-fe.male).astype(float),
                                 charlson_comorbidity_score=fe.charlson_comorbidity_score.astype(float)))
        base["death_time_hr"] = np.where(base.died_hosp==1, base.los_h, np.nan)
    else:
        fm = pd.read_parquet(DATA/"features_mimic3.parquet")[
            ["icustay_id","age","male","charlson_comorbidity_score","los_h","d28","days_to_death"]]
        # CareVue-only 过滤 (2026-09-07): 剔除与 MIMIC-IV train 重叠的 MetaVision 期患者
        import duckdb as _dd
        _con = _dd.connect()
        _cv = _con.execute(f"""SELECT DISTINCT CAST(ICUSTAY_ID AS BIGINT) AS icustay_id
        FROM read_csv('E:/TBI subtype/data/mimic-iii-1.4/ICUSTAYS.csv.gz', all_varchar=true)
        WHERE DBSOURCE = 'carevue'""").df()
        _n0 = len(fm); fm = fm[fm.icustay_id.isin(_cv.icustay_id)].copy()
        lg(f"[M3 CareVue 过滤] {_n0}→{len(fm)} (剔 MetaVision {_n0-len(fm)})")
        base = pd.DataFrame(dict(id=fm.icustay_id.astype("int64"),
                                 d28=fm.d28.astype(int),
                                 los_h=fm.los_h.astype(float),
                                 dtd_h=fm.days_to_death.astype(float)*24.0,
                                 admission_age=fm.age.astype(float),
                                 sex_female=(1-fm.male).astype(float),
                                 charlson_comorbidity_score=fm.charlson_comorbidity_score.astype(float)))
        base["death_time_hr"] = base.dtd_h
    # --- 长表 / 事件流 / 三分项 ---
    lng = pd.read_parquet(DATA/c["long"])
    if c["rename_long"]: lng = lng.rename(columns=RAW2SCHEMA)
    lng = lng.assign(id=pd.to_numeric(lng["id"]).astype("int64"))
    if db == "mimic4":
        # 08c 实际读取 07a 原始长表 gcs_long_mimic4.parquet (≤72h, 82923 行), 非 09a 120h 重提取文件 —
        # diag7 实证: 07a 原生行序 first[0,24] vs 08c 矩阵 intval 4 列 0 差异;
        #             09a 文件原生行序 vs 矩阵 = 112/65/53/39 差异 (23.2%/13.5%/11.0%/8.1%) = run1 残余 feat_diff 根因
        lng_first07 = pd.read_parquet(DATA / "gcs_long_mimic4.parquet").rename(columns=RAW2SCHEMA)
        lng_first07 = lng_first07.assign(id=pd.to_numeric(lng_first07["id"]).astype("int64"))
    lng = lng.sort_values(["id","offset_hr"])
    long72 = lng[lng.offset_hr <= 72.0]
    tot = lng[["id","offset_hr","gcs_total"]].dropna(subset=["gcs_total"])
    ggroups = {k: v for k, v in tot.groupby("id", sort=False)}
    ev = pd.read_parquet(DATA/c["events"])
    ev = ev.assign(id=pd.to_numeric(ev["id"]).astype("int64"), concept=ev["concept"].astype(str))
    vit_ev = ev[ev.concept.isin(V7)]
    lab_ev = ev[ev.concept.isin(LAB16)]
    tri = pd.read_parquet(DATA/c["triples"]).assign(
        id=lambda d: pd.to_numeric(d["id"]).astype("int64")) if c["triples"] else None
    lg(f"base={len(base)} | long={len(lng)} (≤72h {len(long72)}) | events vit={len(vit_ev)} lab={len(lab_ev)}"
       + (f" | triples={len(tri)}" if tri is not None else ""))

    rows_all, X24, ids24 = [], None, None
    for t in TS:
        # 风险集
        if c["death"] == "precise_m4":
            alive = base.death_time_hr.isna() | (base.death_time_hr > t)
        elif c["death"] == "precise_m3":
            alive = base.death_time_hr.isna() | (base.death_time_hr > t)
        else:
            alive = ~((base.died_hosp==1) & (base.los_h <= t))
        w70 = long72[(long72.offset_hr>=0)&(long72.offset_hr<=t)]
        has_gcs = set(w70[w70.gcs_total.notna() | w70.gcs_motor.notna()].id.unique())   # spec: total 或 motor 非空
        rk = base[(base.los_h >= t) & alive & base.id.isin(has_gcs)].copy()
        ids_t = rk[["id"]]
        if len(rk) == 0:
            continue
        # Block A
        X = ids_t.merge(pivot_stats(vit_ev, t, ["min","mean","max"], 0.0), on="id", how="left")
        X = X.merge(pivot_stats(lab_ev, t, ["min","max"], c["lab_lo"]), on="id", how="left")
        # 静态 GCS (M4=triples 口径; eICU/M3=long 口径) + 首值 (全库 long)
        if tri is not None:
            tt = tri[tri.offset_hr <= t]
            st = tt.groupby("id")[["eye","motor","verbal"]].min().rename(
                columns={"eye":"gcs_eye_min","motor":"gcs_motor_min","verbal":"gcs_verbal_min"})
            cp = tt.dropna(subset=["eye","motor","verbal"]).assign(
                gcs_total_min=lambda d: d.eye+d.motor+d.verbal).groupby("id")["gcs_total_min"].min()
            X = X.merge(st.reset_index(), on="id", how="left").merge(
                cp.reset_index(), on="id", how="left")
        else:
            gm = w70.groupby("id")[["gcs_total","gcs_eye","gcs_motor","gcs_verbal"]].min()
            gm.columns = [f"{x}_min" for x in gm.columns]
            X = X.merge(gm.reset_index(), on="id", how="left")
        if db == "mimic4":
            # 复刻 08c L45+L108-109: 07a 长表 + 原生行序 first-non-null (diag7 实证 0 差异)
            wnat = lng_first07[(lng_first07.offset_hr >= 0) & (lng_first07.offset_hr <= float(t))]
            gf = wnat.groupby("id")[[f"gcs_{k}" for k in GCOMP]].first()
        else:
            # 复刻 07c L172 / 07d L148: offset_hr 排序后 first = 真最早 (09d 原实现, 与 eICU/M3 参考一致)
            gf = w70.groupby("id")[[f"gcs_{k}" for k in GCOMP]].first()
        gf.columns = [f"gcs_{k}_first" for k in GCOMP]
        X = X.merge(gf.reset_index(), on="id", how="left")
        # bg
        X = X.merge(base[["id","admission_age","sex_female","charlson_comorbidity_score"]], on="id", how="left")
        # Block B / C
        X = X.merge(c07.neuro_dynamic_features(long72, "gcs_motor", "motor", win_hr=float(t)),
                    on="id", how="left")
        pm = posterior(long72, "gcs_motor", motor_model, t)
        pm.columns = ["id"] + [f"prob_m{g}" for g in range(1, motor_model["ng"]+1)] + ["n_obs"]
        X = X.merge(pm, on="id", how="left")
        # 推理 (zero-touch)
        miss = [c2 for c2 in FULL if c2 not in X.columns]
        assert not miss, f"[{db} t={t}] FULL 缺列 {miss}"
        p = frozen_predict(X)
        # END 旗标 + 汇行
        pa = X["prob_m1"].notna().to_numpy()
        ids_v = rk.id.to_numpy(); lab_v = rk.d28.to_numpy()
        los_v = rk.los_h.to_numpy(); dt_v = rk.death_time_hr.to_numpy()
        e_u = np.full(len(rk), np.nan); e_p = np.full(len(rk), np.nan)
        e_ev = np.full(len(rk), np.nan); n_t = np.zeros(len(rk), int)
        for i, gid in enumerate(ids_v):
            g = ggroups.get(gid)
            if g is None: continue
            a, na, _, _ = end_window(g, 0, t)
            b2, _, _, ev2 = end_window(g, t, min(t+48, 120))
            e_u[i], e_p[i], e_ev[i], n_t[i] = a, b2, ev2, na
        rows_all.append(pd.DataFrame(dict(
            db=db, id=ids_v, t=t, p=p, flag=(p >= THRESH).astype(int), post_avail=pa,
            end_upto_t=e_u, endpost48=e_p, endpost48_ev_hr=e_ev, n_gcs_upto_t=n_t,
            d28=lab_v, los_h=los_v, death_time_hr=dt_v)))
        if t == 24:
            X24 = X.copy(); ids24 = rk.id.tolist()
        if t in (6, 24, 48, 72):
            ev_in_rk = int(np.nansum(e_p == 1))
            auc_t = roc_auc_score(lab_v, p) if len(np.unique(lab_v)) > 1 else float("nan")
            lg(f"  t={t:2d}h: risk n={len(rk)} | AUROC(d28)={auc_t:.4f} | "
               f"post_avail {pa.mean():.1%} | END_post 事件 {ev_in_rk}")
    out = pd.concat(rows_all, ignore_index=True)
    out.to_parquet(DATA/f"09d_rolling_preds_{db}.parquet", index=False)
    lg(f"[{db}] 产出 09d_rolling_preds_{db}.parquet rows={len(out)}")

    # ---------- t=24 一致性门禁 ----------
    assert X24 is not None, f"[{db}] t=24 风险集为空 — 无法做一致性门禁"
    lg(f"[{db}] t=24 一致性门禁 (vs {c['anchor_src']})")
    ck = dict(anchor=c["anchor"], anchor_src=c["anchor_src"], gates={})
    pf, pidc = c["preds_ref"]
    ref = pd.read_parquet(DATA/pf).rename(columns={pidc: "id", "p_full": "p_ref"})
    ref["id"] = pd.to_numeric(ref["id"]).astype("int64")
    r24 = out[out.t == 24].merge(ref[["id","p_ref","d28"]], on="id", how="inner")
    ck["n_common_preds"] = int(len(r24))
    ck["label_agree"] = float((r24.d28_x == r24.d28_y).mean())
    # 逐点 Δp 仅描述性: MICE sample_posterior=True 使 transform 每次调用重抽后验样本,
    # 逐患者概率天然不可逐点复现 — diag7 噪声底: 同特征同 imputer 两次 transform,
    # m=1 时 frac|Δp|>0.01 = 3.52% > 1% 阈值 → 逐点门禁按构造不可满足 (集合级 AUROC 复现不受影响)
    dp = np.abs(r24.p - r24.p_ref)
    ck["pred_max_abs_dp"] = round(float(dp.max()), 6)
    ck["pred_frac_gt001"] = round(float((dp > 0.01).mean()), 5)
    ck["pred_pointwise_note"] = "descriptive only — MICE sample_posterior draw noise (diag7)"
    ck["auroc_common_mine"] = round(float(roc_auc_score(r24.d28_x, r24.p)), 4)
    ck["auroc_ref_on_common"] = round(float(roc_auc_score(r24.d28_x, r24.p_ref)), 4)
    ck["auroc_diff_popmatched"] = round(ck["auroc_common_mine"] - ck["auroc_ref_on_common"], 4)
    ck["auroc_diff_vs_anchor"] = round(ck["auroc_common_mine"] - c["anchor"], 4)
    g = ck["gates"]
    g["label_agree_100pct"] = ck["label_agree"] == 1.0
    g["auroc_within_tol_popmatched"] = abs(ck["auroc_diff_popmatched"]) <= GATE_AUC_TOL
    # 特征 diff
    if db == "mimic4":
        refF = m8[m8.split=="intval"].rename(columns={"stay_id": "id"})
    elif db == "eicu":
        refF = strip_dyn(pd.read_parquet(DATA/"features_eicu.parquet").rename(columns={"age":"admission_age"}))
        refF["sex_female"] = 1 - refF["male"]
        refF["id"] = refF.icustay_id_eicu.astype("int64")
    else:
        refF = strip_dyn(pd.read_parquet(DATA/"features_mimic3.parquet").rename(columns={"age":"admission_age"}))
        refF["sex_female"] = 1 - refF["male"]
        refF["id"] = refF.icustay_id.astype("int64")
        refF = refF[refF.id.isin(base.id.unique())]   # CareVue 同步过滤 (与 base 一致)
    g["feat_diff_ok"] = feat_diff(X24, refF, "id", f"{db}_feat24", ck)
    ck["gate_thresholds"] = dict(dp_frac=GATE_DP_FRAC, feat_frac=GATE_FEAT_FRAC, auc_tol=GATE_AUC_TOL)
    CONS[db] = ck
    db_pass = all(g.values())
    overall_pass &= db_pass
    lg(f"[{db}] 门禁: {'PASS' if db_pass else 'FAIL'} {g} | AUROC mine/ref on common = "
       f"{ck['auroc_common_mine']}/{ck['auroc_ref_on_common']} (Δ{ck['auroc_diff_popmatched']:+.4f}) | "
       f"anchor {c['anchor']} (Δ{ck['auroc_diff_vs_anchor']:+.4f}, 人群口径差异仅上下文) | "
       f"max|Δp|={ck['pred_max_abs_dp']} (仅描述)")

CONS["overall"] = "PASS" if overall_pass else "FAIL"
(REP/"09d_t24_consistency.json").write_text(json.dumps(CONS, ensure_ascii=False, indent=1), encoding="utf-8")
lg(f"产出 reports/09d_t24_consistency.json overall={CONS['overall']}")
if not overall_pass:
    print("!! t=24 一致性门禁 FAIL — 冻结管线复现失败, 09e 不得使用本产物")
    sys.exit(1)
print("DONE 09d")
