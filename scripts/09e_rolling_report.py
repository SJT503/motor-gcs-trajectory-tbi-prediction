# -*- coding: utf-8 -*-
# ============================================================
# 09e — Rolling 报告 (09_rolling_spec.md §四 机械执行)
# 输入: data/09d_rolling_preds_{db}.parquet (09d 产物; 须 09d_t24_consistency overall=PASS)
# 产出: reports/09e_rolling_report.json
#       reports/09e_fig2_d28.csv + 09e_fig2_endpost.csv (AUROC-vs-t + bootstrap 500 95% 带)
#       reports/09e_warning_time.csv
# §四 冻结条款:
#   - 24h = 确证性锚点, 其余 t 描述性不占 α
#   - 预警: 单次超阈即触发(主) + 连续2次(敏感性); 阈值 = 冻结 Youden (card)
#   - warning time: 事件者 首次触发→事件 中位时距+IQR; 28d 死亡只报 lead time
#   - 12h 窗 sens/spec (Zhang 口径操作化, 记 log): 仅 END_post —
#       事件时刻 = 患者最早 END 事件时刻 (各 t 窗最早 ev_hr); sens = 事件者 [ev-12,ev] 内 ≥1 触发占比
#       spec = 无事件者(END([0,72])=0) 全程零触发占比
#   - FAC 操作化(记 log): 非事件者 触发起始次数(0→1 转换, 首检即超阈计1) / (检查点数/24) × 100 患者-日
#   - SOFA 对照: rolling 未建+触发规则未冻结 → 挂旗待 PI, 仅附 24h 静态 SOFA 上下文, 不发明阈值
# ============================================================
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"E:/TBI subtype")
DATA = ROOT/"07_prediction_system/data"; REP = ROOT/"07_prediction_system/reports"
MDL = ROOT/"07_prediction_system/models"
DBS = ["mimic4", "eicu", "mimic3"]
CONF_T = 24
B = 500

cons = json.loads((REP/"09d_t24_consistency.json").read_text(encoding="utf-8"))
assert cons.get("overall") == "PASS", f"09d t=24 门禁未过 ({cons.get('overall')}) — 09e 不得使用"
card = json.loads((MDL/"08d_v2_model_card.json").read_text(encoding="utf-8"))
THRESH = float(card["threshold_youden_train_oof"])
print(f"=== 09e rolling 报告 | 冻结 Youden={THRESH:.4f} | bootstrap B={B} | 确证锚点 t={CONF_T}h ===")

def boot_band(y, p, seed=42):
    y = np.asarray(y, int); p = np.asarray(p, float)
    if y.sum() == 0 or y.sum() == len(y): return [None, None]
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(q), 4) for q in np.percentile(aucs, [2.5, 97.5])]

def trig_seqs(g):
    """患者级触发序列 → dict(t→flag) + 连续2次触发集合"""
    seq = dict(zip(g.t, g.flag))
    two = {t for t, f in seq.items() if f == 1 and seq.get(t - 1) == 1}
    return seq, two

def onsets(seq):
    """0→1 转换数 (首检即超阈计 1) — FAC 操作化"""
    n, prev = 0, 0
    for t in sorted(seq):
        if seq[t] == 1 and prev == 0: n += 1
        prev = seq[t]
    return n

def med_iqr(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0: return dict(n=0, median=None, q1=None, q3=None)
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2),
                q1=round(float(np.percentile(x, 25)), 2), q3=round(float(np.percentile(x, 75)), 2))

res = {"threshold_frozen_youden": THRESH, "bootstrap_B": B, "confirmatory_anchor_h": CONF_T,
       "curves": {}, "warning": {}, "fac": {}, "operationalizations": {
        "12h_window": "事件时刻=各 t 窗最早 END 事件时刻; sens=事件者[ev-12,ev]内≥1触发; spec=END([0,72])=0者全程零触发",
        "fac": "非事件者 0→1 触发转换数(首检超阈计1) / (在险检查点数/24h) × 100 患者-日"},
       "sofa_comparator": {"flag": "rolling SOFA 未建且 SOFA 触发规则未冻结 — 挂旗待 PI, 不静默发明阈值",
                           "static_24h_context": {}}}
cur_d28, cur_end, warn_rows = [], [], []

# 24h 静态 SOFA 上下文 (有则附, 无则缺省)
for fn, key in [("08z_sofa_report.json", "mimic4"), ("08i_sofa_eicu.json", "eicu"), ("08h_sofa_mimic3.json", "mimic3")]:
    p = REP/fn
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        hits = {}
        def walk(d, path=""):
            for k, v in (d.items() if isinstance(d, dict) else []):
                if isinstance(v, (int, float)) and ("auroc" in k.lower()): hits[path+k] = round(float(v), 4)
                elif isinstance(v, dict): walk(v, path+k+".")
        walk(j)
        res["sofa_comparator"]["static_24h_context"][key] = dict(file=fn, auroc_like=hits)

for db in DBS:
    d = pd.read_parquet(DATA/f"09d_rolling_preds_{db}.parquet")
    d = d.sort_values(["id", "t"])
    print(f"\n[{db}] rows={len(d)} ids={d.id.nunique()}")
    # ---------- AUROC-vs-t 曲线 ----------
    for t in sorted(d.t.unique()):
        r = d[d.t == t]
        # 28d
        y = r.d28.to_numpy(int); p = r.p.to_numpy(float)
        if len(np.unique(y)) > 1:
            lo, hi = boot_band(y, p, seed=42 + int(t))
            cur_d28.append(dict(db=db, t=int(t), n=len(r), events=int(y.sum()),
                                auroc=round(float(roc_auc_score(y, p)), 4), lo=lo, hi=hi,
                                anchor_24h=(t == CONF_T)))
        # END_post: 风险集 = end_upto_t==0 且 endpost48 可计算
        e = r[(r.end_upto_t == 0) & r.endpost48.notna()]
        if len(e) and e.endpost48.nunique() > 1:
            y2 = e.endpost48.to_numpy(int); p2 = e.p.to_numpy(float)
            lo2, hi2 = boot_band(y2, p2, seed=42 + int(t))
            cur_end.append(dict(db=db, t=int(t), n=len(e), events=int(y2.sum()),
                                auroc=round(float(roc_auc_score(y2, p2)), 4), lo=lo2, hi=hi2,
                                anchor_24h=(t == CONF_T)))
    a24 = [c for c in cur_d28 if c["db"] == db and c["t"] == CONF_T]
    e24 = [c for c in cur_end if c["db"] == db and c["t"] == CONF_T]
    print(f"  AUROC@24: d28={a24[0]['auroc'] if a24 else None} | endpost={e24[0]['auroc'] if e24 else None}")

    # ---------- 预警 / warning time / 12h 窗 / FAC ----------
    W = {}
    grp = {pid: g for pid, g in d.groupby("id")}   # 预分组 (防逐 pid 全表扫描)
    # END: 事件时刻 = 各窗最早 ev_hr; 无事件 = end_upto_72==0
    ev_time = {}
    for pid, g in grp.items():
        eg = g[(g.endpost48 == 1) & g.endpost48_ev_hr.notna()]
        if len(eg): ev_time[pid] = float(eg.endpost48_ev_hr.min())
    u72 = d[d.t == 72]
    no_end_ids = set(u72[u72.end_upto_t == 0].id) if len(u72) else set()
    for rule in ("single", "two_consec"):
        leads_end, det, sens_12h = [], 0, 0
        for pid, g in grp.items():
            seq, two = trig_seqs(g)
            active = two if rule == "two_consec" else {t for t, f in seq.items() if f == 1}
            if pid in ev_time:
                ev = ev_time[pid]
                pre = [t for t in active if t <= ev]
                if pre:
                    det += 1; leads_end.append(ev - min(pre))
                if any(ev - 12 <= t <= ev for t in active): sens_12h += 1
        n_ev = len(ev_time)
        # spec: 无事件者全程零触发
        spec_cnt, spec_n = 0, 0
        for pid in no_end_ids:
            seq, two = trig_seqs(grp[pid])
            active = two if rule == "two_consec" else {t for t, f in seq.items() if f == 1}
            spec_n += 1
            if not active: spec_cnt += 1
        W[f"end_{rule}"] = dict(
            n_events=n_ev, detected_before_event=det,
            detection_rate=round(det / n_ev, 4) if n_ev else None,
            warning_time_hr=med_iqr(leads_end),
            sens_12h_window=round(sens_12h / n_ev, 4) if n_ev else None,
            spec_no_trigger=round(spec_cnt / spec_n, 4) if spec_n else None,
            n_nonevent_for_spec=int(spec_n))
        # 28d lead time (只报 lead)
        leads_d = []
        for pid, g in grp.items():
            if g.d28.iloc[0] != 1: continue
            dth = g.death_time_hr.iloc[0]
            if not np.isfinite(dth): continue
            seq, two = trig_seqs(g)
            active = two if rule == "two_consec" else {t for t, f in seq.items() if f == 1}
            pre = [t for t in active if t <= dth]
            if pre: leads_d.append(dth - min(pre))
        W[f"d28_{rule}"] = dict(lead_time_hr=med_iqr(leads_d))
        # FAC: 非事件者
        d28_nonev = {pid for pid, g in grp.items() if g.d28.iloc[0] == 0}
        for out, nonev_ids in (("end", no_end_ids), ("d28", d28_nonev)):
            on, pts = 0, 0
            for pid in nonev_ids:
                seq, two = trig_seqs(grp[pid])
                # run2 缺陷修复 (核验代理 F-1): 原写法 seq,_ 丢弃 two 集合 → two_consec FAC 恒等于 single;
                # 正确口径 = 按该规则自身的触发指示序列数 0→1 转换 (首检即触发计 1)
                active = two if rule == "two_consec" else {t for t, f in seq.items() if f == 1}
                ind = {t: (1 if t in active else 0) for t in sorted(seq)}
                on += onsets(ind); pts += len(seq)
            W[f"fac_{out}_{rule}"] = dict(
                trigger_onsets=int(on), patient_checkpoints=int(pts),
                patient_days=round(pts / 24.0, 2),
                false_alarms_per_100_patient_days=round(on / (pts / 24.0) * 100, 2) if pts else None)
    res["warning"][db] = W
    wr = [dict(db=db, **{k: v for k, v in W[r2].items()}) for r2 in
          ("end_single", "end_two_consec", "d28_single", "d28_two_consec")]
    warn_rows.extend(wr)
    print(f"  END(s): det={W['end_single']['detection_rate']} warn_med={W['end_single']['warning_time_hr']['median']}h "
          f"sens12h={W['end_single']['sens_12h_window']} spec={W['end_single']['spec_no_trigger']} | "
          f"FAC_end={W['fac_end_single']['false_alarms_per_100_patient_days']}/100pd")

# ---------- 落盘 ----------
cd = pd.DataFrame(cur_d28); ce = pd.DataFrame(cur_end)
cd.to_csv(REP/"09e_fig2_d28.csv", index=False)
ce.to_csv(REP/"09e_fig2_endpost.csv", index=False)
pd.DataFrame(warn_rows).to_csv(REP/"09e_warning_time.csv", index=False)
res["curves"] = dict(
    d28=dict(n_points=len(cd), by_db={b: int((cd.db == b).sum()) for b in DBS}),
    endpost=dict(n_points=len(ce), by_db={b: int((ce.db == b).sum()) for b in DBS}))
def _scalar(v):
    # pandas Series.iloc[0] 取出的 numpy 标量 → Python 标量 (run1: anchor_24h 的 n/events 为 int64 致 json.dumps 崩)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v
res["anchor_24h"] = {b: dict(
    d28={k: _scalar(cd[(cd.db == b) & (cd.t == CONF_T)][k].iloc[0]) if len(cd[(cd.db == b) & (cd.t == CONF_T)]) else None
         for k in ("auroc", "lo", "hi", "n", "events")},
    endpost={k: _scalar(ce[(ce.db == b) & (ce.t == CONF_T)][k].iloc[0]) if len(ce[(ce.db == b) & (ce.t == CONF_T)]) else None
             for k in ("auroc", "lo", "hi", "n", "events")}) for b in DBS}
res["flags"] = ["M4 曲线人群 = temporal 内验子集 (spec §二预注册)",
                "eICU 死亡时刻 = 出院时刻近似 (Methods 披露)",
                "24h 外各 t 描述性不占 α",
                "SOFA rolling 对照待 PI 决策"]
(REP/"09e_rolling_report.json").write_text(json.dumps(
    res, ensure_ascii=False, indent=1,
    default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o)),
    encoding="utf-8")
print(f"\n产出: reports/09e_rolling_report.json | 09e_fig2_d28.csv ({len(cd)} pts) | "
      f"09e_fig2_endpost.csv ({len(ce)} pts) | 09e_warning_time.csv")
print("DONE 09e")
