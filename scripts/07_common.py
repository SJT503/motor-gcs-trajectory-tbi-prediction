# 07 系列共享库: Block B 神经动态特征全家桶 (SAP_prediction §3.2 冻结清单) + 覆盖率审计
# 依赖 07a 的 gcs_long parquet (id, offset_hr, gcs_total, gcs_motor)
import numpy as np, pandas as pd

def _sampen(x, m=2, r_scale=0.2):
    """样本熵 SampEn(m,r) 标准实现; SD=0 → 0.0(完全恒定=零变异, 报告计 n_sd0); n<m+3 → NaN(太短走MICE)"""
    x = np.asarray(x, float); n = len(x); sd = np.std(x)
    if sd == 0: return 0.0
    if n < m+3: return np.nan
    r = r_scale*sd
    def _phi(mm):
        cnt = 0; tot = 0
        for i in range(n-mm):
            for j in range(i+1, n-mm):
                if np.max(np.abs(x[i:i+mm]-x[j:j+mm])) <= r: cnt += 1
                tot += 1
        return cnt, tot
    A, N = _phi(m+1); B, Nm = _phi(m)
    return np.nan if (A == 0 or B == 0) else -np.log(A/B)

def _apen(x, m=2, r_scale=0.2):
    """近似熵 ApEn(m,r) 标准实现(含 self-match); SD=0 → 0.0; n<m+2 → NaN"""
    x = np.asarray(x, float); n = len(x); sd = np.std(x)
    if sd == 0: return 0.0
    if n < m+2: return np.nan
    r = r_scale*sd
    def _phi(mm):
        cnt = 0; tot = 0
        for i in range(n-mm+1):
            for j in range(n-mm+1):
                if np.max(np.abs(x[i:i+mm]-x[j:j+mm])) <= r: cnt += 1
                tot += 1
        return cnt/tot
    return _phi(m+1)-_phi(m)

def neuro_dynamic_features(long_df, value_col="gcs_motor", prefix="motor",
                           win_hr=24.0, min_obs=2):
    """长表 → 每患者 Block B 特征 (窗口 [0, win_hr], SAP §3.2 冻结清单)
    返回 DataFrame(id, n, first, last, min, max, delta, rate_per_hr, sd, cv, slope, sampen, apen)"""
    sub = long_df[(long_df.offset_hr>=0)&(long_df.offset_hr<=win_hr)].dropna(subset=[value_col])
    rows = []
    for pid, g in sub.groupby("id"):
        g = g.sort_values("offset_hr"); v = g[value_col].values.astype(float); h = g.offset_hr.values; n = len(v)
        r = {"id": pid, f"{prefix}_n": n}
        if n < min_obs:
            rows.append(r); continue
        span = h.max()-h.min()
        slope = float(np.polyfit(h, v, 1)[0]) if (n>=2 and span>0) else 0.0
        sd = float(np.std(v, ddof=1)) if n>=2 else 0.0
        r.update({f"{prefix}_first": v[0], f"{prefix}_last": v[-1], f"{prefix}_min": v.min(),
                  f"{prefix}_max": v.max(), f"{prefix}_delta": v[-1]-v[0],
                  f"{prefix}_rate_per_hr": (v[-1]-v[0])/span if span>0 else np.nan,
                  f"{prefix}_sd": sd, f"{prefix}_cv": sd/v.mean() if v.mean()!=0 else np.nan,
                  f"{prefix}_slope": slope, f"{prefix}_sampen": _sampen(v), f"{prefix}_apen": _apen(v)})
        rows.append(r)
    return pd.DataFrame(rows)

def coverage_audit(df, id_cols=("id",), out_csv=None):
    """特征覆盖率审计 (SAP §3.6: >20% 缺失 → DROP 标记); 返回 DataFrame"""
    feat = [c for c in df.columns if c not in id_cols]
    cov = pd.DataFrame({"feature": feat,
                        "coverage_pct": [round(100*df[c].notna().mean(),1) for c in feat]})
    cov["flag"] = np.where(cov.coverage_pct<80, "DROP_>20%miss", "keep")
    cov = cov.sort_values("coverage_pct").reset_index(drop=True)
    if out_csv: cov.to_csv(out_csv, index=False)
    return cov

def gate_check(name, n_expect, n_got, tol_pct=5.0):
    """入口检验: 行数/患者数与锚点对照, 超容差 → 如实报警不擅改"""
    if n_expect is None: return
    dev = 100*(n_got-n_expect)/n_expect
    flag = "OK" if abs(dev)<=tol_pct else f"⚠️ 偏差{dev:+.1f}% >{tol_pct}% — 报告差异, 不擅改口径"
    print(f"[GATE] {name}: got {n_got} vs expect {n_expect} [{flag}]")
