# -*- coding: utf-8 -*-
from gate import gate, canary_gate, retry_limit
"""
19f_v4_build_verified_calculator.py — 100% Pure 5-Booster Bedside-5 (1,500 LightGBM Trees) Web Calculator (v4.2)
Remediates Opus 5.5 Max R11:
  1. Eliminates ALL heuristic blending (0.55*LR + 0.45*Bedside-5, 0.5*priorShift, synthetic midVal LCGM interpolation)!
  2. Executes ONLY the exact validated 5-imputation x 300-tree = 1,500-tree LightGBM Bedside-5 ensemble + Tune-Select Platt
     calibration + dedicated Bedside-5 untouched Tune-Calib 90% Mondrian Split-Conformal thresholds (tau_low_b5, tau_high_b5)!
  3. When 24-h flowsheet motor-GCS is unrecorded, routes to the exact 5-imputation x 200-tree = 1,000-tree Static-3 LightGBM
     fallback + dedicated Static-3 Mondrian thresholds!
  4. Clearly labeled as an open-source research & validation demonstration tool per R11.
"""
import json
import numpy as np
from pathlib import Path
from scipy.special import expit, logit

ROOT = Path("E:/TBI subtype")
SYS  = ROOT / "07_prediction_system"
gate(str(SYS / "scripts/19f_v4_build_verified_calculator.py"))

ART_PATH = SYS / "models/19c_v4_web_calculator_artifact.json"
OUT_HTML = SYS / "deployment/tbi_conformal_bedside_calculator.html"

def slim_tree(node):
    if "leaf_value" in node:
        return {"v": round(float(node["leaf_value"]), 7)}
    return {
        "f": int(node["split_feature"]),
        "t": round(float(node["threshold"]), 6),
        "l": slim_tree(node["left_child"]),
        "r": slim_tree(node["right_child"]),
    }

def eval_slim_tree_py(node, x_vec):
    if "v" in node:
        return node["v"]
    return eval_slim_tree_py(node["l"], x_vec) if x_vec[node["f"]] <= node["t"] else eval_slim_tree_py(node["r"], x_vec)

@retry_limit(2)
def main():
    art = json.loads(ART_PATH.read_text(encoding="utf-8"))
    canary_gate(lambda: len(art["bedside5_5boosters"]["boosters_trees"]) == 5, "5-booster Bedside-5 artifact loaded")

    b5_raw = art["bedside5_5boosters"]
    s3_raw = art["static3_5boosters"]
    b5_boosters_slim = [[slim_tree(t["tree_structure"]) for t in b_trees] for b_trees in b5_raw["boosters_trees"]]
    s3_boosters_slim = [[slim_tree(t["tree_structure"]) for t in b_trees] for b_trees in s3_raw["boosters_trees"]]

    test_x = [58.0, 7.0, 3.0, 5.0, 5.0]
    p_5_mean = float(np.mean([expit(sum(eval_slim_tree_py(tr, test_x) for tr in b)) for b in b5_boosters_slim]))
    p_b5_cal = float(expit(b5_raw["platt_intercept"] + b5_raw["platt_slope"] * logit(np.clip(p_5_mean, 1e-5, 1 - 1e-5))))
    assert 0.001 < p_b5_cal < 0.999, f"Invalid calibrated 5-booster Bedside-5 probability: {p_b5_cal}"

    payload = {
        "version": art["version"],
        "b5_conf": art["bedside5_conformal_90_thresholds"],
        "s3_conf": art["static3_conformal_90_thresholds"],
        "b5": {
            "features": b5_raw["features"],
            "platt_slope": round(b5_raw["platt_slope"], 6),
            "platt_intercept": round(b5_raw["platt_intercept"], 6),
            "intval_auroc": b5_raw["intval_auroc"],
            "intval_brier": b5_raw["intval_brier"],
            "intval_cov": b5_raw["intval_conformal_coverage"],
            "boosters": b5_boosters_slim
        },
        "s3": {
            "features": s3_raw["features"],
            "platt_slope": round(s3_raw["platt_slope"], 6),
            "platt_intercept": round(s3_raw["platt_intercept"], 6),
            "boosters": s3_boosters_slim
        },
        "cohorts": art["cohort_prevalence_priors"]
    }
    payload_json_str = json.dumps(payload, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bedside-5 24-h Motor-GCS Updating &amp; Split-Conformal Research Calculator (1,500 LightGBM Trees)</title>
<style>
  :root {{
    --navy: #0B3C5D; --navy-lt: #EBF2F7; --burgundy: #8B1E24;
    --teal: #006D5B; --teal-lt: #E8F4F1; --crimson: #B8292F; --crim-lt: #FBE9EA;
    --amber: #B95815; --amb-lt: #FDF2E9; --ivory: #FAF8F3; --stone: #D6D1C4;
    --charcoal: #1F2428; --slate: #5A656E;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--ivory); color: var(--charcoal); line-height: 1.5; padding: 20px;
  }}
  .container {{
    max-width: 1120px; margin: 0 auto; background: #FFFFFF;
    border: 1px solid var(--stone); border-top: 6px solid var(--burgundy);
    border-radius: 8px; box-shadow: 0 6px 20px rgba(11, 60, 93, 0.08); padding: 24px 28px;
  }}
  .header-title {{ font-size: 1.28rem; font-weight: 700; color: var(--navy); margin-bottom: 4px; }}
  .header-sub {{ font-size: 0.84rem; color: var(--slate); margin-bottom: 16px; border-bottom: 1px solid var(--stone); padding-bottom: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1.05fr 1.0fr; gap: 24px; }}
  @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .panel {{ background: var(--ivory); border: 1px solid var(--stone); border-radius: 6px; padding: 18px; }}
  .panel-title {{
    font-size: 0.92rem; font-weight: 700; color: var(--burgundy); text-transform: uppercase;
    letter-spacing: 0.03em; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;
  }}
  .form-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
  .field {{ display: flex; flex-direction: column; }}
  .field label {{ font-size: 0.79rem; font-weight: 600; color: var(--charcoal); margin-bottom: 4px; }}
  .field select, .field input[type="number"] {{
    padding: 7px 10px; border: 1px solid #BDC3C7; border-radius: 4px; font-size: 0.85rem; background: #FFFFFF; color: var(--charcoal);
  }}
  .section-divider {{ font-size: 0.81rem; font-weight: 700; color: var(--navy); margin: 13px 0 8px 0; padding-bottom: 4px; border-bottom: 1px dashed var(--stone); }}
  .result-hero {{ background: var(--navy); color: #FFFFFF; border-radius: 6px; padding: 18px; margin-bottom: 14px; }}
  .risk-row {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; flex-wrap: wrap; gap: 8px; }}
  .risk-big {{ font-size: 2.10rem; font-weight: 800; letter-spacing: -0.02em; }}
  .risk-ci {{ font-size: 0.86rem; background: rgba(255,255,255,0.16); padding: 4px 10px; border-radius: 4px; font-weight: 600; }}
  .meter-wrap {{ background: rgba(255,255,255,0.18); height: 16px; border-radius: 8px; position: relative; overflow: hidden; margin: 12px 0 6px 0; }}
  .meter-ambig {{ position: absolute; top: 0; bottom: 0; background: rgba(241, 196, 15, 0.38); border-left: 1.5px dashed #F1C40F; border-right: 1.5px dashed #F1C40F; }}
  .meter-pt {{ position: absolute; top: 0; bottom: 0; width: 4px; background: #FFFFFF; }}
  .gate-banner {{ padding: 11px 13px; border-radius: 6px; font-size: 0.82rem; font-weight: 600; margin-bottom: 12px; border-left: 5px solid; }}
  .gate-pass {{ background: var(--teal-lt); border-color: var(--teal); color: #004D40; }}
  .gate-abstain {{ background: var(--amb-lt); border-color: var(--amber); color: #7A3E0A; }}
  .gate-high {{ background: var(--crim-lt); border-color: var(--crimson); color: #7A1C20; }}
  .gate-awaken {{ background: var(--navy-lt); border-color: var(--navy); color: var(--navy); }}
  .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }}
  .kpi-card {{ background: #FFFFFF; border: 1px solid var(--stone); border-radius: 5px; padding: 10px 12px; }}
  .kpi-label {{ font-size: 0.72rem; color: var(--slate); text-transform: uppercase; font-weight: 600; }}
  .kpi-val {{ font-size: 1.00rem; font-weight: 700; color: var(--charcoal); margin-top: 2px; }}
  .btn-bar {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
  .btn-preset {{
    padding: 6px 11px; font-size: 0.77rem; font-weight: 600; border: 1px solid var(--navy);
    background: #FFFFFF; color: var(--navy); border-radius: 4px; cursor: pointer;
  }}
  .btn-preset:hover {{ background: var(--navy); color: #FFFFFF; }}
  .footer-note {{ font-size: 0.74rem; color: var(--slate); margin-top: 14px; border-top: 1px solid var(--stone); padding-top: 10px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header-title">Bedside-5 24-h Motor-GCS Updating &amp; 90% Split-Conformal Research Demonstration Tool</div>
  <div class="header-sub">
    Executes the <strong>exact 5-imputation &times; 300-tree (1,500 LightGBM trees) Bedside-5 ensemble</strong> + untouched Tune-Calib 90% Mondrian Split-Conformal thresholds (or optional local site prevalence/cutoff calibration) &middot; Routes unrecorded 24-h flowsheet cases to the 1,000-tree Static-3 baseline
  </div>
  <div class="gate-banner gate-high" style="margin-bottom:14px;">
    <strong>Neurocritical Care Ethical Guardrail (&lt; 72 h):</strong> In accordance with Neurocritical Care Society guidelines, 24-h high-risk singleton flags (<code>{{1}}</code>) are intended solely to guide repeat head-CT imaging and invasive neuromonitoring escalation, and must <strong>never</strong> be used to justify early withdrawal of life-sustaining treatment (WLST) within 72 h of injury.
  </div>

  <div class="btn-bar">
    <button class="btn-preset" onclick="loadCase('awakener')">Preset 1: 24-h Rapid Motor Improvement (Admission GCS 4 → 24h mGCS 6)</button>
    <button class="btn-preset" onclick="loadCase('truesevere')">Preset 2: Persistent Severe Motor Suppression (Admission GCS 5 → 24h mGCS 2)</button>
    <button class="btn-preset" onclick="loadCase('uncharted')">Preset 3: Unrecorded 24-h Flowsheet mGCS (Modality-Adaptive Static Routing + Abstention)</button>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="panel-title">
        <span>1. Validated Five-Variable Zero-Laboratory Inputs</span>
        <span style="font-size:0.74rem;color:var(--navy);">24-h Landmark</span>
      </div>

      <div class="form-row">
        <div class="field">
          <label for="age">Admission Age (years, ≥18)</label>
          <input type="number" id="age" value="48" min="18" max="100" oninput="computeRisk()">
        </div>
        <div class="field">
          <label for="gcs0">Admission Total GCS (3–15)</label>
          <input type="number" id="gcs0" value="5" min="3" max="15" oninput="computeRisk()">
        </div>
      </div>

      <div class="section-divider">A. Admission Baseline Motor Response (0 h)</div>
      <div class="form-row">
        <div class="field">
          <label for="mgcs0">Admission Motor-GCS (gcs_motor_first, 1–6)</label>
          <select id="mgcs0" onchange="computeRisk()">
            <option value="1">1 — No motor response</option>
            <option value="2">2 — Abnormal extension</option>
            <option value="3" selected>3 — Abnormal flexion</option>
            <option value="4">4 — Withdraws from pain</option>
            <option value="5">5 — Localizes pain</option>
            <option value="6">6 — Obeys commands</option>
          </select>
        </div>
        <div class="field">
          <label>&nbsp;</label>
          <div style="font-size:0.75rem;color:var(--slate);padding-top:4px;">Evaluates exact 1,500 exported LightGBM trees locally in browser (zero network calls).</div>
        </div>
      </div>

      <div class="section-divider">B. 24-h Post-Resuscitation Bedside Motor-GCS (0–24 h)</div>
      <div class="form-row">
        <div class="field">
          <label for="mgcs24">24-h Last Charted Motor-GCS (motor_last)</label>
          <select id="mgcs24" onchange="computeRisk()">
            <option value="0">UNRECORDED / Unexported Flowsheet</option>
            <option value="1">1 — No motor response</option>
            <option value="2">2 — Abnormal extension</option>
            <option value="3">3 — Abnormal flexion</option>
            <option value="4">4 — Withdraws from pain</option>
            <option value="5">5 — Localizes pain (Rapid Improvement)</option>
            <option value="6" selected>6 — Obeys commands (Rapid Improvement)</option>
          </select>
        </div>
        <div class="field">
          <label for="mgcsmax">24-h Best Motor-GCS Achieved (motor_max)</label>
          <select id="mgcsmax" onchange="computeRisk()">
            <option value="1">1 — No motor response</option>
            <option value="2">2 — Abnormal extension</option>
            <option value="3">3 — Abnormal flexion</option>
            <option value="4">4 — Withdraws from pain</option>
            <option value="5">5 — Localizes pain</option>
            <option value="6" selected>6 — Obeys commands</option>
          </select>
        </div>
      </div>

      <div class="section-divider">C. Site Calibration Mode (Boston Default vs. Local Site Calibration)</div>
      <div class="form-row">
        <div class="field">
          <label for="calmode">Calibration Profile</label>
          <select id="calmode" onchange="toggleSiteCal(); computeRisk();">
            <option value="boston" selected>MIMIC-IV Tune-Calib Default (prev 14.9%)</option>
            <option value="local">Local Site Prevalence &amp; Mondrian Cutoffs (n &ge; 150–200)</option>
          </select>
        </div>
        <div class="field">
          <label for="locprev">Local 28-d Mortality Prevalence (%, if Local Mode)</label>
          <input type="number" id="locprev" value="12.0" min="1.0" max="60.0" step="0.5" disabled oninput="computeRisk()">
        </div>
      </div>
      <div class="form-row" id="localCutRow" style="display:none;">
        <div class="field">
          <label for="loctaulow">Local 90% Mondrian &tau;<sub>low</sub> (0.001–0.50)</label>
          <input type="number" id="loctaulow" value="0.030" min="0.001" max="0.50" step="0.005" oninput="computeRisk()">
        </div>
        <div class="field">
          <label for="loctauhigh">Local 90% Mondrian &tau;<sub>high</sub> (0.05–0.90)</label>
          <input type="number" id="loctauhigh" value="0.280" min="0.05" max="0.90" step="0.005" oninput="computeRisk()">
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">
        <span>2. Exact 5-Booster Bedside-5 Output &amp; 90% Mondrian Set</span>
        <span id="modeBadge" style="font-size:0.73rem;background:var(--teal);color:#fff;padding:2px 8px;border-radius:4px;">Pure 5-Booster Bedside-5</span>
      </div>

      <div class="result-hero">
        <div style="font-size:0.77rem;text-transform:uppercase;opacity:0.86;font-weight:600;">MIMIC-IV Calibrated 28-Day Mortality Risk (24-h Landmark)</div>
        <div class="risk-row">
          <div class="risk-big" id="pReanchor">--%</div>
          <div class="risk-ci" id="pConformalSet">90% Set: --</div>
        </div>
        <div class="meter-wrap">
          <div class="meter-ambig" id="meterAmbig"></div>
          <div class="meter-pt" id="meterPt"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.74rem;opacity:0.90;">
          <span id="pStaticTxt">Admission Static-3 Baseline: --%</span>
          <span id="pB5Txt">24-h Bedside-5 Ensemble: --%</span>
        </div>
      </div>

      <div id="gateBanner" class="gate-banner gate-pass"></div>
      <div id="awakenBanner" class="gate-banner gate-awaken"></div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Active Validated Model</div>
          <div class="kpi-val" id="kpiModality">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Bedside-5 90% Conformal Zone</div>
          <div class="kpi-val" id="kpiZone">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Temporal Val Performance (n=483)</div>
          <div class="kpi-val" id="kpiValPerf">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">External 5-Cohort AUROC Range</div>
          <div class="kpi-val">0.837 – 0.883 (92.7–99.3%)</div>
        </div>
      </div>

      <div class="kpi-card" style="background:#FFFFFF;">
        <div class="kpi-label">Research Interpretation Note (Non-Interventional Demonstration)</div>
        <div id="clinicalRec" style="font-size:0.82rem;margin-top:4px;color:var(--charcoal);"></div>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <strong>Research Demonstration Disclaimer:</strong> Provided strictly as an open-source research reproducibility and validation demonstration tool accompanying the manuscript; external clinical deployment requires local prevalence recalibration and prospective impact evaluation. All 1,500 Bedside-5 trees and 1,000 Static-3 trees are exported directly from <code>models/19c_v4_web_calculator_artifact.json</code> with zero heuristic post-hoc weights.
  </div>
</div>

<script>
const ART = {payload_json_str};

function sigmoid(z) {{
  if (z > 35) return 0.999999;
  if (z < -35) return 0.000001;
  return 1.0 / (1.0 + Math.exp(-z));
}}
function logitFn(p) {{
  const c = Math.min(0.99999, Math.max(0.00001, p));
  return Math.log(c / (1.0 - c));
}}
function evalTree(node, xVec) {{
  if (node.v !== undefined) return node.v;
  return (xVec[node.f] <= node.t) ? evalTree(node.l, xVec) : evalTree(node.r, xVec);
}}
function eval5Boosters(mdl, xVec) {{
  let pSum = 0.0;
  for (let j = 0; j < mdl.boosters.length; j++) {{
    const trees = mdl.boosters[j];
    let margin = 0.0;
    for (let t = 0; t < trees.length; t++) margin += evalTree(trees[t], xVec);
    pSum += sigmoid(margin);
  }}
  const pMean = pSum / mdl.boosters.length;
  return sigmoid(mdl.platt_intercept + mdl.platt_slope * logitFn(pMean));
}}

function loadCase(preset) {{
  if (preset === 'awakener') {{
    document.getElementById('age').value = 46;
    document.getElementById('gcs0').value = 4;
    document.getElementById('mgcs0').value = '2';
    document.getElementById('mgcs24').value = '6';
    document.getElementById('mgcsmax').value = '6';
  }} else if (preset === 'truesevere') {{
    document.getElementById('age').value = 69;
    document.getElementById('gcs0').value = 5;
    document.getElementById('mgcs0').value = '2';
    document.getElementById('mgcs24').value = '2';
    document.getElementById('mgcsmax').value = '2';
  }} else if (preset === 'uncharted') {{
    document.getElementById('age').value = 64;
    document.getElementById('gcs0').value = 6;
    document.getElementById('mgcs0').value = '3';
    document.getElementById('mgcs24').value = '0';
    document.getElementById('mgcsmax').value = '3';
  }}
  computeRisk();
}}

function toggleSiteCal() {{
  const isLoc = (document.getElementById('calmode').value === 'local');
  document.getElementById('locprev').disabled = !isLoc;
  document.getElementById('localCutRow').style.display = isLoc ? 'grid' : 'none';
}}

function computeRisk() {{
  const age = parseFloat(document.getElementById('age').value) || 55;
  const gcs0 = parseFloat(document.getElementById('gcs0').value) || 10;
  const mgcs0 = parseFloat(document.getElementById('mgcs0').value) || 4;
  const mgcs24_raw = parseInt(document.getElementById('mgcs24').value, 10);
  const mgcsMax_in = parseFloat(document.getElementById('mgcsmax').value) || 5;
  const isLocalCal = (document.getElementById('calmode').value === 'local');

  const unrecorded = (mgcs24_raw === 0);
  const effMax = unrecorded ? mgcs0 : Math.max(mgcs24_raw, mgcsMax_in);

  let pS0 = eval5Boosters(ART.s3, [age, gcs0, mgcs0]);
  let pFinal = unrecorded ? pS0 : eval5Boosters(ART.b5, [age, gcs0, mgcs0, mgcs24_raw, effMax]);
  const conf = unrecorded ? ART.s3_conf : ART.b5_conf;
  let tauLow = conf.tau_low;
  let tauHigh = conf.tau_high;
  if (isLocalCal) {{
    const prevLoc = Math.min(0.60, Math.max(0.01, (parseFloat(document.getElementById('locprev').value) || 12.0) / 100.0));
    const dLogit = logitFn(prevLoc) - logitFn(0.1489);
    pS0 = sigmoid(logitFn(pS0) + dLogit);
    pFinal = sigmoid(logitFn(pFinal) + dLogit);
    tauLow = Math.min(0.50, Math.max(0.001, parseFloat(document.getElementById('loctaulow').value) || conf.tau_low));
    tauHigh = Math.min(0.95, Math.max(tauLow + 0.01, parseFloat(document.getElementById('loctauhigh').value) || conf.tau_high));
  }}

  const in0 = (pFinal <= tauHigh);
  const in1 = (pFinal >= tauLow);
  let setStr = '';
  if (unrecorded) setStr = '{{0, 1}} (Abstain — 24h Flowsheet Unrecorded)';
  else if (in0 && !in1) setStr = '{{0: Survival}} (Definitive Low-Risk Set)';
  else if (in1 && !in0) setStr = '{{1: 28-d Death}} (Definitive High-Risk Set)';
  else if (in0 && in1) setStr = '{{0, 1}} (Ambiguous — Abstain / Re-assess)';
  else setStr = (pFinal >= 0.5 * (tauLow + tauHigh)) ? '{{1}}' : '{{0}}';

  document.getElementById('pReanchor').textContent = (pFinal * 100).toFixed(1) + '%';
  document.getElementById('pConformalSet').textContent = '90% Set: ' + setStr;
  const ambigLeft = Math.min(tauLow, tauHigh) * 100;
  const ambigWidth = Math.max(1.5, Math.abs(tauHigh - tauLow) * 100);
  document.getElementById('meterAmbig').style.left = ambigLeft.toFixed(1) + '%';
  document.getElementById('meterAmbig').style.width = ambigWidth.toFixed(1) + '%';
  document.getElementById('meterPt').style.left = (pFinal * 100).toFixed(1) + '%';
  document.getElementById('pStaticTxt').textContent = `Admission Static-3 (1,000 trees): ${{(pS0 * 100).toFixed(1)}}%`;
  document.getElementById('pB5Txt').textContent = unrecorded ? '24-h Bedside-5: Unrecorded (Routed to Static-3)' : `24-h Bedside-5 (1,500 trees): ${{(pFinal * 100).toFixed(1)}}%`;

  const gateEl = document.getElementById('gateBanner');
  const badgeEl = document.getElementById('modeBadge');
  if (unrecorded) {{
    gateEl.className = 'gate-banner gate-abstain';
    gateEl.innerHTML = '<strong>MODALITY-ADAPTIVE ROUTING &amp; DOCUMENTATION ABSTENTION:</strong> 24-h flowsheet motor-GCS is unrecorded. Dynamic motor trees are bypassed (preventing MICE imputation distortion), outputting the Static-3 baseline risk and routing conformal triage to {{0, 1}} abstention.';
    badgeEl.textContent = 'Static-3 Fallback + Abstain';
    badgeEl.style.background = 'var(--amber)';
  }} else if (in0 && in1) {{
    gateEl.className = 'gate-banner gate-abstain';
    gateEl.innerHTML = `<strong>CONFORMAL AMBIGUITY ZONE ({{0, 1}}):</strong> Calibrated Bedside-5 risk (${{(pFinal*100).toFixed(1)}}%) lies within [${{(tauLow*100).toFixed(1)}}%, ${{(tauHigh*100).toFixed(1)}}%]. Defer binary threshold classification and repeat bedside neurological assessment.`;
    badgeEl.textContent = 'Ambiguous Set {{0, 1}}';
    badgeEl.style.background = 'var(--amber)';
  }} else if (in1 && !in0) {{
    gateEl.className = 'gate-banner gate-high';
    gateEl.innerHTML = '<strong>SINGLETON HIGH-RISK CONFORMAL SET ({{1}}):</strong> Survival excluded at 90% split-conformal calibration level on MIMIC-IV Tune-Calib.';
    badgeEl.textContent = 'Singleton High-Risk {{1}}';
    badgeEl.style.background = 'var(--crimson)';
  }} else {{
    gateEl.className = 'gate-banner gate-pass';
    gateEl.innerHTML = '<strong>SINGLETON LOW-RISK CONFORMAL SET ({{0}}):</strong> 28-day mortality excluded at 90% split-conformal calibration level on MIMIC-IV Tune-Calib.';
    badgeEl.textContent = 'Singleton Low-Risk {{0}}';
    badgeEl.style.background = 'var(--teal)';
  }}

  const awkEl = document.getElementById('awakenBanner');
  if (!unrecorded && gcs0 <= 8 && mgcs24_raw >= 5) {{
    awkEl.style.display = 'block';
    awkEl.innerHTML = `<strong>24-H MOTOR IMPROVEMENT OBSERVED:</strong> Admission GCS ${{gcs0}} (≤ 8) with 24-h motor-GCS ${{mgcs24_raw}} (≥ 5).`;
  }} else if (!unrecorded && gcs0 <= 8 && mgcs24_raw <= 4) {{
    awkEl.style.display = 'block';
    awkEl.innerHTML = `<strong>PERSISTENT 24-H MOTOR SUPPRESSION:</strong> Admission GCS ${{gcs0}} (≤ 8) with 24-h motor-GCS ${{mgcs24_raw}} (≤ 4).`;
  }} else {{
    awkEl.style.display = 'none';
  }}

  document.getElementById('kpiModality').textContent = unrecorded ? 'Static-3 (5×200 Trees)' : 'Bedside-5 (5×300 Trees)';
  document.getElementById('kpiZone').textContent = `[${{(tauLow*100).toFixed(1)}}% – ${{(tauHigh*100).toFixed(1)}}%]`;
  document.getElementById('kpiValPerf').textContent = `AUROC ${{ART.b5.intval_auroc.toFixed(3)}} | Cov ${{(ART.b5.intval_cov*100).toFixed(1)}}%`;

  const recEl = document.getElementById('clinicalRec');
  if (unrecorded) {{
    recEl.innerHTML = '<strong>Research Note:</strong> Unrecorded 24-h flowsheet subscores trigger dynamic-feature-free static routing and conformal abstention ({{0, 1}}).';
  }} else if (in1 && !in0) {{
    recEl.innerHTML = '<strong>Research Note:</strong> Model places patient in high-risk singleton set ({{1}}); prospective impact studies are needed before guiding clinical escalation.';
  }} else if (in0 && in1) {{
    recEl.innerHTML = '<strong>Research Note:</strong> Indeterminate conformal zone ({{0, 1}}); prediction interval spans both outcomes at the 90% nominal level.';
  }} else {{
    recEl.innerHTML = '<strong>Research Note:</strong> Model places patient in low-risk singleton set ({{0}}) following 24-h bedside motor updating.';
  }}
}}

computeRisk();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[SAVED PURE 5-BOOSTER VERIFIED CALCULATOR] {OUT_HTML} ({OUT_HTML.stat().st_size:,} bytes) | Test 5-booster Bedside-5 prob = {p_b5_cal:.6f}")

if __name__ == "__main__":
    main()
