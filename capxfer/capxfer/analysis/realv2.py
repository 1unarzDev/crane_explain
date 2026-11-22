"""Frozen analysis of the PROTOCOL_V2 real-robot bonus study (docs/PROTOCOL_V2.md): R1-R4 + descriptive validation.

Entry point: `run_realv2(manifest_path) -> results dict` (also written to `<output_dir>/results_v2.json`).

Manifest (JSON or YAML; relative paths resolve against the manifest's directory):
  output_dir: out/
  freeze_record: freeze_v2.json      # {files: {relpath: sha256}, seed_commitment, freeze_time_utc, git_commit, root}
  seed_hex: ...                      # revealed sealed seed (after collection); optional until then
  mask_trials: mask.csv              # one row per MASK trial (columns: see MASK_COLS + rho_<key> for the 14 keys)
  predictions: pred.csv              # FROZEN per-condition sim predictions (hashed at the freeze; PRED_COLS)
  r3_units: r3.csv                   # one row per (MASK assembly trial, module) (R3_COLS)
  demo: demo.csv                     # one row per docking episode (DEMO_COLS)
  simval_day2: day2.csv              # optional, same columns as mask_trials (build = "single_M1")
  apriltag_static: tag_static.csv    # optional (pose, err_x_m, err_y_m, err_yaw_rad, std_x_m, std_y_m, std_yaw_rad, ref_unc_m)
  apriltag_motion: tag_motion.csv    # optional (trial_id, latency_s, speed_ratio)
  lambda_v2: lambda_v2.json          # optional (WS-I3F): the field-day calibration file, checked against R3's lambda_hat
  config: {...}                      # overrides of DEFAULTS (must equal the prereg)

Every numeric constant used by a decision rule is in DEFAULTS; if PROTOCOL_V2 and DEFAULTS disagree, the code wins and
the disagreement is a reported deviation. AI agents never create or edit field files; the CSVs above are produced by
the field loader/featurizer (truth from the evaluator-only AprilTag pose) and by the frozen certificate pipeline.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from ..probes import TARGET_KEYS
from . import realv2_design as D
from . import stats as S

KEYS = tuple(TARGET_KEYS)
RHO_COLS = tuple(f"rho_{k}" for k in KEYS)
STRAFE = ("rho_P2+", "rho_P2-")

MASK_COLS = ("trial_id", "session", "build", "assembly_type", "module", "wheel", "firmware", "excluded")
PRED_COLS = ("build", "module", "wheel", "L_pred", "z_pred", "k_star")      # module/wheel empty for healthy rows
R3_COLS = ("trial_id", "build", "assembly_type", "unit_module", "fault_mw", "r3_split", "score", "sim_lb",
           "carry_over_lb", "excluded")
DEMO_COLS = (("episode", "faulted_modules", "carry_over_lb", "nearest_goal_module", "random_module", "cert_tie_coin",
              "oracle_tie_coin", "carry_over_tie_coin", "excluded")
             + tuple(f"{c}_{m}" for m in D.DEMO_MODULES for c in ("y", "s", "z_alone")))
# y_<M>: dock success of module M (NaN iff no arm chose M); s_<M>: certificate score; z_alone_<M>: measured lone strafe;
# carry_over_lb: black-box current-body (L3) lower bound on z, applied unchanged to every module (D2, SPLIT (b)).
# PROTOCOL_V2 v3 (D16, WS-KIT2): optional DEMO columns pm_<M> (uncalibrated structured posterior-mean strafe of module M,
# from the same pre-split assembly probe; the POSTERIOR RANKING arm) and dock_design ("all3": every module docked in
# every episode -> an episode is valid iff all three docks are valid, a rule blind to the arm choices).
# Optional R3 column u_q05: the uncalibrated default one-sided 95 % posterior lower quantile of the module's strafe
# (go iff u_q05 >= tau), compared with the certificate per module (secondary, no verdict).
DEMO_OPT_COLS = tuple(f"pm_{m}" for m in D.DEMO_MODULES) + ("dock_design",)

DEFAULTS = dict(
    alpha_cert=0.10,            # certificate level (D10 primary)
    alpha_cert_sens=0.05,       # sensitivity
    test_alpha=0.05,            # one-sided level of every hypothesis test
    conf=0.95,                  # CP / bootstrap one-sided confidence
    tau=0.8,                    # primary strafe threshold for R2/R3 (one passive fault on a lone module)
    tau_secondary=0.5,          # reported, no verdict
    tau_margin=0.10,            # G-tau gate: every module's healthy lone strafe z must be >= tau + margin
    tau_gate_abs_min=0.75,      # v4 (D17): ... and its ABSOLUTE healthy lone strafe >= this (healthy-tracking floor)
    mask_loss_max=0.10,         # "masked in the assembly": worst-direction loss <= this AND assembly strafe z >= tau
    r1_delta=0.05,              # R1 sim-agreement equivalence margin on L (absolute)
    r1_equiv_conf=0.90,         # two-sided CI for the TOST-equivalent rule (= two one-sided 0.05 tests)
    r2_pi0=0.25,                # R2 null reveal proportion (carry-over predicts 0)
    r2_agree_min=0.75,          # R2 secondary: per-condition reveal agreement real vs sim (reported against this)
    n_boot=4000, seed=20261121,
    min_null_cal=None,          # None -> ceil(1/alpha) - 1 (finite conformal threshold)
    capability="relative",      # v4 (SPINE D17): rho / median rho of the same build's healthy MASK trials; "absolute"
                                # = the v1-v3 definition (sensitivity / legacy only)
    demo_dock_rule="auto",      # v3: "all3" (every module docked; valid iff all 3 docks valid), "chosen" (D15 dedup),
                                # "auto" = "all3" iff demo.csv has dock_design == "all3" on every row
    bands=dict(
        fidelity_multi=0.05,            # median |rho_real - rho_pred| healthy / masked multi-module
        fidelity_single_passive=0.10,   # lone module with one passive fault
        fidelity_roller=0.15,           # lone module F4x2d (roller contact is the weakest physics)
        repeat_sd=0.03,                 # pooled within-condition SD of rho per leg
        day_shift=0.05,                 # |day-2 mean - day-1 mean| per leg, same condition
        tag_static_std_m=0.002, tag_static_std_rad=0.004, tag_bias_extra_m=0.005,
        tag_latency_spread_s=0.03, tag_speed_ratio=(0.95, 1.02)),
)
STATUS_OK = "CONFIRMATORY_PROTOCOL_V2"
STATUS_BAD = "NOT_CONFIRMATORY_PROVENANCE_FAIL"
CODE_FILES = ("analysis/realv2.py", "analysis/realv2_design.py", "analysis/stats.py")


# ------------------------------------------------------------------------------------------------------ helpers
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else ("inf" if x == math.inf else ("-inf" if x == -math.inf else x))
    return x


def _prop(k, n, conf=0.95):
    lo, hi = S.clopper_pearson(k, n, conf) if n else (math.nan, math.nan)
    return dict(k=int(k), n=int(n), est=(k / n if n else math.nan), cp_lo=lo, cp_hi=hi,
                cp_lower_one_sided=S.cp_lower(k, n, conf), cp_upper_one_sided=S.cp_upper(k, n, conf))


def _mw(m, w):
    return f"{m}:{w}" if isinstance(m, str) and isinstance(w, str) and m and w else "none"


def strafe(rho):
    """Decision quantity z = min(rho_P2+, rho_P2-) (row-wise for a frame, or for a dict/Series)."""
    if isinstance(rho, pd.DataFrame):
        return rho[list(STRAFE)].min(axis=1)
    return float(min(rho[STRAFE[0]], rho[STRAFE[1]]))


def worst_direction_loss(rho_healthy, rho_fault):
    """L = max over the legs of (healthy mean - faulted mean); legs undefined in either are skipped."""
    a = np.asarray(rho_healthy, float) - np.asarray(rho_fault, float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else math.nan


# ------------------------------------------------------------------------------ relative capability (v4, D17)
ABS_COLS = tuple(f"abs_{c}" for c in RHO_COLS)


def _is_healthy(df):
    m = df["module"]
    return m.isna() | (m.astype(str).isin(["", "nan", "None"]))


def healthy_baseline(trials: pd.DataFrame) -> pd.DataFrame:
    """Per build: the per-leg MEDIAN rho over the build's non-excluded healthy trials (PROTOCOL_V2 v4, SPINE D17). For a
    lone build single_Mk these are that module's own healthy MASK trials. Index build, columns RHO_COLS."""
    t = trials[~trials["excluded"].astype(bool)]
    t = t[_is_healthy(t)]
    src = list(ABS_COLS) if set(ABS_COLS) <= set(t.columns) else list(RHO_COLS)
    b = t.groupby("build")[src].median()
    b.columns = list(RHO_COLS)
    return b


def to_relative(trials: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Relative capability: rho_<key> / healthy-baseline median of the same build (absolute copies in abs_rho_<key>).
    base: healthy_baseline() to use (default: from `trials` itself; SIMVAL day 2 uses the MASK baseline). A build
    without a baseline gets NaN. Idempotent (an already relative frame is returned unchanged)."""
    if set(ABS_COLS) <= set(trials.columns):
        return trials
    base = healthy_baseline(trials) if base is None else base
    t = trials.copy()
    for a, c in zip(ABS_COLS, RHO_COLS):
        t[a] = t[c].astype(float)
    B = base.reindex(t["build"].astype(str).to_numpy())
    t[list(RHO_COLS)] = t[list(ABS_COLS)].to_numpy(float) / B[list(RHO_COLS)].to_numpy(float)
    return t


def relative_trial_strafe(trials: pd.DataFrame) -> dict:
    """{trial_id: relative strafe z} of every trial (its build's healthy baseline)."""
    t = to_relative(trials)
    return dict(zip(t["trial_id"].astype(str), strafe(t).to_numpy(float)))


def relative_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """v4: when the prediction file carries the predicted rho_<key> and one healthy row per build (capxfer.field.predict
    pred.csv), L_pred / z_pred / k_star are recomputed on relative capability (predicted rho / predicted healthy rho of
    the same build: the estimator applied identically to sim and real). Otherwise returned unchanged."""
    if pred is None or not set(RHO_COLS) <= set(pred.columns):
        return pred
    p = pred.copy()
    h = p[_is_healthy(p)].groupby("build")[list(RHO_COLS)].median()
    if h.empty:
        return pred
    H = h.reindex(p["build"].astype(str).to_numpy())[list(RHO_COLS)].to_numpy(float)
    rel = p[list(RHO_COLS)].to_numpy(float) / H
    p["L_pred_abs"], p["z_pred_abs"] = p.get("L_pred"), p.get("z_pred")
    loss = 1.0 - rel
    fault = ~_is_healthy(p).to_numpy()
    ok = np.isfinite(loss).any(1)
    Lr = np.where(ok, np.nanmax(np.where(np.isfinite(loss), loss, -np.inf), 1), np.nan)
    ks = [KEYS[int(np.nanargmax(np.where(np.isfinite(r), r, -np.inf)))] if o else "" for r, o in zip(loss, ok)]
    zi = [KEYS.index("P2+"), KEYS.index("P2-")]
    p["L_pred"] = np.where(fault, Lr, 0.0)
    p["k_star"] = [k if f else "" for k, f in zip(ks, fault)]
    p["z_pred"] = np.min(rel[:, zi], 1)
    return p


# ---------------------------------------------------------------------------------------------- condition table
def condition_table(trials: pd.DataFrame, relative: bool = True, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (build, module, wheel) means over non-excluded trials, with the worst-direction loss L against the same
    build's healthy baseline, the loss l_star on the predicted worst leg (if `k_star` merged later), and strafe z.
    v4 (D17): relative=True (default) first divides every trial's rho by its build's healthy median (to_relative)."""
    if relative:
        trials = to_relative(trials, base)
    t = trials[~trials["excluded"].astype(bool)].copy()
    t["module"] = t["module"].fillna("").astype(str)
    t["wheel"] = t["wheel"].fillna("").astype(str)
    t["mw"] = [_mw(m, w) for m, w in zip(t["module"], t["wheel"])]
    g = t.groupby(["build", "assembly_type", "mw", "firmware"], dropna=False)
    means = g[list(RHO_COLS)].mean()
    n = g.size().rename("n_trials")
    c = means.join(n).reset_index()
    base = c[c["mw"] == "none"].set_index("build")
    out = []
    for _, r in c.iterrows():
        if r["mw"] == "none":
            continue
        if r["build"] not in base.index:
            L = math.nan
        else:
            L = worst_direction_loss(base.loc[r["build"], list(RHO_COLS)].to_numpy(float), r[list(RHO_COLS)].to_numpy(float))
        d = r.to_dict()
        d["L"] = L
        d["z"] = strafe(r)
        out.append(d)
    cond = pd.DataFrame(out)
    hb = base.reset_index()
    hb["z"] = strafe(hb)
    if set(ABS_COLS) <= set(t.columns):                            # v4: absolute healthy strafe for the G-tau floor
        ha = t[t["mw"] == "none"].groupby("build")[list(ABS_COLS)].mean()
        hb["z_abs"] = [float(min(ha.loc[b, "abs_rho_P2+"], ha.loc[b, "abs_rho_P2-"])) if b in ha.index else math.nan
                       for b in hb["build"]]
    else:
        hb["z_abs"] = hb["z"]
    cond.attrs["healthy"] = hb
    return cond


def lone_healthy_abs(cond: pd.DataFrame) -> dict:
    hb = cond.attrs["healthy"]
    return {b.split("_", 1)[1]: float(z) for b, z in zip(hb["build"], hb["z_abs"]) if str(b).startswith("single_")}


def lone_truth(cond: pd.DataFrame) -> tuple[dict, dict]:
    """(z_alone per 'M:w' faulted lone module, z_alone_healthy per module) from the single_* builds."""
    hb = cond.attrs["healthy"]
    zh = {b.split("_", 1)[1]: float(z) for b, z in zip(hb["build"], hb["z"]) if str(b).startswith("single_")}
    zf = {r.mw: float(r.z) for r in cond.itertuples() if str(r.build).startswith("single_")}
    return zf, zh


# --------------------------------------------------------------------------------------------------------- R1
def stratified_trend_test(blocks, types, values, order=D.MULTI):
    """Exact one-sided test of a decreasing trend of `values` along `order` (smaller assembly -> larger loss),
    stratified by block (the same faulted wheel in several assemblies). Statistic T = sum over blocks of the number
    of size-ordered pairs (i smaller than j) with v_i > v_j (ties 1/2). Under H0 the values are exchangeable across
    the types inside each block; the exact null distribution is the convolution of the per-block permutation
    distributions. Blocks with < 2 types carry no information and are dropped."""
    rank = {t: i for i, t in enumerate(order)}
    df = pd.DataFrame(dict(b=list(blocks), t=list(types), v=np.asarray(values, float)))
    df = df[df["t"].isin(rank) & np.isfinite(df["v"])]
    pmf = np.array([1.0])
    T = 0
    used = 0
    for _, g in df.groupby("b"):
        if len(g) < 2:
            continue
        g = g.assign(r=g["t"].map(rank)).sort_values("r")
        v = g["v"].to_numpy()
        k = len(v)
        T += int(round(2 * _conc(v)))
        perms = [v[list(p)] for p in itertools.permutations(range(k))]
        cnt = np.bincount([int(round(2 * _conc(p))) for p in perms])
        pmf = np.convolve(pmf, cnt / cnt.sum())
        used += 1
    if used == 0:
        return dict(T=math.nan, p=math.nan, n_blocks=0, T_max=0)
    p = float(pmf[T:].sum())
    return dict(T=T / 2, p=min(1.0, p), n_blocks=used, T_max=(len(pmf) - 1) / 2, E_T_H0=float(np.arange(len(pmf)) @ pmf) / 2)


def _conc(v):
    s = 0.0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            s += 1.0 if v[i] > v[j] else (0.5 if v[i] == v[j] else 0.0)
    return s


def equivalence_bootstrap(d, delta, conf=0.90, n_boot=4000, seed=0):
    """Percentile bootstrap CI (two-sided `conf`) of mean(d); pass iff the CI lies inside [-delta, +delta]
    (equivalent to two one-sided tests at (1 - conf) / 2). Units resampled iid (one value per condition)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": math.nan, "lo": math.nan, "hi": math.nan, "pass": False, "delta": delta,
                "status": "TOO_FEW"}
    rng = np.random.default_rng(seed)
    reps = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.quantile(reps, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    ok = bool(lo >= -delta and hi <= delta)
    return {"n": n, "mean": float(d.mean()), "median": float(np.median(d)), "lo": float(lo), "hi": float(hi),
            "pass": ok, "delta": delta, "status": "AVAILABLE"}


def r1(cond: pd.DataFrame, pred: pd.DataFrame, cfg) -> dict:
    multi = cond[cond["assembly_type"].isin(D.MULTI)].copy()
    pm = pred.copy()
    pm["mw"] = [_mw(m, w) for m, w in zip(pm["module"].fillna(""), pm["wheel"].fillna(""))]
    multi = multi.merge(pm[["build", "mw", "L_pred", "z_pred", "k_star"]], on=["build", "mw"], how="left")
    trend = stratified_trend_test(multi["mw"], multi["assembly_type"], multi["L"], D.MULTI)
    by_type, match_all = {}, True
    for i, t in enumerate(D.MULTI):
        s = multi[multi["assembly_type"] == t]
        d = (s["L"] - s["L_pred"]).to_numpy(float)
        eq = equivalence_bootstrap(d, cfg["r1_delta"], cfg["r1_equiv_conf"], cfg["n_boot"], cfg["seed"] + i)
        match_all &= eq["pass"]
        hb = cond.attrs["healthy"].set_index("build")
        lstar = [float(hb.loc[r["build"], f"rho_{r['k_star']}"] - r[f"rho_{r['k_star']}"])
                 for _, r in s.iterrows() if isinstance(r["k_star"], str) and f"rho_{r['k_star']}" in s.columns
                 and r["build"] in hb.index]
        Lr = s["L"].to_numpy(float)
        by_type[t] = dict(n_conditions=int(len(s)), L_median=float(np.nanmedian(Lr)) if len(s) else math.nan,
                          L_pred_median=float(np.nanmedian(s["L_pred"])) if len(s) else math.nan,
                          frac_L_le_mask_max=_prop(int(np.nansum(Lr <= cfg["mask_loss_max"])), int(np.isfinite(Lr).sum())),
                          equivalence=eq, l_star_median=(float(np.nanmedian(lstar)) if lstar else math.nan))
    trend_ok = bool(np.isfinite(trend["p"]) and trend["p"] < cfg["test_alpha"])
    verdict = "SUPPORTED" if (trend_ok and match_all) else "NOT_SUPPORTED"
    return dict(trend=trend, trend_holds=trend_ok, sim_match_all_types=bool(match_all), by_type=by_type,
                verdict=verdict, rule="trend p < test_alpha AND 90% bootstrap CI of mean(L_real - L_pred) inside "
                                      "+-r1_delta for each of pair, L3, square4")


# --------------------------------------------------------------------------------------------------------- R2
def tau_gate(zh: dict, tau, margin, zh_abs: dict | None = None, abs_min: float | None = None):
    """G-tau. v1-v3: every module's healthy lone strafe >= tau + margin. v4 (relative capability, where the healthy
    relative strafe is ~1 by construction): additionally every module's ABSOLUTE healthy lone strafe >= abs_min (the
    healthy-tracking floor; below it the normalisation divides by a module that barely follows its command)."""
    bad = {m: z for m, z in zh.items() if not (np.isfinite(z) and z >= tau + margin)}
    if zh_abs is not None and abs_min is not None:
        bad.update({f"{m}(absolute)": z for m, z in zh_abs.items() if not (np.isfinite(z) and z >= abs_min)})
    return dict(pass_=not bad and len(zh) > 0, failing_modules=bad)


def r2(cond: pd.DataFrame, pred: pd.DataFrame, cfg, tau=None) -> dict:
    tau = cfg["tau"] if tau is None else tau
    zf, zh = lone_truth(cond)
    gate = tau_gate(zh, tau, cfg["tau_margin"], lone_healthy_abs(cond), cfg.get("tau_gate_abs_min"))
    sq = cond[cond["assembly_type"] == "square4"].set_index("mw")
    pm = pred.copy()
    pm["mw"] = [_mw(m, w) for m, w in zip(pm["module"].fillna(""), pm["wheel"].fillna(""))]
    zp = {r.mw: r.z_pred for r in pm.itertuples() if str(r.build).startswith("single_")}
    rows = []
    for mw in sorted(zf):
        if mw not in sq.index:
            continue
        masked = bool(sq.loc[mw, "z"] >= tau and sq.loc[mw, "L"] <= cfg["mask_loss_max"])
        rows.append(dict(mw=mw, masked=masked, z_assembly=float(sq.loc[mw, "z"]), L_assembly=float(sq.loc[mw, "L"]),
                         z_alone=zf[mw], reveal=bool(zf[mw] < tau), z_pred_alone=zp.get(mw, math.nan)))
    df = pd.DataFrame(rows)
    out = dict(tau=tau, gate=dict(pass_=bool(gate["pass_"]), failing_modules=gate["failing_modules"]),
               n_conditions=len(df))
    if not gate["pass_"]:
        out["verdict"] = "NOT_EVALUABLE_TAU_GATE"
        return out
    M = df[df["masked"]] if len(df) else df
    k, n = int(M["reveal"].sum()) if len(M) else 0, int(len(M))
    p = float(sps.binom.sf(k - 1, n, cfg["r2_pi0"])) if n else math.nan
    agree = M[np.isfinite(M["z_pred_alone"])] if len(M) else M
    ka = int(((agree["z_pred_alone"] < tau) == agree["reveal"]).sum()) if len(agree) else 0
    out.update(n_masked=n, reveal=_prop(k, n), binom_p=p, pi0=cfg["r2_pi0"],
               carry_over_predicted_reveal=0.0,
               sim_agreement=dict(**_prop(ka, int(len(agree))), threshold=cfg["r2_agree_min"]),
               rows=df.to_dict("records"))
    if n == 0:
        out["verdict"] = "NOT_EVALUABLE_NO_MASKED_CONDITION"
    else:
        out["verdict"] = "SUPPORTED" if p < cfg["test_alpha"] else "NOT_SUPPORTED"
    return out


# --------------------------------------------------------------------------------------------------------- R3
def classcond_threshold(null_scores, alpha):
    """Null-calibrated (class-conditional) conformal threshold: certify iff score > lambda. lambda is the
    ceil((1 - alpha)(n0 + 1))-th smallest null calibration score; +inf (never certify) if n0 < ceil(1/alpha) - 1.
    Under exchangeability of a test null with the calibration nulls, P(score > lambda | null) <= alpha."""
    s = np.sort(np.asarray(null_scores, float))
    s = s[np.isfinite(s)] if s.size else s
    n0 = len(s)
    r = math.ceil((1 - alpha) * (n0 + 1))
    if r > n0:
        return math.inf
    return float(s[r - 1])


def nhg_validity_p(n0, n1, k_go, alpha):
    """Exact P(K >= k_go) for the number K of n1 test nulls certified by `classcond_threshold` calibrated on n0 nulls,
    under exchangeability of all n0 + n1 null scores (continuous, no ties). K = n1 - X, X = test items ranked below the
    r-th calibration item ~ negative hypergeometric."""
    r = math.ceil((1 - alpha) * (n0 + 1))
    if r > n0 or n1 == 0:
        return math.nan
    x = np.arange(n1 + 1)
    # P(X = x): among the first r + x - 1 items there are r - 1 calibration items and x test items, next item is cal
    lp = (sps.hypergeom.logpmf(x, n0 + n1, n1, r - 1 + x) + np.log((n0 - r + 1) / (n0 + n1 - r + 1 - x)))
    pX = np.exp(lp)
    pK = pX[::-1]                                   # K = n1 - X
    return float(min(1.0, pK[k_go:].sum()))


def r3_verdict(k, n, alpha, conf=0.95):
    """Three-way reading of a false-go count among unsafe test modules (no claim beyond 'not refuted')."""
    if n == 0:
        return "NOT_EVALUABLE_NO_UNSAFE_TEST_MODULE"
    lo, hi = S.cp_lower(k, n, conf), S.cp_upper(k, n, conf)
    if lo > alpha:
        return "REFUTED"
    return "NOT_REFUTED_STRONG" if hi <= 2 * alpha else "NOT_REFUTED"


def r3_units_truth(units: pd.DataFrame, zf: dict, zh: dict) -> pd.DataFrame:
    u = units[~units["excluded"].astype(bool)].copy()
    z = []
    for r in u.itertuples():
        fm = str(r.fault_mw)
        z.append(zf.get(fm, math.nan) if fm != "none" and fm.split(":")[0] == r.unit_module else zh.get(r.unit_module, math.nan))
    u["z_alone"] = z
    u["cluster"] = [str(f) if str(f) != "none" else f"healthy:{m}" for f, m in zip(u["fault_mw"], u["unit_module"])]
    return u


def r3(units: pd.DataFrame, cond: pd.DataFrame, cfg, alpha=None, tau=None) -> dict:
    alpha = cfg["alpha_cert"] if alpha is None else alpha
    tau = cfg["tau"] if tau is None else tau
    zf, zh = lone_truth(cond)
    gate = tau_gate(zh, tau, cfg["tau_margin"], lone_healthy_abs(cond), cfg.get("tau_gate_abs_min"))
    u = r3_units_truth(units, zf, zh)
    u = u[np.isfinite(u["z_alone"])]
    u["unsafe"] = u["z_alone"] < tau
    cal_null = u[(u["r3_split"] == "CAL") & u["unsafe"]]
    test = u[u["r3_split"] == "TEST"]
    lam = classcond_threshold(cal_null["score"], alpha)
    test = test.assign(go=test["score"] > lam, go_sim=test["sim_lb"] >= tau, go_carry=test["carry_over_lb"] >= tau)
    tn, ts = test[test["unsafe"]], test[~test["unsafe"]]
    k, n = int(tn["go"].sum()), int(len(tn))
    out = dict(alpha=alpha, tau=tau, gate=dict(pass_=bool(gate["pass_"]), failing_modules=gate["failing_modules"]),
               lambda_hat=lam, n_null_cal=int(len(cal_null)), n_null_test=n, n_safe_test=int(len(ts)),
               false_go=_prop(k, n, cfg["conf"]),
               verdict=("NOT_EVALUABLE_TAU_GATE" if not gate["pass_"] else
                        ("NOT_EVALUABLE_TOO_FEW_CAL_NULLS" if not np.isfinite(lam) else r3_verdict(k, n, alpha, cfg["conf"]))),
               nhg_exchangeability_p=nhg_validity_p(len(cal_null), n, k, alpha) if np.isfinite(lam) else math.nan)
    if n:
        W = S.bootstrap_weights(tn["cluster"].to_numpy(), cfg["n_boot"], cfg["seed"])
        out["false_go"]["cluster_boot"] = S.boot_ci(tn["go"].astype(float).to_numpy(), W, "mean", cfg["conf"])
        out["false_go"]["n_clusters"] = int(tn["cluster"].nunique())
        p, n10, n01 = S.mcnemar_greater(tn["go_carry"].to_numpy(), tn["go"].to_numpy())
        out["vs_carry_over"] = dict(carry_over_false_go=_prop(int(tn["go_carry"].sum()), n), mcnemar_p=p, n10=n10, n01=n01)
        out["sim_only_certificate"] = dict(false_go=_prop(int(tn["go_sim"].sum()), n),
                                           reading=r3_verdict(int(tn["go_sim"].sum()), n, alpha, cfg["conf"]))
    if "u_q05" in test.columns and np.isfinite(test["u_q05"].astype(float)).any():
        # v3 secondary (no verdict): the uncalibrated default posterior quantile (l = 0.05) as a per-module go/no-go
        gu = test["u_q05"].astype(float) >= tau
        gu_n = gu[test["unsafe"]]
        o = dict(rule="go iff uncalibrated posterior 0.05 quantile of z >= tau",
                 false_go=_prop(int(gu_n.sum()), n, cfg["conf"]),
                 yield_safe=_prop(int(gu[~test["unsafe"]].sum()), int(len(ts))))
        if n:
            p, n10, n01 = _mcnemar_two_sided(gu_n.to_numpy(), tn["go"].to_numpy())
            o["mcnemar_two_sided_vs_cert"] = dict(p=p, n10_uq05_only=n10, n01_cert_only=n01)
        out["vs_uncalibrated_q05"] = o
    out["certification_yield_safe"] = _prop(int(ts["go"].sum()), int(len(ts)))
    out["marginal_unsafe_go"] = _prop(k, int(len(test)))
    out["lb_rule"] = "certify iff score > lambda_hat (strict); lambda_hat from unsafe CAL units only"
    return out


# ------------------------------------------------------------------------------- field-day calibration (WS-I3F)
def cal_calibration(units: pd.DataFrame, trials: pd.DataFrame, cfg=None) -> dict:
    """PROTOCOL_V2 §3 / §9, between MASK and DEMO: lambda_hat (alpha_cert, alpha_cert_sens) and the black-box carry-over
    offset from the CAL units ONLY, by the same rule as r3 (`classcond_threshold` on the unsafe CAL units, z_alone from
    the lone MASK builds). Reads no TEST unit and computes no R1-R3 metric. The carry-over offset is the same order
    statistic of (bb_q05 - z_assembly) over the non-excluded CAL assembly trials (one per trial) when `units` carries
    those columns (capxfer.field.builders r3.csv). Written as lambda_v2.json by capxfer.field.builders."""
    cfg = {**DEFAULTS, **(cfg or {})}
    lone = trials[trials["build"].astype(str).str.startswith("single_")]
    cond = condition_table(lone, relative=cfg.get("capability", "relative") == "relative")
    zf, zh = lone_truth(cond)
    cal = units[units["r3_split"] == "CAL"]
    u = r3_units_truth(cal, zf, zh)
    u = u[np.isfinite(u["z_alone"])]
    null = u[u["z_alone"] < cfg["tau"]]["score"]
    out = dict(lambda_hat=classcond_threshold(null, cfg["alpha_cert"]),
               lambda_hat_alpha05=classcond_threshold(null, cfg["alpha_cert_sens"]), n_null_cal=int(len(null)),
               alpha=cfg["alpha_cert"], alpha_sens=cfg["alpha_cert_sens"], tau=cfg["tau"],
               carry_over_offset=None, n_carry_cal=0)
    if {"bb_q05", "z_assembly"} <= set(cal.columns):
        t = cal[~cal["excluded"].astype(bool)].drop_duplicates("trial_id")
        za = t["z_assembly"].astype(float).to_numpy()
        if cfg.get("capability", "relative") == "relative" and "trial_id" in trials.columns:
            zr = relative_trial_strafe(trials)                     # v4: the assembly trial's RELATIVE strafe
            za = np.array([zr.get(str(i), z) for i, z in zip(t["trial_id"], za)], float)
        d = (t["bb_q05"].astype(float).to_numpy() - za)
        d = d[np.isfinite(d)]
        out.update(carry_over_offset=classcond_threshold(d, cfg["alpha_cert"]), n_carry_cal=int(len(d)))
    return out


def lambda_v2_check(path, res: dict) -> dict:
    """WS-I3F: the field-day lambda_v2.json (capxfer.lambda_v2/v1) against the lambda_hat that R3 recomputes from the
    same CAL units. A mismatch is a reported deviation (R4 uses the recomputed value, as frozen)."""
    from ..field.bundle import LambdaFormatError, read_lambda
    try:
        lam = read_lambda(path)
    except LambdaFormatError as e:
        return dict(status="INVALID", error=str(e))
    r3l = res.get("R3", {}).get("lambda_hat", math.nan)
    r3s = res.get("R3_alpha_sensitivity", {}).get("lambda_hat", math.nan)
    same = lambda a, b: (a == b) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-9)    # noqa: E731
    ok = same(lam["lambda_hat"], r3l) and same(lam["lambda_hat_alpha05"], r3s)
    return dict(status="MATCH" if ok else "MISMATCH", file_status=lam["status"], sha256=lam["sha256"],
                lambda_hat_file=lam["lambda_hat"], lambda_hat_r3=r3l, lambda_hat_alpha05_file=lam["lambda_hat_alpha05"],
                lambda_hat_alpha05_r3=r3s, carry_over_offset=lam["carry_over_offset"])


# --------------------------------------------------------------------------------------------------------- R4
def demo_choices(r, lam, tau, modules=D.DEMO_MODULES) -> dict:
    """Arm choices of one DEMO episode (None = abstain). Rules fixed in PROTOCOL_V2 R4 (D15, CHOICE sec. 1; v3 adds
    post_rank when pm_<M> is present):
    certificate: among modules with s > lam (real-calibrated, R3), the largest s (tie: sealed coin); none -> abstain.
    carry-over: iff the current-body black-box bound >= tau, the module nearest the goal; else abstain.
    random: the sealed uniform module. oracle: the largest measured lone z if >= tau (tie: sealed coin); else abstain."""
    s = {m: float(r[f"s_{m}"]) for m in modules}
    z = {m: float(r[f"z_alone_{m}"]) for m in modules}
    cert = [m for m in modules if s[m] > lam]
    if cert:
        best = max(s[m] for m in cert)
        top = [m for m in cert if s[m] == best]
        c = top[0] if len(top) == 1 else r["cert_tie_coin"]
    else:
        c = None
    rank_best = max(s.values())
    rk = [m for m in modules if s[m] == rank_best]
    c_rank = rk[0] if len(rk) == 1 else r["cert_tie_coin"]
    b = r["nearest_goal_module"] if float(r["carry_over_lb"]) >= tau else None
    zb = max(z.values())
    zt = [m for m in modules if z[m] == zb]
    o = (zt[0] if len(zt) == 1 else r["oracle_tie_coin"]) if zb >= tau else None
    out = dict(cert=c, carry_over=b, random=r["random_module"], oracle=o, cert_ranking=c_rank)
    pm = {m: r.get(f"pm_{m}") for m in modules}
    if all(v is not None and np.isfinite(float(v)) for v in pm.values()):
        # v3 fifth arm, POSTERIOR RANKING (docs/ideas/H1B.md U-mean): the largest uncalibrated posterior-mean strafe,
        # never abstains (tie: the certificate's sealed coin)
        pb = max(float(v) for v in pm.values())
        pt = [m for m in modules if float(pm[m]) == pb]
        out["post_rank"] = pt[0] if len(pt) == 1 else r["cert_tie_coin"]
    return out


def _mcnemar_two_sided(a, b):
    """Exact two-sided McNemar on paired binary outcomes -> (p, n10, n01)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2 * sps.binom.cdf(min(n10, n01), n, 0.5)))
    return p, n10, n01


def demo_dock_rule(demo: pd.DataFrame, cfg) -> str:
    rule = cfg.get("demo_dock_rule", "auto")
    if rule == "auto":
        dd = demo.get("dock_design")
        rule = "all3" if dd is not None and len(demo) and (dd.astype(str) == "all3").all() else "chosen"
    if rule not in ("all3", "chosen"):
        raise ValueError(f"demo_dock_rule {rule}")
    return rule


def r4(demo: pd.DataFrame, lam: float, cfg, tau=None) -> dict:
    tau = cfg["tau"] if tau is None else tau
    mods = D.DEMO_MODULES
    rule = demo_dock_rule(demo, cfg)
    d = demo[~demo["excluded"].astype(bool)]
    rows, invalid = [], []
    has_pm = all(f"pm_{m}" in d.columns for m in mods) and len(d) > 0 and \
        np.isfinite(d[[f"pm_{m}" for m in mods]].to_numpy(float)).all()
    arms = ("cert", "carry_over", "random", "oracle", "cert_ranking") + (("post_rank",) if has_pm else ())
    for _, r in d.iterrows():
        ch = demo_choices(r, lam, tau, mods)
        need = {m for a, m in ch.items() if m is not None and a not in ("cert_ranking", "post_rank")}
        y = {m: float(r[f"y_{m}"]) for m in mods}
        if rule == "all3":
            need = set(mods)                       # v3: blind to the arm choices
        if any(not np.isfinite(y[m]) for m in need):
            invalid.append(int(r["episode"]))
            continue
        faulted = set(str(r["faulted_modules"]).split("|"))
        row = dict(episode=int(r["episode"]), faulted=sorted(faulted), docks=len(need))
        for a in arms:
            m = ch[a]
            yy = y.get(m, math.nan) if m is not None else 0.0
            row[f"choice_{a}"] = m
            row[f"y_{a}"] = 0.0 if m is None else yy
            row[f"unsafe_{a}"] = bool(m is not None and float(r[f"z_alone_{m}"]) < tau)
        rows.append(row)
    e = pd.DataFrame(rows)
    n = len(e)
    if n == 0:
        return dict(verdict="NOT_EVALUABLE_NO_EPISODES", n_episodes=0, invalid_episodes=invalid)
    p_c, a_c, b_c = S.mcnemar_greater(e["y_cert"].astype(bool), e["y_carry_over"].astype(bool))
    p_r, a_r, b_r = S.mcnemar_greater(e["y_cert"].astype(bool), e["y_random"].astype(bool))
    h = S.holm([p_c, p_r])
    succ = {a: _prop(int(np.nansum(e[f"y_{a}"])), int(np.isfinite(e[f"y_{a}"]).sum())) for a in arms}
    disp = {}
    for a in arms:
        dm = e[e[f"choice_{a}"].notna()]
        disp[a] = dict(dispatched=_prop(len(dm), n), success_among_dispatched=_prop(int(dm[f"y_{a}"].sum()), len(dm)),
                       unsafe_dispatch=_prop(int(e[f"unsafe_{a}"].sum()), n))
    kc, ko = succ["cert"]["k"], succ["oracle"]["k"]
    secondary = {}
    if has_pm:                                     # v3: reported, two-sided, no superiority claim
        for name, (a, b) in (("post_rank_vs_carry_over", ("post_rank", "carry_over")),
                             ("cert_vs_post_rank", ("cert", "post_rank"))):
            p, n10, n01 = _mcnemar_two_sided(e[f"y_{a}"].astype(bool), e[f"y_{b}"].astype(bool))
            secondary[name] = dict(test="exact two-sided McNemar (abstain = failure); reported, no verdict", p=p,
                                   n10=n10, n01=n01)
    out = dict(n_episodes=n, invalid_episodes=invalid, tau=tau, lambda_hat=lam, dock_rule=rule,
               secondary_contrasts=secondary,
               primary=dict(test="exact one-sided McNemar on paired episodes (abstain = failure), Holm over 2",
                            vs_carry_over=dict(p=p_c, n10=a_c, n01=b_c, p_holm=float(h[0])),
                            vs_random=dict(p=p_r, n10=a_r, n01=b_r, p_holm=float(h[1]))),
               success=succ, dispatch=disp,
               cert_minus_oracle_newcombe90=S.newcombe_diff(kc, n, ko, n, 0.90),
               docks_per_episode=dict(mean=float(e["docks"].mean()), max=int(e["docks"].max())),
               docked_per_episode=(len(mods) if rule == "all3" else None),
               episodes=e.to_dict("records"))
    out["verdict"] = "SUPPORTED" if bool((h < cfg["test_alpha"]).all()) else "NOT_SUPPORTED_INCONCLUSIVE"
    return out


# ------------------------------------------------------------------------------------------- descriptive checks
def band(value, hi=None, lo=None):
    ok = np.isfinite(value) and (hi is None or value <= hi) and (lo is None or value >= lo)
    return dict(value=value, lo=lo, hi=hi, within=bool(ok))


def fidelity(cond: pd.DataFrame, pred_rho: pd.DataFrame | None, cfg) -> dict:
    """Median |rho_real - rho_pred| over legs and conditions per (assembly type, fault class). `pred_rho`: per-condition
    predicted rho_<key> (same build/module/wheel keys as the predictions file); absent -> not evaluable."""
    if pred_rho is None:
        return dict(status="NOT_EVALUABLE_NO_PREDICTED_RHO")
    b = cfg["bands"]
    p = pred_rho.copy()
    p["mw"] = [_mw(m, w) for m, w in zip(p["module"].fillna(""), p["wheel"].fillna(""))]
    c = cond.merge(p[["build", "mw"] + list(RHO_COLS)], on=["build", "mw"], suffixes=("", "_pred"))
    out = {}
    for (t, f), g in c.groupby(["assembly_type", "firmware"]):
        err = np.abs(g[list(RHO_COLS)].to_numpy(float) - g[[f"{k}_pred" for k in RHO_COLS]].to_numpy(float))
        med = float(np.nanmedian(err))
        lim = (b["fidelity_multi"] if t in D.MULTI else
               b["fidelity_roller"] if str(f).startswith("F4") else b["fidelity_single_passive"])
        out[f"{t}|{f}"] = dict(n_conditions=int(len(g)), median_abs_err=band(med, hi=lim),
                               p90_abs_err=float(np.nanquantile(err, 0.9)))
    return out


def repeatability(trials: pd.DataFrame, day2: pd.DataFrame | None, cfg) -> dict:
    b = cfg["bands"]
    t = trials[~trials["excluded"].astype(bool)].copy()
    t["mw"] = [_mw(m, w) for m, w in zip(t["module"].fillna(""), t["wheel"].fillna(""))]
    sds, dfs = [], []
    for _, g in t.groupby(["build", "mw"]):
        if len(g) >= 2:
            x = g[list(RHO_COLS)].to_numpy(float)
            sds.append(np.nanvar(x, axis=0, ddof=1) * (len(g) - 1))
            dfs.append(len(g) - 1)
    pooled = np.sqrt(np.nansum(sds, 0) / max(1, sum(dfs))) if sds else np.full(14, math.nan)
    out = dict(pooled_within_sd_per_leg=dict(zip(KEYS, pooled.tolist())),
               max_pooled_sd=band(float(np.nanmax(pooled)), hi=b["repeat_sd"]))
    if day2 is not None and len(day2):
        d2 = day2[~day2["excluded"].astype(bool)].copy()
        d2["mw"] = [_mw(m, w) for m, w in zip(d2["module"].fillna(""), d2["wheel"].fillna(""))]
        shifts = {}
        for mw, g in d2.groupby("mw"):
            g1 = t[(t["build"] == "single_M1") & (t["mw"] == mw)]
            if len(g1):
                shifts[mw] = float(np.nanmax(np.abs(g[list(RHO_COLS)].mean() - g1[list(RHO_COLS)].mean())))
        out["day_shift_max_abs"] = {k: band(v, hi=b["day_shift"]) for k, v in shifts.items()}
    return out


def apriltag(static: pd.DataFrame | None, motion: pd.DataFrame | None, cfg) -> dict:
    b = cfg["bands"]
    out = {}
    if static is not None and len(static):
        out["static_std_m"] = band(float(np.nanmax(static[["std_x_m", "std_y_m"]].to_numpy())), hi=b["tag_static_std_m"])
        out["static_std_rad"] = band(float(np.nanmax(static["std_yaw_rad"])), hi=b["tag_static_std_rad"])
        bias = np.hypot(static["err_x_m"], static["err_y_m"]) - static["ref_unc_m"] - b["tag_bias_extra_m"]
        out["static_bias_excess_m"] = band(float(np.nanmax(bias)), hi=0.0)
    if motion is not None and len(motion):
        lat = motion["latency_s"].to_numpy(float)
        out["latency_spread_s"] = band(float(np.nanmax(lat) - np.nanmin(lat)), hi=b["tag_latency_spread_s"])
        sr = motion["speed_ratio"].to_numpy(float)
        out["speed_ratio_min"] = band(float(np.nanmin(sr)), lo=b["tag_speed_ratio"][0])
        out["speed_ratio_max"] = band(float(np.nanmax(sr)), hi=b["tag_speed_ratio"][1])
    return out or dict(status="NOT_EVALUABLE_NO_INPUT")


# ------------------------------------------------------------------------------------------------ provenance gate
def provenance_gate(freeze: dict | None, root: Path, trials: pd.DataFrame | None = None, seed: bytes | None = None,
                    schedule: dict | None = None) -> dict:
    """PASS iff (i) every file in the freeze record hashes to its recorded SHA-256; (ii) the revealed seed matches the
    commitment and regenerates the schedule (if given); (iii) every analysed trial is FIELD, from a FIELD_OK session,
    started after the freeze time, and its fault echo matches the sealed schedule."""
    fails = []
    if not freeze:
        return dict(status="FAIL", failures=["no freeze record"])
    for rel, h in (freeze.get("files") or {}).items():
        p = Path(freeze.get("root", root)) / rel
        if not p.exists():
            fails.append(f"missing {rel}")
        elif _sha(p) != h:
            fails.append(f"hash mismatch {rel}")
    if seed is not None:
        if D.commitment(seed) != freeze.get("seed_commitment"):
            fails.append("seed does not match commitment")
        elif schedule is not None and D.schedule_sha256(D.expand_schedule(seed)) != D.schedule_sha256(schedule):
            fails.append("schedule not reproduced by the revealed seed")
    if trials is not None and len(trials):
        if "source" in trials and (trials["source"] != "FIELD").any():
            fails.append(f"{int((trials['source'] != 'FIELD').sum())} non-FIELD trials")
        if "session_status" in trials and (trials["session_status"] != "FIELD_OK").any():
            fails.append(f"{int((trials['session_status'] != 'FIELD_OK').sum())} trials from non-FIELD_OK sessions")
        if "start_utc" in trials and freeze.get("freeze_time_utc"):
            t0 = pd.Timestamp(freeze["freeze_time_utc"])
            early = (pd.to_datetime(trials["start_utc"], utc=True) <= t0.tz_localize("UTC") if t0.tzinfo is None
                     else pd.to_datetime(trials["start_utc"], utc=True) <= t0)
            if early.any():
                fails.append(f"{int(early.sum())} trials before the freeze")
        if "echo_ok" in trials and (~trials["echo_ok"].astype(bool)).any():
            fails.append(f"{int((~trials['echo_ok'].astype(bool)).sum())} fault-echo mismatches")
    return dict(status="PASS" if not fails else "FAIL", failures=fails)


# --------------------------------------------------------------------------------------- paper-facing paths
def paper_paths(res: dict, cond: pd.DataFrame, pred_rho: pd.DataFrame | None) -> dict:
    """Keys bound by paper/results_map.json (WS-PAPER2). Pure re-exports of the results above; no new statistics,
    except `fidelity.mae_passive_roller` = mean |rho_real - rho_pred| over legs and faulted MASK conditions."""
    out = {}
    if pred_rho is not None:
        p = pred_rho.copy()
        p["mw"] = [_mw(m, w) for m, w in zip(p["module"].fillna(""), p["wheel"].fillna(""))]
        c = cond.merge(p[["build", "mw"] + list(RHO_COLS)], on=["build", "mw"], suffixes=("", "_pred"))
        err = np.abs(c[list(RHO_COLS)].to_numpy(float) - c[[f"{k}_pred" for k in RHO_COLS]].to_numpy(float))
        out["fidelity"] = dict(mae_passive_roller=float(np.nanmean(err)) if err.size else math.nan,
                               definition="mean |rho_real - rho_pred| over 14 legs and all faulted MASK conditions")
    m = {}
    if res.get("R2", {}).get("reveal"):
        m["split_someunsafe"] = res["R2"]["reveal"]
    if res.get("R3", {}).get("vs_carry_over"):
        m["carry_go_unsafe"] = res["R3"]["vs_carry_over"]["carry_over_false_go"]
    if m:
        out["masking"] = m
    if res.get("R4", {}).get("success"):
        sc = res["R4"]["success"]
        out["docking"] = {a: dict(successes=sc[k]["k"], runs=sc[k]["n"]) for a, k in
                          (("certificate", "cert"), ("carry_over", "carry_over"), ("random", "random"), ("oracle", "oracle"))}
        out["prereg"] = dict(docking=dict(runs_per_arm=int(res["R4"]["n_episodes"])))
    return out


# ------------------------------------------------------------------------------------------------------- driver
def _read(p):
    return pd.read_csv(p) if p is not None and Path(p).exists() else None


def run_realv2(manifest_path):
    mp = Path(manifest_path).resolve()
    man = yaml.safe_load(mp.read_text())
    rp = lambda k: (mp.parent / man[k]) if man.get(k) else None                     # noqa: E731
    cfg = {**DEFAULTS, **(man.get("config") or {})}
    cfg["bands"] = {**DEFAULTS["bands"], **((man.get("config") or {}).get("bands") or {})}
    inputs = {k: {"path": str(rp(k)), "sha256": _sha(rp(k))} for k in
              ("mask_trials", "predictions", "r3_units", "demo", "simval_day2", "apriltag_static", "apriltag_motion",
               "freeze_record", "predicted_rho", "lambda_v2") if rp(k) is not None and rp(k).exists()}
    trials = _read(rp("mask_trials"))
    pred = _read(rp("predictions"))
    freeze = json.loads(rp("freeze_record").read_text()) if rp("freeze_record") and rp("freeze_record").exists() else None
    seed = bytes.fromhex(man["seed_hex"]) if man.get("seed_hex") else None
    gate = provenance_gate(freeze, mp.parent, trials, seed)
    rel = cfg.get("capability", "relative") == "relative"
    cond = condition_table(trials, relative=rel)                   # v4: relative capability (D17) for R1-R4
    cond_abs = condition_table(trials, relative=False)             # absolute: simulator fidelity only
    if rel:
        pred = relative_predictions(pred)
    res = dict(status=STATUS_OK if gate["status"] == "PASS" else STATUS_BAD, provenance=gate, config=cfg, inputs=inputs,
               counts=dict(mask_trials=int(len(trials)), excluded=int(trials["excluded"].astype(bool).sum())),
               R1=r1(cond, pred, cfg),
               R2=r2(cond, pred, cfg), R2_tau_secondary=r2(cond, pred, cfg, tau=cfg["tau_secondary"]))
    units = _read(rp("r3_units"))
    if units is not None:
        res["R3"] = r3(units, cond, cfg)
        res["R3_alpha_sensitivity"] = r3(units, cond, cfg, alpha=cfg["alpha_cert_sens"])
        res["R3_tau_secondary"] = r3(units, cond, cfg, tau=cfg["tau_secondary"])
    demo = _read(rp("demo"))
    if demo is not None:
        lam = res.get("R3", {}).get("lambda_hat", math.inf)
        res["R4"] = r4(demo, lam, cfg)
    if rp("lambda_v2") is not None and rp("lambda_v2").exists():                                    # WS-I3F
        res["lambda_v2_check"] = lambda_v2_check(rp("lambda_v2"), res)
    res["capability"] = dict(definition=cfg.get("capability", "relative"),
                             healthy_baseline=(healthy_baseline(trials).reset_index().to_dict("records") if rel else None))
    res["descriptive"] = dict(fidelity=fidelity(cond_abs, _read(rp("predicted_rho")), cfg),
                              repeatability=repeatability(trials, _read(rp("simval_day2")), cfg),
                              apriltag=apriltag(_read(rp("apriltag_static")), _read(rp("apriltag_motion")), cfg))
    res.update(paper_paths(res, cond_abs, _read(rp("predicted_rho"))))
    pkg = Path(__file__).resolve().parents[1]
    res["code_sha256"] = {f: _sha(pkg / f) for f in CODE_FILES}
    res["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res = _clean(res)
    out = mp.parent / man.get("output_dir", "out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_v2.json").write_text(json.dumps(res, indent=1))
    return res
