"""The frozen analysis: `run_frozen(manifest_path)` computes every H1-H5 metric of docs/PROTOCOL_DRAFT.md.

Manifest (JSON or YAML; relative paths resolve against the manifest's directory):
  output_dir: out/
  pred_tables: {method: {rung: table.npz}}          # real FIELD trials, analysis/table.py format
  sim_tables: {method: table.npz}                   # sim held-out, in-model trials (H5); optional
  explain_scores: explain.csv                       # WS-E automatic scores (H3); optional
  docking: docking.csv                              # arm,run,success (H4); optional
  probe_policy: probe.csv                           # offline probe-policy table; optional (secondary)
  probe_ladder: [{label: P1P5, pids: [P1, P5], table: t.npz}, ...]   # H2 redesign (ordered rungs); optional
  config: {...}                                     # overrides of DEFAULTS (must be frozen in the prereg)

Leakage guards (asserted, not assumed): the test assembly type never appears in a fold's calibration set; only
FIELD, in-model (ood == none) trials calibrate; sim tables must be SIMULATION; PredTables of all methods/rungs
contain exactly the same trial ids.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..conformal import capability as cap, ood as oodm, wheel_sets as ws
from ..probes import TARGET_KEYS
from . import probe_policy, stats as S, table as tb
from .paper_view import paper_view as _paper_view

DEFAULTS = dict(
    alpha=0.10, conf=0.95, test_alpha=0.05, n_boot=2000, seed=20261120,
    folds=["single", "pair", "L3", "square4"],
    seen_pids=["P1", "P5"],
    primary_method="structured_npe", blackbox_method="blackbox", nominal_method="nominal_sim",
    primary_rung="E2", ladder_rungs=["E0", "E1", "E2"],          # confirmatory; E3 widths still reported (exploratory)
    h1_fold_threshold=0.85, h1_min_folds=3, coverage_hold=0.85,
    dock_proposed="proposed", dock_omni="omni", dock_oracle="oracle", dock_margin=0.20, dock_ci_conf=0.90,
    explain_calibrated="calibrated", explain_point="point_estimate",
    cluster_oneway="session", cluster_crossed=["session", "module_set"],
    sensitivity_group_keys=["assembly_type", "module_set"],
    probe_budget=3,
    aps_pooling="joint",
    # FIX (H3 redesign support): capability threshold tau and decision-relevant legs, identical to the values passed
    # to explain()/score() (pipeline default: strafe P2+/P2-, tau from configs/study_constants.yaml).
    tau=0.5, decision_targets=["P2+", "P2-"],
    # H1-H3 redesign, approved by the study owner on 2026-11-20 (docs/PROTOCOL_DRAFT.md section 4). "original" (the
    # superseded earlier draft rules) is kept only as a sensitivity switch; every original metric is still reported.
    rule_set="redesign",           # "redesign" (binding) | "original" (superseded; sensitivity only)
    # H1(a) capability: the deployed capability method (black-box deep ensemble), split_joint, joint coverage over the
    # 10 untested legs. H1(b) diagnosis: structured ensemble vs analytic Bayes, McNemar on "all wheels' argmax correct".
    h1r_cap_method="ensemble", h1r_diag_method="structured_ensemble", h1r_diag_baseline="analytic_bayes",
    h1r_diag_merge_cb=True,        # primary diagnosis definition merges COAST and BRAKE; the unmerged one is reported
    h1r_aps_min=0.85,              # H1(b): pooled in-model APS all-wheels coverage of h1r_diag_method must reach this
    h1r_diag_rate_min=None,        # H1(b): optional floor on the faulted-trial diagnosed rate (None = reported only)
    h1r_secondary_methods=["structured_ensemble", "structured_npe", "blackbox"],   # descriptive capability rows
    # H3: the capability certificate (table-rule overclaim, coverage degradation headline) comes from the deployed
    # capability method; falls back to primary_method when that PredTable is absent.
    h3_cap_method="ensemble",
    probe_ladder_targets="all",    # "all" 14 legs | "unprobed_by_first" (legs not seen at the first ladder rung)
    # H2 confirmatory trial set (owner-approved 2026-11-20, PROTOCOL 4.1 D3-1): "double_fault" = in-model trials with
    # exactly two injected wheel faults (F5); "faulted" = all in-model faulted trials (then descriptive only).
    h2_trials="double_fault",
)
RULE_TAG = "redesign (approved by the study owner on 2026-11-20)"
STATUS = "CONFIRMATORY_APPROVED_2026-11-20"
JOINT_METRIC = ("joint coverage over the 10 untested directions: fraction of in-model test trials whose split_joint "
                "interval contains all 10 untested-leg ratios (P2+-, P3+-, P4+-, P6x+-, P6y+-) simultaneously")
PER_KEY_METRIC = ("secondary: per-key coverage = mean over the 10 untested legs of the per-leg coverage indicator "
                  "(of the split_joint intervals, and of separately calibrated split_per_target intervals)")
ROC_ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
SCHEMES = ("split_joint", "split_per_target", "group_ss1_type", "group_pooled_type",
           "group_pooled_instance", "raw_joint")
CODE_FILES = ("conformal/_common.py", "conformal/capability.py", "conformal/wheel_sets.py", "conformal/ood.py",
              "analysis/stats.py", "analysis/table.py", "analysis/frozen.py", "analysis/probe_policy.py",
              "analysis/paper_view.py")


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
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else ("inf" if x == math.inf else ("-inf" if x == -math.inf else x))
    return x


def eval_targets(seen_pids):
    return np.array([i for i, k in enumerate(TARGET_KEYS) if k[:-1] not in seen_pids])


# ------------------------------------------------------------------------------------------------ per fold
def _joint_hit(lo, hi, y, T):
    yy = y[:, T]
    c = (yy >= lo[:, T]) & (yy <= hi[:, T])
    return np.all(c | np.isnan(yy), 1), c


def _width(lo, hi, T):
    return (hi[:, T] - lo[:, T]).mean(1)


def fold_eval(t, fold, cfg, T):
    """Calibrate on the other assembly types' in-model FIELD trials, evaluate on every trial of `fold`."""
    a = cfg["alpha"]
    cal = (t["assembly_type"] != fold) & (t["ood"] == "none") & (t["source"] == t["source"][0])
    test = t["assembly_type"] == fold
    assert not np.any(cal & test) and fold not in set(t["assembly_type"][cal].tolist()), "calibration leakage"
    tc, tt = tb.subset(t, np.flatnonzero(cal)), tb.subset(t, np.flatnonzero(test))
    pc, pt = tb.as_pred(tc), tb.as_pred(tt)
    yc, yt = tc["y"], tt["y"]
    out = dict(test_idx=np.flatnonzero(test), n_cal=int(cal.sum()), schemes={}, thresholds={})
    inst = np.array(["|".join(r) for r in zip(*[tc[k] for k in cfg["sensitivity_group_keys"]])])
    specs = {
        "split_joint": dict(method="split", joint=True),
        "split_per_target": dict(method="split", joint=False),
        "group_ss1_type": dict(method="group", joint=True, group_variant="ss1", groups=tc["assembly_type"]),
        "group_pooled_type": dict(method="group", joint=True, group_variant="pooled", groups=tc["assembly_type"]),
        "group_pooled_instance": dict(method="group", joint=True, group_variant="pooled", groups=inst),
    }
    for name, kw in specs.items():
        c = cap.calibrate(pc, yc, a, targets=T, seed=cfg["seed"], **kw)
        lo, hi = (np.asarray(v) for v in c.interval(pt))
        jh, ph = _joint_hit(lo, hi, yt, T)
        out["schemes"][name] = dict(hit=jh, per_target=ph.mean(1), width=_width(lo, hi, T))
        if name == "split_joint":
            out["split_joint_lo"] = lo
        out["thresholds"][name] = dict(value=c.threshold.tolist(), finite=c.info["finite"], n_groups=c.n_groups)
    lo, hi = pt["y_q"][..., np.argmin(t["q_levels"])], pt["y_q"][..., np.argmax(t["q_levels"])]
    jh, ph = _joint_hit(lo, hi, yt, T)
    out["schemes"]["raw_joint"] = dict(hit=jh, per_target=ph.mean(1), width=_width(lo, hi, T))
    out["raw_lo"] = lo
    q = np.asarray(t["q_levels"], float)
    out["point"] = pt["y_q"][..., int(np.argmin(np.abs(q - 0.5)))]
    if "mode_prob" in pc:
        sc = ws.calibrate(pc, tc["mode_true"], a, pooling=cfg["aps_pooling"], seed=cfg["seed"])
        sets = sc.sets(pt)
        m = tt["wheel_mask"]
        hit = np.take_along_axis(sets, np.where(m, tt["mode_true"], 0)[..., None], -1)[..., 0] & m
        size = sets.sum(-1)
        nw = m.sum(1)
        out["aps"] = dict(frac_covered=hit.sum(1) / nw, all_covered=(hit | ~m).all(1),
                          mean_size=np.where(m, size, 0).sum(1) / nw,
                          frac_singleton=np.where(m, size == 1, False).sum(1) / nw,
                          frac_empty=np.where(m, size == 0, False).sum(1) / nw)
    oc = oodm.calibrate(tc["ppc_score"], a)
    out["ood_flag"] = oc.flag(tt["ppc_score"])
    out["ood_threshold"] = oc.threshold
    out["ood_roc"] = {str(ag): oodm.calibrate(tc["ppc_score"], ag).flag(tt["ppc_score"]) for ag in ROC_ALPHAS}
    return out


# ------------------------------------------------------------------------------------------------ helpers
def _clusters(t, idx, cfg):
    one = t[cfg["cluster_oneway"]][idx]
    crossed = [t[k][idx] for k in cfg["cluster_crossed"]]
    return one, crossed


def _cov_row(t, idx, hit, cfg, seed_off=0):
    one, crossed = _clusters(t, idx, cfg)
    return S.coverage_summary(hit, one, crossed, cfg["n_boot"], cfg["seed"] + seed_off, cfg["conf"])


def _per_trial(t, per_fold, key, sub=None):
    """Assemble a (N,) per-trial array from fold outputs (each trial is test exactly once)."""
    N = t["trial_id"].shape[0]
    arr = np.full(N, np.nan)
    for f in per_fold.values():
        v = f["schemes"][key][sub] if sub else f[key]
        arr[f["test_idx"]] = v
    return arr


# ------------------------------------------------------------------------------------------------ main
def run_frozen(manifest_path):
    mp = Path(manifest_path).resolve()
    with open(mp) as fh:
        man = yaml.safe_load(fh) if mp.suffix in (".yaml", ".yml") else json.load(fh)
    base = mp.parent
    cfg = dict(DEFAULTS, **(man.get("config") or {}))
    outdir = (base / man.get("output_dir", "out")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rp = lambda p: (base / p).resolve()
    inputs = {"manifest": {"path": str(mp), "sha256": _sha(mp)}}

    # ---- load and align tables
    tables = {}
    for m, rungs in man["pred_tables"].items():
        for r, p in rungs.items():
            tables.setdefault(m, {})[r] = tb.load(rp(p))
            inputs[f"pred_tables/{m}/{r}"] = {"path": str(rp(p)), "sha256": _sha(rp(p))}
    ref = tables[cfg["primary_method"]][cfg["primary_rung"]]
    ids = ref["trial_id"].tolist()
    for m in tables:
        for r in tables[m]:
            if set(tables[m][r]["trial_id"].tolist()) != set(ids):
                raise ValueError(f"PredTable {m}/{r} does not cover exactly the reference trials")
            tables[m][r] = tb.align(tables[m][r], ids)
            for k in tb.STR_KEYS:
                if not np.array_equal(tables[m][r][k], ref[k]):
                    raise ValueError(f"PredTable {m}/{r} disagrees with the reference on {k}")
    srcs = set(ref["source"].tolist())
    if len(srcs) != 1:
        raise ValueError(f"pred_tables must have a single source (FIELD for the real analysis), got {srcs}")
    source = srcs.pop()
    if source != "FIELD" and man.get("frozen", False):
        raise ValueError("a frozen analysis must read FIELD pred_tables only")
    T = eval_targets(cfg["seen_pids"])
    # prereg exclusion: a trial with any undefined evaluated target leaves every capability analysis (counted)
    bad_y = np.isnan(ref["y"][:, T]).any(1)
    excluded = ref["trial_id"][bad_y].tolist()
    if bad_y.any():
        keep = np.flatnonzero(~bad_y)
        tables = {m: {r: tb.subset(t_, keep) for r, t_ in v.items()} for m, v in tables.items()}
        ref = tables[cfg["primary_method"]][cfg["primary_rung"]]
        ids = ref["trial_id"].tolist()
    folds = [f for f in cfg["folds"] if np.any(ref["assembly_type"] == f)]
    inmodel = ref["ood"] == "none"

    # ---- per method x rung x fold
    R = {m: {r: {f: fold_eval(tables[m][r], f, cfg, T) for f in folds} for r in tables[m]} for m in tables}

    cov_rows, aps_rows, ood_rows = [], [], []
    results = dict(config=cfg, eval_targets=[TARGET_KEYS[i] for i in T], folds=folds, n_trials=len(ids),
                   excluded_missing_target=dict(n=len(excluded), trial_ids=excluded),
                   n_inmodel=int(inmodel.sum()), reused_statistics=S.REUSED)
    for m in R:
        for r in R[m]:
            for f, fo in R[m][r].items():
                ti = fo["test_idx"]
                im = inmodel[ti]
                for sc, v in fo["schemes"].items():
                    row = _cov_row(ref, ti[im], v["hit"][im], cfg)
                    oo = ~im
                    cov_rows.append(dict(method=m, rung=r, fold=f, scheme=sc, n_cal=fo["n_cal"], n=row["n"],
                                         coverage=row["coverage"], cp_lo=row["cp_lo"], cp_hi=row["cp_hi"],
                                         boot_lo=row["boot_oneway"].get("lo"), boot_hi=row["boot_oneway"].get("hi"),
                                         crossed_lo=row["boot_crossed"].get("lo"),
                                         crossed_hi=row["boot_crossed"].get("hi"),
                                         conservative_lower=row["conservative_lower"],
                                         per_target_coverage=float(np.mean(v["per_target"][im])),
                                         median_width=float(np.median(v["width"][im])),
                                         ood_coverage=float(np.mean(v["hit"][oo])) if oo.any() else math.nan,
                                         threshold_finite=fo["thresholds"].get(sc, {}).get("finite", True)))
                if "aps" in fo:
                    A = fo["aps"]
                    aps_rows.append(dict(method=m, rung=r, fold=f, n=int(im.sum()),
                                         wheel_coverage=float(np.mean(A["frac_covered"][im])),
                                         all_wheels_covered=float(np.mean(A["all_covered"][im])),
                                         mean_set_size=float(np.mean(A["mean_size"][im])),
                                         frac_singleton=float(np.mean(A["frac_singleton"][im])),
                                         frac_empty=float(np.mean(A["frac_empty"][im]))))
                fl = fo["ood_flag"]
                kind = ref["ood"][ti]
                k_fa, n_fa = int(fl[im].sum()), int(im.sum())
                row = dict(method=m, rung=r, fold=f, false_alarm=k_fa / n_fa if n_fa else math.nan,
                           false_alarm_cp_hi=S.cp_upper(k_fa, n_fa, cfg["conf"]), n_inmodel=n_fa,
                           shift_test_p=float(S.sps.binom.sf(k_fa - 1, n_fa, cfg["alpha"])) if n_fa else math.nan)
                for o in ("payload", "low_mu"):
                    s = kind == o
                    row[f"detect_{o}"] = float(fl[s].mean()) if s.any() else math.nan
                    row[f"n_{o}"] = int(s.sum())
                ood_rows.append(row)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(outdir / "fig_coverage_by_fold.csv", index=False)
    pd.DataFrame(aps_rows).to_csv(outdir / "fig_wheel_sets.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(outdir / "fig_ood_flag.csv", index=False)
    results["per_fold"] = cov_rows
    results["wheel_set_rows"] = aps_rows
    results["ood_flag_rows"] = ood_rows

    # ---- H1
    pm, pr, bm = cfg["primary_method"], cfg["primary_rung"], cfg["blackbox_method"]
    prim = cov[(cov.method == pm) & (cov.rung == pr) & (cov.scheme == "split_joint")]
    passes = int((prim.coverage >= cfg["h1_fold_threshold"]).sum())
    strong = int((prim.conservative_lower >= cfg["h1_fold_threshold"]).sum())
    h1 = dict(metric="joint split-conformal coverage of untested-direction capability, in-model test trials",
              per_fold={r.fold: dict(coverage=r.coverage, conservative_lower=r.conservative_lower, n=r.n)
                        for r in prim.itertuples()},
              folds_passing_point=passes, folds_passing_lower_bound=strong,
              supported=passes >= cfg["h1_min_folds"], strongly_supported=strong >= cfg["h1_min_folds"])
    # prespecified pivot analysis: does the OOD test detect the folds whose coverage fails?
    fa = {r["fold"]: r for r in ood_rows if r["method"] == pm and r["rung"] == pr}
    h1["shift_detection"] = {
        r.fold: dict(coverage_fails=bool(r.coverage < cfg["h1_fold_threshold"]),
                     inmodel_flag_rate=fa[r.fold]["false_alarm"], shift_test_p=fa[r.fold]["shift_test_p"],
                     shift_detected=bool(fa[r.fold]["shift_test_p"] < cfg["test_alpha"]))
        for r in prim.itertuples() if r.fold in fa}
    sbb_rows = []
    if bm in R and pr in R[bm]:
        ps = []
        for f in folds:
            a_, b_ = R[pm][pr][f], R[bm][pr][f]
            im = inmodel[a_["test_idx"]]
            hs, hb = a_["schemes"]["split_joint"]["hit"][im], b_["schemes"]["split_joint"]["hit"][im]
            ws_, wb = a_["schemes"]["split_joint"]["width"][im], b_["schemes"]["split_joint"]["width"][im]
            p_cov, n10, n01 = S.mcnemar_greater(hs, hb)
            p_w, npos, nneg = S.sign_test_greater(wb - ws_)
            ps += [p_cov, p_w]
            sbb_rows.append(dict(fold=f, cov_structured=hs.mean(), cov_blackbox=hb.mean(), mcnemar_p=p_cov,
                                 n10=n10, n01=n01, medw_structured=float(np.median(ws_)),
                                 medw_blackbox=float(np.median(wb)), sign_p=p_w, n_bb_wider=npos, n_bb_narrower=nneg))
        adj = S.holm(ps)
        for i, row in enumerate(sbb_rows):
            row["mcnemar_p_holm"], row["sign_p_holm"] = adj[2 * i], adj[2 * i + 1]
            row["blackbox_worse"] = bool(min(adj[2 * i], adj[2 * i + 1]) < cfg["test_alpha"])
        h1["blackbox"] = dict(per_fold=sbb_rows,
                              folds_blackbox_worse=int(sum(r["blackbox_worse"] for r in sbb_rows)),
                              supported=sum(r["blackbox_worse"] for r in sbb_rows) >= cfg["h1_min_folds"])
    pd.DataFrame(sbb_rows).to_csv(outdir / "fig_struct_vs_bb.csv", index=False)
    results["H1"] = h1

    # ---- H2 ladder (pooled over folds; every trial is a test trial exactly once)
    rungs = [r for r in cfg["ladder_rungs"] if r in R[pm]]
    lad_rows, h2 = [], dict(rungs=rungs)
    W = {r: _per_trial(ref, R[pm][r], "split_joint", "width") for r in rungs}
    Hh = {r: _per_trial(ref, R[pm][r], "split_joint", "hit") for r in rungs}
    for r in rungs:
        idx = np.flatnonzero(inmodel)
        row = _cov_row(ref, idx, Hh[r][idx].astype(bool), cfg)
        lad_rows.append(dict(rung=r, fold="all", coverage=row["coverage"], conservative_lower=row["conservative_lower"],
                             median_width=float(np.median(W[r][idx])), n=row["n"]))
        for f in folds:
            fi = idx[ref["assembly_type"][idx] == f]
            lad_rows.append(dict(rung=r, fold=f, coverage=float(Hh[r][fi].mean()), conservative_lower=math.nan,
                                 median_width=float(np.median(W[r][fi])), n=int(fi.size)))
    steps, ps = [], []
    for r0, r1 in zip(rungs[:-1], rungs[1:]):
        d = (W[r0] - W[r1])[inmodel]
        p, npos, nneg = S.sign_test_greater(d)
        Wb = S.bootstrap_weights(ref[cfg["cluster_oneway"]][inmodel], cfg["n_boot"], cfg["seed"])
        ci = S.boot_ci(np.where(np.isfinite(d), d, np.nan), Wb, "median", cfg["conf"])
        steps.append(dict(step=f"{r0}>{r1}", p=p, n_narrower=npos, n_wider=nneg, median_diff=float(np.nanmedian(
            np.where(np.isfinite(d), d, np.nan))), ci=ci))
        ps.append(p)
    adj = S.holm(ps) if ps else []
    for s, pa in zip(steps, adj):
        s["p_holm"] = float(pa)
    holds = all(r["coverage"] >= cfg["coverage_hold"] for r in lad_rows if r["fold"] == "all")
    h2.update(steps=steps, coverage_holds=holds,
              supported=bool(steps) and holds and all(s["p_holm"] < cfg["test_alpha"] for s in steps))
    pd.DataFrame(lad_rows).to_csv(outdir / "fig_ladder.csv", index=False)
    results["H2"] = h2
    results["ladder"] = lad_rows

    # ---- H1 / H2 redesign (approved 2026-11-20; binding with rule_set == "redesign", the default)
    h1["redesign"] = _h1_redesign(R, tables, ref, inmodel, cfg, folds, outdir, ood_rows)
    if man.get("probe_ladder"):
        lad = []
        for e in man["probe_ladder"]:
            p = rp(e["table"])
            inputs[f"probe_ladder/{e['label']}"] = {"path": str(p), "sha256": _sha(p)}
            t_ = tb.align(tb.load(p), ids)
            for k in tb.STR_KEYS:
                if not np.array_equal(t_[k], ref[k]):
                    raise ValueError(f"probe_ladder table {e['label']} disagrees with the reference on {k}")
            lad.append((str(e["label"]), list(e.get("pids", [])), t_))
        h2["probe_ladder"] = _h2_probe_ladder(lad, ref, inmodel, cfg, folds, outdir)
    if cfg["rule_set"] == "redesign":
        rd = h1["redesign"]
        # the superseded earlier-draft H1 (structured_npe coverage, H1b black-box comparison) becomes descriptive
        orig = {k: h1.pop(k) for k in ("metric", "per_fold", "folds_passing_point", "folds_passing_lower_bound",
                                       "supported", "strongly_supported", "shift_detection", "blackbox") if k in h1}
        orig.update(method=pm, status="SUPERSEDED_DESCRIPTIVE")
        capb = rd.get("capability") or {}
        h1.update(original=orig, original_supported=orig.get("supported"), metric=rd["primary_coverage_metric"],
                  method=rd["cap_method"], per_fold=capb.get("per_fold", {}),
                  folds_passing_point=capb.get("folds_passing_point"),
                  folds_passing_lower_bound=capb.get("folds_passing_lower_bound"),
                  shift_detection=capb.get("shift_detection", {}),
                  supported=rd.get("supported"), strongly_supported=rd.get("strongly_supported"), rule=RULE_TAG)
        rd2 = h2.get("probe_ladder")
        h2["original_supported"] = h2.get("supported")
        h2["supported"] = None if rd2 is None else rd2.get("supported")
        h2["strongly_supported"] = None if rd2 is None else rd2.get("strongly_supported")
        h2["rule"] = RULE_TAG
    else:
        h1["rule"] = h2["rule"] = "original (superseded earlier draft; sensitivity only)"

    # ---- H3 explanations (WS-E scores)
    ex_df = None
    if man.get("explain_scores"):
        p = rp(man["explain_scores"])
        inputs["explain_scores"] = {"path": str(p), "sha256": _sha(p)}
        ex_df = pd.read_csv(p)
        results["H3"] = _h3(ex_df, ref, cfg, outdir)
    # ---- H3: consequential OOD conditions, coverage degradation and overclaim
    results["H3_ood"] = _h3_ood(R, ref, inmodel, cfg, T, outdir)
    if ex_df is not None:
        results["H3"]["redesign"] = _h3_redesign(ex_df, ref, R, results["H3_ood"], cfg, T, outdir)
        if cfg["rule_set"] == "redesign":
            rd = results["H3"]["redesign"]
            results["H3"]["original_supported"] = results["H3"].get("supported")
            results["H3"]["supported"] = rd["supported"]
            if rd.get("verdict_override"):
                results["H3"]["verdict_override"] = rd["verdict_override"]
            results["H3"]["strongly_supported"] = rd["strongly_supported"]
            results["H3"]["rule"] = RULE_TAG
        else:
            results["H3"]["rule"] = "original (superseded earlier draft; sensitivity only)"

    # ---- H4 docking
    dock_df = None
    if man.get("docking"):
        p = rp(man["docking"])
        inputs["docking"] = {"path": str(p), "sha256": _sha(p)}
        dock_df = pd.read_csv(p)
        results["H4"] = _h4(dock_df, cfg, outdir)

    # ---- H5 sim-vs-real coverage gap
    if man.get("sim_tables"):
        sims = {}
        for m, p in man["sim_tables"].items():
            sims[m] = tb.load(rp(p))
            inputs[f"sim_tables/{m}"] = {"path": str(rp(p)), "sha256": _sha(rp(p))}
        results["H5"] = _h5(sims, tables, ref, inmodel, cfg, T, outdir)

    # ---- probe policy (secondary, exploratory)
    if man.get("probe_policy"):
        p = rp(man["probe_policy"])
        inputs["probe_policy"] = {"path": str(p), "sha256": _sha(p)}
        summ, tests, _ = probe_policy.evaluate(pd.read_csv(p, dtype={"seen": str, "candidate": str}),
                                               cfg["probe_budget"], cfg["seed"], cfg["n_boot"], cfg["conf"])
        summ.to_csv(outdir / "fig_probe_policy.csv", index=False)
        results["probe_policy"] = dict(status="SECONDARY_EXPLORATORY", summary=summ.to_dict("records"),
                                       tests=tests.to_dict("records"))

    pkg = Path(__file__).resolve().parents[1]
    results["code_sha256"] = {f: _sha(pkg / f) for f in CODE_FILES}
    results["inputs"] = inputs
    results["source"] = source
    results["status"] = ("FROZEN_ANALYSIS_OUTPUT" if man.get("frozen", False) and source == "FIELD"
                         else "DRY_RUN_NOT_FROZEN")
    _paper_view(results, R, ref, inmodel, cfg, man, rp, inputs, sims if man.get("sim_tables") else None, T,
                ex_df, dock_df)
    results = _clean(results)
    with open(outdir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True, allow_nan=False)
    results["results_path"] = str(outdir / "results.json")
    return results


def _h3(df, ref, cfg, outdir):
    a = cfg["alpha"]
    pos = {tid: i for i, tid in enumerate(ref["trial_id"].tolist())}
    df = df[df.trial_id.isin(pos)].copy()
    df["ood"] = [ref["ood"][pos[t]] for t in df.trial_id]
    df["fold"] = [ref["assembly_type"][pos[t]] for t in df.trial_id]
    df["unsupported"] = df["n_unsupported_cause"] > 0
    rows = []
    for (ex, fold), g in list(df.groupby(["explainer", "fold"])) + [((e, "all"), g) for e, g in df.groupby("explainer")]:
        o = g[g.ood != "none"]
        k, n = int(g.unsupported.sum()), len(g)
        km, nm = int(o.wheel_cause_asserted.astype(bool).sum()), len(o)
        row = dict(explainer=ex, fold=fold, n=n, unsupported_rate=k / n if n else math.nan,
                   unsupported_cp_hi=S.cp_upper(k, n, cfg["conf"]), n_ood=nm,
                   misattribution_rate=km / nm if nm else math.nan,
                   misattribution_cp_lo=S.cp_lower(km, nm, cfg["conf"]),
                   misattribution_cp_hi=S.cp_upper(km, nm, cfg["conf"]))
        for col in ("correct_claim_rate", "informativeness"):
            if col in g:
                row[col] = float(g[col].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "fig_explain_ood.csv", index=False)
    allr = {r["explainer"]: r for r in rows if r["fold"] == "all"}
    c, pe = allr.get(cfg["explain_calibrated"]), allr.get(cfg["explain_point"])
    out = dict(per_explainer=rows)
    if c and pe:
        w = df[df.ood != "none"].pivot_table(index="trial_id", columns="explainer", values="wheel_cause_asserted")
        w = w[[cfg["explain_point"], cfg["explain_calibrated"]]].dropna().astype(bool)
        p, n10, n01 = S.mcnemar_greater(w[cfg["explain_point"]].to_numpy(), w[cfg["explain_calibrated"]].to_numpy())
        out.update(calibrated_unsupported_rate=c["unsupported_rate"], calibrated_unsupported_cp_hi=c["unsupported_cp_hi"],
                   point_misattribution=pe["misattribution_rate"], point_misattribution_cp_lo=pe["misattribution_cp_lo"],
                   calibrated_misattribution=c["misattribution_rate"], mcnemar_p=p, n10=n10, n01=n01,
                   supported=bool(c["unsupported_rate"] <= a and c["misattribution_rate"] <= a
                                  and pe["misattribution_cp_lo"] > a and p < cfg["test_alpha"]),
                   strongly_supported=bool(c["unsupported_cp_hi"] <= a))
    return out


# ------------------------------------------------------------------------------------------------ H1 / H2 redesign
def _diagnosed(t, merge_cb=False):
    """Per trial: argmax mode equals the injected mode on every real wheel (NaN-safe; None without mode_prob).
    merge_cb=True merges COAST (2) and BRAKE (3) into one class before the argmax (probabilities summed, labels
    mapped 3 -> 2), the primary H1(b) definition; the two are nearly indistinguishable from P1/P5 in simulation."""
    if "mode_prob" not in t:
        return None
    mp = np.nan_to_num(np.asarray(t["mode_prob"], float), nan=-1.0)
    lab = np.asarray(t["mode_true"])
    if merge_cb:
        mp = mp.copy()
        mp[..., 2] = np.where(mp[..., 2] >= 0, mp[..., 2] + np.clip(mp[..., 3], 0, None), -1.0)
        mp[..., 3] = -1.0
        lab = np.where(lab == 3, 2, lab)
    am = mp.argmax(-1)
    m = t["wheel_mask"]
    return np.all((am == lab) | ~m, 1)


def _cap_summary(R_m, ref, inmodel, cfg, folds, ood_rows=None, method=None):
    """Capability-interval summary of one method x rung (LOAO, split_joint). Primary: joint coverage over the
    untested legs (per fold + pooled, with the conservative lower bound). Secondary: per-key coverage, width."""
    idx = np.flatnonzero(inmodel)
    thr = cfg["h1_fold_threshold"]
    per_fold = {}
    for f in folds:
        fo = R_m[f]
        im = inmodel[fo["test_idx"]]
        sj = fo["schemes"]["split_joint"]
        row = _cov_row(ref, fo["test_idx"][im], sj["hit"][im], cfg)
        per_fold[f] = dict(coverage=row["coverage"], conservative_lower=row["conservative_lower"], n=row["n"],
                           cp_lo=row["cp_lo"], cp_hi=row["cp_hi"],
                           per_key_coverage=float(np.mean(sj["per_target"][im])),
                           per_key_coverage_split_per_target=float(np.mean(
                               fo["schemes"]["split_per_target"]["per_target"][im])),
                           median_width=float(np.median(sj["width"][im])))
    hit = _per_trial(ref, R_m, "split_joint", "hit").astype(bool)
    pk = _per_trial(ref, R_m, "split_joint", "per_target")
    pkt = _per_trial(ref, R_m, "split_per_target", "per_target")
    w = _per_trial(ref, R_m, "split_joint", "width")
    pooled = _cov_row(ref, idx, hit[idx], cfg)
    npass = int(sum(v["coverage"] >= thr for v in per_fold.values()))
    nstrong = int(sum(v["conservative_lower"] >= thr for v in per_fold.values()))
    out = dict(primary_metric=JOINT_METRIC, secondary_metric=PER_KEY_METRIC, per_fold=per_fold,
               pooled=dict(coverage=pooled["coverage"], conservative_lower=pooled["conservative_lower"],
                           n=pooled["n"], per_key_coverage=float(np.mean(pk[idx])),
                           per_key_coverage_split_per_target=float(np.mean(pkt[idx])),
                           median_width=float(np.median(w[idx]))),
               folds_passing_point=npass, folds_passing_lower_bound=nstrong,
               passes=npass >= cfg["h1_min_folds"], strong=nstrong >= cfg["h1_min_folds"])
    if ood_rows is not None and method is not None:
        fa = {r["fold"]: r for r in ood_rows if r["method"] == method and r["rung"] == cfg["primary_rung"]}
        out["shift_detection"] = {
            f: dict(coverage_fails=bool(v["coverage"] < thr), inmodel_flag_rate=fa[f]["false_alarm"],
                    shift_test_p=fa[f]["shift_test_p"], shift_detected=bool(fa[f]["shift_test_p"] < cfg["test_alpha"]))
            for f, v in per_fold.items() if f in fa}
    return out


def _h1_redesign(R, tables, ref, inmodel, cfg, folds, outdir, ood_rows=None):
    """Approved by the study owner on 2026-11-20. Primary rung cfg primary_rung (E2); LOAO; in-model test trials.
      (a) capability: the deployed capability method cfg h1r_cap_method (`ensemble`, 5 x black-box), split_joint
          calibrated. Primary metric: joint coverage over the 10 untested legs. A fold passes if the point estimate is
          >= h1_fold_threshold (0.85); (a) holds if >= h1_min_folds (3 of 4) folds pass. Strong: the conservative
          lower bound (min of CP and both cluster-bootstrap one-sided 95 % lower bounds) passes in >= 3 folds.
      (b) diagnosis: cfg h1r_diag_method (`structured_ensemble`) vs h1r_diag_baseline (`analytic_bayes`). 'Diagnosed'
          = argmax mode correct on every wheel of the trial, COAST and BRAKE merged (h1r_diag_merge_cb; the unmerged
          definition is reported). Exact one-sided McNemar on in-model trials (method diagnosed, baseline not),
          p < test_alpha; AND the method's pooled in-model APS all-wheels coverage >= h1r_aps_min (0.85); AND, only if
          h1r_diag_rate_min is set, its faulted-trial diagnosed rate >= that value (default: reported only).
    Supported iff (a) and (b). Strongly supported iff, in addition, (a) holds on the conservative lower bound.
    Secondary / descriptive (no verdict): capability coverage and width of h1r_secondary_methods (structured_ensemble,
    structured_npe, blackbox), the structured_ensemble vs ensemble calibrated-width comparison, per-key coverage."""
    cm, dm, db, pr = cfg["h1r_cap_method"], cfg["h1r_diag_method"], cfg["h1r_diag_baseline"], cfg["primary_rung"]
    out = dict(status=STATUS, rule=_h1_redesign.__doc__.strip(), cap_method=cm, diag_method=dm, diag_baseline=db,
               rung=pr, primary_coverage_metric=JOINT_METRIC, secondary_coverage_metric=PER_KEY_METRIC,
               supported=None, strongly_supported=None, notes=[])
    has = lambda m: m in R and pr in R[m]                                                    # noqa: E731
    idx = np.flatnonzero(inmodel)
    a_ok = a_strong = None
    if has(cm):
        out["capability"] = _cap_summary(R[cm][pr], ref, inmodel, cfg, folds, ood_rows, cm)
        a_ok, a_strong = out["capability"]["passes"], out["capability"]["strong"]
    else:
        out["notes"].append(f"(a) not evaluated: no PredTable for {cm}/{pr}")
    # ---- (b) diagnosis
    b_ok = None
    tdm, tdb = tables.get(dm, {}).get(pr), tables.get(db, {}).get(pr)
    if tdm is not None and tdb is not None and "mode_prob" in tdm and "mode_prob" in tdb:
        merge = bool(cfg["h1r_diag_merge_cb"])
        flt_mask = ((tdm["mode_true"] > 0) & tdm["wheel_mask"]).any(1)
        flt = idx[flt_mask[idx]]
        defs = {}
        for name, mg in (("coast_brake_merged", True), ("unmerged", False)):
            dh, dbb = _diagnosed(tdm, mg), _diagnosed(tdb, mg)
            p, n10, n01 = S.mcnemar_greater(dh[idx], dbb[idx])
            defs[name] = dict(mcnemar_p=p, n10=n10, n01=n01,
                              diagnosed_rate=_prop(ref, idx, dh[idx], cfg),
                              diagnosed_rate_baseline=_prop(ref, idx, dbb[idx], cfg),
                              faulted_diagnosed_rate=_prop(ref, flt, dh[flt], cfg),
                              faulted_diagnosed_rate_baseline=_prop(ref, flt, dbb[flt], cfg))
        prim = defs["coast_brake_merged" if merge else "unmerged"]
        aps = {}
        for m in (dm, db):
            A_ = [R[m][pr][f].get("aps") for f in folds] if has(m) else [None]
            if all(a_ is not None for a_ in A_):
                allc = np.full(ref["trial_id"].shape[0], np.nan)
                size = np.full_like(allc, np.nan)
                for f, a_ in zip(folds, A_):
                    allc[R[m][pr][f]["test_idx"]] = a_["all_covered"]
                    size[R[m][pr][f]["test_idx"]] = a_["mean_size"]
                aps[m] = dict(all_wheels_covered=_prop(ref, idx, allc[idx] > 0.5, cfg, boot=True),
                              mean_set_size=float(np.nanmean(size[idx])))
        aps_ok = (aps[dm]["all_wheels_covered"]["est"] >= cfg["h1r_aps_min"]) if dm in aps else None
        rmin = cfg.get("h1r_diag_rate_min")
        rate_ok = True if rmin is None else bool(prim["faulted_diagnosed_rate"]["est"] >= rmin)
        p_ok = bool(prim["mcnemar_p"] < cfg["test_alpha"])
        if aps_ok is None:
            out["notes"].append(f"(b) not evaluated: no APS wheel sets for {dm}/{pr}")
        else:
            b_ok = bool(p_ok and aps_ok and rate_ok)
        out["diagnosis"] = dict(primary_definition="coast_brake_merged" if merge else "unmerged", definitions=defs,
                                mcnemar_p=prim["mcnemar_p"], n10=prim["n10"], n01=prim["n01"], mcnemar_passes=p_ok,
                                diagnosed_rate=prim["diagnosed_rate"]["est"],
                                diagnosed_rate_baseline=prim["diagnosed_rate_baseline"]["est"],
                                faulted_diagnosed_rate=prim["faulted_diagnosed_rate"]["est"],
                                faulted_diagnosed_rate_baseline=prim["faulted_diagnosed_rate_baseline"]["est"],
                                faulted_diagnosed_rate_min=rmin, wheel_sets=aps, aps_min=cfg["h1r_aps_min"],
                                aps_passes=aps_ok, passes=b_ok)
    else:
        out["notes"].append(f"(b) not evaluated: PredTable with mode_prob missing for {dm}/{pr} or {db}/{pr}")
    if a_ok is not None and b_ok is not None:
        out["supported"] = bool(a_ok and b_ok)
        out["strongly_supported"] = bool(out["supported"] and a_strong)
    # ---- secondary / descriptive (no verdict)
    sec = {}
    for m in cfg["h1r_secondary_methods"]:
        if has(m) and m != cm:
            sec[m] = _cap_summary(R[m][pr], ref, inmodel, cfg, folds)
    dg = {}
    for m, rr in tables.items():
        t_ = rr.get(pr)
        if t_ is not None and "mode_prob" in t_ and np.isfinite(t_["mode_prob"]).any():
            dg[m] = dict(diagnosed_rate_cb=float(_diagnosed(t_, True)[idx].mean()),
                         diagnosed_rate_unmerged=float(_diagnosed(t_, False)[idx].mean()))
    out["secondary"] = dict(status="DESCRIPTIVE_NO_VERDICT", capability=sec, diagnosis_by_method=dg)
    if has(dm) and has(cm):
        W1 = _per_trial(ref, R[dm][pr], "split_joint", "width")[idx]
        W0 = _per_trial(ref, R[cm][pr], "split_joint", "width")[idx]
        d = W1 - W0
        out["secondary"]["width_comparison"] = dict(
            status="DESCRIPTIVE_NO_VERDICT", methods=[dm, cm], median_width={dm: float(np.median(W1)),
                                                                              cm: float(np.median(W0))},
            median_paired_diff=float(np.median(d)), n_wider=int((d > 0).sum()), n_narrower=int((d < 0).sum()),
            note=f"positive diff = {dm} wider than {cm}")
    rows = []
    for role, m, cs in ([("confirmatory_a", cm, out.get("capability"))] if "capability" in out else []) + \
            [("secondary", m, v) for m, v in sec.items()]:
        for f, v in cs["per_fold"].items():
            rows.append(dict(role=role, method=m, fold=f, **v))
    pd.DataFrame(rows).to_csv(outdir / "fig_h1_redesign.csv", index=False)
    return out


def _h2_probe_ladder(lad, ref, inmodel, cfg, folds, outdir):
    """H2 probe ladder (approved 2026-11-20; metric changed to diagnosis by the pre-freeze change of 2026-11-20, PROTOCOL
    section 4 'Pre-freeze changes' D3-1). Rungs (e.g. P1+P5 -> +P2 -> +P3/P4 -> all) are PredTables of the ladder
    method (structured_ensemble) at the primary sensing rung, one per probe set.
    PRIMARY: per-trial 'diagnosed' (argmax mode correct on every wheel; COAST+BRAKE merged per h1r_diag_merge_cb) on
    the in-model DOUBLE-FAULT test trials (exactly two injected wheel faults = F5; cfg h2_trials = "double_fault",
    owner-approved 2026-11-20; single-fault diagnosis is at its ceiling with P1+P5 in simulation). The same ladder
    statistics on ALL in-model faulted trials are reported descriptively (`all_faulted`). For each step r0 -> r1:
    exact one-sided McNemar for an
    increase (r1 diagnosed and r0 not, against the reverse); Holm over all steps. Also, for every LATER step (all but
    the first), an exact one-sided McNemar for a DECREASE; Holm over the later steps.
    Supported iff (1) the first step's Holm-adjusted increase p < test_alpha AND (2) no later step has a Holm-adjusted
    decrease p < test_alpha. Later steps need not increase (they may saturate). Not evaluable (None) if the ladder
    tables carry no mode posterior or there are no faulted in-model trials.
    SECONDARY (descriptive, no verdict): APS mean set size and all-wheels coverage per rung (LOAO); the capability
    width ladder (split_joint on a FIXED leg set, cfg probe_ladder_targets, with the former sign tests and Holm, now
    descriptive: probes do not change capability width in simulation, C2); joint coverage per rung; the unmerged
    diagnosed rate; and the diagnosed rate over all in-model trials."""
    if cfg["probe_ladder_targets"] == "all":
        TL = np.arange(len(TARGET_KEYS))
    else:
        TL = eval_targets(lad[0][1] or cfg["seen_pids"])
    idx = np.flatnonzero(inmodel)
    merge = bool(cfg["h1r_diag_merge_cb"])
    if "mode_true" in ref:
        nfw = ((ref["mode_true"] > 0) & ref["wheel_mask"]).sum(1)
        flt_all, dbl = nfw > 0, nfw == 2
    else:
        flt_all = trial_conditions(ref, cfg, TL)["inmodel_fault"]
        dbl = np.asarray(ref["fault"]) == "F5"
    aidx = idx[flt_all[idx]]                                   # all in-model faulted trials (descriptive)
    fidx = idx[dbl[idx]] if cfg.get("h2_trials", "double_fault") == "double_fault" else aidx
    rows, W, Hh, DG = [], {}, {}, {}
    for label, pids, t_ in lad:
        pf = {f: fold_eval(t_, f, cfg, TL) for f in folds}
        W[label] = _per_trial(ref, pf, "split_joint", "width")
        Hh[label] = _per_trial(ref, pf, "split_joint", "hit").astype(bool)
        dg = _diagnosed(t_, merge)
        dgu = _diagnosed(t_, False)
        DG[label] = dg
        aps_size = aps_cov = math.nan
        if all("aps" in pf[f] for f in folds):
            sz = np.full(ref["trial_id"].shape[0], np.nan)
            ac = np.full_like(sz, np.nan)
            for f in folds:
                sz[pf[f]["test_idx"]] = pf[f]["aps"]["mean_size"]
                ac[pf[f]["test_idx"]] = pf[f]["aps"]["all_covered"]
            aps_size, aps_cov = float(np.nanmean(sz[idx])), float(np.nanmean(ac[idx]))
        row = _cov_row(ref, idx, Hh[label][idx], cfg)
        fr = _prop(ref, fidx, dg[fidx], cfg) if dg is not None else None
        fa = _prop(ref, aidx, dg[aidx], cfg) if dg is not None else None
        rows.append(dict(label=label, pids="+".join(pids),
                         confirmatory_diagnosed_rate=fr["est"] if fr else math.nan,
                         n_confirmatory=int(fidx.size),
                         all_faulted_diagnosed_rate=fa["est"] if fa else math.nan, n_all_faulted=int(aidx.size),
                         faulted_diagnosed_rate=fr["est"] if fr else math.nan,
                         faulted_diagnosed_cp_lo=fr["cp_lo"] if fr else math.nan,
                         faulted_diagnosed_cp_hi=fr["cp_hi"] if fr else math.nan, n_faulted=int(fidx.size),
                         faulted_diagnosed_rate_unmerged=float(dgu[fidx].mean()) if dgu is not None and fidx.size
                         else math.nan,
                         diagnosed_rate=float(dg[idx].mean()) if dg is not None else math.nan,
                         diagnosed_rate_unmerged=float(dgu[idx].mean()) if dgu is not None else math.nan,
                         aps_mean_set_size=aps_size, aps_all_wheels_covered=aps_cov,
                         coverage=row["coverage"], conservative_lower=row["conservative_lower"],
                         median_width=float(np.median(W[label][idx])),
                         per_key_coverage=float(np.mean(_per_trial(ref, pf, "split_joint", "per_target")[idx])),
                         n=row["n"]))
    labels = [l_ for l_, _, _ in lad]
    pairs = list(zip(labels[:-1], labels[1:]))
    # ---- secondary: the capability-width ladder (formerly the H2 metric; descriptive now)
    wsteps, wps = [], []
    for r0, r1 in pairs:
        d = (W[r0] - W[r1])[idx]
        p, npos, nneg = S.sign_test_greater(d)
        wsteps.append(dict(step=f"{r0}>{r1}", p=p, n_narrower=npos, n_wider=nneg, median_diff=float(np.median(d))))
        wps.append(p)
    for s_, pa in zip(wsteps, S.holm(wps) if wps else []):
        s_["p_holm"] = float(pa)
    holds = all(r["coverage"] >= cfg["coverage_hold"] for r in rows)
    pd.DataFrame(rows).to_csv(outdir / "fig_probe_ladder.csv", index=False)
    out = dict(status=STATUS, rule=_h2_probe_ladder.__doc__.strip(),
               primary_metric=("per-trial diagnosed (all wheels argmax correct, "
                               + ("COAST+BRAKE merged" if merge else "unmerged") + ") on in-model "
                               + ("double-fault (F5) trials" if cfg.get("h2_trials", "double_fault") == "double_fault"
                                  else "faulted trials")),
               confirmatory_trials=cfg.get("h2_trials", "double_fault"),
               coverage_metric="joint coverage over the fixed ladder leg set (probe_ladder_targets); descriptive",
               targets=[TARGET_KEYS[i] for i in TL], rungs=rows, steps=[], n_faulted=int(fidx.size),
               width_steps=dict(status="DESCRIPTIVE_NO_VERDICT", steps=wsteps, coverage_holds=holds,
                                all_steps_narrower=bool(wsteps) and all(s_["p_holm"] < cfg["test_alpha"]
                                                                       for s_ in wsteps)),
               supported=None, strongly_supported=None, notes=[])
    if any(DG[l_] is None for l_ in labels) or not pairs or fidx.size == 0:
        out["notes"].append("not evaluable: ladder tables without mode_prob, fewer than 2 rungs, or no faulted "
                            "in-model trials")
        return out
    allf = []
    for r0, r1 in pairs:                                       # descriptive: the ladder on all in-model faulted trials
        a1, a0 = DG[r1][aidx], DG[r0][aidx]
        p_inc, n10, n01 = S.mcnemar_greater(a1, a0)
        allf.append(dict(step=f"{r0}>{r1}", rate_from=float(a0.mean()) if aidx.size else math.nan,
                         rate_to=float(a1.mean()) if aidx.size else math.nan, n_gained=n10, n_lost=n01,
                         p_increase=p_inc))
    out["all_faulted"] = dict(status="DESCRIPTIVE_NO_VERDICT", n=int(aidx.size), steps=allf)
    steps, pinc, pdec = [], [], []
    for r0, r1 in pairs:
        a1, a0 = DG[r1][fidx], DG[r0][fidx]
        p_inc, n10, n01 = S.mcnemar_greater(a1, a0)
        p_dec = S.mcnemar_greater(a0, a1)[0]
        steps.append(dict(step=f"{r0}>{r1}", rate_from=float(a0.mean()), rate_to=float(a1.mean()),
                          n_gained=n10, n_lost=n01, p_increase=p_inc, p_decrease=p_dec))
        pinc.append(p_inc)
        pdec.append(p_dec)
    for s_, pa in zip(steps, S.holm(pinc)):
        s_["p_increase_holm"] = float(pa)
        s_["p"] = s_["p_holm"] = float(pa)                     # backward-compatible keys = the increase test
    later = S.holm(pdec[1:]) if len(pdec) > 1 else []
    steps[0]["p_decrease_holm_later"] = None
    for s_, pa in zip(steps[1:], later):
        s_["p_decrease_holm_later"] = float(pa)
        s_["significant_decrease"] = bool(pa < cfg["test_alpha"])
    first_ok = bool(steps[0]["p_increase_holm"] < cfg["test_alpha"])
    no_drop = not any(s_.get("significant_decrease", False) for s_ in steps[1:])
    out.update(steps=steps, first_step_increase=first_ok, no_later_significant_decrease=no_drop,
               supported=bool(first_ok and no_drop),
               all_steps_increase=bool(all(s_["p_increase_holm"] < cfg["test_alpha"] for s_ in steps)))
    return out


# ------------------------------------------------------------------------------------------------ H3 redesign support
CERT_RULES = ("conformal_gated", "conformal", "raw", "point")


def trial_conditions(t, cfg, T):
    """Boolean trial masks used by the H3 redesign (all computed from evaluator-only labels / truth):
      inmodel            ood == none
      ood                ood != none
      ood_fault          ood != none AND >= 1 injected in-model wheel fault (mode_true > 0)
      ood_nofault        ood != none AND no injected wheel fault
      ood_consequential  ood != none AND the truth contradicts 'go': some decision leg has y < tau
      ood_inconsequential ood != none AND every decision leg has y >= tau
    """
    ood = t["ood"] != "none"
    if "mode_true" in t:
        faulted = ((t["mode_true"] > 0) & t["wheel_mask"]).any(1)
    elif "condition" in t:
        # D3 (pre-freeze 2026-11-20): the schedule `condition` distinguishes F6w/F7w (OOD + firmware wheel fault)
        # from F6/F7 (OOD only); `fault` stays F6/F7 for both, so it cannot be used for OOD trials.
        faulted = np.array([str(c) not in ("F0", "F6", "F7") for c in t["condition"]])
    else:
        faulted = np.array([f not in ("F0", "F6", "F7") for f in t["fault"]])
    D = decision_idx(cfg)
    yd = t["y"][:, D]
    conseq = np.any(yd < cfg["tau"], 1)
    return dict(inmodel=~ood, ood=ood, ood_fault=ood & faulted, ood_nofault=ood & ~faulted,
                ood_consequential=ood & conseq, ood_inconsequential=ood & ~conseq,
                inmodel_fault=~ood & faulted)


def decision_idx(cfg):
    return np.array([TARGET_KEYS.index(k) for k in cfg["decision_targets"]])


def overclaim(lo_cert, y, idx, tau, gate=None):
    """Certification = 'capability d is reliable' (lower bound >= tau). Per trial: any certified leg in `idx` whose
    true ratio is < tau (an overclaim), and the number of certified legs. `gate` (B,) bool withholds all
    certificates on flagged trials (the calibrated explainer withholds when the OOD flag fires)."""
    cert = lo_cert[:, idx] >= tau
    if gate is not None:
        cert &= ~gate[:, None]
    yy = y[:, idx]
    wrong = cert & (yy < tau) & np.isfinite(yy)
    return wrong.any(1), cert.sum(1), cert.any(1)


def _per_trial_arr(t, per_fold, key):
    N = t["trial_id"].shape[0]
    out = None
    for f in per_fold.values():
        v = np.asarray(f[key])
        if out is None:
            out = np.full((N,) + v.shape[1:], np.nan if v.dtype.kind == "f" else False, dtype=v.dtype)
        out[f["test_idx"]] = v
    return out


def _prop(t, idx, x, cfg, boot=False):
    """Proportion with one-sided CP bounds; boot=True adds the one-way session-cluster bootstrap CI."""
    x = np.asarray(x, bool)
    k, n = int(x.sum()), int(x.size)
    r = None
    if boot and n:
        one, _ = _clusters(t, idx, cfg)
        r = S.coverage_summary(x, one, None, cfg["n_boot"], cfg["seed"], cfg["conf"])
    return dict(est=k / n if n else math.nan, k=k, n=n, cp_lo=S.cp_lower(k, n, cfg["conf"]) if n else math.nan,
                cp_hi=S.cp_upper(k, n, cfg["conf"]) if n else math.nan,
                boot_lo=(r or {}).get("boot_oneway", {}).get("lo"), boot_hi=(r or {}).get("boot_oneway", {}).get("hi"))


def _h3_ood(R, ref, inmodel, cfg, T, outdir):
    """Per method x rung, pooled over LOAO folds (each trial tested once), per trial condition:
    joint coverage of the split-joint (and raw) intervals over the unprobed legs (PRIMARY coverage metric; `per_target`
    = per-key coverage, secondary), coverage degradation (joint) vs the in-model trials, and overclaim rates for four certification rules:
      conformal_gated  split-joint lower bound >= tau, withheld when the OOD flag fires (= the calibrated explainer)
      conformal        split-joint lower bound >= tau (no OOD gate)
      raw              the model's own lowest predicted quantile >= tau (uncalibrated interval)
      point            the median prediction >= tau (= the point-estimate explainer's capability claim)
    over two leg sets: 'unprobed' (the evaluated legs T) and 'decision' (cfg decision_targets)."""
    conds = trial_conditions(ref, cfg, T)
    D = decision_idx(cfg)
    y, tau = ref["y"], cfg["tau"]
    rows, ovr = [], []
    out = dict(definitions=dict(conditions=trial_conditions.__doc__.strip(), tau=tau,
                                decision_targets=list(cfg["decision_targets"]),
                                cert_rules=_h3_ood.__doc__.strip(), coverage_primary=JOINT_METRIC,
                                coverage_secondary=PER_KEY_METRIC),
               n=dict((c, int(m.sum())) for c, m in conds.items()), per_method={})
    for m in R:
        for r in R[m]:
            pf = R[m][r]
            hit = _per_trial(ref, pf, "split_joint", "hit").astype(bool)
            raw_hit = _per_trial(ref, pf, "raw_joint", "hit").astype(bool)
            ptc = _per_trial(ref, pf, "split_joint", "per_target")
            lo_c, lo_r, pt = (_per_trial_arr(ref, pf, k) for k in ("split_joint_lo", "raw_lo", "point"))
            flag = _per_trial_arr(ref, pf, "ood_flag").astype(bool)
            los = dict(conformal_gated=(lo_c, flag), conformal=(lo_c, None), raw=(lo_r, None), point=(pt, None))
            ent = dict(coverage={}, overclaim={})
            for c, msk in conds.items():
                idx = np.flatnonzero(msk)
                cov = _prop(ref, idx, hit[idx], cfg, boot=True)
                cov.update(raw=float(raw_hit[idx].mean()) if idx.size else math.nan,
                           per_target=float(np.nanmean(ptc[idx])) if idx.size else math.nan,
                           ood_flag_rate=float(flag[idx].mean()) if idx.size else math.nan)
                ent["coverage"][c] = cov
                rows.append(dict(method=m, rung=r, condition=c, n=cov["n"], coverage=cov["est"], cp_lo=cov["cp_lo"],
                                 cp_hi=cov["cp_hi"], raw_coverage=cov["raw"], per_target_coverage=cov["per_target"],
                                 ood_flag_rate=cov["ood_flag_rate"]))
                ent["overclaim"][c] = {}
                for rule, (lo, gate) in los.items():
                    for lset, li in (("unprobed", T), ("decision", D)):
                        wrong, ncert, anyc = overclaim(lo[idx], y[idx], li, tau, None if gate is None else gate[idx])
                        pr = _prop(ref, idx, wrong, cfg)
                        pr.update(certify_rate=float(anyc.mean()) if idx.size else math.nan,
                                  mean_certified=float(ncert.mean()) if idx.size else math.nan)
                        ent["overclaim"][c][f"{rule}.{lset}"] = pr
                        ovr.append(dict(method=m, rung=r, condition=c, rule=rule, legs=lset, n=pr["n"],
                                        overclaim=pr["est"], cp_lo=pr["cp_lo"], cp_hi=pr["cp_hi"],
                                        certify_rate=pr["certify_rate"]))
            base = ent["coverage"]["inmodel"]["est"]
            ent["coverage_degradation"] = {c: (base - v["est"]) if v["n"] else math.nan
                                           for c, v in ent["coverage"].items() if c != "inmodel"}
            out["per_method"].setdefault(m, {})[r] = ent
    pd.DataFrame(rows).to_csv(outdir / "fig_ood_coverage.csv", index=False)
    pd.DataFrame(ovr).to_csv(outdir / "fig_overclaim.csv", index=False)
    return out


def _h3_redesign(df, ref, R, h3o, cfg, T, outdir):
    """Approved 2026-11-20 (H3). Uses the WS-E explain CSV plus `_h3_ood`. The capability certificate and the
    coverage-degradation headline come from cfg h3_cap_method (the deployed capability method, `ensemble`), falling
    back to primary_method if that PredTable is absent. Coverage = joint coverage over the 10 untested legs.
    misattribution (per trial): the explanation asserts an unsupported cause, or names a wheel fault on a trial with
    no injected wheel fault. On OOD + wheel-fault trials naming the (truly faulted) wheel's mode is not penalised; a
    cause claim is (WS-E scores every wheel cause on an OOD trial as unsupported, since the deficit is not explained by
    the wheel alone). overclaim: the CSV column `capability_overclaim` if WS-E/WS-I provide it, else the table rule
    (calibrated -> conformal_gated, point_estimate -> point) of the primary method/rung over the decision legs.
    Rule: supported iff, on the consequential OOD trials (pooled, with and without wheel faults),
      (i)   calibrated overclaim <= alpha,
      (ii)  point-estimate overclaim CP lower bound > alpha,
      (iii) exact one-sided McNemar (point overclaims, calibrated does not) p < test_alpha,
      (iv)  calibrated misattribution on OOD + wheel-fault trials <= alpha,
      (v)   calibrated unsupported-cause rate over all trials <= alpha (= original condition (i)).
    Strongly supported iff additionally the CP upper bounds of (i) and (iv) are <= alpha. Coverage degradation on the
    consequential OOD trials is reported alongside (descriptive; no threshold)."""
    a = cfg["alpha"]
    conds = trial_conditions(ref, cfg, T)
    pos = {tid: i for i, tid in enumerate(ref["trial_id"].tolist())}
    df = df[df.trial_id.isin(pos)].copy()
    ii = np.array([pos[t] for t in df.trial_id], int)
    faulted = conds["inmodel_fault"] | conds["ood_fault"]
    df["unsupported"] = df["n_unsupported_cause"].astype(int) > 0
    df["misattrib"] = df["unsupported"] | (df["wheel_cause_asserted"].astype(bool) & ~faulted[ii])
    pr = cfg["primary_rung"]
    pm = cfg.get("h3_cap_method") or cfg["primary_method"]
    if pm not in R or pr not in R[pm]:
        pm = cfg["primary_method"]
    ent = h3o["per_method"].get(pm, {}).get(pr)
    rule_of = {cfg["explain_calibrated"]: "conformal_gated", cfg["explain_point"]: "point"}
    D = decision_idx(cfg)
    rows, ind = [], {}
    for ex, g in df.groupby("explainer"):
        gi = np.array([pos[t] for t in g.trial_id], int)
        if "capability_overclaim" in g:
            oc = g["capability_overclaim"].astype(bool).to_numpy()
            src = "explain_csv"
        else:
            oc = np.full(len(g), np.nan)
            src = "table_rule:" + rule_of.get(ex, "none")
        ind[ex] = dict(idx=gi, mis=g["misattrib"].to_numpy(), uns=g["unsupported"].to_numpy(), oc=oc, src=src)
    tabind = _cert_wrong(ref, R[pm][pr], cfg, D) if (pm in R and pr in R[pm]) else None
    for ex, d in ind.items():
        if np.isnan(d["oc"].astype(float)).all() and rule_of.get(ex) in (tabind or {}):
            d["oc"] = tabind[rule_of[ex]][d["idx"]]
        for c in ("inmodel", "ood", "ood_fault", "ood_nofault", "ood_consequential", "ood_inconsequential"):
            s = conds[c][d["idx"]]
            sel = d["idx"][s]
            rr = dict(explainer=ex, condition=c, n=int(s.sum()),
                      misattribution=_prop(ref, sel, d["mis"][s], cfg),
                      unsupported=_prop(ref, sel, d["uns"][s], cfg),
                      overclaim=(_prop(ref, sel, d["oc"][s].astype(bool), cfg)
                                 if not np.isnan(d["oc"].astype(float)).any() else None),
                      overclaim_source=d["src"])
            rows.append(rr)
    flat = [dict(explainer=r["explainer"], condition=r["condition"], n=r["n"],
                 misattribution=r["misattribution"]["est"], misattribution_cp_hi=r["misattribution"]["cp_hi"],
                 unsupported=r["unsupported"]["est"],
                 overclaim=(r["overclaim"] or {}).get("est"), overclaim_cp_lo=(r["overclaim"] or {}).get("cp_lo"),
                 overclaim_cp_hi=(r["overclaim"] or {}).get("cp_hi"), overclaim_source=r["overclaim_source"])
            for r in rows]
    pd.DataFrame(flat).to_csv(outdir / "fig_explain_h3_redesign.csv", index=False)
    out = dict(status=STATUS, rule=_h3_redesign.__doc__.strip(), per_explainer=rows, cap_method=pm,
               coverage_metric=JOINT_METRIC,
               supported=None, strongly_supported=None)
    ce, pe = cfg["explain_calibrated"], cfg["explain_point"]
    if ce in ind and pe in ind:
        get = lambda ex, c: next(r for r in rows if r["explainer"] == ex and r["condition"] == c)  # noqa: E731
        cc, pc_ = get(ce, "ood_consequential"), get(pe, "ood_consequential")
        cf = get(ce, "ood_fault")
        call = _prop(ref, ind[ce]["idx"], ind[ce]["uns"], cfg)
        # paired McNemar on consequential OOD trials
        common = sorted(set(ind[ce]["idx"][conds["ood_consequential"][ind[ce]["idx"]]].tolist())
                        & set(ind[pe]["idx"][conds["ood_consequential"][ind[pe]["idx"]]].tolist()))
        mp = {}
        for ex in (ce, pe):
            o = dict(zip(ind[ex]["idx"].tolist(), ind[ex]["oc"].tolist()))
            mp[ex] = np.array([bool(o[i]) for i in common], bool) if common else np.zeros(0, bool)
        p, n10, n01 = S.mcnemar_greater(mp[pe], mp[ce]) if common else (math.nan, 0, 0)
        have = cc["overclaim"] is not None and pc_["overclaim"] is not None and cc["n"] > 0 and cf["n"] > 0
        out.update(n_consequential=cc["n"], n_ood_fault=cf["n"],
                   calibrated_overclaim=(cc["overclaim"] or {}).get("est"),
                   calibrated_overclaim_cp_hi=(cc["overclaim"] or {}).get("cp_hi"),
                   point_overclaim=(pc_["overclaim"] or {}).get("est"),
                   point_overclaim_cp_lo=(pc_["overclaim"] or {}).get("cp_lo"),
                   calibrated_ood_fault_misattribution=cf["misattribution"]["est"],
                   calibrated_ood_fault_misattribution_cp_hi=cf["misattribution"]["cp_hi"],
                   point_ood_fault_misattribution=get(pe, "ood_fault")["misattribution"]["est"],
                   calibrated_unsupported_rate=call["est"], mcnemar_overclaim_p=p, n10=n10, n01=n01)
        if ent is not None:
            out["coverage_consequential"] = ent["coverage"]["ood_consequential"]
            out["coverage_degradation_consequential"] = ent["coverage_degradation"]["ood_consequential"]
            out["coverage_ood_fault"] = ent["coverage"]["ood_fault"]
        # D3 (pre-freeze 2026-11-20): per assembly type evaluability. A type with no consequential OOD trials (with
        # the safe field specs, L3/square4 are physically never consequential in simulation) is reported as 'not
        # evaluable for this type', never as a failure; the verdict is the pooled rule above.
        by_type = {}
        for f in cfg["folds"]:
            ty = ref["assembly_type"] == f
            if not ty.any():
                continue
            e = dict(n_ood=int((conds["ood"] & ty).sum()), n_consequential=int((conds["ood_consequential"] & ty).sum()),
                     n_ood_fault=int((conds["ood_fault"] & ty).sum()))
            for ex, nm_ in ((ce, "calibrated"), (pe, "point")):
                d_ = ind[ex]
                tsel = ty[d_["idx"]]
                cs_ = conds["ood_consequential"][d_["idx"]] & tsel
                fs_ = conds["ood_fault"][d_["idx"]] & tsel
                oc_ = d_["oc"].astype(float)
                e[f"{nm_}_overclaim"] = (float(np.nanmean(oc_[cs_])) if cs_.any() and np.isfinite(oc_[cs_]).all()
                                         else None)
                e[f"{nm_}_ood_fault_misattribution"] = float(d_["mis"][fs_].mean()) if fs_.any() else None
            e["evaluable"] = e["n_consequential"] > 0
            e["status"] = ("evaluable" if e["evaluable"] else
                           "not evaluable for this type: 0 consequential OOD trials (not a failure)")
            by_type[f] = e
        out["by_type"] = by_type
        out["types_not_evaluable"] = [f for f, e in by_type.items() if not e["evaluable"]]
        out["evaluable"] = bool(have)
        if have:
            sup = bool(out["calibrated_overclaim"] <= a and out["point_overclaim_cp_lo"] > a
                       and p < cfg["test_alpha"] and out["calibrated_ood_fault_misattribution"] <= a
                       and call["est"] <= a)
            out.update(supported=sup, strongly_supported=bool(
                sup and out["calibrated_overclaim_cp_hi"] <= a
                and out["calibrated_ood_fault_misattribution_cp_hi"] <= a))
        else:
            out["note"] = "not evaluable: no consequential OOD or no OOD + wheel-fault trials, or no overclaim input"
            out["verdict_override"] = "not evaluable (no consequential OOD trials)"
    return out


def _cert_wrong(ref, per_fold, cfg, idx):
    """Per-trial overclaim indicators (N,) over legs `idx`, by certification rule, for one method x rung."""
    lo_c, lo_r, pt = (_per_trial_arr(ref, per_fold, k) for k in ("split_joint_lo", "raw_lo", "point"))
    flag = _per_trial_arr(ref, per_fold, "ood_flag").astype(bool)
    y, tau = ref["y"], cfg["tau"]
    return dict(conformal_gated=overclaim(lo_c, y, idx, tau, flag)[0], conformal=overclaim(lo_c, y, idx, tau)[0],
                raw=overclaim(lo_r, y, idx, tau)[0], point=overclaim(pt, y, idx, tau)[0])


def _h4(df, cfg, outdir):
    rows = []
    for arm, g in df.groupby("arm"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = S.clopper_pearson(k, n, cfg["conf"])
        rows.append(dict(arm=arm, n=n, k=k, rate=k / n, cp_lo=lo, cp_hi=hi))
    pd.DataFrame(rows).to_csv(outdir / "fig_docking.csv", index=False)
    a = {r["arm"]: r for r in rows}
    P, O, X = a.get(cfg["dock_proposed"]), a.get(cfg["dock_omni"]), a.get(cfg["dock_oracle"])
    out = dict(arms=rows)
    if P and O:
        out["fisher_p_proposed_gt_omni"] = S.fisher_greater(P["k"], P["n"], O["k"], O["n"])
        out["boschloo_p_proposed_gt_omni"] = S.boschloo_greater(P["k"], P["n"], O["k"], O["n"])
    if P and X:
        lo, hi = S.newcombe_diff(P["k"], P["n"], X["k"], X["n"], cfg["dock_ci_conf"])
        out.update(diff_proposed_minus_oracle=P["rate"] - X["rate"], newcombe_ci=[lo, hi],
                   noninferior=bool(lo > -cfg["dock_margin"]))
    if P and O and X:
        out["supported"] = bool(out["fisher_p_proposed_gt_omni"] < cfg["test_alpha"] and out["noninferior"])
    return out


def _h5(sims, tables, ref, inmodel, cfg, T, outdir):
    """Calibrate split-joint conformal on half of each method's sim held-out set; coverage on the other sim half
    (cov_sim) and on all in-model real trials at the primary rung (cov_real); both = joint coverage over the 10
    untested legs. gap = cov_sim - cov_real (signed, reported). PRIMARY (pre-freeze change D3-2, 2026-11-20): the
    under-coverage shortfall s = max(0, cov_sim - cov_real); supported iff the one-sided (1 - conf) bootstrap lower
    bound of s_nominal - s_fitted is > 0 (fitted = primary_method, nominal = nominal_method; real trials resampled by
    session cluster, paired across methods; sim trials iid). Over-coverage on real data is conservative and is not
    penalised. The former |gap| contrast is reported as `abs_gap_*` (descriptive)."""
    rng = np.random.default_rng(cfg["seed"])
    nb = cfg["n_boot"]
    Wr = S.bootstrap_weights(ref[cfg["cluster_oneway"]][inmodel], nb, cfg["seed"])
    rows, reps = [], {}
    for m, st in sims.items():
        if not np.all(st["source"] == "SIMULATION"):
            raise ValueError(f"sim table {m} contains non-SIMULATION rows")
        st = tb.subset(st, np.flatnonzero(st["ood"] == "none"))
        perm = rng.permutation(st["y"].shape[0])
        h = perm.size // 2
        sc, stt = tb.subset(st, perm[:h]), tb.subset(st, perm[h:])
        rt = tb.subset(tables[m][cfg["primary_rung"]], np.flatnonzero(inmodel))
        c = cap.calibrate(tb.as_pred(sc), sc["y"], cfg["alpha"], joint=True, targets=T)
        pks, hs = c.covered(tb.as_pred(stt), stt["y"])
        pkr, hr = c.covered(tb.as_pred(rt), rt["y"])
        lo = lambda t_: t_["y_q"][..., np.argmin(t_["q_levels"])]
        hi = lambda t_: t_["y_q"][..., np.argmax(t_["q_levels"])]
        raw_s = _joint_hit(lo(stt), hi(stt), stt["y"], T)[0]
        raw_r = _joint_hit(lo(rt), hi(rt), rt["y"], T)[0]
        Ws = rng.multinomial(hs.size, np.full(hs.size, 1 / hs.size), size=nb).astype(float)
        cs = (Ws @ hs) / hs.size
        cr = (Wr @ hr) / Wr.sum(1) if Wr is not None else np.full(nb, np.nan)
        reps[m] = cs - cr
        rows.append(dict(method=m, cov_sim=float(hs.mean()), cov_real=float(hr.mean()),
                         gap=float(hs.mean() - hr.mean()), raw_cov_sim=float(raw_s.mean()),
                         raw_cov_real=float(raw_r.mean()), raw_gap=float(raw_s.mean() - raw_r.mean()),
                         per_key_cov_sim=float(np.nanmean(np.where(np.isnan(stt["y"][:, T]), np.nan, pks))),
                         per_key_cov_real=float(np.nanmean(np.where(np.isnan(rt["y"][:, T]), np.nan, pkr))),
                         n_sim_test=int(hs.size), n_real=int(hr.size)))
    for r_ in rows:
        r_["shortfall"] = max(0.0, r_["gap"])
    pd.DataFrame(rows).to_csv(outdir / "fig_sim_real.csv", index=False)
    out = dict(per_method=rows, coverage_metric=JOINT_METRIC + " (cov_sim, cov_real, gap)",
               secondary_metric="per_key_cov_sim / per_key_cov_real (per-key coverage), raw_* (uncalibrated)")
    pm, nm = cfg["primary_method"], cfg["nominal_method"]
    if pm in reps and nm in reps:
        a = 1 - cfg["conf"]
        sf = lambda x: np.clip(x, 0, None)                                       # noqa: E731
        d = sf(reps[nm]) - sf(reps[pm])
        d = d[np.isfinite(d)]
        lo1 = float(np.quantile(d, a)) if d.size else math.nan
        da = np.abs(reps[nm]) - np.abs(reps[pm])
        da = da[np.isfinite(da)]
        g = {r["method"]: r for r in rows}
        out.update(metric="under-coverage shortfall s = max(0, cov_sim - cov_real), joint over the 10 untested legs",
                   shortfall={nm: g[nm]["shortfall"], pm: g[pm]["shortfall"]},
                   signed_gap={nm: g[nm]["gap"], pm: g[pm]["gap"]},
                   shortfall_nominal_minus_fitted=float(g[nm]["shortfall"] - g[pm]["shortfall"]),
                   one_sided_lower=lo1, supported=bool(lo1 > 0),
                   abs_gap_nominal_minus_calibrated=float(abs(g[nm]["gap"]) - abs(g[pm]["gap"])),
                   abs_gap_one_sided_lower=float(np.quantile(da, a)) if da.size else math.nan,
                   note="real trials resampled by session cluster (paired across methods); sim trials iid; "
                        "abs_gap_* = the superseded |gap| contrast (descriptive)")
    return out
