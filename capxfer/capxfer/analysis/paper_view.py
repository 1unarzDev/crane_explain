"""Paper-facing block of results.json, matching paper/results_map.json (WS-G).

Adds these top-level keys: counts, coverage, width, wheel_sets, explanation, ood, docking, hypotheses, plus
sysid / provenance counts passed through from an optional manifest `provenance` JSON (WS-A/WS-C write it; hashed).
Proportions use {est, ci_lo, ci_hi, n, k, ci_method}. The CI is the 95 % one-way session-cluster bootstrap
percentile interval. Where that bootstrap is withheld (< 2 clusters), the CI falls back to Clopper-Pearson and
ci_method says so. Coverage throughout is the joint (all untested targets) split-conformal indicator on in-model
trials.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..conformal import capability as cap
from . import stats as S
from . import table as tb


def _pci(hit, clusters, cfg, seed_off=0, cluster_name="session"):
    h = np.asarray(hit, float)
    h = h[~np.isnan(h)]
    n, k = int(h.size), float(h.sum())
    if n == 0:
        return dict(est=None, ci_lo=None, ci_hi=None, n=0, k=0, ci_method="no data")
    est = k / n
    W = S.bootstrap_weights(clusters, cfg["n_boot"], cfg["seed"] + seed_off) if clusters is not None else None
    ci = S.boot_ci(h, W, "mean", cfg["conf"]) if W is not None else {"status": "WITHHELD"}
    binary = np.all((h == 0) | (h == 1))
    if ci.get("status") == "AVAILABLE":
        lo, hi, meth = ci["lo"], ci["hi"], f"{cluster_name}-cluster bootstrap 95% percentile"
    elif binary:
        lo, hi = S.clopper_pearson(int(k), n, cfg["conf"])
        meth = "Clopper-Pearson 95% (cluster bootstrap withheld: <2 clusters)"
    else:
        lo, hi, meth = math.nan, math.nan, "withheld"
    return dict(est=est, ci_lo=lo, ci_hi=hi, n=n, k=int(round(k)), ci_method=meth)


def _q(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return dict(median=None, q25=None, q75=None)
    return dict(median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)))


def _verdict(h):
    if h is None:
        return dict(verdict="not evaluated (input missing)", supported=None)
    sup = h.get("supported")
    if sup is None:
        return dict(verdict=h.get("verdict_override") or "not evaluated (input missing)", supported=None)
    if sup and h.get("strongly_supported"):
        v = "supported (strong criterion met)"
    else:
        v = "supported" if sup else "not supported"
    return dict(verdict=v, supported=bool(sup), strongly_supported=h.get("strongly_supported"))


def _sim_heldout(st, cfg, T, seed):
    """split: random trial halves; group: family-level halves, Dunn et al. ss1 (one draw per calibration family)."""
    st = tb.subset(st, np.flatnonzero(st["ood"] == "none"))
    rng = np.random.default_rng(seed)
    fam = st["assembly_type"]
    size = st["n_wheels"] // 4 if "n_wheels" in st else np.zeros(fam.size, int)
    out = {}
    perm = rng.permutation(fam.size)
    h = perm.size // 2
    fams = np.unique(fam)
    fperm = rng.permutation(fams)
    cal_f = set(fperm[: fams.size // 2].tolist())
    is_cal = np.array([f in cal_f for f in fam])
    for name, ci, ti, kw in (
            ("split", perm[:h], perm[h:], dict(method="split")),
            ("group", np.flatnonzero(is_cal), np.flatnonzero(~is_cal),
             dict(method="group", group_variant="ss1"))):
        sc, stt = tb.subset(st, ci), tb.subset(st, ti)
        c = cap.calibrate(tb.as_pred(sc), sc["y"], cfg["alpha"], joint=True, targets=T,
                          groups=sc["assembly_type"] if kw["method"] == "group" else None, seed=seed, **kw)
        hit = c.covered(tb.as_pred(stt), stt["y"])[1]
        sz = size[ti]
        out[name] = dict(pooled=_pci(hit, stt["assembly_type"], cfg, cluster_name="family"),
                         by_size=[dict(n_modules=int(m), **_pci(hit[sz == m], stt["assembly_type"][sz == m], cfg,
                                                                cluster_name="family"))
                                  for m in np.unique(sz)],
                         threshold_finite=c.info["finite"], n_cal_families=int(np.unique(sc["assembly_type"]).size),
                         n_test_families=int(np.unique(stt["assembly_type"]).size))
    out["group"]["variant"] = "ss1 (Dunn, Wasserman & Ramdas 2023, one draw per calibration family)"
    return out


def paper_view(results, R, ref, inmodel, cfg, man, rp, inputs, sims, T, ex_df, dock_df):
    sess = ref[cfg["cluster_oneway"]]
    N = ref["trial_id"].size
    # ---- counts
    counts = dict(n_trials_real=int(N), n_sessions=int(np.unique(sess).size), n_trials_inmodel=int(inmodel.sum()),
                  n_trials_F6=int((ref["fault"] == "F6").sum()), n_trials_F7=int((ref["fault"] == "F7").sum()))
    if dock_df is not None:
        runs = dock_df.groupby("arm").size()
        counts["n_dock_runs_per_arm"] = int(runs.min())
        counts["dock_runs_by_arm"] = {str(k): int(v) for k, v in runs.items()}
    if sims:
        st = sims.get(cfg["primary_method"], next(iter(sims.values())))
        counts["n_sim_heldout_families"] = int(np.unique(st["assembly_type"]).size)
    if man.get("provenance"):
        p = rp(man["provenance"])
        from .frozen import _sha
        inputs["provenance"] = {"path": str(p), "sha256": _sha(p)}
        prov = json.loads(open(p).read())
        for k, v in (prov.get("counts") or {}).items():
            counts.setdefault(k, v)
        if "sysid" in prov:
            results["sysid"] = prov["sysid"]
    results["counts"] = counts

    # ---- coverage / width / wheel sets (LOAO, pooled = every trial tested once)
    cov, wid, wsets, ood = {}, {}, {}, {}
    for m in R:
        for r in R[m]:
            hit = np.full(N, np.nan)
            w = np.full(N, np.nan)
            ac = np.full(N, np.nan)
            fc = np.full(N, np.nan)
            sg = np.full(N, np.nan)
            em = np.full(N, np.nan)
            scheme_hits = {}
            for f, fo in R[m][r].items():
                ti = fo["test_idx"]
                hit[ti] = fo["schemes"]["split_joint"]["hit"]
                w[ti] = fo["schemes"]["split_joint"]["width"]
                for sc, v in fo["schemes"].items():
                    scheme_hits.setdefault(sc, np.full(N, np.nan))[ti] = v["hit"]
                if "aps" in fo:
                    ac[ti] = fo["aps"]["all_covered"]
                    fc[ti] = fo["aps"]["frac_covered"]
                    sg[ti] = fo["aps"]["frac_singleton"]
                    em[ti] = fo["aps"]["frac_empty"]
            im = inmodel
            folds = list(R[m][r])
            fold_of = ref["assembly_type"]
            cov.setdefault(m, {})[r] = dict(
                pooled=_pci(hit[im], sess[im], cfg),
                by_fold={f: _pci(hit[im & (fold_of == f)], sess[im & (fold_of == f)], cfg) for f in folds},
                by_scheme={sc: _pci(v[im], sess[im], cfg) for sc, v in scheme_hits.items()},
                ood_trials=_pci(hit[~im], sess[~im], cfg) if (~im).any() else None)
            wid.setdefault(m, {})[r] = dict(**_q(w[im]), by_fold={f: _q(w[im & (fold_of == f)]) for f in folds})
            if not np.all(np.isnan(ac)):
                wsets.setdefault(m, {})[r] = dict(
                    coverage=_pci(ac[im], sess[im], cfg), coverage_definition="all wheels of the trial covered "
                    "(trial-level APS, pooling=%s)" % cfg["aps_pooling"],
                    wheel_coverage=float(np.nanmean(fc[im])), singleton_rate=float(np.nanmean(sg[im])),
                    empty_rate=float(np.nanmean(em[im])))
    results["coverage"] = dict(loao=cov)
    results["width"] = dict(loao=wid)
    results["wheel_sets"] = dict(loao=wsets)

    # ---- OOD flag (primary method / rung), pooled over folds
    pm, pr = cfg["primary_method"], cfg["primary_rung"]
    if pm in R and pr in R[pm]:
        fl = np.full(N, np.nan)
        roc = {}
        for fo in R[pm][pr].values():
            fl[fo["test_idx"]] = fo["ood_flag"]
            for ag, v in fo["ood_roc"].items():
                roc.setdefault(ag, np.full(N, np.nan))[fo["test_idx"]] = v
        oo = ~inmodel
        ood = dict(method=pm, rung=pr, alpha=cfg["alpha"],
                   false_alarm_rate=_pci(fl[inmodel], sess[inmodel], cfg),
                   detection_rate=_pci(fl[oo], sess[oo], cfg) if oo.any() else None,
                   by_kind={k: _pci(fl[ref["ood"] == k], sess[ref["ood"] == k], cfg)
                            for k in ("payload", "low_mu") if (ref["ood"] == k).any()},
                   roc=[dict(alpha=float(ag), false_alarm=float(np.nanmean(v[inmodel])),
                             detection=float(np.nanmean(v[oo])) if oo.any() else None)
                        for ag, v in sorted(roc.items(), key=lambda kv: float(kv[0]))])
    results["ood"] = ood

    # ---- explanations
    if ex_df is not None:
        pos = {t: i for i, t in enumerate(ref["trial_id"].tolist())}
        df = ex_df[ex_df.trial_id.isin(pos)].copy()
        ii = np.array([pos[t] for t in df.trial_id])
        df["ood"], df["fault"], df["sess"] = ref["ood"][ii], ref["fault"][ii], sess[ii]
        df["unsup"] = (df["n_unsupported_cause"] > 0).astype(float)
        ex = {}
        for e, g in df.groupby("explainer"):
            o = g[g.ood != "none"]
            d = dict(unsupported_cause_rate=_pci(g.unsup, g.sess, cfg),
                     ood_misattribution=_pci(o.wheel_cause_asserted.astype(float), o.sess, cfg) if len(o) else None)
            if "correct_claim_rate" in g:
                d["correct_claim_rate"] = _pci(g.correct_claim_rate.astype(float), g.sess, cfg)
            if "informativeness" in g:
                d["informativeness"] = float(g.informativeness.mean())
            ex[str(e)] = d
        ex["by_condition"] = {
            str(f): {str(e): dict(unsupported_cause_rate=float(gg.unsup.mean()),
                                  wheel_cause_asserted_rate=float(gg.wheel_cause_asserted.mean()),
                                  **({"correct_claim_rate": float(gg.correct_claim_rate.mean())}
                                     if "correct_claim_rate" in gg else {}), n=int(len(gg)))
                     for e, gg in g.groupby("explainer")}
            for f, g in df.groupby("fault")}
        if "text" in df:
            f6 = sorted(df[df.fault == "F6"].trial_id.unique())
            if f6:
                ex["examples"] = [dict(trial_id=f6[0], explainer=str(r.explainer), text=str(r.text))
                                  for r in df[df.trial_id == f6[0]].itertuples()]
        results["explanation"] = ex

    # ---- docking
    if dock_df is not None:
        dk = {}
        for arm, g in dock_df.groupby("arm"):
            k, n = int(g.success.sum()), int(len(g))
            lo, hi = S.clopper_pearson(k, n, cfg["conf"])
            dk[str(arm)] = dict(successes=k, runs=n, rate=k / n, ci_lo=lo, ci_hi=hi, ci_method="Clopper-Pearson 95%")
        if "final_pose_error" in dock_df:
            dk["final_pose_error"] = dock_df[["arm", "run", "final_pose_error"]].to_dict("records")
        h4 = results.get("H4", {})
        dk["margin_test"] = dict(margin=cfg["dock_margin"], ci_conf=cfg["dock_ci_conf"],
                                 diff_proposed_minus_oracle=h4.get("diff_proposed_minus_oracle"),
                                 newcombe_ci=h4.get("newcombe_ci"), noninferior=h4.get("noninferior"),
                                 fisher_p_proposed_gt_omni=h4.get("fisher_p_proposed_gt_omni"))
        results["docking"] = dk

    # ---- simulation held-out (split and group conformal)
    if sims:
        results["coverage"]["sim_heldout"] = {m: _sim_heldout(st, cfg, T, cfg["seed"]) for m, st in sims.items()}

    results["hypotheses"] = {h: _verdict(results.get(h)) for h in ("H1", "H2", "H3", "H4", "H5")}
