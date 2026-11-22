"""Task 2 evaluation: per-family coverage / width / diagnosis on the free-polyomino benchmark (52 families of sizes
4-6 + sampled heptominoes), raw and calibrated, for every method, plus the in-distribution reference (devW random
sizes 1-3, the training families).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.bench_eval --methods W:structured_ensemble,W:ensemble,...
  -> ~/.cache/capxfer/I4/bench_v1_<method>.json and bench_v1_<method>_trials.npz

Calibration variants (alpha 0.1; legs = the 10 untested legs; marginal = per-key CQR, joint = normalized max CQR):
  raw            the model's own 5-95 % quantiles
  split_indist   split CQR on devW random sizes 1-3 (training families; what sim-only calibration gives)
  lofo_split     for each benchmark family f: split CQR pooled over all OTHER benchmark families, test on f
  lofo_group     same, Dunn-Wasserman-Ramdas two-layer group conformal with families as groups (ss1 = one draw per
                 family, exact finite-sample guarantee for a new exchangeable family; pooled = all trials)
  lsizeo_group   leave-size-out: calibrate group-ss1 on benchmark families of the OTHER sizes only, test size k
                 (the extrapolation-in-size version: no family of the test size was seen in calibration)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from capxfer.conformal import capability as CC
from capxfer.conformal import wheel_sets as WS

from . import eval_core as E
from .common import ALPHA, I4, UNTESTED, json_dump


def _iv(c, rec):
    lo, hi = c.interval(dict(y_q=rec["y_q"].numpy()))
    return np.asarray(lo), np.asarray(hi)


def per_trial_cover(lo, hi, y, legs=UNTESTED):
    c = (y[:, legs] >= lo[:, legs]) & (y[:, legs] <= hi[:, legs])
    w = (hi[:, legs] - lo[:, legs]).mean(1)
    return c.mean(1), c.all(1), w


def lofo(rec, fam, sizes, alpha=ALPHA, seeds=(0, 1, 2), log=print):
    """-> dict variant -> (marg (N,), joint (N,), width (N,)) per trial (test = own family left out)."""
    y = rec["y"].numpy()
    yq = rec["y_q"].numpy()
    N = len(y)
    out = {}
    variants = [("lofo_split", "split", None, False), ("lofo_split_joint", "split", None, True),
                ("lofo_group_ss1", "group", "ss1", False), ("lofo_group_ss1_joint", "group", "ss1", True),
                ("lofo_group_pooled", "group", "pooled", False), ("lofo_group_pooled_joint", "group", "pooled", True),
                ("lsizeo_group_ss1_joint", "group", "ss1", True), ("lsizeo_split_joint", "split", None, True)]
    for name, meth, gv, joint in variants:
        M, J, W = np.zeros(N), np.zeros(N, bool), np.zeros(N)
        reps = seeds if gv == "ss1" else (0,)
        units = np.unique(sizes) if name.startswith("lsizeo") else np.unique(fam)
        Macc, Jacc, Wacc = np.zeros(N), np.zeros(N), np.zeros(N)
        for sd in reps:
            for u in units:
                te = (sizes == u) if name.startswith("lsizeo") else (fam == u)
                tr = ~te
                kw = dict(method=meth, joint=joint, targets=UNTESTED, seed=sd)
                if meth == "group":
                    kw.update(groups=fam[tr], group_variant=gv)
                c = CC.calibrate(dict(y_q=yq[tr]), y[tr], alpha, **kw)
                lo, hi = (np.asarray(a) for a in c.interval(dict(y_q=yq[te])))
                m_, j_, w_ = per_trial_cover(lo, hi, y[te])
                Macc[te] += m_
                Jacc[te] += j_
                Wacc[te] += w_
        out[name] = (Macc / len(reps), Jacc / len(reps), Wacc / len(reps))
        log(f"   {name}: marg {out[name][0].mean():.3f} joint {out[name][1].mean():.3f}")
    return out


def summarize(per, fam, sizes, sel=None):
    """per: variant -> (marg, joint, width) per trial. Family-averaged (each family weight 1) and trial-averaged."""
    sel = np.ones(len(fam), bool) if sel is None else sel
    res = {}
    for v, (m, j, w) in per.items():
        fams = np.unique(fam[sel])
        fm = np.array([m[sel & (fam == f)].mean() for f in fams])
        fj = np.array([j[sel & (fam == f)].mean() for f in fams])
        fw = np.array([np.median(w[sel & (fam == f)]) for f in fams])
        res[v] = dict(marg=float(m[sel].mean()), joint=float(j[sel].mean()), width=float(np.median(w[sel])),
                      fam_mean_joint=float(fj.mean()), fam_min_joint=float(fj.min()),
                      fam_p10_joint=float(np.quantile(fj, 0.1)), fam_frac_joint_lt_0p8=float((fj < 0.8).mean()),
                      fam_mean_marg=float(fm.mean()), fam_min_marg=float(fm.min()),
                      fam_frac_marg_lt_0p85=float((fm < 0.85).mean()), fam_median_width=float(np.median(fw)),
                      n_families=int(len(fams)), n_trials=int(sel.sum()))
    return res


def run(spec, bench, dev_small, ab_subset=False, log=print):
    t0 = time.time()
    m = E.load_model(spec)
    if ab_subset:                                   # analytic Bayes: one assembly per family (hypothesis tables)
        bench = [F for F in bench if str(F["labels"]["asm"]).endswith("_0")]
        dev_small = dev_small[::6]
    nmax = max(F["wheel_x"].shape[1] for F in bench + dev_small)
    tp = time.time()
    rb = E.predict(m, bench, "bench", nmax)
    t_pred = (time.time() - tp) / len(rb["y"])
    rd = E.predict(m, dev_small, "indist", nmax)
    log(f"{spec}: predicted {len(rb['y'])} bench + {len(rd['y'])} in-dist trials in {time.time() - t0:.0f}s")
    fam = np.asarray(rb["family"])
    sizes = np.asarray([E.size_of(f) for f in fam])
    y = rb["y"].numpy()
    per = {"raw": per_trial_cover(*E.bounds(rb), y)}
    cal_m = CC.calibrate(dict(y_q=rd["y_q"].numpy()), rd["y"].numpy(), ALPHA, method="split", targets=UNTESTED)
    cal_j = CC.calibrate(dict(y_q=rd["y_q"].numpy()), rd["y"].numpy(), ALPHA, method="split", joint=True,
                         targets=UNTESTED)
    pm = per_trial_cover(*_iv(cal_m, rb), y)
    pj = per_trial_cover(*_iv(cal_j, rb), y)
    per["split_indist"] = (pm[0], pm[1], pm[2])
    per["split_indist_joint"] = (pj[0], pj[1], pj[2])
    per.update(lofo(rb, fam, sizes, log=log))
    res = dict(method=spec, predict_ms_per_trial=1000 * t_pred, n_bench_trials=int(len(y)),
               families=sorted(set(fam)), overall=summarize(per, fam, sizes),
               by_size={int(k): summarize(per, fam, sizes, sizes == k) for k in np.unique(sizes)})
    # in-distribution reference (devW sizes 1-3; raw and split on itself would be trivially exchangeable)
    fd = np.asarray(rd["family"])
    sd = np.asarray([E.size_of(f) for f in fd])
    yd = rd["y"].numpy()
    pd = {"raw": per_trial_cover(*E.bounds(rd), yd)}
    res["indist"] = {int(k): summarize(pd, fd, sd, sd == k) for k in np.unique(sd)}
    # diagnosis
    res["diag"] = {"overall": E.diag_stats(rb), **{int(k): E.diag_stats(E.subset(rb, sizes == k))
                                                   for k in np.unique(sizes)},
                   "indist": E.diag_stats(rd)}
    if not np.isnan(rb["mode_prob"].numpy()[rb["wheel_mask"].numpy()]).all():
        res["aps"] = E.aps(rd, {"bench": rb, **{f"k{k}": E.subset(rb, sizes == k) for k in np.unique(sizes)}})
        res["aps_lofo"] = aps_lofo(rb, fam, sizes)
        res["diag_per_family"] = {f: E.diag_stats(E.subset(rb, fam == f)).get("diag_trial") for f in np.unique(fam)}
    res["per_family"] = {f: {v: dict(marg=float(per[v][0][fam == f].mean()), joint=float(per[v][1][fam == f].mean()),
                                     width=float(np.median(per[v][2][fam == f])))
                             for v in ("raw", "split_indist_joint", "lofo_split_joint", "lofo_group_ss1_joint",
                                       "lofo_group_ss1")}
                         for f in np.unique(fam)}
    tag = spec.replace(":", "_").replace("#", "m")
    json_dump(res, I4 / f"bench_v1_{tag}.json")
    np.savez_compressed(I4 / f"bench_v1_{tag}_trials.npz", family=fam, size=sizes,
                        consequential=np.asarray(rb["consequential"]),
                        **{f"{v}_{k}": per[v][i] for v in per for i, k in enumerate(("marg", "joint", "width"))})
    log(f"{spec}: done {time.time() - t0:.0f}s; overall " + str({v: (round(r['marg'], 3), round(r['joint'], 3),
                                                                      round(r['width'], 3))
                                                                  for v, r in res["overall"].items()}))
    return res


def aps_lofo(rec, fam, sizes, alpha=ALPHA):
    """APS wheel sets calibrated on the OTHER benchmark families (joint pooling), per test family."""
    mp = torch.nan_to_num(rec["mode_prob"]).numpy()
    wm = rec["wheel_mask"].numpy()
    lab = rec["mode_true"].numpy()
    hit_all = np.zeros(len(lab), bool)
    size_all = np.zeros(len(lab))
    for f in np.unique(fam):
        te, tr = fam == f, fam != f
        sc = WS.calibrate(dict(mode_prob=mp[tr], wheel_mask=wm[tr]), np.where(lab[tr] < 0, 0, lab[tr]), alpha,
                          pooling="joint")
        sets = np.asarray(sc.sets(dict(mode_prob=mp[te], wheel_mask=wm[te])))
        hit = np.take_along_axis(sets, np.clip(lab[te], 0, 4)[..., None], -1)[..., 0] | ~wm[te]
        hit_all[te] = hit.all(1)
        size_all[te] = np.where(wm[te], sets.sum(-1), 0).sum(1) / wm[te].sum(1)
    out = {"overall": dict(trial_cover=float(hit_all.mean()), set_size=float(size_all.mean()))}
    for k in np.unique(sizes):
        s = sizes == k
        out[int(k)] = dict(trial_cover=float(hit_all[s].mean()), set_size=float(size_all[s].mean()))
    fc = np.array([hit_all[fam == f].mean() for f in np.unique(fam)])
    out["fam_min_trial_cover"] = float(fc.min())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="W:structured_ensemble,W:ensemble,W:structured_npe,W:blackbox")
    ap.add_argument("--mem-frac", type=float, default=0.06)
    a = ap.parse_args()
    E.MEM_FRAC = a.mem_frac
    import os
    from .common import I4 as _I4
    bench = E.bench_feats()
    dev = E.devw_feats()
    small = [F for F in dev["random"] if (E.size_of(F["labels"]["family"]) or 99) <= 3]
    for spec in a.methods.split(","):
        if (_I4 / f"bench_v1_{spec.replace(':', '_').replace('#', 'm')}.json").exists():
            continue
        run(spec, bench, small, ab_subset=spec.endswith("analytic_bayes"), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
