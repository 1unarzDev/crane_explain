"""Task 3(a): independent Fleet L1 truth model as the "real" plant (target swap, single module).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.l1_swap     # -> ~/.cache/capxfer/I4/l1_swap.json

Feasible version: L1 (capplan/truth, imported read-only through capxfer.xcheck_l1) is quasi-static and produces
steady twists, not encoder/duty/IMU streams, so the model's EVIDENCE is the capxfer simulation of the fault
(P1+P5, E2) while the capability TRUTH it is scored against is L1's rho for the same fault (48 L1 worlds per case,
matched = K_v 30 + capxfer back-drive range; native = L1's own priors). This asks: if reality's contact physics were
L1's instead of capxfer's (with P1/P5 behaving alike, which A.md D4 shows within ~0.05), would the intervals still
contain the truth? It is a lower bound on the harm of model-form error only for the unprobed legs; a full L1
observation model would be needed for the probed-signal side. Coverage = mean over (capxfer trial, L1 world) pairs
of the indicator rho_L1 in [lo, hi], untested legs, NaN (non-converged L1 solves) excluded.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from capxfer import assembly as A
from capxfer import datagen, sensors, sim, xcheck_l1 as X
from capxfer.conformal import capability as CC

from . import eval_core as E
from .common import ALPHA, I4, UNTESTED, feat_from_raw, json_dump

METHODS = ("W:structured_ensemble", "W:structured_npe", "W:ensemble", "W:blackbox", "W:analytic_bayes",
           "W:nominal_sim")


def gen_capxfer(params, B=256, seed=5):
    asm = A.physical("single", params)
    feats = []
    for ci, case in enumerate(X.CASES):
        name, mode_id, g, _ = case
        rng = np.random.default_rng(seed + ci)
        dom = sim.sample_domain(B, rng, sim.DR_WIDE)
        mode = torch.zeros(B, 4, dtype=torch.long)
        mode[:, 0] = mode_id
        gg = torch.ones(B, 4)
        if g is not None:
            gg[:, 0] = g
        R = sim.simulate(asm, params, dom, mode, gg, device="cuda", engine="graph", seed=seed + ci)
        raw = sensors.to_raw(R, params, dom, rng, trial_ids=[f"l1_{name}_{i}" for i in range(B)], ood="none")
        lab = raw["meta"]["labels"]
        lab.update(rho_twist=sim.targets_from_twist(R)[0].cpu(), family="phys_single", asm_type="single",
                   stress=name)
        F = feat_from_raw(raw)
        F["labels"]["stress"] = name
        feats.append(F)
    return feats


def cover_pairs(lo, hi, yl, legs=UNTESTED):
    """lo/hi (B,14) intervals, yl (W,14) L1 truths -> marginal and joint coverage over all (trial, world) pairs."""
    lo, hi, yl = lo[:, legs][:, None], hi[:, legs][:, None], yl[:, legs][None]
    c = (yl >= lo) & (yl <= hi)
    ok = np.isfinite(yl)
    marg = float(c[ok.repeat(lo.shape[0], 0)].mean())
    wok = ok[0].all(1)
    joint = float(c[:, wok].all(-1).mean()) if wok.any() else None
    return dict(marg=marg, joint=joint, n_worlds_joint=int(wok.sum()))


def main():
    import json, os
    frac = float(os.environ.get("I4_MEM_FRAC", 0.06))
    torch.cuda.set_per_process_memory_fraction(frac)
    E.MEM_FRAC = frac
    params = A.load_params(datagen.PARAMS)
    t0 = time.time()
    l1 = {}
    for case in X.CASES:
        for variant, matched in (("matched", True), ("native", False)):
            r, conv = X.l1_ratios(params, case, n_worlds=48, seed=0, matched=matched)
            l1[(case[0], variant)] = r
    print(f"L1 solved {time.time() - t0:.0f}s", flush=True)
    feats = gen_capxfer(params)
    dev = E.devw_feats()
    out = dict(cases=[c[0] for c in X.CASES], l1_mean={f"{k[0]}|{k[1]}": np.nanmean(v, 0).tolist() for k, v in l1.items()},
               capxfer_mean={F["labels"]["stress"]: F["labels"]["rho_twist"].mean(0).tolist() if "rho_twist" in F["labels"]
                             else None for F in feats}, methods={})
    prev = json.loads((I4 / "l1_swap.json").read_text()) if (I4 / "l1_swap.json").exists() else {}
    out["methods"].update(prev.get("methods", {}))
    for spec in METHODS:
        if spec in out["methods"]:
            continue
        m = E.load_model(spec)
        cal_f = dev["random"][::3] if spec.endswith("analytic_bayes") else dev["random"]
        nmax = max(F["wheel_x"].shape[1] for F in cal_f)
        cal = E.predict(m, cal_f, "cal", nmax)
        cj = CC.calibrate(dict(y_q=cal["y_q"].numpy()), cal["y"].numpy(), ALPHA, method="split", joint=True,
                          targets=UNTESTED)
        cm = CC.calibrate(dict(y_q=cal["y_q"].numpy()), cal["y"].numpy(), ALPHA, method="split", targets=UNTESTED)
        st = E.predict(m, feats, "l1", nmax)
        res = {}
        for name in out["cases"]:
            sel = np.asarray(st["stress"]) == name
            t = E.subset(st, sel)
            y = t["y"].numpy()
            row = {}
            for cn, c in (("raw", None), ("split", cm), ("split_joint", cj)):
                if c is None:
                    lo, hi = E.bounds(t)
                else:
                    lo, hi = (np.asarray(a) for a in c.interval(dict(y_q=t["y_q"].numpy())))
                row[cn] = dict(capxfer_truth=E.cov_stats(lo, hi, y),
                               **{f"L1_{v}": cover_pairs(lo, hi, l1[(name, v)]) for v in ("matched", "native")})
            row["diag"] = E.diag_stats(t)
            res[name] = row
        out["methods"][spec] = res
        print(spec, {n: (round(res[n]["split_joint"]["capxfer_truth"]["joint"], 2),
                         round(res[n]["split_joint"]["L1_matched"]["joint"] or -1, 2)) for n in res}, flush=True)
        json_dump(out, I4 / "l1_swap.json")
        del m


if __name__ == "__main__":
    main()
