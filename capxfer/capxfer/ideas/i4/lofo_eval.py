"""Task 1 evaluation: in-family vs leave-family-out models on each physical type (devW physical trials).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.lofo_eval   # -> ~/.cache/capxfer/I4/lofo_results.json

Generations: W (C's 5 x full-steps, trainW incl. all 4 small families), I4_lofo_none (matched reduced recipe, all
families), I4_lofo_<type> (same reduced recipe, the type's family removed). For square4 every generation is already
leave-family-out (the 2x2 square never occurs in sizes 1-3). Methods per generation: structured_ensemble and
ensemble, plus member 0 of each as the single-network structured_npe / blackbox. analytic_bayes has no training
(W tables) and is reported once.
Calibration pool (sim): devW random groups (every 2nd group, all sizes 1-6) MINUS the held-out type's family, so no
calibration trial shares the test morphology either. Variants: raw, split (per-key), split_joint (10 legs), group
conformal over shape families (ss1 / pooled, joint), APS wheel sets (joint pooling) on the same pool.
"""
from __future__ import annotations

import time

import numpy as np

from . import eval_core as E
from .common import I4, TYPE_FAMILY, TYPES, json_dump

GENS = ("W", "I4_lofo_none", "I4_lofo_single", "I4_lofo_pair", "I4_lofo_L3")


def evaluate_records(rec_pool, rec_phys, out):
    fam_pool = np.asarray(rec_pool["family"])
    fam_phys = np.asarray(rec_phys["family"])
    for t in TYPES:
        cal = E.subset(rec_pool, fam_pool != TYPE_FAMILY[t])
        test = E.subset(rec_phys, fam_phys == f"phys_{t}")
        r = E.conformal(cal, {"t": test}, groups="family")["t"]
        r["diag"] = E.diag_stats(test)
        r["aps"] = E.aps(cal, {"t": test}).get("t")
        r["n_cal"] = int(len(cal["y"]))
        r["n_cal_families"] = int(len(np.unique(np.asarray(cal["family"]))))
        out[t] = r
    return out


def main(gens=GENS):
    import os
    E.MEM_FRAC = float(os.environ.get("I4_MEM_FRAC", 0.06))
    dev = E.devw_feats()
    pool = dev["random"][::2]
    phys = dev["physical"]
    nmax = max(F["wheel_x"].shape[1] for F in pool + phys)
    import json
    path = I4 / f"lofo_results_{'_'.join(gens)}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for gen in [g for g in gens if g != "AB"]:
        for ens_name, single_name in (("structured_ensemble", "structured_npe"), ("ensemble", "blackbox")):
            t0 = time.time()
            ens = E.load_model(f"{gen}:{ens_name}")
            pe, p0 = E.predict_ens_and_m0(ens, pool, "pool", nmax)
            qe, q0 = E.predict_ens_and_m0(ens, phys, "phys", nmax)
            for name, rp, rq in ((ens_name, pe, qe), (single_name + "(m0)", p0, q0)):
                res[f"{gen}:{name}"] = evaluate_records(rp, rq, {})
            print(gen, ens_name, f"{time.time() - t0:.0f}s",
                  {t: (round(res[f'{gen}:{ens_name}'][t]['raw']['joint'], 3),
                       round(res[f'{gen}:{ens_name}'][t]['split_joint']['joint'], 3),
                       res[f'{gen}:{ens_name}'][t]['diag'].get('diag_trial')) for t in TYPES}, flush=True)
            json_dump(res, path)
            del ens
    if "AB" not in gens or "W:analytic_bayes" in res:
        return
    ab = E.load_model("W:analytic_bayes")
    sub_pool = pool[::4]
    rp = E.predict(ab, sub_pool, "pool", nmax)
    rq = E.predict(ab, phys, "phys", nmax)
    res["W:analytic_bayes"] = evaluate_records(rp, rq, {})
    json_dump(res, path)
    print("done", flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else GENS)
