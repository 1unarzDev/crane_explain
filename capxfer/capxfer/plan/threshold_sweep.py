"""Capability-threshold evidence (integrator request): for each run, the SAME domain/scene/fault is docked twice with
true-ratio knowledge, once forced sideways (yaw primitives removed) and once forced yaw+forward (lateral and diagonal
primitives removed). Success vs the true strafe ratio rho(P2+) shows where the alternative maneuver starts to beat
strafing; the crossover is the recommended capability_threshold.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.plan.threshold_sweep --runs 120 --procs 6
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np

FAULTS = ("F4", "F4x2d", "F1", "F2", "F3")


def _worker(task):
    import torch
    torch.set_num_threads(1)
    from .. import assembly as A, sim
    from . import dock as D, capability as CAP
    asm_name, run, base_seed = task
    P = A.load_params(os.path.join(os.path.dirname(__file__), "..", "..", "params", "module_measured.yaml"))
    asm = A.physical(asm_name, P)
    seed = int(np.random.default_rng([base_seed, zlib.crc32(asm_name.encode()), run]).integers(2 ** 31))
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(1, rng)
    fk = FAULTS[run % len(FAULTS)]
    wheel = int(rng.integers(asm.n))
    fault = D.fault_spec(asm, fk, wheel=wheel, g=float(rng.uniform(0.3, 0.9)))
    rho, dr = CAP.true_ratios(asm, P, dom, *fault, seed=seed)
    rho, dr = rho[0], dr[0]
    scene = D.make_scene(asm, seed, "wide")
    out = dict(assembly=asm_name, run=run, fault=fk, wheel=wheel, seed=seed, rho=rho.round(4).tolist(),
               drift=dr.round(4).tolist())
    for route, kill in (("sideways", ()), ("yaw+forward", ("P2", "P3", "P4", "P6y"))):
        r = rho.copy()
        for i, k in enumerate(CAP.KEYS):
            if kill and k.startswith(kill):
                r[i] = 0.0
        if route == "sideways":   # no gating: every direction usable, so the (shorter) sideways route is chosen
            cs = CAP.from_point(np.maximum(r, 1e-2), asm, np.zeros(14), name="forced_sideways", rho_min=0.0,
                                drift_max=np.inf, drift_max_yaw=np.inf)
        else:
            cs = CAP.from_point(np.maximum(r, 1e-3), asm, np.zeros(14), name="forced_yaw_forward")
        res = D.execute(cs, asm, P, scene, dom, *fault, loop="closed", seed=seed,
                        rng=np.random.default_rng([seed, 99]))
        out[route] = dict(success=res["success"], status=res["status"], time=res["time"], pos_err=res["pos_err"],
                          route=res["route"], collision=res["collision"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=120)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--assemblies", default="single,pair")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/capxfer/F/threshold_v1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(asm, i, a.seed) for asm in a.assemblies.split(",") for i in range(a.runs)]
    import multiprocessing as mp
    rows, t0 = [], time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool, open(os.path.join(a.out, "rows.jsonl"), "w") as fh:
        for k, r in enumerate(pool.imap_unordered(_worker, tasks)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.2]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r["rho"][CAP_KEYS_P2P] < hi]
        if not sel:
            continue
        table.append(dict(rho_bin=[lo, hi], n=len(sel),
                          sideways=sum(r["sideways"]["success"] for r in sel) / len(sel),
                          yaw_forward=sum(r["yaw+forward"]["success"] for r in sel) / len(sel),
                          t_sideways=float(np.median([r["sideways"]["time"] for r in sel])),
                          t_yaw_forward=float(np.median([r["yaw+forward"]["time"] for r in sel]))))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(dict(args=vars(a), table=table, elapsed_s=time.time() - t0), fh, indent=1)
    for t in table:
        print(t)


CAP_KEYS_P2P = 2   # index of "P2+" in probes.TARGET_KEYS (the bay is entered moving +y)

if __name__ == "__main__":
    main()
