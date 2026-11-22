"""Sim pilot of the docking arms (WS-F deliverable 4). Paired design: per run one randomised domain, scene and fault
wheel, shared by all arms.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.plan.pilot --runs 50 --procs 6 --out ~/.cache/capxfer/F/pilot_v1

Writes results.jsonl (one line per run x arm) and summary.json (success rate + Wilson 95% CI per config x arm).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib

import numpy as np

CONFIGS = (("square4", "F0"), ("square4", "F4"), ("pair", "F0"), ("pair", "F4"),
           ("single", "F4"), ("pair", "F4x2d"), ("single", "F4x2d"))
_NOM = {}


def _worker(task):
    import torch
    torch.set_num_threads(1)
    from .. import assembly as A, sim
    from . import dock as D, capability as CAP
    asm_name, fk, run, variant, loop, base_seed, ckw = task
    P = A.load_params(os.path.join(os.path.dirname(__file__), "..", "..", "params", "module_measured.yaml"))
    asm = A.physical(asm_name, P)
    seed = int(np.random.default_rng([base_seed, zlib.crc32(f"{asm_name}:{fk}".encode()), run]).integers(2 ** 31))
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(1, rng)
    if fk == "F4x2d":
        wheel = int(rng.integers(asm.n))
    elif fk == "F0":
        wheel = 0
    else:
        wheel = int(rng.integers(asm.n))
    fault = D.fault_spec(asm, fk, wheel=wheel)
    t0 = time.time()
    rho, dr = CAP.true_ratios(asm, P, dom, *fault, seed=seed)
    key = (asm_name, fk, wheel if fk != "F4x2d" else int(asm.wheel_module[wheel]))
    if key not in _NOM:
        nr, nd = CAP.true_ratios(asm, P, D.nominal_domain(), *fault, seed=0)
        _NOM[key] = (nr[0], nd[0])
    lo, hi = D.stub_intervals(rho[0], rng)
    dhi = D.stub_drift_bound(dr[0], rng)
    scene = D.make_scene(asm, seed, variant)
    out = []
    for arm in D.ARMS:
        t1 = time.time()
        r = D.simulate_arm(arm, asm, fault, dom, P, scene, truth=(rho[0], dr[0]), stub=(lo, hi, dhi),
                           nominal=_NOM[key], loop=loop, seed=seed, capset_kw=ckw)
        r.update(assembly=asm_name, fault=fk, wheel=wheel, run=run, seed=seed, variant=variant, loop=loop,
                 wall_s=time.time() - t1, rho_true=np.round(rho[0], 4).tolist(), drift_true=np.round(dr[0], 4).tolist())
        out.append(r)
    out[0]["truth_s"] = time.time() - t0
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarise(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["assembly"], r["fault"], r["variant"], r["loop"], r["arm"])].append(r)
    out = []
    for (a, f, v, l, arm), rs in sorted(g.items()):
        k = sum(r["success"] for r in rs)
        n = len(rs)
        st = {}
        for r in rs:
            st[r["status"]] = st.get(r["status"], 0) + 1
        ro = {}
        for r in rs:
            ro[r["route"]] = ro.get(r["route"], 0) + 1
        out.append(dict(assembly=a, fault=f, variant=v, loop=l, arm=arm, n=n, success=k, rate=k / n,
                        ci95=wilson(k, n), collisions=sum(r["collision"] for r in rs), status=st, routes=ro,
                        median_time=float(np.median([r["time"] for r in rs])),
                        median_pos_err=float(np.median([r["pos_err"] for r in rs]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--variant", default="wide")
    ap.add_argument("--loop", default="closed")
    ap.add_argument("--configs", default=",".join(f"{a}:{f}" for a, f in CONFIGS))
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--drift-max", type=float, default=None)
    ap.add_argument("--drift-max-yaw", type=float, default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/capxfer/F/pilot_v1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfgs = [tuple(c.split(":")) for c in a.configs.split(",")]
    ckw = {k: v for k, v in (("drift_max", a.drift_max), ("drift_max_yaw", a.drift_max_yaw)) if v is not None}
    tasks = [(asm, fk, i, a.variant, a.loop, a.seed, ckw) for asm, fk in cfgs for i in range(a.runs)]
    path = os.path.join(a.out, f"results_{a.variant}_{a.loop}.jsonl")
    rows = []
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.procs) as pool, open(path, "w") as fh:
        for k, res in enumerate(pool.imap_unordered(_worker, tasks)):
            for r in res:
                fh.write(json.dumps(r) + "\n")
                rows.append(r)
            fh.flush()
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(tasks)} runs, {time.time() - t0:.0f}s", flush=True)
    summ = summarise(rows)
    with open(os.path.join(a.out, f"summary_{a.variant}_{a.loop}.json"), "w") as fh:
        json.dump(dict(args=vars(a), elapsed_s=time.time() - t0, summary=summ), fh, indent=1)
    for s in summ:
        print(f"{s['assembly']:8s} {s['fault']:6s} {s['arm']:15s} {s['success']:3d}/{s['n']:3d} "
              f"[{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}] coll={s['collisions']} {s['status']} {s['routes']}")


if __name__ == "__main__":
    main()
