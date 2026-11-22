"""Task 2 data: held-out benchmark of genuinely unseen morphologies.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.gen_bench [--asm-per-family 4 --trials 48 --k7 16]

Families = FREE polyominoes (equivalence under rotation AND reflection, i.e. "symmetric equivalents" are one family):
all 5 tetrominoes, 12 pentominoes, 35 hexominoes (52 families of sizes 4-6), plus a random sample of heptominoes
(size 7, extrapolation point for the degradation curve). None of them is in any training set (trainW = sizes 1-3).
Per family `asm_per_family` assemblies, each a random fixed orientation of the shape, random module ids and random
module yaws (k*90 deg per module: mixed roller patterns), `trials` trials each, DR_WIDE domain, the default (devW)
fault prior. Output: ~/.cache/capxfer/I4/bench_v1/shard_*.pt in WS-A compact format (load with datagen.load_shard)
+ manifest.json. labels: family = free family id (k<size>_f<idx>), bbox_family = assembly.shape_family label.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from capxfer import assembly as A
from capxfer import datagen, sensors, sim

from .common import I4, free_name, free_polyominoes


def plan(rng, asm_per_family, k7):
    params = A.load_params(datagen.PARAMS)
    items = []
    for k in (4, 5, 6, 7):
        fams = free_polyominoes(k)
        keys = list(fams)
        if k == 7:
            keys = [keys[i] for i in sorted(rng.choice(len(keys), size=min(k7, len(keys)), replace=False))]
        for canon in keys:
            for j in range(asm_per_family if k < 7 else max(2, asm_per_family // 2)):
                cells = fams[canon][int(rng.integers(len(fams[canon])))]
                mods = [(str(rng.choice(("M1", "M2", "M3", "M4"))), i, jj, int(rng.integers(4))) for i, jj in cells]
                a = A.build(f"bench_{free_name(cells)}_{j}", mods, params)
                a.meta.update(family=free_name(cells), bbox_family=A.shape_family(cells), type="random",
                              cells=[list(c) for c in cells])
                items.append(a)
    return params, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bench_v1")
    ap.add_argument("--asm-per-family", type=int, default=4)
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--k7", type=int, default=16)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--wheel-budget", type=int, default=4096)
    ap.add_argument("--gpu-frac", type=float, default=0.08)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    rng = np.random.default_rng(a.seed)
    params, items = plan(rng, a.asm_per_family, a.k7)
    out = I4 / a.name
    out.mkdir(parents=True, exist_ok=True)
    man = dict(name=a.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(a),
               params_sha256=datagen.sha256_file(datagen.PARAMS), code_sha256=datagen.code_hash(), dr="DR_WIDE",
               fault_prior="sim.sample_faults defaults (devW recipe)", shards=[],
               families=sorted({x.meta["family"] for x in items}), counts={})
    by_n = {}
    for x in items:
        by_n.setdefault(x.n, []).append(x)
    t0, done, si, gseed = time.time(), 0, 0, 0
    for n, asms in sorted(by_n.items()):
        per = max(1, a.wheel_budget // (n * a.trials))
        for s in range(0, len(asms), per):
            grp = asms[s:s + per]
            doms, modes, gs = [], [], []
            for _ in grp:
                doms.append(sim.sample_domain(a.trials, rng, sim.DR_WIDE))
                m, g = sim.sample_faults(a.trials, n, rng)
                modes.append(m)
                gs.append(g)
            Rs = sim.simulate_multi(grp, params, doms, modes, gs, device="cuda", engine="graph",
                                    seed=a.seed * 100003 + gseed)
            pending = []
            for R, asm, dom in zip(Rs, grp, doms):
                ids = [f"{a.name}_g{gseed}_{i}" for i in range(a.trials)]
                raw = sensors.to_raw(R, params, dom, rng, trial_ids=ids, ood="none")
                rho, drift = sim.targets_from_twist(R)
                lab = raw["meta"]["labels"]
                lab.update(rho_twist=rho.cpu(), drift_twist=drift.cpu(), family=asm.meta["family"],
                           bbox_family=asm.meta["bbox_family"], asm_type="random", size=len(asm.modules))
                pending.append(datagen.compact(raw))
                gseed += 1
                done += a.trials
                man["counts"][asm.meta["family"]] = man["counts"].get(asm.meta["family"], 0) + a.trials
            p = out / f"shard_{si:05d}.pt"
            torch.save(pending, p)
            man["shards"].append(dict(file=p.name, groups=len(pending), sha256=datagen.sha256_file(p)))
            si += 1
            (out / "manifest.json").write_text(json.dumps(man, indent=1))
            print(f"n={n} {done} trials {time.time() - t0:.0f}s", flush=True)
            del Rs
    man.update(complete=True, trials=done, elapsed_s=time.time() - t0)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
