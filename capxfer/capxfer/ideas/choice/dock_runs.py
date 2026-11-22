"""CHOICE dock simulations (SIMULATION ONLY): closed-loop SIDEWAYS dock of one post-split single module.

Task: the module starts in front of a NARROW bay (width = module x-length + 0.12 m, so only a sideways entry fits:
the task is strafe-dependent by construction) and must reach the bay with WS-F's executor (capxfer.plan.dock.execute,
closed loop, stop-and-go replanning with a noisy localisation fix). Every dispatched module is planned with the omni
CapSet: every policy only sends a module it believes capable, so the planner differs in nothing but the module chosen.

Two task sets (written to ~/.cache/capxfer/CHOICE/dock_rows.jsonl, one JSON row per run, resumable):
  demo : single modules with the physical demo faults (healthy, F2 single coast, F2x2d coast fl+rr, F4 single
         roller-lock, F4x2d roller-lock fl+rr) at theta ~ DR_WIDE; 40 per type.
  ep   : every module of a size-stratified sample of faulted SPLIT test episodes (true health + true plant theta of
         the episode), plus faulted modules stratified over the strafe-ratio range (proxy calibration).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.choice.dock_runs [--procs 2] [--per-size 40]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import time

import numpy as np

OUT = pathlib.Path.home() / ".cache/capxfer/CHOICE"
ROWS = OUT / "dock_rows.jsonl"
DEMO_TYPES = ("F0", "F2", "F2x2d", "F4", "F4x2d")
COAST, ROLLER = 2, 4


def demo_health(kind, rng):
    mode = np.zeros(4, np.int64)
    if kind == "F2":
        mode[rng.integers(4)] = COAST
    elif kind == "F4":
        mode[rng.integers(4)] = ROLLER
    elif kind == "F2x2d":
        mode[[0, 3]] = COAST                    # corner order fl, fr, rl, rr
    elif kind == "F4x2d":
        mode[[0, 3]] = ROLLER
    return mode, np.ones(4, np.float32)


def tasks(per_size=24, per_bin=30, n_demo=20, seed=2026):
    from capxfer.ideas.split import common as C
    rng = np.random.default_rng(seed)
    out = []
    for ki, kind in enumerate(DEMO_TYPES):
        for i in range(n_demo):
            rng_d = np.random.default_rng([seed, ki, i])
            mode, g = demo_health(kind, rng_d)
            out.append(dict(key=f"demo|{kind}|{i}", set="demo", kind=kind, mid=str(rng_d.choice(C.MIDS)),
                            mode=mode.tolist(), g=g.tolist(), theta=None, dom_seed=int(rng_d.integers(1 << 30)),
                            seed=int(rng_d.integers(1 << 30))))
    E = C.episodes("P1P5P2")
    Gs = C.split_groups(E)
    T = np.load(C.OUT / "truth_singles.npz")
    ep, mod = T["ep"], T["mod"]
    z = np.minimum(T["rho"][:, 2], T["rho"][:, 3])
    test = np.setdiff1d(np.arange(len(E)), Gs["calA"])
    faulted = np.array([k for k in test if (E[k]["mode"] > 0).any()])
    ep_rows = []
    chosen = set()
    for nm in range(2, 7):
        ks = faulted[np.array([E[k]["nmod"] for k in faulted]) == nm]
        for k in rng.choice(ks, min(per_size, len(ks)), replace=False):
            chosen.add(int(k))
            ep_rows += [int(r) for r in np.flatnonzero(ep == k)]
    # strafe-ratio-stratified faulted modules from other test episodes (proxy calibration in the transition range)
    nf = np.array([(E[k]["mode"][4 * m:4 * m + 4] > 0).sum() for k, m in zip(ep, mod)])
    pool = np.flatnonzero(np.isin(ep, faulted) & ~np.isin(ep, list(chosen)) & (nf > 0))
    bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    extra = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = pool[(z[pool] >= lo) & (z[pool] < hi)]
        extra += [int(r) for r in rng.choice(b, min(per_bin, len(b)), replace=False)]
    for tag, rows in (("ep", ep_rows), ("zbin", extra)):
        for r in rows:
            e = E[ep[r]]
            m = int(mod[r])
            out.append(dict(key=f"{tag}|{r}", set=tag, row=r, ep=int(ep[r]), mid=e["modules"][m][0],
                            mode=e["mode"][4 * m:4 * m + 4].tolist(), g=e["g"][4 * m:4 * m + 4].tolist(),
                            theta=e["theta"], yaw_std=e["yaw_std"], z=float(z[r]),
                            seed=int(rng.integers(1 << 30))))
    # demo + z-bin rows interleaved first (proxy calibration), then whole episodes, small ones first, so partial
    # results give complete episodes for the direct validation
    head = [t for t in out if t["set"] != "ep"]
    head = [head[i] for i in rng.permutation(len(head))]
    eps = sorted({t["ep"] for t in out if t["set"] == "ep"}, key=lambda k: (E[k]["nmod"], k))
    order = {k: i for i, k in enumerate(eps)}
    tail = sorted([t for t in out if t["set"] == "ep"], key=lambda t: order[t["ep"]])
    return head + tail


def _worker(t):
    import torch
    torch.set_num_threads(1)
    from capxfer import sim
    from capxfer.plan import dock as D, capability as CAP
    from capxfer.ideas.split import common as C, sims as SM
    prm = C.params()
    asm = C.single_asm(t["mid"], prm)
    if t["theta"] is None:
        dom = sim.sample_domain(1, np.random.default_rng(t["dom_seed"]), sim.DR_WIDE)
    else:
        dom = SM.dom_from([t["theta"]], t["yaw_std"])
    mode = torch.tensor([t["mode"]], dtype=torch.long)
    g = torch.tensor([t["g"]], dtype=torch.float32)
    t0 = time.time()
    res = {}
    # truth strafe ratio of THIS module at this plant (noise-free probe, as in the SPLIT truth)
    if t["theta"] is None:
        rho, dr = CAP.true_ratios(asm, prm, dom, mode, g, seed=t["seed"])
        res["rho"] = np.asarray(rho[0], float).round(4).tolist()
        res["z"] = float(min(rho[0][2], rho[0][3]))
    scene = D.make_scene(asm, t["seed"], "narrow")
    r = D.execute(CAP.omni(asm), asm, prm, scene, dom, mode, g, loop="closed", seed=t["seed"],
                  rng=np.random.default_rng([t["seed"], 99]))
    res.update({k: r[k] for k in ("success", "status", "time", "pos_err", "head_err", "collision", "iters")})
    res["wall_s"] = time.time() - t0
    return {**{k: v for k, v in t.items() if k not in ("theta",)}, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--per-size", type=int, default=40)
    ap.add_argument("--per-bin", type=int, default=40)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T = tasks(a.per_size, a.per_bin)
    done = set()
    if ROWS.exists():
        done = {json.loads(l)["key"] for l in ROWS.open()}
    todo = [t for t in T if t["key"] not in done]
    print(f"{len(T)} tasks, {len(todo)} to do", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool, ROWS.open("a") as fh:
        for k, r in enumerate(pool.imap(_worker, todo, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
