"""SPLIT simulations (SIMULATION ONLY): post-split truth at the true plant, and the single-module stage-2 library.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.split.sims truth|library
Outputs ~/.cache/capxfer/SPLIT/{truth_singles.npz, truth_rem.npz, library.npz}.
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import torch

from capxfer import sim
from capxfer.ideas.split import common as C

G_RANGE = (0.3, 0.9)
DEV = "cuda"


def setup():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.3)          # <= ~4.8 GB of 16


def dom_from(thetas, yaw_std=0.03):
    d = {k: torch.tensor([t[k] for t in thetas], dtype=torch.float32) for k in C.PLANT_KEYS}
    d["wheel_yaw_err_std"] = float(yaw_std)
    return d


def run_multi(asms, doms, modes, gs, seed, max_rows=16384):
    """simulate_multi over lists, chunked so that sum(B)*n <= max_rows*4; returns list of (B_i,14) rho arrays."""
    out = [None] * len(asms)
    n = asms[0].n
    cap = max(64, max_rows * 4 // n)
    i = 0
    while i < len(asms):
        j, tot = i, 0
        while j < len(asms) and (tot + modes[j].shape[0] <= cap or j == i):
            tot += modes[j].shape[0]
            j += 1
        Rs = sim.simulate_multi(asms[i:j], C.params(), doms[i:j], modes[i:j], gs[i:j], device=DEV, engine="graph",
                                record_hz=50, store_device="cpu", seed=seed + i)
        for k, R in zip(range(i, j), Rs):
            out[k] = sim.targets_from_twist(R)[0].cpu().numpy()
        del Rs
        i = j
    return out


def split_rows(asm, dom, mode, g, chunk):
    """one big batch for one assembly -> list of chunked (asm, dom, mode, g)"""
    B = mode.shape[0]
    res = []
    for s in range(0, B, chunk):
        sl = slice(s, min(B, s + chunk))
        res.append((asm, {k: (v[sl] if torch.is_tensor(v) else v) for k, v in dom.items()}, mode[sl], g[sl]))
    return res


def truth():
    prm = C.params()
    E = C.episodes("P1P5")
    t0 = time.time()
    # singles
    keys, rhos = [], []
    for mid in C.MIDS:
        sel = [(k, m) for k, e in enumerate(E) for m in range(e["nmod"]) if e["modules"][m][0] == mid]
        th = [E[k]["theta"] for k, _ in sel]
        mode = torch.tensor(np.stack([E[k]["mode"][4 * m:4 * m + 4] for k, m in sel]))
        g = torch.tensor(np.stack([E[k]["g"][4 * m:4 * m + 4] for k, m in sel]))
        parts = split_rows(C.single_asm(mid, prm), dom_from(th), mode, g, 8192)
        r = run_multi([p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts], [p[3] for p in parts],
                      seed=1000 + C.MIDS.index(mid) * 100)
        keys += sel
        rhos.append(np.concatenate(r))
        print(f"[{time.time() - t0:.0f}s] singles {mid}: {len(sel)}", flush=True)
    keys = np.array(keys)
    np.savez(C.OUT / "truth_singles.npz", ep=keys[:, 0], mod=keys[:, 1], rho=np.concatenate(rhos))
    # remainders: one asm per devW group
    by_gi = {}
    for k, e in enumerate(E):
        by_gi.setdefault(e["gi"], []).append(k)
    jobs = {}
    for gi, ks in by_gi.items():
        asm, kept = C.remainder_asm(E[ks[0]], prm)
        w = C.wheels_of(kept)
        mode = torch.tensor(np.stack([E[k]["mode"][w] for k in ks]))
        g = torch.tensor(np.stack([E[k]["g"][w] for k in ks]))
        jobs.setdefault(asm.n, []).append((ks, asm, dom_from([E[k]["theta"] for k in ks]), mode, g))
    ep, rho = [], []
    for n, js in sorted(jobs.items()):
        r = run_multi([j[1] for j in js], [j[2] for j in js], [j[3] for j in js], [j[4] for j in js], seed=5000 + n)
        for j, rr in zip(js, r):
            ep += j[0]
            rho.append(rr)
        print(f"[{time.time() - t0:.0f}s] rem n={n}: {sum(len(j[0]) for j in js)}", flush=True)
    np.savez(C.OUT / "truth_rem.npz", ep=np.array(ep), rho=np.concatenate(rho))


def configs4(K=2):
    """all single-module health configs with <= K faults; WEAK at both g corners. Returns modes, gs, mode-vector key."""
    modes, gs, keys = [], [], []
    for k in range(K + 1):
        for wh in itertools.combinations(range(4), k):
            for ms in itertools.product(range(1, 5), repeat=k):
                weak = [j for j, m in zip(wh, ms) if m == 1]
                for gc in itertools.product(G_RANGE, repeat=len(weak)):
                    mo = np.zeros(4, np.int64)
                    gg = np.ones(4, np.float32)
                    for j, m in zip(wh, ms):
                        mo[j] = m
                    for j, gv in zip(weak, gc):
                        gg[j] = gv
                    modes.append(mo)
                    gs.append(gg)
                    keys.append(int(sum(int(m) * 5 ** j for j, m in enumerate(mo))))
    return np.stack(modes), np.stack(gs), np.array(keys)


def library(M=500, seed=7):
    """Z[mid] (n_cfg, M, 14): rho of every config at M plant draws theta ~ DR_WIDE (same draws for every config)."""
    prm = C.params()
    modes, gs, keys = configs4()
    rng = np.random.default_rng(seed)
    out = dict(modes=modes, gs=gs, keys=keys)
    t0 = time.time()
    for mid in C.MIDS:
        dom = sim.sample_domain(M, rng, sim.DR_WIDE)
        nc = len(modes)
        idx = np.tile(np.arange(M), nc)
        D = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in dom.items()}
        mo = torch.tensor(np.repeat(modes, M, 0))
        gg = torch.tensor(np.repeat(gs, M, 0))
        parts = split_rows(C.single_asm(mid, prm), D, mo, gg, 8192)
        r = run_multi([p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts], [p[3] for p in parts],
                      seed=int(rng.integers(1 << 30)))
        out[f"Z_{mid}"] = np.concatenate(r).reshape(nc, M, 14).astype(np.float32)
        out[f"theta_{mid}"] = np.stack([dom[k].numpy() for k in C.PLANT_KEYS], 1)
        print(f"[{time.time() - t0:.0f}s] library {mid}: {nc} cfg x {M}", flush=True)
    np.savez(C.OUT / "library.npz", **out)


if __name__ == "__main__":
    setup()
    {"truth": truth, "library": library}[sys.argv[1]]()
