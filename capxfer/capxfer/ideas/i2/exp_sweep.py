"""I2 sweep: every single-wheel fault on hundreds of random morphologies, paired with a healthy twin (centred domain),
plus healthy DR trials per assembly for the nuisance spread. Simulator ground truth + QS predictions.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i2.exp_sweep --n-asm 300 --out ~/.cache/capxfer/I2/sweep_v1.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from capxfer import assembly as A, sim
from capxfer.ideas.i2 import simrun as S, theory as T

G_WEAK = 0.5
N_DR = 16


def configs(n):
    modes = [np.zeros(n, np.int64)]
    gs = [np.ones(n, np.float32)]
    tags = [(-1, 0)]
    for i in range(n):
        for m in (sim.WEAK, sim.COAST, sim.BRAKE, sim.ROLLER):
            mo = np.zeros(n, np.int64)
            g = np.ones(n, np.float32)
            mo[i] = m
            if m == sim.WEAK:
                g[i] = G_WEAK
            modes.append(mo)
            gs.append(g)
            tags.append((i, m))
    return np.stack(modes), np.stack(gs), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-asm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/capxfer/I2/sweep_v1.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-per-n", type=int, default=10**6)
    ap.add_argument("--budget", type=int, default=6000, help="max B*n per simulate_multi call")
    a = ap.parse_args()
    P = A.load_params("params/module_measured.yaml")
    rng = np.random.default_rng(a.seed)
    asms = [A.physical(k, P) for k in ("single", "pair", "L3", "square4")]
    for k in asms:
        k.meta["family"] = "phys_" + k.name
    sizes = [1, 2, 3, 4, 5, 6]
    for j in range(a.n_asm):
        asms.append(A.random_assembly(rng, P, sizes=(sizes[j % 6],)))
    dev = S.gpu()
    results = [None] * len(asms)
    if a.resume and os.path.exists(a.resume):
        old = torch.load(a.resume, weights_only=False)["results"]
        for i, r in enumerate(old[:len(asms)]):
            if r is not None and r["name"] == asms[i].name and np.allclose(r["wheel_pos"], asms[i].wheel_pos):
                results[i] = r
        print("resumed", sum(r is not None for r in results), flush=True)
    by_n = {}
    for idx, asm in enumerate(asms):
        if results[idx] is None and len(by_n.get(asm.n, [])) < a.max_per_n:
            by_n.setdefault(asm.n, []).append(idx)
    t0 = time.time()
    for n, idxs in sorted(by_n.items()):
        modes, gs, tags = configs(n)
        per = (len(modes) + N_DR) * n
        step = max(1, a.budget // per)
        for c in range(0, len(idxs), step):
            grp = idxs[c:c + step]
            L, D, M, Gs = [], [], [], []
            for idx in grp:
                B = len(modes) + N_DR
                dom = S.centred_domain(B)
                dr = sim.sample_domain(N_DR, np.random.default_rng(1000 + idx))
                for k in dom:
                    if torch.is_tensor(dom[k]) and k in dr:
                        dom[k][len(modes):] = dr[k]
                L.append(asms[idx])
                D.append(dom)
                M.append(torch.as_tensor(np.concatenate([modes, np.zeros((N_DR, n), np.int64)])))
                Gs.append(torch.as_tensor(np.concatenate([gs, np.ones((N_DR, n), np.float32)])))
            Rs = sim.simulate_multi(L, P, D, M, Gs, device=dev, seed=idx, store_device="cpu")
            for idx, R in zip(grp, Rs):
                w = S.windows_reduce(R)
                asm = asms[idx]
                xi, om, du = T.qs_solve(asm, P, modes, gs, T.legs(asm))
                results[idx] = dict(
                    name=asm.name, family=asm.meta.get("family"), n=n, modules=asm.modules,
                    wheel_pos=asm.wheel_pos, w_hat=asm.w_hat, r_hat=asm.r_hat, tags=tags, nc=len(modes),
                    sim={k: v.astype(np.float32) for k, v in w.items()},
                    qs=dict(rho=T.rho_of(xi, T.legs(asm), asm.char_length()).astype(np.float32),
                            om=om.astype(np.float32), duty=du.astype(np.float32), twist=xi.astype(np.float32),
                            resid=T.qs_solve.last_resid.astype(np.float32)),
                    E=T.excitation(asm))
            del Rs
            print(f"n={n} {c + len(grp)}/{len(idxs)} t={time.time() - t0:.0f}s", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
        torch.save(dict(results=results, g_weak=G_WEAK, n_dr=N_DR, partial=True), a.out + ".partial")
    torch.save(dict(results=results, g_weak=G_WEAK, n_dr=N_DR, params_sha=None), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
