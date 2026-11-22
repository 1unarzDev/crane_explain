"""'off1' split, REMAINDER sub-assembly (>= 2 modules): COMP certificate with per-episode stage-2 simulation.
(SIMULATION ONLY.) Stage-1 sparse sets (structured_ensemble, calA wheels) projected onto the remainder wheels, I1
stage2.configs (<= K - forced faults outside, WEAK at g corners), M fresh DR_WIDE draws per episode; one-sided
order-statistic lower bound on z per mode vector, union (min) over mode vectors. Also the oracle-health band.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.split.rem_comp [n_eps] [pids]
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from capxfer import sim
from capxfer.ideas.i1 import stage1 as S1, stage2 as I1S2
from capxfer.ideas.split import common as C, evaluate as EV, sims as SM


MAX_CFG = 200


def main(n_eps=300, p="P1P5", M=79, seed=4):
    SM.setup()
    prm = C.params()
    E = C.episodes(p)
    Gs = C.split_groups(E)
    D = S1.load("structured_ensemble", p)
    Tw = S1.calibrate(D, np.array([E[k]["b"] for k in Gs["calA"]]), EV.ALPHA1, K=EV.K, seed=0)
    test = np.concatenate([Gs[g] for g in ("calB", "heldout", "L3", "square4")])
    test = test[np.array([E[k]["nmod"] >= 3 for k in test])]
    rng = np.random.default_rng(seed)
    # oversample episodes with a fault on the remainder
    keep = lambda e: C.wheels_of([m for m in range(e["nmod"]) if m != e["off1"]])
    remf = np.array([(E[k]["mode"][keep(E[k])] > 0).any() for k in test])
    pick = np.concatenate([rng.choice(test[remf], n_eps * 2 // 3, replace=False),
                           rng.choice(test[~remf], n_eps - n_eps * 2 // 3, replace=False)])
    jobs = []
    for k in pick:
        e = E[k]
        asm, kept = C.remainder_asm(e, prm)
        w = C.wheels_of(kept)
        Sw = S1.sets_for(D, e["b"], Tw, "sparse")
        outside = np.setdiff1d(np.arange(e["n"]), w)
        kk = EV.K - int((~Sw[outside, 0]).sum())
        cf = I1S2.configs(Sw[w], kk) if kk >= 0 else []
        if len(cf) > MAX_CFG:                     # composition cost cap (I1 K2 criterion): abstain
            cf = []
        orc = (e["mode"][w].astype(np.int64), e["g"][w].astype(np.float32))
        jobs.append((k, asm, cf + [orc], sim.sample_domain(M, rng, sim.DR_WIDE)))
    t0 = time.time()
    Z = {}
    by_n = {}
    for ji, j in enumerate(jobs):
        by_n.setdefault(j[1].n, []).append(ji)
    for n, jis in sorted(by_n.items()):
        budget = 40000 // n
        chunk, cnt = [], 0
        for ji in jis + [None]:
            if ji is not None:
                chunk.append(ji)
                cnt += len(jobs[ji][2]) * M
            if chunk and (ji is None or cnt >= budget):
                asms, doms, modes, gs = [], [], [], []
                for c in chunk:
                    _, asm, cf, dom = jobs[c]
                    idx = np.tile(np.arange(M), len(cf))
                    asms.append(asm)
                    doms.append({kk: (v[idx] if torch.is_tensor(v) else v) for kk, v in dom.items()})
                    modes.append(torch.tensor(np.repeat(np.stack([m for m, _ in cf]), M, 0)))
                    gs.append(torch.tensor(np.repeat(np.stack([g for _, g in cf]), M, 0)))
                Rs = sim.simulate_multi(asms, prm, doms, modes, gs, device="cuda", engine="graph", record_hz=50,
                                        store_device="cpu", seed=int(rng.integers(1 << 30)))
                for c, R in zip(chunk, Rs):
                    Z[c] = sim.targets_from_twist(R)[0].cpu().numpy().reshape(len(jobs[c][2]), M, 14)
                del Rs
                print(f"[{time.time() - t0:.0f}s] n={n} {len(Z)}/{len(jobs)}", flush=True)
                chunk, cnt = [], 0
    k1 = math.floor((M + 1) * EV.ALPHA2)
    T = np.load(C.OUT / "truth_rem.npz")
    zt_all = dict(zip(T["ep"].tolist(), EV.zmin(T["rho"]).tolist()))
    rows = []
    for ji, (k, asm, cf, _) in enumerate(jobs):
        Zj = Z[ji]
        nc = len(cf) - 1
        z = EV.zmin(Zj)                                                   # (cfg+1, M)

        def lb_of(ix):
            return float(np.sort(z[ix].min(0))[k1 - 1])
        if nc == 0:
            lb = -np.inf
        else:
            keys = [cf[i][0].tobytes() for i in range(nc)]
            lb = min(lb_of([i for i in range(nc) if keys[i] == u]) for u in dict.fromkeys(keys))
        rows.append(dict(ep=int(k), nmod_rem=E[k]["nmod"] - 1, ncfg=nc, lb_comp=lb, lb_orc=lb_of([nc]),
                         z=zt_all[int(k)], rem_faulted=bool(remf[np.flatnonzero(test == k)[0]])))
    out = dict(pids=p, M=M, alpha2=EV.ALPHA2, rows=rows)
    import json
    (C.OUT / f"rem_comp_{p}.json").write_text(json.dumps(out, indent=1, default=float))
    for tau in EV.TAUS:
        z = np.array([r["z"] for r in rows])
        for m in ("lb_comp", "lb_orc"):
            lb = np.array([r[m] for r in rows])
            go = lb >= tau
            print(f"tau={tau} {m}: n={len(rows)} unsafe={int((z < tau).sum())} cert={go.mean():.3f} "
                  f"unsafe_go={(go & (z < tau)).mean():.4f} lb_cov={(z >= lb).mean():.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 300, a[1] if len(a) > 1 else "P1P5")
