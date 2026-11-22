"""Check the corner (coordinatewise-monotonicity) assumption for WEAK gains: with common random numbers, is
rho(theta, g) for g in (0.3, 0.9) inside [min, max] of the corner values g = 0.3, 0.9 (per domain, per target)?"""
import pickle
import pathlib

import numpy as np
import torch

from capxfer import assembly as A, datagen, sim

torch.cuda.set_per_process_memory_fraction(0.12)
G = pickle.load(open(pathlib.Path.home() / ".cache/capxfer/I1/devW_meta.pkl", "rb"))["_groups"]
params = A.load_params(datagen.PARAMS)
rng = np.random.default_rng(3)
GRID = [0.3, 0.45, 0.6, 0.75, 0.9]
viol, tot, gaps = 0, 0, []
seen = set()
for gidx in rng.permutation(len(G)):
    n = G[gidx]["assembly"]["wheel_pos"].shape[0]
    if n in seen:
        continue
    seen.add(n)
    asm = datagen.asm_from_dict(G[gidx]["assembly"])
    B = 64
    dom = sim.sample_domain(B, rng, sim.DR_WIDE)
    base_mode, base_g = sim.sample_faults(B, n, rng, p_healthy=0.6, p_double=0.0)   # maybe one other fault
    w = rng.integers(0, n, size=B)
    rhos = []
    for gv in GRID:
        mode, g = base_mode.clone(), base_g.clone()
        mode[torch.arange(B), torch.tensor(w)] = sim.WEAK
        g[torch.arange(B), torch.tensor(w)] = gv
        R = sim.simulate(asm, params, dom, mode, g, device="cuda", engine="graph", record_hz=50,
                         store_device="cpu", seed=11)
        rhos.append(sim.targets_from_twist(R)[0].cpu().numpy())
    Rh = np.stack(rhos)                                   # (5,B,14)
    lo, hi = Rh[[0, -1]].min(0), Rh[[0, -1]].max(0)
    out = np.maximum(lo - Rh[1:-1], Rh[1:-1] - hi).max(0)  # (B,14) worst interior excursion
    gaps.append(out.ravel())
    viol += int((out > 0.005).sum())
    tot += out.size
    print(f"n={n}: excursion q99={np.quantile(out, .99):.4f} max={out.max():.4f}, "
          f"P(>0.005)={np.mean(out > 0.005):.4f}, span med={np.median(hi - lo):.4f}", flush=True)
    if len(seen) >= 6:
        break
gaps = np.concatenate(gaps)
print(f"ALL: P(excursion>0.005)={viol / tot:.4f}  max={gaps.max():.4f}")
