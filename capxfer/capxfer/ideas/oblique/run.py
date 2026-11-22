"""WS-OBL: full-simulator test of THEORY Prop. 4(b) (one oblique translation at 22.5 deg covers every channel).

One custom schedule per trial: P1, P2, P5 (library twists, P5 yaw-scaled as in probes.Schedule), O22 (0.3 m/s at
22.5 deg) and O26 (I2's (0.27, 0.13), 25.7 deg), each probe = two legs with the library ramp/hold/rest timing.
Per assembly: healthy twin + every single-wheel fault (weak g=0.5, coast, brake, roller-lock) at the centred domain
(exact pairing) + 16 healthy DR trials for the nuisance spread (same recipe as i2/exp_sweep). Also QS predictions.
Assemblies: 4 physical types + random 1-6-module assemblies with random module yaws (seed 0, sizes cycled; the
first 100 random ones coincide with I2's sweep design).

  nice -n 15 env OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m capxfer.ideas.oblique.run --n-asm 206
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from capxfer import assembly as A, probes, sim
from capxfer.ideas.i2 import simrun as S, theory as T
from capxfer.ideas.i2.exp_sweep import configs, N_DR

V = probes.V_NOM
OBL = {"O22": (V * math.cos(math.radians(22.5)), V * math.sin(math.radians(22.5)), 0.0),
       "O26": (0.27, 0.13, 0.0)}
OUT = os.path.expanduser("~/.cache/capxfer/OBL")


class Custom:
    """probes.Schedule-compatible: library P1, P2, P5 then the oblique probes, library leg timing."""

    def __init__(self, asm):
        lib = {s.pid: s.twist for s in probes.Schedule(asm.char_length()).segments if s.leg > 0}
        self.probes = [("P1", lib["P1"]), ("P2", lib["P2"]), ("P5", lib["P5"])] + \
                      [(k, np.array(v)) for k, v in OBL.items()]
        self.segments = []
        t = probes.SETTLE
        R_, H = probes.RAMP, 1.0
        for pid, tw in self.probes:
            for leg in (+1, -1):
                seg = probes.Segment(pid, leg, t, t + R_, t + R_ + H, t + R_ + H + R_ + probes.REST,
                                     leg * np.asarray(tw, float))
                self.segments.append(seg)
                t = seg.t1
        self.T = t + 0.2

    twist = probes.Schedule.twist
    windows = probes.Schedule.windows

    def pids(self):
        return [p for p, _ in self.probes]

    def legs(self):
        return np.array([s.twist for s in self.segments])


def run_one(asm, P, idx, dev, with_qs=False):
    n = asm.n
    modes, gs, tags = configs(n)
    nc = len(modes)
    B = nc + N_DR
    dom = S.centred_domain(B)
    dr = sim.sample_domain(N_DR, np.random.default_rng(1000 + idx))
    for k in dom:
        if torch.is_tensor(dom[k]) and k in dr:
            dom[k][nc:] = dr[k]
    M = np.concatenate([modes, np.zeros((N_DR, n), np.int64)])
    G = np.concatenate([gs, np.ones((N_DR, n), np.float32)])
    sch = Custom(asm)
    R = sim.simulate(asm, P, dom, torch.as_tensor(M), torch.as_tensor(G), schedule=sch, device=dev,
                     store_device="cpu", seed=idx)
    rho, drift = sim.targets_from_twist(R, schedule=sch)
    t = R["t"]
    om, du, tw = [], [], []
    for pid, leg, ta, tb, xc in sch.windows():
        m = (t >= ta) & (t <= tb)
        om.append(R["omega"][:, m].float().mean(1))
        du.append(R["duty"][:, m].float().mean(1))
        tw.append(R["twist"][:, m].float().mean(1))
    w = dict(rho=rho.numpy(), om=torch.stack(om, 1).numpy(), duty=torch.stack(du, 1).numpy(),
             twist=torch.stack(tw, 1).numpy())
    del R
    qs = None
    if with_qs:                                                             # slow on CPU (~45 s at n=24)
        legs = sch.legs()
        xi, qom, qdu = T.qs_solve(asm, P, modes, gs, legs)
        qs = dict(rho=T.rho_of(xi, legs, asm.char_length()).astype(np.float32), om=qom.astype(np.float32),
                  duty=qdu.astype(np.float32), twist=xi.astype(np.float32),
                  resid=T.qs_solve.last_resid.astype(np.float32))
    g, h = T.rows(asm)
    E = np.zeros((n, 5, len(sch.probes)))                                   # |speed demand| per channel, probe
    for k, (pid, xc) in enumerate(sch.probes):
        a, b = np.abs(g @ xc), np.abs(h @ xc)
        E[:, sim.WEAK, k] = E[:, sim.COAST, k] = E[:, sim.BRAKE, k] = a
        E[:, sim.ROLLER, k] = b
    return dict(name=asm.name, family=asm.meta.get("family"), n=n, modules=asm.modules, wheel_pos=asm.wheel_pos,
                w_hat=asm.w_hat, r_hat=asm.r_hat, tags=tags, nc=nc, pids=sch.pids(), probe_twists=dict(sch.probes),
                sim={k: v.astype(np.float32) for k, v in w.items()}, qs=qs, E=E,
                p5_wz=float(sch.probes[2][1][2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-asm", type=int, default=206)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "obl_v1.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    P = A.load_params("params/module_measured.yaml")
    rng = np.random.default_rng(a.seed)
    asms = [A.physical(k, P) for k in ("single", "pair", "L3", "square4")]
    for k in asms:
        k.meta["family"] = "phys_" + k.name
    sizes = [1, 2, 3, 4, 5, 6]
    for j in range(a.n_asm):
        asms.append(A.random_assembly(rng, P, sizes=(sizes[j % 6],)))
    asms = asms[:a.limit] if a.limit else asms
    dev = a.device or "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(1.0 / 16.0)              # CONTRACTS rule 4: <= 1 GB
    res = []
    if os.path.exists(a.out + ".partial"):
        res = torch.load(a.out + ".partial", weights_only=False)["results"]
        print("resumed", len(res), flush=True)
    t0 = time.time()
    for idx in range(len(res), len(asms)):
        res.append(run_one(asms[idx], P, idx, dev))
        print(f"{idx + 1}/{len(asms)} n={asms[idx].n} t={time.time() - t0:.0f}s", flush=True)
        if (idx + 1) % 10 == 0:
            torch.save(dict(results=res), a.out + ".partial")
            if dev == "cuda":
                torch.cuda.empty_cache()
    torch.save(dict(results=res, g_weak=0.5, n_dr=N_DR, obl=OBL), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
