"""I2 Prop. 5 test: one oblique translation probe (heading ~25.7 deg, outside the library) excites every drive and
roller channel of any 90-deg X-pattern grid, so it should match the observability of P1+P5+P2 in 1/3 of the time.
Runs every single fault + healthy twin + 16 DR-healthy trials under the custom schedule on the sweep's assemblies.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from capxfer import assembly as A, probes, sim
from capxfer.ideas.i2 import simrun as S, theory as T
from capxfer.ideas.i2.exp_sweep import configs, N_DR

XI_OBL = np.array([0.27, 0.13, 0.0])


class OneProbe:
    """Schedule-compatible object with a single probe (both legs), same ramp/hold/rest as the library."""

    def __init__(self, xi, pid="PO", hold=1.0):
        self.segments = []
        t = probes.SETTLE
        for leg in (+1, -1):
            seg = probes.Segment(pid, leg, t, t + probes.RAMP, t + probes.RAMP + hold,
                                 t + probes.RAMP + hold + probes.RAMP + probes.REST, leg * np.asarray(xi, float))
            self.segments.append(seg)
            t = seg.t1
        self.T = t + 0.2

    twist = probes.Schedule.twist
    windows = probes.Schedule.windows


def main(sweep=os.path.expanduser("~/.cache/capxfer/I2/sweep_v1.pt"), every=3):
    P = A.load_params("params/module_measured.yaml")
    d = torch.load(sweep, weights_only=False)
    R = [r for r in d["results"] if r is not None][::int(every)]
    sch = OneProbe(XI_OBL)
    out = []
    for r in R:
        asm = A.build(r["name"], [tuple(m) for m in r["modules"]], P)
        assert np.allclose(asm.wheel_pos, r["wheel_pos"])
        n = asm.n
        modes, gs, tags = configs(n)
        B = len(modes) + N_DR
        dom = S.centred_domain(B)
        idx = len(out)
        dr = sim.sample_domain(N_DR, np.random.default_rng(1000 + idx))
        for k in dom:
            if torch.is_tensor(dom[k]) and k in dr:
                dom[k][len(modes):] = dr[k]
        M = np.concatenate([modes, np.zeros((N_DR, n), np.int64)])
        G = np.concatenate([gs, np.ones((N_DR, n), np.float32)])
        chunk = max(1, 6000 // n)
        ws = []
        for s in range(0, B, chunk):
            dd = {k: (v[s:s + chunk] if torch.is_tensor(v) else v) for k, v in dom.items()}
            Rs = sim.simulate(asm, P, dd, torch.as_tensor(M[s:s + chunk]), torch.as_tensor(G[s:s + chunk]),
                              schedule=sch, device=S.gpu(), store_device="cpu")
            t = Rs["t"]
            w = dict(om=[], duty=[], twist=[])
            for pid, leg, ta, tb, xc in sch.windows():
                m = (t >= ta) & (t <= tb)
                w["om"].append(Rs["omega"][:, m].mean(1))
                w["duty"].append(Rs["duty"][:, m].mean(1))
                w["twist"].append(Rs["twist"][:, m].mean(1))
            ws.append({k: torch.stack(v, 1).numpy() for k, v in w.items()})
            del Rs
        w = {k: np.concatenate([x[k] for x in ws]) for k in ws[0]}
        g, h = T.rows(asm)
        e_obl = np.stack([np.abs(g @ XI_OBL), np.abs(h @ XI_OBL)], 1)
        out.append(dict(name=r["name"], n=n, nc=len(modes), tags=tags, sim=w, e_obl=e_obl))
        print(len(out), r["name"], flush=True)
    torch.save(out, os.path.expanduser("~/.cache/capxfer/I2/oblique_v1.pt"))


if __name__ == "__main__":
    main(*sys.argv[1:])
