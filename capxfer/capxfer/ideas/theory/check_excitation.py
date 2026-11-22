"""Lemma 1 (excitation) check.

(a) Idealised QS (c_roller = 0): unexcited (fault, leg) pairs must leave twist, wheel speeds and duties unchanged to solver
    precision; excited ones must change at least one robot-visible signal (body twist or a wheel speed/duty).
(b) Nominal QS (c_roller = 2.25): size of the O(c_r) violation for unexcited pairs.
(c) Full simulator (sweeps n = 4..24): |Delta rho| and nuisance-normalised signal z for unexcited pairs; share of
    consequential legs (|Delta rho| > 0.05) that are excited.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.theory.check_excitation
"""
from __future__ import annotations

import numpy as np
import torch

from capxfer.ideas.i2 import theory as T, analyze as An
from capxfer.ideas.i2.exp_sweep import configs
from capxfer.ideas.theory.common import rebuild, params, load_sweeps, dump, MODE_NAME

torch.set_num_threads(4)


def qs_part(R, P, dom, per_n):
    un, ex, ex_rho = [], [], []
    seen = {}
    for r in R:
        if r["n"] > 16 or seen.get(r["n"], 0) >= per_n:
            continue
        seen[r["n"]] = seen.get(r["n"], 0) + 1
        asm = rebuild(r, P)
        modes, gs, tags = configs(asm.n)
        L = T.legs(asm)
        xi, om, du = T.qs_solve(asm, P, modes, gs, L, dom=dom)
        rho = T.rho_of(xi, L, asm.char_length())
        E = r["E"]
        for c, (i, m) in enumerate(tags):
            if i < 0:
                continue
            e = E[i, m][An.LEG_PID]
            sig = np.maximum.reduce([np.abs(xi[c] - xi[0]).max(-1), np.abs(om[c] - om[0]).max(-1),
                                     np.abs(du[c] - du[0]).max(-1)])                     # (14,)
            drho = np.abs(rho[c] - rho[0])
            u = e < 1e-9
            un.append(np.stack([sig[u], drho[u]], 1))
            ex.append(np.stack([sig[~u], drho[~u], e[~u]], 1))
        print(dom, asm.name, asm.n, flush=True)
    un, ex = np.concatenate(un), np.concatenate(ex)
    return dict(n_unexcited=len(un), n_excited=len(ex),
                unexcited_max_signal_change=float(un[:, 0].max()), unexcited_max_drho=float(un[:, 1].max()),
                excited_min_signal_change=float(ex[:, 0].min()),
                excited_signal_change_p01=float(np.quantile(ex[:, 0], 0.01)),
                excited_frac_signal_gt_1em6=float((ex[:, 0] > 1e-6).mean()))


def sim_part(R):
    un_d, un_z, ex_d, by_n = [], [], [], {}
    for r in R:
        z = An.signals_z(r)
        rho = r["sim"]["rho"]
        for c, i, m, e in An.fault_rows(r):
            el = e[An.LEG_PID]
            dr = np.abs(rho[c] - rho[0])
            u = el < 1e-9
            un_d.append(dr[u]); un_z.append(z[c][u]); ex_d.append(dr[~u])
            b = by_n.setdefault(r["n"], [0, 0])
            b[0] += int(u.sum()); b[1] += int((~u).sum())
    un_d, un_z, ex_d = map(np.concatenate, (un_d, un_z, ex_d))
    cons = np.concatenate([un_d, ex_d]) > 0.05
    return dict(n_unexcited=len(un_d), n_excited=len(ex_d), unexcited_drho_max=float(un_d.max()),
                unexcited_drho_p99=float(np.quantile(un_d, .99)), unexcited_frac_z_lt_3=float((un_z < 3).mean()),
                unexcited_z_max=float(un_z.max()),
                consequential_legs=int(cons.sum()), consequential_excited_frac=float((ex_d > 0.05).sum() / cons.sum()),
                by_n={int(k): dict(unexcited=v[0], excited=v[1]) for k, v in sorted(by_n.items())})


def main():
    P = params()
    R = load_sweeps()
    res = dict(qs_idealised=qs_part(R, P, dict(c_roller=0.0), per_n=4),
               qs_nominal=qs_part(R, P, None, per_n=4),
               sim=sim_part(R))
    dump("check_excitation.json", res)
    print(res)


if __name__ == "__main__":
    main()
