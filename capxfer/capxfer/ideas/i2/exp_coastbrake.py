"""I2 Prop. 4: coast and brake are observationally and consequentially equivalent iff the back-drive torque exceeds
the largest traction torque the contact can apply, tau_bd > c1 mu N_i (c1 = r w.r). Sweep tau_bd and mu (CPU)."""
import json
import os

import numpy as np
import torch

from capxfer import assembly as A, sim
from capxfer.ideas.i2 import simrun as S, theory as T


def main():
    P = A.load_params("params/module_measured.yaml")
    nom = sim.nominal(P)
    c1 = nom["radius"] * abs(float(np.dot([1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)])))
    taus = np.array([0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2])
    out = []
    for name in ("single", "pair"):
        asm = A.physical(name, P)
        N = T.normal_loads(asm)
        for mu in (0.6, 1.0):
            B = 2 * len(taus)
            modes = np.zeros((B, asm.n), np.int64)
            modes[: len(taus), 0] = sim.COAST
            modes[len(taus):, 0] = sim.BRAKE
            gs = np.ones((B, asm.n), np.float32)
            tb = np.concatenate([taus, taus])
            dom = S.centred_domain(B, dict(mu=np.full(B, mu), tau_bd=tb))
            R = sim.simulate(asm, P, dom, torch.as_tensor(modes), torch.as_tensor(gs), device="cpu", engine="eager")
            w = S.windows_reduce(R)
            qx, qom, _ = T.qs_solve(asm, P, modes, gs, T.legs(asm), dom=dict(mu=mu), N=N) if False else (None, None, None)
            rows_q = []
            for k, t in enumerate(tb):
                xi, om, _ = T.qs_solve(asm, P, modes[k:k + 1], gs[k:k + 1], T.legs(asm), dom=dict(mu=mu, tau_bd=t))
                rows_q.append((T.rho_of(xi, T.legs(asm), asm.char_length())[0], np.abs(om[0, :, 0]).max()))
            L = len(taus)
            for k, t in enumerate(taus):
                out.append(dict(asm=name, mu=mu, tau_bd=float(t), tau_star=float(c1 * mu * N[0]),
                                drho_sim=float(np.abs(w["rho"][k] - w["rho"][L + k]).max()),
                                om_coast_sim=float(np.abs(w["om"][k, :, 0]).max()),
                                om_brake_sim=float(np.abs(w["om"][L + k, :, 0]).max()),
                                drho_qs=float(np.abs(rows_q[k][0] - rows_q[L + k][0]).max()),
                                om_coast_qs=float(rows_q[k][1])))
                print(out[-1], flush=True)
    json.dump(out, open(os.path.expanduser("~/.cache/capxfer/I2/coastbrake.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
