"""Prop. 4 (reveal on split) check.

For every single-wheel fault in the I2 sweeps' multi-module assemblies (n = 8..24, simulator + QS stored), compare
  parent:  Delta z_parent = z(healthy) - z(fault), z = min(rho_P2+, rho_P2-) in the parent frame (simulator), and the
           worst-leg loss over all 14 legs;
  child:   z of the module that carries the fault, driven alone in its own frame (the SPLIT 'singles' convention),
           from the simulator (CPU, centred domain, same as the sweep) and the QS model.
Reports how often the parent is masked (Delta z_parent <= 0.02, worst-leg <= 0.05) while the child cannot strafe
(z_child < 0.5 / 0.8), and the exact-invisibility subcase (parent strafe leg unexcited: roller locks on odd-yaw
modules), where Lemma 1 gives Delta z_parent = 0 in the idealised model.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.theory.check_split
"""
from __future__ import annotations

import numpy as np
import torch

from capxfer import assembly as A, probes, sim
from capxfer.ideas.i2 import theory as T, simrun as S
from capxfer.ideas.i2.exp_sweep import configs
from capxfer.ideas.theory.common import ROLLER, WEAK, COAST, BRAKE, MODE_NAME, params, load_sweeps, dump

torch.set_num_threads(4)
STRAFE = [probes.TARGET_KEYS.index("P2+"), probes.TARGET_KEYS.index("P2-")]


def child_tables(P):
    """z of a lone module (own frame) for every single fault: {mid: (tags, z_sim (17,), z_qs (17,))}."""
    out = {}
    for mid in ("M1", "M2", "M3", "M4"):
        asm = A.build(f"single_{mid}", [(mid, 0, 0, 0)], P)
        modes, gs, tags = configs(asm.n)
        dom = S.centred_domain(len(modes))
        Rr = sim.simulate(asm, P, dom, torch.as_tensor(modes), torch.as_tensor(gs), device="cpu")
        rho_s = sim.targets_from_twist(Rr)[0].numpy()
        xi, _, _ = T.qs_solve(asm, P, modes, gs, T.legs(asm))
        rho_q = T.rho_of(xi, T.legs(asm), asm.char_length())
        out[mid] = (tags, rho_s[:, STRAFE].min(1), rho_q[:, STRAFE].min(1))
        print(mid, "child z_sim", np.round(rho_s[:, STRAFE].min(1), 3), flush=True)
    return out


def main():
    P = params()
    child = child_tables(P)
    R = [r for r in load_sweeps() if len(r["modules"]) >= 2]
    rows = []
    for r in R:
        rs = r["sim"]["rho"]
        z0 = rs[0, STRAFE].min()
        E = r["E"]
        for c, (i, m) in enumerate(r["tags"]):
            if i < 0:
                continue
            k = i // 4
            mid, _, _, yk = r["modules"][k]
            ctags, zs, zq = child[mid]
            cc = ctags.index((i % 4, m))
            e_par = E[i, m][probes.PIDS.index("P2")]
            rows.append(dict(n=r["n"], mode=int(m), yaw_odd=int(yk) % 2, dz_par=float(z0 - rs[c, STRAFE].min()),
                             worst_par=float((rs[0] - rs[c]).max()), e_par=float(e_par),
                             z_child_sim=float(zs[cc]), z_child_qs=float(zq[cc]), z_child_healthy=float(zs[0])))
    arr = lambda k, G: np.array([x[k] for x in G])
    res = dict(n_faults=len(rows), child_healthy_z={m: float(v[1][0]) for m, v in child.items()})
    for gname, ms in dict(all=(WEAK, COAST, BRAKE, ROLLER), passive=(COAST, BRAKE), roller=(ROLLER,), weak=(WEAK,)).items():
        G = [x for x in rows if x["mode"] in ms]
        masked = (arr("dz_par", G) <= 0.02) & (arr("worst_par", G) <= 0.05)
        d = dict(count=len(G), parent_masked_frac=float(masked.mean()),
                 parent_dz_max=float(arr("dz_par", G).max()), parent_worst_med=float(np.median(arr("worst_par", G))))
        for tau in (0.5, 0.8):
            bad = arr("z_child_sim", G) < tau
            d[f"child_unsafe_{tau}"] = float(bad.mean())
            d[f"masked_and_child_unsafe_{tau}"] = float((masked & bad).mean())
            d[f"child_unsafe_given_parent_masked_{tau}"] = float(bad[masked].mean()) if masked.any() else None
            d[f"qs_sim_child_decision_agree_{tau}"] = float(((arr("z_child_qs", G) < tau) == bad).mean())
        un = arr("e_par", G) < 1e-9
        d["parent_strafe_unexcited"] = int(un.sum())
        if un.any():
            d["unexcited_parent_dz_absmax"] = float(np.abs(arr("dz_par", G)[un]).max())
            d["unexcited_child_z_sim_max"] = float(arr("z_child_sim", G)[un].max())
            d["unexcited_child_unsafe_0.5"] = float((arr("z_child_sim", G)[un] < 0.5).mean())
        # rank correlation parent loss vs child loss (information the parent carries about the child)
        a, b = arr("worst_par", G), arr("z_child_healthy", G) - arr("z_child_sim", G)
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        d["spearman_parent_worst_vs_child_loss"] = float(np.corrcoef(ra, rb)[0, 1])
        res[gname] = d
    dump("check_split.json", res)
    for k, v in res.items():
        print(k, v)




def double_roller(n_parents=40):
    """Exact-invisibility case: two diagonal roller locks (F4x2d) on an odd-yaw module of a parent. Lemma 1 =>
    parent strafe unchanged (idealised); the lone module (own frame) loses strafe. Parent: QS; child: sim + QS."""
    from capxfer.ideas.theory.common import rebuild
    P = params()
    out = dict(child={}, parent=[])
    for mid in ("M1", "M2", "M3", "M4"):
        asm = A.build(f"single_{mid}", [(mid, 0, 0, 0)], P)
        modes = np.array([[0, 0, 0, 0], [ROLLER, 0, 0, ROLLER], [0, ROLLER, ROLLER, 0]])
        gs = np.ones((3, 4), np.float32)
        Rr = sim.simulate(asm, P, S.centred_domain(3), torch.as_tensor(modes), torch.as_tensor(gs), device="cpu")
        zs = sim.targets_from_twist(Rr)[0].numpy()[:, STRAFE].min(1)
        xi, _, _ = T.qs_solve(asm, P, modes, gs, T.legs(asm))
        zq = T.rho_of(xi, T.legs(asm), asm.char_length())[:, STRAFE].min(1)
        out["child"][mid] = dict(z_sim=zs.tolist(), z_qs=zq.tolist())
        print(mid, zs, zq, flush=True)
    R = [r for r in load_sweeps() if len(r["modules"]) >= 2 and r["n"] <= 16
         and any(int(m[3]) % 2 for m in r["modules"])][:n_parents]
    for r in R:
        asm = rebuild(r, P)
        L = T.legs(asm)
        cfg = []
        for k, m in enumerate(r["modules"]):
            if int(m[3]) % 2:
                for pair in ((0, 3), (1, 2)):
                    mo = np.zeros(asm.n, np.int64)
                    mo[[4 * k + pair[0], 4 * k + pair[1]]] = ROLLER
                    cfg.append((k, pair, mo))
        modes = np.stack([np.zeros(asm.n, np.int64)] + [c[2] for c in cfg])
        gs = np.ones(modes.shape, np.float32)
        for dom, tag in ((dict(c_roller=0.0), "ideal"), (None, "nominal")):
            xi, _, _ = T.qs_solve(asm, P, modes, gs, L, dom=dom)
            rho = T.rho_of(xi, L, asm.char_length())
            for c, (k, pair, _) in enumerate(cfg, 1):
                mid = r["modules"][k][0]
                out["parent"].append(dict(model=tag, n=asm.n, dz_par=float(rho[0, STRAFE].min() - rho[c, STRAFE].min()),
                                          worst_par=float((rho[0] - rho[c]).max()),
                                          z_child_sim=out["child"][mid]["z_sim"][1 + (pair == (1, 2))]))
        print(r["name"], flush=True)
    for tag in ("ideal", "nominal"):
        G = [x for x in out["parent"] if x["model"] == tag]
        out[f"summary_{tag}"] = dict(count=len(G), dz_par_absmax=max(abs(x["dz_par"]) for x in G),
                                     worst_par_med=float(np.median([x["worst_par"] for x in G])),
                                     worst_par_max=max(x["worst_par"] for x in G),
                                     child_z_sim_max=max(x["z_child_sim"] for x in G))
    dump("check_split_double.json", out)
    print({k: v for k, v in out.items() if k.startswith("summary")}, out["child"])


if __name__ == "__main__":
    import sys
    double_roller() if sys.argv[1:] == ["double"] else main()
