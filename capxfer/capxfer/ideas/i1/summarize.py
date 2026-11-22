"""Pool stage-2 results over unseen morphologies; bootstrap CIs over trials. Usage: summarize.py [pids]"""
import pathlib
import sys

import numpy as np
import torch

from capxfer.ideas.i1 import mapping, stage1 as S1
from capxfer.ideas.i1.stage2 import UNTESTED, cqr_scores

OUT = pathlib.Path.home() / ".cache/capxfer/I1"
UNSEEN = ("tetromino I (1x4)", "tetromino L/T/S (2x3)", "tetromino O (2x2)", "k5", "k6", "phys_square4")
SEEN = ("phys_single", "phys_pair", "phys_L3", "in-dist (1-3)")


def ci(x, rng, B=2000):
    x = np.asarray(x, float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def main(pids="P1P5P2"):
    D = S1.load("structured_npe", pids)
    E = S1.load("ensemble", pids)
    d_raw = torch.load(S1.PRED.format(m="structured_npe", p=pids), weights_only=False)
    G, gmap = mapping.build(d_raw)
    gid = np.array(d_raw["group_id"])
    rho = np.zeros_like(D["y"])
    for u in np.unique(gid):
        idx = np.flatnonzero(gid == u)
        rho[idx] = G[gmap[idx[0]]]["rho"]
    ix = UNTESTED[pids]
    rng = np.random.default_rng(0)
    print(f"| variant | pool | n | per-target cov | per-target width | joint-untested cov [95% CI] | joint width |")
    print("|---|---|---|---|---|---|---|")
    for mode in ("prior", "box0.1", "known"):
        f = OUT / (f"stage2_{pids}.npy" if mode == "prior" else f"stage2_{pids}_{mode}.npy")
        if not f.exists():
            continue
        Z = np.load(f, allow_pickle=True).item()
        res, calA, alpha = Z["res"], Z["calA"], Z["alpha1"] + Z["alpha2"]
        for pool_name, pool in (("unseen", UNSEEN), ("seen", SEEN)):
            rows = [r for g in pool for r in res.get(g, [])]
            b = np.array([r["b"] for r in rows])
            R = rho[b]

            big = lambda k, s: np.array([r["comp"][k][s] if r["comp"] else np.full(14, 9. * (2 * s - 1))
                                         for r in rows])
            # per-target cov/width from 'pt'; joint from 'jun'
            lo, hi, jlo, jhi = big("pt", 0), big("pt", 1), big("jun", 0), big("jun", 1)
            ins = (R >= lo) & (R <= hi)
            j = ((R >= jlo) & (R <= jhi))[:, ix].all(1)
            m, l, h = ci(j, rng)
            print(f"| COMP-{mode} | {pool_name} | {len(b)} | {ins.mean():.3f} | {np.median(hi - lo):.3f} | "
                  f"{m:.3f} [{l:.2f}, {h:.2f}] | {np.median((jhi - jlo)[:, ix]):.3f} |")
            olo = np.array([r["orc"]["pt"][0] for r in rows])
            ohi = np.array([r["orc"]["pt"][1] for r in rows])
            ojl = np.array([r["orc"]["jun"][0] for r in rows])
            ojh = np.array([r["orc"]["jun"][1] for r in rows])
            ins = (R >= olo) & (R <= ohi)
            m, l, h = ci(((R >= ojl) & (R <= ojh))[:, ix].all(1), rng)
            print(f"| oracle-h-{mode} | {pool_name} | {len(b)} | {ins.mean():.3f} | {np.median(ohi - olo):.3f} | "
                  f"{m:.3f} [{l:.2f}, {h:.2f}] | {np.median((ojh - ojl)[:, ix]):.3f} |")
            if mode == "prior":
                for name, P in (("ENS-split", E), ("SNPE-split", D)):
                    s, w = cqr_scores(P, rho, calA)
                    tpt = np.array([S1.cq(s[:, k], alpha) for k in range(14)])
                    tj = S1.cq((s[:, ix] / w[:, ix]).max(1), alpha)
                    ql, qh = P["yq"][b, :, 0], P["yq"][b, :, 2]
                    wb = np.maximum(qh - ql, 0.02)
                    ins = (R >= ql - tpt) & (R <= qh + tpt)
                    jl, jh = ql - tj * wb, qh + tj * wb
                    m, l, h = ci(((R >= jl) & (R <= jh))[:, ix].all(1), rng)
                    print(f"| {name} | {pool_name} | {len(b)} | {ins.mean():.3f} | {np.median(2 * tpt + qh - ql):.3f}"
                          f" | {m:.3f} [{l:.2f}, {h:.2f}] | {np.median((jh - jl)[:, ix]):.3f} |")

if __name__ == "__main__":
    main(*(sys.argv[1:2]))
