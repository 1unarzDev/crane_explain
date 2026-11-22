"""I1 stage 2: compose unit-level health sets through the (morphology-general) simulator.

For a test trial with stage-1 product set S (<= K faulted wheels; WEAK wheels expanded to the g-range corners), draw M
domains theta_m ~ DR_WIDE, simulate every h in S at every theta_m (independent per-run wheel-yaw noise), and form the
per-draw envelope Zlo_m = min_h rho(theta_m, h), Zhi_m = max_h rho(theta_m, h). Draws 1..M1 fit centre/scale; draws
M1+1..M fix the conformal threshold of the envelope score r = max_k max((c_k - Zlo_k)/s_k, (Zhi_k - c_k)/s_k).
On the stage-1 event {h* in S}, rho_test lies in the test trial's own (unobserved) envelope, whose score is
exchangeable with the M2 calibration envelope scores -> P(miss) <= alpha1 + alpha2.
Also computes the oracle-health band (S = {h*}) and split / group conformal baselines on the same trials.

Usage: PYTHONPATH=. python capxfer/ideas/i1/stage2.py [n_per_group] [pids]
Writes ~/.cache/capxfer/I1/stage2_<pids>.npz and prints tables.
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys
import time

import numpy as np
import torch

from capxfer import assembly as A, datagen, sim
from capxfer.ideas.i1 import mapping, stage1 as S1

OUT = pathlib.Path.home() / ".cache/capxfer/I1"
G_RANGE = (0.3, 0.9)                 # WEAK gain prior support (sim.sample_faults); corners used under monotonicity
UNTESTED = {"P1P5P2": [4, 5, 6, 7, 10, 11, 12, 13], "P1P5": [2, 3, 4, 5, 6, 7, 10, 11, 12, 13]}


def configs(Sw, K=2):
    """Sw (n,5) bool -> list of (mode (n,), g (n,)) with <= K faulted wheels, WEAK at both g corners."""
    n = Sw.shape[0]
    forced = [j for j in range(n) if not Sw[j, 0]]
    free = [j for j in range(n) if Sw[j, 0] and Sw[j, 1:].any()]
    if len(forced) > K:
        return []
    out = []
    for k in range(K - len(forced) + 1):
        for extra in itertools.combinations(free, k):
            wh = forced + list(extra)
            choices = [[m for m in range(1, 5) if Sw[j, m]] for j in wh]
            for ms in itertools.product(*choices):
                weak = [j for j, m in zip(wh, ms) if m == 1]
                for gs in itertools.product(G_RANGE, repeat=len(weak)):
                    mode = np.zeros(n, np.int64)
                    g = np.ones(n, np.float32)
                    for j, m in zip(wh, ms):
                        mode[j] = m
                    for j, gv in zip(weak, gs):
                        g[j] = gv
                    out.append((mode, g))
    return out


def rep_dom(dom, idx):
    return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in dom.items()}


def band(Z, M1, alpha2, sets):
    """Z (C, M, 14) rho over configs x draws -> per-target (lo, hi) and joint boxes for each target index set."""
    Zlo, Zhi = Z.min(0), Z.max(0)                                    # (M,14)
    c = 0.5 * (np.median(Zlo[:M1], 0) + np.median(Zhi[:M1], 0))
    s = np.quantile(Zhi[:M1], 0.9, 0) - np.quantile(Zlo[:M1], 0.1, 0) + 0.01
    r = np.maximum((c - Zlo[M1:]) / s, (Zhi[M1:] - c) / s)          # (M2,14)
    M2 = r.shape[0]
    k = math.ceil((M2 + 1) * (1 - alpha2))
    tau = np.sort(r, 0)[k - 1]                                       # per-target
    out = {"pt": (c - tau * s, c + tau * s)}
    for name, ix in sets.items():
        tj = np.sort(r[:, ix].max(1))[k - 1]
        out[name] = (c - tj * s, c + tj * s)
    return out


def select(D, n_per, rng):
    Gs = S1.groups(D)
    sel = {}
    for name in ("  tetromino I (1x4)", "  tetromino L/T/S (2x3)", "  tetromino O (2x2)", "  k5", "  k6",
                 "  phys_single", "  phys_pair", "  phys_L3", "  phys_square4"):
        idx = Gs[name]
        sel[name.strip()] = rng.choice(idx, size=min(n_per, len(idx)), replace=False)
    return sel


SIM_KEYS = ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale", "mass_scale",
            "izz_scale", "radius_scale", "kp_scale", "ki_scale", "v_batt")


def make_dom(dommode, gdom, i, M, rng, frac=0.1):
    """prior: theta ~ DR_WIDE (no plant knowledge). known: theta = the trial's true plant (ideal sys-ID; only the
    per-run wheel-yaw noise varies). box<f>: theta ~ U(theta* +- f * DR width), clipped (an idealised sys-ID set)."""
    if dommode == "prior":
        return sim.sample_domain(M, rng, sim.DR_WIDE)
    d = {k: torch.full((M,), float(gdom[k][i])) for k in SIM_KEYS + ("cg_dx", "cg_dy")}
    d["wheel_yaw_err_std"] = float(gdom["wheel_yaw_err_std"][i])
    if dommode.startswith("box"):
        f = float(dommode[3:] or frac)
        for k in SIM_KEYS:
            lo, hi = sim.DR_WIDE[k]
            c, h = float(gdom[k][i]), f * (hi - lo)
            d[k] = torch.tensor(rng.uniform(max(lo, c - h), min(hi, c + h), M), dtype=torch.float32)
        for k in ("cg_dx", "cg_dy"):
            d[k] = d[k] + torch.tensor(rng.normal(0, f * sim.DR_WIDE["cg_off"], M), dtype=torch.float32)
    return d


def main(n_per=40, pids="P1P5P2", M1=40, M2=59, alpha1=0.05, alpha2=0.05, K=2, seed=0, dommode="prior"):
    torch.cuda.set_per_process_memory_fraction(0.12)                   # <= 2 GB
    params = A.load_params(datagen.PARAMS)
    D = S1.load("structured_npe", pids)
    E = S1.load("ensemble", pids)
    d_raw = torch.load(S1.PRED.format(m="structured_npe", p=pids), weights_only=False)
    G, gmap = mapping.build(d_raw)
    gid = np.array(d_raw["group_id"])
    rho = np.zeros_like(D["y"])
    within = np.zeros(len(gid), int)
    for u in np.unique(gid):
        idx = np.flatnonzero(gid == u)
        rho[idx] = G[gmap[idx[0]]]["rho"]
        within[idx] = np.arange(len(idx))
    rng = np.random.default_rng(seed)
    cal = np.flatnonzero(D["set"] == "cal")
    perm = rng.permutation(cal)
    calA, calB = perm[: len(perm) // 2], perm[len(perm) // 2:]
    T = S1.calibrate(D, calA, alpha1, K=K, seed=seed)
    sel = select(D, n_per, rng)
    sel["in-dist (1-3)"] = rng.choice(calB, size=n_per, replace=False)
    tsets = {"j14": list(range(14)), "jun": UNTESTED[pids]}
    res = {}
    t0 = time.time()
    jobs = []                                    # (gname, b, asm, configs, dom)
    for gname, trials in sel.items():
        for b in trials:
            gi = gmap[b]
            asm = datagen.asm_from_dict(G[gi]["assembly"])
            n = asm.n
            cf = configs(S1.sets_for(D, b, T, "sparse"), K)
            orc = (D["mt"][b, :n].astype(np.int64), np.nan_to_num(D["g_true"][b, :n], nan=1.0).astype(np.float32))
            jobs.append((gname, b, asm, cf + [orc], make_dom(dommode, G[gi]["domain"], within[b], M1 + M2, rng)))
    M = M1 + M2
    Zs = {}
    by_n = {}
    for ji, jb in enumerate(jobs):
        by_n.setdefault(jb[2].n, []).append(ji)
    for n, jis in sorted(by_n.items()):
        budget = 60000 // n                                           # sims per simulate_multi call
        chunk, cnt = [], 0
        for k, ji in enumerate(jis + [None]):
            if ji is not None:
                chunk.append(ji)
                cnt += len(jobs[ji][3]) * M
            if chunk and (ji is None or cnt >= budget):
                asms, doms, modes, gs = [], [], [], []
                for c in chunk:
                    _, b, asm, cf, dom = jobs[c]
                    asms.append(asm)
                    idx = np.tile(np.arange(M), len(cf))
                    doms.append({kk: (v[idx] if torch.is_tensor(v) else v) for kk, v in dom.items()})
                    modes.append(torch.tensor(np.repeat(np.stack([m for m, _ in cf]), M, 0)))
                    gs.append(torch.tensor(np.repeat(np.stack([g for _, g in cf]), M, 0)))
                Rs = sim.simulate_multi(asms, params, doms, modes, gs, device="cuda", engine="graph", record_hz=50,
                                        store_device="cpu", seed=int(rng.integers(1 << 30)))
                for c, R in zip(chunk, Rs):
                    Zs[c] = sim.targets_from_twist(R)[0].cpu().numpy().reshape(len(jobs[c][3]), M, 14)
                del Rs
                print(f"[{time.time() - t0:.0f}s] n={n}: {len(Zs)}/{len(jobs)} trials", flush=True)
                chunk, cnt = [], 0
    for ji, (gname, b, asm, cf, dom) in enumerate(jobs):
        Z = Zs[ji]
        nc = len(cf) - 1
        row = dict(b=b, ncfg=nc, rho=rho[b], y=D["y"][b])
        if nc == 0:
            row["comp"] = None
        else:
            # union over MODE vectors of per-mode bands (WEAK g-corners enveloped inside one mode vector)
            keys = [cf[i][0].tobytes() for i in range(nc)]
            parts = [band(Z[[i for i in range(nc) if keys[i] == u]], M1, alpha2, tsets) for u in dict.fromkeys(keys)]
            row["comp"] = {nm: (np.min([p[nm][0] for p in parts], 0), np.max([p[nm][1] for p in parts], 0))
                           for nm in parts[0]}
            row["n_modecfg"] = len(parts)
        row["orc"] = band(Z[nc:], M1, alpha1 + alpha2, tsets)
        res.setdefault(gname, []).append(row)
    np.save(OUT / f"stage2_{pids}_{dommode}.npy", dict(res=res, calA=calA, pids=pids, alpha1=alpha1, alpha2=alpha2),
            allow_pickle=True)
    report(res, D, E, rho, calA, tsets, alpha1 + alpha2, within)


def cqr_scores(P, rho, idx, lo_i=0, hi_i=2):
    ql, qh = P["yq"][idx, :, lo_i], P["yq"][idx, :, hi_i]
    return np.maximum(ql - rho[idx], rho[idx] - qh), np.maximum(qh - ql, 0.02)


def report(res, D, E, rho, calA, tsets, alpha, within):
    rng = np.random.default_rng(5)
    # baselines: split CQR (per-target + normalized joint), calibrated on calA against rho
    base = {}
    for name, P in (("ens", E), ("snpe", D)):
        s, w = cqr_scores(P, rho, calA)
        tpt = np.array([S1.cq(s[:, k], alpha) for k in range(14)])
        tj = {nm: S1.cq((s[:, ix] / w[:, ix]).max(1), alpha) for nm, ix in tsets.items()}
        # group ss1 over calibration ASSEMBLIES (one trial per assembly)
        gids = np.array([str(x) for x in np.array(D["gid"])[calA]])
        one = np.array([rng.choice(np.flatnonzero(gids == u)) for u in np.unique(gids)])
        tg = {nm: S1.cq((s[one][:, ix] / w[one][:, ix]).max(1), alpha) for nm, ix in tsets.items()}
        base[name] = (tpt, tj, tg, len(one))
    fams = sorted({D["fam"][r["b"]] for rows in res.values() for r in rows})
    print(f"\n# alpha_total={alpha}; group ss1 K(assemblies)={base['ens'][3]}; K(shape families in cal)=4 -> "
          f"ss1 threshold at alpha={alpha}: {S1.cq(np.zeros(4), alpha)}")
    hdr = ("group | n | COMP pt cov / width | COMP joint-untested cov / width | ORACLE-h pt cov/width | "
           "ENS split pt cov/width | ENS split joint-unt cov/width | ENS group-asm joint-unt cov/width | "
           "SNPE split pt cov/width | abstain")
    print(hdr)
    for gname, rows in res.items():
        b = np.array([r["b"] for r in rows])
        R = rho[b]
        ix = tsets["jun"]

        def cw(lo, hi, joint=None):
            inside = (R >= lo) & (R <= hi)
            if joint is None:
                return inside.mean(), np.median(hi - lo)
            return inside[:, joint].all(1).mean(), np.median((hi - lo)[:, joint])
        lo = np.array([r["comp"]["pt"][0] if r["comp"] else np.full(14, -9.) for r in rows])
        hi = np.array([r["comp"]["pt"][1] if r["comp"] else np.full(14, 9.) for r in rows])
        jlo = np.array([r["comp"]["jun"][0] if r["comp"] else np.full(14, -9.) for r in rows])
        jhi = np.array([r["comp"]["jun"][1] if r["comp"] else np.full(14, 9.) for r in rows])
        olo = np.array([r["orc"]["pt"][0] for r in rows])
        ohi = np.array([r["orc"]["pt"][1] for r in rows])
        out = [cw(lo, hi), cw(jlo, jhi, ix), cw(olo, ohi)]
        for name, P in (("ens", E),):
            tpt, tj, tg, _ = base[name]
            ql, qh = P["yq"][b, :, 0], P["yq"][b, :, 2]
            w = np.maximum(qh - ql, 0.02)
            out.append(cw(ql - tpt, qh + tpt))
            out.append(cw(ql - tj["jun"] * w, qh + tj["jun"] * w, ix))
            out.append(cw(ql - tg["jun"] * w, qh + tg["jun"] * w, ix))
        tpt = base["snpe"][0]
        out.append(cw(D["yq"][b, :, 0] - tpt, D["yq"][b, :, 2] + tpt))
        ab = np.mean([r["comp"] is None for r in rows])
        print(f"{gname} | {len(rows)} | " + " | ".join(f"{c:.3f} / {w:.3f}" for c, w in out) + f" | {ab:.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 40, a[1] if len(a) > 1 else "P1P5P2", dommode=a[2] if len(a) > 2 else "prior")
