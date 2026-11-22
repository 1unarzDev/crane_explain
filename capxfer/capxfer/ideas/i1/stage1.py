"""I1 stage 1: unit-level (per-wheel) conformal health sets, calibrated on small assemblies, tested on unseen ones.

Reads WS-C prediction files (read-only). Compares ways to turn per-wheel scores into an "all wheels covered" event:
  joint   : trial-level max score calibrated on cal trials (WS-D default; n-agnostic, no guarantee under size shift)
  bonf    : per-wheel marginal score, one random wheel per cal trial, level 1 - a/n (n of the TEST assembly)
  sparse  : label-conditional (Mondrian by true class) thresholds; OK wheels at a/(2n), fault classes at a/(2K)
            with K = max #faulted wheels (sparsity assumption). Union bound: (n-K)*a/(2n) + K*a/(2K) <= a.
Score: s_c = 1 - p_c (LAC). Set = {c : 1 - p_c <= t_c}.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

PRED = "~/.cache/capxfer/C/preds_{m}_{p}_W_devW.pt"


def load(m="structured_npe", p="P1P5P2"):
    d = torch.load(PRED.format(m=m, p=p), weights_only=False)
    out = dict(P=d["mode_prob"].numpy().astype(float), mt=d["mode_true"].numpy(), wm=d["wheel_mask"].numpy(),
               n=d["n_wheels"].numpy(), set=np.array(d["set"]), fam=np.array(d["family"]),
               gid=np.array(d["group_id"]), tid=np.array(d["trial_id"]), y=d["y"].numpy().astype(float),
               yq=d["y_q"].numpy().astype(float), g_mean=d["g_mean"].numpy(), g_std=d["g_std"].numpy(),
               g_true=d["g_true"].numpy())
    out["P"] = np.nan_to_num(out["P"], nan=0.0)
    return out


def cq(s, a):
    s = np.sort(np.asarray(s, float))
    k = math.ceil((s.size + 1) * (1 - a))
    return math.inf if k > s.size else float(s[k - 1])


def subsample_one(D, idx, cls=None, rng=None):
    """one wheel per trial (optionally restricted to wheels of true class `cls` / faulted wheels)."""
    sc = []
    for b in idx:
        n = D["n"][b]
        mt = D["mt"][b, :n]
        ok = np.ones(n, bool) if cls is None else (mt == 0 if cls == "ok" else mt != 0)
        if not ok.any():
            continue
        j = rng.choice(np.flatnonzero(ok))
        sc.append(1 - D["P"][b, j, mt[j]])
    return np.array(sc)


def calibrate(D, cal, alpha=0.05, K=2, seed=0):
    rng = np.random.default_rng(seed)
    T = {}
    # joint (trial max)
    smax = [max(1 - D["P"][b, j, D["mt"][b, j]] for j in range(D["n"][b])) for b in cal]
    T["joint"] = cq(smax, alpha)
    T["s_any"] = subsample_one(D, cal, None, rng)
    T["s_ok"] = subsample_one(D, cal, "ok", rng)
    T["s_f"] = subsample_one(D, cal, "fault", rng)
    T["alpha"], T["K"] = alpha, K
    return T


def sets_for(D, b, T, scheme):
    n = D["n"][b]
    p = D["P"][b, :n]
    a = T["alpha"]
    if scheme == "joint":
        t = np.full(5, T["joint"])
    elif scheme == "bonf":
        t = np.full(5, cq(T["s_any"], a / n))
    elif scheme == "sidak":
        t = np.full(5, cq(T["s_any"], 1 - (1 - a) ** (1 / n)))
    elif scheme == "sparse":
        t = np.array([cq(T["s_ok"], a / (2 * n))] + [cq(T["s_f"], a / (2 * T["K"]))] * 4)
    else:
        raise ValueError(scheme)
    return (1 - p) <= t[None, :]                                      # (n,5)


def n_configs(S, K=2):
    """#health configurations in prod(S) with at most K faulted wheels (wheels whose set lacks OK must fault)."""
    has_ok = S[:, 0]
    nf = S[:, 1:].sum(1)
    forced = np.flatnonzero(~has_ok)
    if len(forced) > K:
        return 0
    free = np.flatnonzero(has_ok & (nf > 0))
    base = int(np.prod([nf[j] for j in forced])) if len(forced) else 1
    tot = 0
    # choose extra faulted wheels among `free`
    k_left = K - len(forced)
    from itertools import combinations
    for k in range(k_left + 1):
        for c in combinations(free, k):
            tot += base * int(np.prod([nf[j] for j in c])) if c else base
            if tot > 10 ** 6:
                return tot
    return tot


def evaluate(D, test, T, scheme, K=2):
    cov, size, ncfg, wheel_cov = [], [], [], []
    for b in test:
        n = D["n"][b]
        S = sets_for(D, b, T, scheme)
        mt = D["mt"][b, :n]
        hit = S[np.arange(n), mt]
        cov.append(hit.all())
        wheel_cov.append(hit.mean())
        size.append(S.sum(1).mean())
        ncfg.append(n_configs(S, K))
    ncfg = np.array(ncfg)
    return dict(cov=float(np.mean(cov)), wheel_cov=float(np.mean(wheel_cov)), mean_set=float(np.mean(size)),
                cfg_med=float(np.median(ncfg)), cfg_q90=float(np.quantile(ncfg, 0.9)),
                cfg_empty=float(np.mean(ncfg == 0)), n=len(test))


def groups(D):
    s, f, n = D["set"], D["fam"], D["n"]
    G = {"cal(1-3)": np.flatnonzero(s == "cal"), "heldout 4-6": np.flatnonzero(s == "heldout")}
    for k in (4, 5, 6):
        G[f"  k{k}"] = np.flatnonzero((s == "heldout") & np.char.startswith(f, f"k{k}"))
    G["  tetromino I (1x4)"] = np.flatnonzero(f == "k4_bb1x4")
    G["  tetromino L/T/S (2x3)"] = np.flatnonzero(f == "k4_bb2x3")
    G["  tetromino O (2x2)"] = np.flatnonzero(f == "k4_bb2x2")
    for ph in ("phys_single", "phys_pair", "phys_L3", "phys_square4"):
        G["  " + ph] = np.flatnonzero(f == ph)
    return G


def main(m="structured_npe", p="P1P5P2", alpha=0.05):
    D = load(m, p)
    G = groups(D)
    cal = G["cal(1-3)"]
    # use half of cal for calibration, other half as in-distribution test
    rng = np.random.default_rng(1)
    perm = rng.permutation(cal)
    calA, calB = perm[: len(perm) // 2], perm[len(perm) // 2:]
    T = calibrate(D, calA, alpha)
    print(f"# {m} {p} alpha_1={alpha}; cal trials={len(calA)}; s_ok n={len(T['s_ok'])}, s_f n={len(T['s_f'])}")
    G2 = {"in-dist (cal half B)": calB, **{k: v for k, v in G.items() if k != "cal(1-3)"}}
    for scheme in ("joint", "bonf", "sidak", "sparse"):
        print(f"\n## scheme={scheme}")
        print("group | n | all-wheel cov | wheel cov | mean |set| | #cfg med | q90 | P(0 cfg)")
        for g, idx in G2.items():
            if len(idx) == 0:
                continue
            r = evaluate(D, idx, T, scheme)
            print(f"{g} | {r['n']} | {r['cov']:.3f} | {r['wheel_cov']:.4f} | {r['mean_set']:.2f} | "
                  f"{r['cfg_med']:.0f} | {r['cfg_q90']:.0f} | {r['cfg_empty']:.3f}")
    # unit-exchangeability check: distribution of faulted-wheel score by size
    print("\n## per-wheel score shift (unit exchangeability): faulted-wheel 1-p_true quantiles by group")
    for g, idx in G2.items():
        sc = []
        for b in idx:
            n = D["n"][b]
            mt = D["mt"][b, :n]
            for j in np.flatnonzero(mt != 0):
                sc.append(1 - D["P"][b, j, mt[j]])
        sc = np.array(sc)
        if sc.size:
            print(f"{g}: n={sc.size} q50={np.quantile(sc, .5):.3f} q80={np.quantile(sc, .8):.3f} "
                  f"q95={np.quantile(sc, .95):.3f} P(>t_f)={np.mean(sc > cq(T['s_f'], alpha / 4)):.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ()), **({"alpha": float(sys.argv[3])} if len(sys.argv) > 3 else {}))
