"""SPLIT benchmark evaluation (SIMULATION ONLY): predict post-split capability BEFORE splitting.

Methods (per post-split single module; 'singles' split and the split-off module of 'off1'):
  comp_se    (a)  structured_ensemble per-wheel posterior -> I1 sparse wheel sets (calibrated on wheels, calA) ->
                  projected onto the module (<= K - #forced-faults-elsewhere) -> union of per-mode-vector stage-2
                  bands from the single-module library (plant prior DR_WIDE, 500 draws). Guarantee alpha1 + alpha2.
  post_se    (a') structured_ensemble posterior mixture composed through the library, then one-sided / CQR conformal
                  calibrated on POST-SPLIT outcomes of calA episodes (one random module per episode).
  comp_ab / post_ab (d) the same with analytic_bayes posteriors.
  bb_carry   (b)  ensemble current-body interval, calibrated on current-body truth, carried over unchanged.
  bb_post    (b') ensemble current-body quantiles recalibrated on post-split outcomes (calA).
  healthy    (c)  assume healthy: library band of the healthy module (no diagnosis).
  orc_h           oracle health, unknown plant (library band at the true mode vector).
  oracle     (e)  truth.
Decision: go iff lower bound on z = min(rho_P2+, rho_P2-) >= tau. alpha = 0.10 (alpha1 = alpha2 = 0.05).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.split.evaluate [P1P5|P1P5P2]
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

from capxfer.ideas.i1 import stage1 as S1
from capxfer.ideas.split import common as C

ALPHA1 = ALPHA2 = 0.05
ALPHA = ALPHA1 + ALPHA2
K = 2
TAUS = (0.5, 0.8)
GRID = np.linspace(-0.3, 1.4, 341)


def cq(s, a):
    return S1.cq(s, a)


def zmin(y):
    return np.minimum(y[..., 2], y[..., 3])


# ----------------------------------------------------------------------------- library bands
class Library:
    def __init__(self):
        L = np.load(C.OUT / "library.npz")
        self.modes, self.gs, self.keys = L["modes"], L["gs"], L["keys"]
        self.Z = {mid: L[f"Z_{mid}"].astype(float) for mid in C.MIDS}
        M = self.Z["M1"].shape[1]
        k1 = math.floor((M + 1) * ALPHA2)                       # one-sided order statistic (k-th smallest)
        k2 = math.floor((M + 1) * ALPHA2 / 2)                   # two-sided per target
        self.band = {}
        for mid in C.MIDS:
            Z = self.Z[mid]
            for u in np.unique(self.keys):
                ix = np.flatnonzero(self.keys == u)
                Zu = Z[ix]                                       # (c, M, 14) WEAK corners
                zlo = zmin(Zu).min(0)                            # (M,) envelope lower of z
                lo = np.sort(Zu.min(0), 0)[k2 - 1]
                hi = -np.sort(-Zu.max(0), 0)[k2 - 1]
                self.band[(mid, int(u))] = (float(np.sort(zlo)[k1 - 1]), lo, hi)
        # mixture machinery: config index by (mode-vector key, WEAK corner bits)
        self.ukeys = np.unique(self.keys)
        self.umodes = np.stack([self.modes[np.flatnonzero(self.keys == u)[0]] for u in self.ukeys])
        self.tab = np.zeros((len(self.ukeys), 16), int)
        for a, u in enumerate(self.ukeys):
            ix = np.flatnonzero(self.keys == u)
            for bits in range(16):
                want = np.array([0.9 if (bits >> j) & 1 else 0.3 for j in range(4)])
                best = None
                for c in ix:
                    w = self.modes[c] == 1
                    if np.allclose(self.gs[c][w], want[w]):
                        best = c
                self.tab[a, bits] = best
        # per-config CDFs on GRID for the 14 targets + z
        self.cdf = {}
        for mid in C.MIDS:
            Y = np.concatenate([self.Z[mid], zmin(self.Z[mid])[..., None]], -1)   # (cfg, M, 15)
            self.cdf[mid] = (Y[..., None] <= GRID).mean(1)                       # (cfg, 15, G)

    def comp(self, mid, mvs):
        if not mvs:
            return -np.inf, np.full(14, -np.inf), np.full(14, np.inf)
        bs = [self.band[(mid, u)] for u in mvs]
        return min(b[0] for b in bs), np.min([b[1] for b in bs], 0), np.max([b[2] for b in bs], 0)

    def mixture_q(self, mid, P4, gmean4, levels=(0.05, 0.5, 0.95)):
        """P4 (N,4,5), gmean4 (N,4) -> quantiles (N,15,len(levels)) of the posterior-predictive mixture
        (mode vectors with <= K faults, weights prod_j P_j(m_j), WEAK gain -> nearest library corner, theta ~ prior)."""
        N = P4.shape[0]
        w = np.ones((N, len(self.ukeys)))
        for j in range(4):
            w *= P4[:, j, self.umodes[:, j]]
        w /= w.sum(1, keepdims=True) + 1e-300
        bits = ((gmean4 >= 0.6).astype(int) * (1 << np.arange(4))).sum(1)
        cfg = self.tab[:, bits].T                                                # (N, U)
        out = np.zeros((N, 15, len(levels)))
        F = self.cdf[mid]
        for s in range(0, N, 32):
            sl = slice(s, min(N, s + 32))
            mix = np.einsum("nu,nutg->ntg", w[sl], F[cfg[sl]])
            for li, lv in enumerate(levels):
                idx = np.clip((mix < lv).sum(-1), 0, len(GRID) - 1)
                out[sl, :, li] = GRID[idx]
        return out


def mode_key(m):
    return int(sum(int(v) * 5 ** j for j, v in enumerate(m)))


def projected_mvs(Sw, w):
    """mode vectors on wheels w consistent with the sparse product set S (<= K faults over the assembly)."""
    outside = np.setdiff1d(np.arange(Sw.shape[0]), w)
    forced = int((~Sw[outside, 0]).sum())
    kk = K - forced
    if kk < 0:
        return []
    choices = [np.flatnonzero(Sw[j]) for j in w]
    out = []
    for mv in itertools.product(*choices):
        if sum(1 for v in mv if v) <= kk:
            out.append(mode_key(mv))
    return out


# ----------------------------------------------------------------------------- main
def build(p="P1P5"):
    E = C.episodes(p)
    Gs = C.split_groups(E)
    calA = Gs["calA"]
    T = np.load(C.OUT / "truth_singles.npz")
    ep, mod, rho = T["ep"], T["mod"], T["rho"].astype(float)
    mids = np.array([E[k]["modules"][m][0] for k, m in zip(ep, mod)])
    zt = zmin(rho)
    lib = Library()
    R = dict(ep=ep, mod=mod, mid=mids, rho=rho, z=zt,
             nf=np.array([(E[k]["mode"][4 * m:4 * m + 4] > 0).sum() for k, m in zip(ep, mod)]),
             nmod=np.array([E[k]["nmod"] for k in ep]), lb={}, lo={}, hi={})
    rowsets = {}
    for k, e in enumerate(E):
        rowsets.setdefault(k, [])
    for r, k in enumerate(ep):
        rowsets[int(k)].append(r)
    calrows = np.array([C_rng_pick(rowsets[k], k) for k in calA])
    # --- oracle-health and healthy
    for name in ("orc_h", "healthy"):
        lb, lo, hi = [], [], []
        for r in range(len(ep)):
            m = E[ep[r]]["mode"][4 * mod[r]:4 * mod[r] + 4] if name == "orc_h" else np.zeros(4, int)
            a, b, c = lib.comp(mids[r], [mode_key(m)])
            lb.append(a); lo.append(b); hi.append(c)
        R["lb"][name], R["lo"][name], R["hi"][name] = np.array(lb), np.array(lo), np.array(hi)
    # --- compositional (a), (d)
    for tag, meth in (("se", "structured_ensemble"), ("ab", "analytic_bayes")):
        D = S1.load(meth, p)
        Tw = S1.calibrate(D, np.array([E[k]["b"] for k in calA]), ALPHA1, K=K, seed=0)
        sets = {}
        lb, lo, hi = [], [], []
        for r in range(len(ep)):
            k = int(ep[r])
            if k not in sets:
                sets[k] = S1.sets_for(D, E[k]["b"], Tw, "sparse")
            mv = projected_mvs(sets[k], np.arange(4 * mod[r], 4 * mod[r] + 4))
            a, b, c = lib.comp(mids[r], mv)
            lb.append(a); lo.append(b); hi.append(c)
        R["lb"]["comp_" + tag], R["lo"]["comp_" + tag], R["hi"]["comp_" + tag] = np.array(lb), np.array(lo), np.array(hi)
        # stage-1 bookkeeping: did the projected set contain the true module mode vector?
        R.setdefault("s1cov", {})["comp_" + tag] = np.array([
            sets[int(ep[r])][np.arange(4 * mod[r], 4 * mod[r] + 4), E[ep[r]]["mode"][4 * mod[r]:4 * mod[r] + 4]].all()
            for r in range(len(ep))])
        # --- post-split-calibrated mixture (a'), (d')
        Q = np.zeros((len(ep), 15, 3))
        bidx = np.array([E[k]["b"] for k in ep])
        for mid in C.MIDS:
            sel = np.flatnonzero(mids == mid)
            w = (4 * mod[sel])[:, None] + np.arange(4)
            P4 = D["P"][bidx[sel][:, None], w]
            gm = D["g_mean"][bidx[sel][:, None], w]
            Q[sel] = lib.mixture_q(mid, P4, gm)
        lbz, lo, hi = conformalize(Q[:, 14, 0], Q[:, :14, 0], Q[:, :14, 2], zt, rho, calrows)
        R["lb"]["post_" + tag], R["lo"]["post_" + tag], R["hi"]["post_" + tag] = lbz, lo, hi
        R.setdefault("raw", {})["post_" + tag] = Q
    # --- black-box ensemble on the current body (b), (b')
    Pe = C.load_pred("ensemble", p)
    bidx = np.array([E[k]["b"] for k in ep])
    zs = zmin(Pe["ys"])                                           # (Ntrial, S)
    qz = np.quantile(zs, 0.05, 1)
    pre_rho = np.stack([e["rho"] for e in E])
    pre_b = np.array([e["b"] for e in E])
    # (b) carry-over: calibrated on current-body truth of calA
    s = qz[pre_b[calA]] - zmin(pre_rho[calA])
    t = cq(s, ALPHA)
    lo_t = np.array([cq(Pe["yq"][pre_b[calA], kk, 0] - pre_rho[calA, kk], ALPHA) for kk in range(14)])
    hi_t = np.array([cq(pre_rho[calA, kk] - Pe["yq"][pre_b[calA], kk, 2], ALPHA) for kk in range(14)])
    R["lb"]["bb_carry"] = qz[bidx] - t
    R["lo"]["bb_carry"] = Pe["yq"][bidx, :, 0] - lo_t
    R["hi"]["bb_carry"] = Pe["yq"][bidx, :, 2] + hi_t
    R["pre"] = dict(lb=qz[pre_b] - t, z=zmin(pre_rho))           # per episode, current body
    # (b') recalibrated on post-split outcomes
    lbz, lo, hi = conformalize(qz[bidx], Pe["yq"][bidx, :, 0], Pe["yq"][bidx, :, 2], zt, rho, calrows)
    R["lb"]["bb_post"], R["lo"]["bb_post"], R["hi"]["bb_post"] = lbz, lo, hi
    R["lb"]["oracle"] = zt.copy()
    R["lo"]["oracle"], R["hi"]["oracle"] = rho.copy(), rho.copy()
    R["calrows"] = calrows
    return E, Gs, R


def C_rng_pick(rows, k):
    return rows[np.random.default_rng(1000 + int(k)).integers(len(rows))]


def conformalize(q05z, ql, qh, zt, rho, calrows):
    t = cq(q05z[calrows] - zt[calrows], ALPHA)
    lo_t = np.array([cq(ql[calrows, kk] - rho[calrows, kk], ALPHA) for kk in range(14)])
    hi_t = np.array([cq(rho[calrows, kk] - qh[calrows, kk], ALPHA) for kk in range(14)])
    return q05z - t, ql - lo_t, qh + hi_t


METHODS = ("comp_se", "post_se", "comp_ab", "post_ab", "bb_carry", "bb_post", "healthy", "orc_h", "oracle")


def metrics(R, rows, tau, m):
    z = R["z"][rows]
    go = R["lb"][m][rows] >= tau
    uns = z < tau
    lo, hi, y = R["lo"][m][rows], R["hi"][m][rows], R["rho"][rows]
    inside = (y >= lo) & (y <= hi)
    fin = np.isfinite(lo) & np.isfinite(hi)
    return dict(n=int(len(rows)), n_unsafe=int(uns.sum()), unsafe_go=float((go & uns).mean()),
                false_go=float(go[uns].mean()) if uns.any() else float("nan"),
                cert=float(go.mean()), useful=float(go[~uns].mean()) if (~uns).any() else float("nan"),
                cov_pt=float(inside.mean()), cov_strafe=float(inside[:, [2, 3]].all(1).mean()),
                width=float(np.median(np.where(fin, hi - lo, np.nan)[fin])) if fin.any() else float("inf"),
                lb_cov=float((z >= R["lb"][m][rows]).mean()))


def task_choice(R, eps, tau, m, rng):
    """'Send one module to strafe-dock after the split': pick the certified module with the largest lower bound
    (ties random). Outcome per episode: ok / unsafe pick / abstain."""
    res = []
    for k in eps:
        rows = np.flatnonzero(R["ep"] == k)
        lb = R["lb"][m][rows] + rng.uniform(0, 1e-9, len(rows))
        if m == "oracle":
            lb = R["z"][rows]
        j = int(np.argmax(lb))
        if lb[j] < tau:
            res.append("abstain")
        else:
            res.append("ok" if R["z"][rows[j]] >= tau else "unsafe")
    res = np.array(res)
    return dict(ok=float((res == "ok").mean()), unsafe=float((res == "unsafe").mean()),
                abstain=float((res == "abstain").mean()))


def main(p="P1P5"):
    E, Gs, R = build(p)
    out = dict(pids=p, alpha=ALPHA, alpha1=ALPHA1, alpha2=ALPHA2, K=K, taus=TAUS, tables={})
    ep = R["ep"]
    groups = {g: np.flatnonzero(np.isin(ep, Gs[g])) for g in ("calB", "heldout", "pair", "L3", "square4")}
    groups["test_all"] = np.flatnonzero(np.isin(ep, np.concatenate([Gs[g] for g in ("calB", "heldout", "pair", "L3",
                                                                                     "square4")])))
    test = groups["test_all"]
    for nm in range(2, 7):
        groups[f"nmod{nm}"] = test[R["nmod"][test] == nm]
    groups["test_faulted_modules"] = test[R["nf"][test] > 0]
    for tau in TAUS:
        for g, rows in groups.items():
            for m in METHODS:
                out["tables"][f"{tau}|{g}|{m}"] = metrics(R, rows, tau, m)
    # stage-1 projected-set coverage
    out["stage1_module_cov"] = {m: {g: float(R["s1cov"][m][rows].mean()) for g, rows in groups.items()}
                                for m in R["s1cov"]}
    # pre-split view (per episode): current body certified by the black box, and its truth
    teste = np.concatenate([Gs[g] for g in ("calB", "heldout", "pair", "L3", "square4")])
    rng = np.random.default_rng(0)
    out["pre"], out["task"] = {}, {}
    for tau in TAUS:
        for nm in range(2, 7):
            ks = teste[np.array([E[k]["nmod"] for k in teste]) == nm]
            faulted = ks[np.array([(E[k]["mode"] > 0).any() for k in ks])]
            post_unsafe = np.array([(R["z"][ep == k] < tau).any() for k in faulted])
            out["pre"][f"{tau}|nmod{nm}"] = dict(
                n_faulted_eps=int(len(faulted)),
                pre_body_unsafe=float((R["pre"]["z"][faulted] < tau).mean()),
                bb_certifies_body=float((R["pre"]["lb"][faulted] >= tau).mean()),
                post_any_module_unsafe=float(post_unsafe.mean()),
                bb_certifies_body_and_post_unsafe=float(((R["pre"]["lb"][faulted] >= tau) & post_unsafe).mean()))
        fe = teste[np.array([(E[k]["mode"] > 0).any() for k in teste])]
        for m in METHODS:
            out["task"][f"{tau}|{m}"] = task_choice(R, fe, tau, m, rng)
    path = C.OUT / f"results_singles_{p}.json"
    path.write_text(json.dumps(out, indent=1))
    np.savez_compressed(C.OUT / f"rows_singles_{p}.npz", ep=R["ep"], mod=R["mod"], z=R["z"], nf=R["nf"],
                        nmod=R["nmod"], **{f"lb_{m}": R["lb"][m] for m in METHODS})
    print("wrote", path)
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
