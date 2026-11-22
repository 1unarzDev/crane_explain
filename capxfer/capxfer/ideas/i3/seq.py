"""I3: probe-until-certified. Offline replay of probe policies on the view tables written by `extract.py`, with
class-conditional conformal certification that stays valid under the policy's own (optional) stopping.

Decision: may the robot strafe-dock? Feasible iff z = min(rho_P2+, rho_P2-) >= tau. H0 (unsafe): z < tau.
Per view v and trial i the predictor gives mixture quantiles q05/q50/q95 on the two decision legs. Statistics
  certify  lam*_{i,v} = min_leg (q50 - tau) / (q50 - q05 + eps)     certify at v iff lam* >= lam_hat
  refute   nu*_{i,v}  = max_leg (tau - q50) / (q95 - q50 + eps)     refute  at v iff nu*  >= mu_hat
Calibration uses only H0 (infeasible) trials for lam_hat (type-I / false-go control) and only feasible trials for
mu_hat (false-no-go control, efficiency only). Methods for a probe chain with stages t = 0..T:
  raw        uncalibrated model quantiles (q05 >= tau certify, q95 < tau refute), stop at first crossing
  naive      per-stage conformal thresholds at alpha, stop at first crossing   (no correction for peeking: INVALID)
  bonf       per-stage thresholds at alpha/(T+1)                     (valid, conservative)
  anytime    one threshold on M_i = max_t lam*_{i,t}                 (valid for ANY stopping rule along the chain)
  exact      one threshold on M_i = max_{t <= R_i} lam*_{i,t}, R_i = the policy's refute time (valid for THIS policy)
Calibration is Mondrian by module count (1, 2, 3+; the robot knows its own morphology). Within a group with
>= 1/alpha infeasible calibration trials the guarantee is P(certify | infeasible, group) <= alpha; otherwise the
group falls back to the joint guarantee P(certify & infeasible | group) <= alpha (CRC, binary loss).
All thresholds are conformal: lam_hat = the m-th largest calibration statistic, m = floor(alpha (n0+1)); certify iff
statistic > lam_hat  <=>  conformal p-value (1 + #{M_j >= M})/(n0+1) <= alpha.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from capxfer.probes import TARGET_KEYS
from .extract import CHAINS, FIXED, chain_views

ROOT = Path(os.path.expanduser("~/.cache/capxfer/I3"))
DEC = [TARGET_KEYS.index("P2+"), TARGET_KEYS.index("P2-")]
COST = dict(P1=4.0, P2=4.0, P3=4.0, P4=4.0, P5=4.0, P6x=4.4, P6y=4.4, P2o=2.0)
EPS = 0.02


def load(name):
    d = torch.load(ROOT / name, weights_only=False)
    ql = list(d["q_levels"])
    k05, k50, k95 = ql.index(0.05), ql.index(0.5), ql.index(0.95)
    q = d["yq"][:, :, DEC].float().numpy()            # (N,V,2,K)
    d["q05"], d["q50"], d["q95"] = q[..., k05], q[..., k50], q[..., k95]
    d["z"] = d["y"][:, DEC].min(1).values.numpy()
    d["vidx"] = {tuple(v): j for j, v in enumerate(d["views"])}
    d["family"] = np.asarray(d["family"])
    del d["yq"]
    return d


def stats(d, view, tau):
    j = d["vidx"][tuple(sorted(view))]
    q05, q50, q95 = d["q05"][:, j], d["q50"][:, j], d["q95"][:, j]
    lam = ((q50 - tau) / (q50 - q05 + EPS)).min(1)
    nu = ((tau - q50) / (q95 - q50 + EPS)).max(1)
    return lam, nu


def chain_cost(chain):
    return np.cumsum([0.0] + [COST[a] if not (a == "P2") else COST["P2"] - (COST["P2o"] if "P2o" in chain[:i] else 0)
                              for i, a in enumerate(chain)])


def conf_thr(cal_stat, alpha):
    """Conformal threshold: certify iff stat > thr guarantees P(certify | null) <= alpha (exchangeable nulls)."""
    s = np.sort(np.asarray(cal_stat))[::-1]
    m = int(np.floor(alpha * (len(s) + 1)))
    return np.inf if m < 1 else s[m - 1]


def score(dec, time_s, z, tau):
    inf = z < tau
    feas = ~inf
    def r(m, c):
        return float((m & c).sum() / max(c.sum(), 1))
    return dict(n=int(len(z)), n_inf=int(inf.sum()), false_go_given_inf=r(dec == 1, inf),
                false_go_marginal=float(((dec == 1) & inf).mean()), certify_given_feas=r(dec == 1, feas),
                refute_given_inf=r(dec == -1, inf), false_nogo_given_feas=r(dec == -1, feas),
                abstain=float((dec == 0).mean()), decision_error=float((((dec == 1) & inf) | ((dec == -1) & feas)).mean()),
                time_mean=float(np.mean(time_s)), time_median=float(np.median(time_s)))


METHODS = ("naive", "bonf", "anytime", "exact", "std", "raw")


def conf_thr_joint(stat, inf, alpha):
    """CRC with binary loss 1{certify & infeasible} (Angelopoulos et al.): smallest threshold thr such that
    (#{inf & stat > thr} + 1)/(n+1) <= alpha. Usable when the group has (almost) no infeasible calibration trials."""
    m = int(np.floor(alpha * (len(stat) + 1)))
    si = np.sort(np.asarray(stat)[np.asarray(inf, bool)])[::-1]
    if m < 1:
        return np.inf
    return -np.inf if len(si) < m else si[m - 1]


def group_of(d):
    """Mondrian group = module count the robot knows about itself: 1, 2, 3+ (n wheels / 4)."""
    k = np.asarray(d["n"]) // 4
    return np.minimum(k, 3)


def _mats(dd, views, tau):
    L, N = zip(*[stats(dd, v, tau) for v in views])
    return np.stack(L, 1), np.stack(N, 1)


def _thresholds(LAMc, NUc, inf, method, alpha, beta, nmin):
    """Per-stage (lthr, nthr) for one calibration group. Returns also which guarantee was used."""
    T1 = LAMc.shape[1]
    ci, cf = inf, ~inf
    cond = ci.sum() >= nmin
    if method in ("naive", "bonf"):
        a, b = (alpha, beta) if method == "naive" else (alpha / T1, beta / T1)
        lthr = np.array([conf_thr(LAMc[ci, t], a) if cond else conf_thr_joint(LAMc[:, t], ci, a) for t in range(T1)])
        nthr = np.array([conf_thr(NUc[cf, t], b) for t in range(T1)])
        return lthr, nthr, cond
    nthr = np.full(T1, conf_thr(NUc[cf].max(1), beta))
    if method == "anytime":
        M = LAMc.max(1)
    else:
        ref = NUc > nthr[None]
        R = np.where(ref.any(1), ref.argmax(1), T1 - 1)
        M = np.where(np.arange(T1)[None] <= R[:, None], LAMc, -np.inf).max(1)
    thr = conf_thr(M[ci], alpha) if cond else conf_thr_joint(M, ci, alpha)
    return np.full(T1, thr), nthr, cond


def decide(LAMt, NUt, lthr, nthr):
    T1 = LAMt.shape[1]
    cert = LAMt > lthr[None]
    refu = NUt > nthr[None]
    stop_c = np.where(cert.any(1), cert.argmax(1), T1)
    stop_r = np.where(refu.any(1), refu.argmax(1), T1)
    dec = np.zeros(len(LAMt), int)
    dec[(stop_c <= stop_r) & (stop_c < T1)] = 1
    dec[stop_r < stop_c] = -1
    return dec, np.minimum(np.minimum(stop_c, stop_r), T1 - 1)


def run_policy(d, dtest, cal_idx, test_idx, views, cost, tau, alpha, beta, method, mondrian=True, nmin=None):
    """Sequential (len(views) > 1) or fixed (one view) policy; per-group (Mondrian) calibration."""
    nmin = nmin or int(np.ceil(1 / alpha))
    LAMc, NUc = _mats(d, views, tau)
    LAMt, NUt = _mats(dtest, views, tau)
    gc = group_of(d)[cal_idx] if mondrian else np.zeros(len(cal_idx), int)
    gt = group_of(dtest)[test_idx] if mondrian else np.zeros(len(test_idx), int)
    dec = np.zeros(len(test_idx), int)
    st = np.zeros(len(test_idx), int)
    used = {}
    for g in np.unique(gt):
        ct, tt = cal_idx[gc == g], test_idx[gt == g]
        sel = gt == g
        if method == "raw":
            q05 = np.stack([d_q(dtest, v, "q05")[tt].min(1) for v in views], 1)
            q95 = np.stack([d_q(dtest, v, "q95")[tt].min(1) for v in views], 1)
            dec[sel], st[sel] = decide(q05, -q95, np.full(len(views), tau - 1e-12), np.full(len(views), -tau))
            continue
        if len(ct) == 0:
            continue
        Lc_, Lt_ = LAMc[ct], LAMt[tt]
        meth = method
        if method == "std":                       # stage-standardised max statistic (dev half fixes the ranks)
            dev = (np.asarray(d["gid"])[ct] % 2) == 0
            infd = d["z"][ct][dev] < tau
            ref = Lc_[dev][infd] if infd.sum() >= nmin else Lc_[dev]
            def tr(L):
                return np.stack([(np.searchsorted(np.sort(ref[:, t]), L[:, t], side="right") + 1e-3 * np.tanh(L[:, t]))
                                 / (len(ref) + 1) for t in range(L.shape[1])], 1)
            ct, Lc_, Lt_ = ct[~dev], tr(Lc_[~dev]), tr(Lt_)
            meth = "anytime"
        inf = d["z"][ct] < tau
        lthr, nthr, cond = _thresholds(Lc_, NUc[ct], inf, meth, alpha, beta, nmin)
        used[int(g)] = dict(conditional=bool(cond), n_cal=int(len(ct)), n_cal_inf=int(inf.sum()))
        dec[sel], st[sel] = decide(Lt_, NUt[tt], lthr, nthr)
    return dec, np.asarray(cost)[st], used


def d_q(dd, v, k):
    return dd[k][:, dd["vidx"][tuple(sorted(v))]]


def evaluate(d, cal_idx, test_idx, tau, alpha=0.05, beta=0.10, chains=None, fixed=FIXED, dtest=None, extra=True,
             mondrian=True):
    dtest = dtest or d
    out = {}
    zt = dtest["z"][test_idx]
    gt = group_of(dtest)[test_idx]
    def add(key, dec, ts, used):
        r = score(dec, ts, zt, tau)
        r["by_group"] = {int(g): score(dec[gt == g], ts[gt == g], zt[gt == g], tau) for g in np.unique(gt)}
        r["calibration"] = used
        out[key] = r
    for k, v in fixed.items():
        dec, ts, used = run_policy(d, dtest, cal_idx, test_idx, [v], [sum(COST[a] for a in v)], tau, alpha, beta,
                                   "naive", mondrian)
        add(f"fixed/{k}", dec, ts, used)
    for cname in (chains or CHAINS):
        ch = CHAINS[cname]
        views, cost = chain_views(ch), chain_cost(ch)
        for m in METHODS if extra else METHODS[:-1]:
            dec, ts, used = run_policy(d, dtest, cal_idx, test_idx, views, cost, tau, alpha, beta, m, mondrian)
            add(f"{cname}/{m}", dec, ts, used)
    return out
