"""EXPL2 evidence (SIMULATION ONLY): per post-split module, the inputs each explainer is allowed to use.

Unit = one post-split single module of one episode (all-singles split). Decision = "this module can strafe after the
split": z = min(rho_P2+, rho_P2-) >= tau, tau in {0.5, 0.8} (the SPLIT benchmark decision).

  CERT   (calibrated, pre-split certificate; SPLIT arm (a)): I1 sparse label-conditional wheel sets at alpha1,
         projected onto the module (<= K faults over the assembly); stage-2 library bands at alpha2 for every
         surviving mode vector; lower bound on z = min over vectors, per-direction upper bound = max over vectors.
         Also the BINDING mode vector (the one that sets the lower bound) and the projected per-wheel mode sets.
  PE     (point estimate of the SAME posterior): posterior-predictive mixture median of rho_P2+-, argmax wheel modes.
  UNCAL  (uncalibrated uncertainty of the SAME posterior): mixture quantiles at a nominal level l (no conformal
         step) and wheel sets {m : p_m >= p_min} plus the argmax.
  PEC    (point estimate, black-box carry-over): the ensemble's current-body median of z (no wheel latent).
All arms see exactly the same pre-split assembly evidence (E1, P1+P5+P2 on the assembly).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from capxfer.ideas.i1 import stage1 as S1
from capxfer.ideas.split import common as C, evaluate as EV

K = 2
LEVELS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
QLEV = tuple(sorted(set(LEVELS) | {1 - l for l in LEVELS}))
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60)


def decode(key):
    return np.array([(key // 5 ** j) % 5 for j in range(4)], np.int64)


class Bands:
    """Stage-2 single-module library bands at an arbitrary alpha2 (same order statistics as SPLIT's Library)."""

    def __init__(self, lib, alpha2):
        self.band = {}
        for mid in C.MIDS:
            Z = lib.Z[mid]
            M = Z.shape[1]
            k1 = math.floor((M + 1) * alpha2)
            k2 = math.floor((M + 1) * alpha2 / 2)
            for u in np.unique(lib.keys):
                Zu = Z[np.flatnonzero(lib.keys == u)]
                zlo = EV.zmin(Zu).min(0)
                lb = float(np.sort(zlo)[k1 - 1]) if k1 >= 1 else -np.inf
                hi = -np.sort(-Zu.max(0), 0)[k2 - 1] if k2 >= 1 else np.full(14, np.inf)
                lo = np.sort(Zu.min(0), 0)[k2 - 1] if k2 >= 1 else np.full(14, -np.inf)
                self.band[(mid, int(u))] = (lb, lo, hi)


def cert_module(Sw, w, mid, bands):
    """-> lb_z, hi(14), projected per-wheel sets (4,5), binding mode vector (4,) or None, #mode vectors."""
    mvs = EV.projected_mvs(Sw, w)
    sets = np.zeros((4, 5), bool)
    if not mvs:
        return -np.inf, np.full(14, np.inf), sets, None, 0
    lbs = []
    hi = np.full(14, -np.inf)
    for u in mvs:
        b = bands.band[(mid, u)]
        lbs.append(b[0])
        hi = np.maximum(hi, b[2])
        sets[np.arange(4), decode(u)] = True
    j = int(np.argmin(lbs))
    return float(lbs[j]), hi, sets, decode(mvs[j]), len(mvs)


def cert_all(D, E, ep, mod, mids, calA_b, alpha, lib, bands_cache):
    """CERT evidence for every module row at total alpha (alpha1 = alpha2 = alpha / 2)."""
    a1 = a2 = alpha / 2
    if a2 not in bands_cache:
        bands_cache[a2] = Bands(lib, a2)
    bands = bands_cache[a2]
    Tw = S1.calibrate(D, calA_b, a1, K=K, seed=0)
    N = len(ep)
    out = dict(lb=np.zeros(N), hi=np.zeros((N, 14)), sets=np.zeros((N, 4, 5), bool), bind=np.full((N, 4), -1),
               nmv=np.zeros(N, int))
    cache = {}
    for r in range(N):
        k = int(ep[r])
        if k not in cache:
            cache[k] = S1.sets_for(D, E[k]["b"], Tw, "sparse")
        lb, hi, s, bmv, nmv = cert_module(cache[k], np.arange(4 * mod[r], 4 * mod[r] + 4), mids[r], bands)
        out["lb"][r], out["hi"][r], out["sets"][r], out["nmv"][r] = lb, hi, s, nmv
        if bmv is not None:
            out["bind"][r] = bmv
    return out


def mixture(lib, P4, gm4, mids):
    """posterior-predictive mixture quantiles (N,15,len(QLEV)) per module (targets 0..13, z at 14)."""
    Q = np.zeros((len(mids), 15, len(QLEV)))
    for mid in C.MIDS:
        sel = np.flatnonzero(mids == mid)
        if len(sel):
            Q[sel] = lib.mixture_q(mid, P4[sel], gm4[sel], levels=QLEV)
    return Q


def qi(level):
    return QLEV.index(min(QLEV, key=lambda q: abs(q - level)))
