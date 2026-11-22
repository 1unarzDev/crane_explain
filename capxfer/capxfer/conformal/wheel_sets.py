"""APS prediction sets over per-wheel modes OK/WEAK/COAST/BRAKE/ROLLER (CONTRACTS C4).

Adaptive prediction sets (Romano, Sesia, Candes 2020). For a wheel with mode posterior p (5,), sort classes by
decreasing p (ties by class index); the score of class c is
    nonrandomized: sum of p over classes ranked at or above c
    randomized:    sum of p over classes ranked strictly above c + U * p_c,  U ~ U(0,1)
A class enters the set iff its score <= t (an empty set means "no known mode fits"). `include_top=True` (default
                                 False) always keeps the argmax class (this can
only increase coverage).

Units and guarantees (`pooling`):
  * "subsample": one uniformly chosen wheel per calibration trial, so calibration scores are i.i.d. trial
    draws. Guarantee: P(true mode of a uniformly chosen wheel of a new exchangeable trial in its set) >= 1 - alpha.
  * "weighted": all wheels, each trial weight 1 split over its wheels (pooled CDF). Uses more data; the finite-sample
    guarantee is weaker than 1 - alpha in the worst case (within-trial dependence). Sensitivity analysis only.
  * "joint" (default; the pre-specified choice): trial-level score = max over the trial's wheels; guarantee
    P(every wheel of a new exchangeable trial covered) >= 1 - alpha. This is what bounds the explanation layer's
    wrong-wheel claims at the trial level.
`method="class"` gives Mondrian / class-conditional thresholds (one per true mode), i.e. coverage >= 1 - alpha
conditional on each true mode. Otherwise coverage is marginal over wheels (not per-wheel, not per-fault).

Padded inputs (any number of wheels; FIX M10): if a Pred carries `wheel_mask (B,n)` (bool), masked-out wheels are
ignored and get empty sets. Without a mask, wheels whose mode_prob row is not finite (NaN padding) and wheels whose
calibration label is negative (-1 padding) are treated as padding. Preds of different n can be mixed in one list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._common import as_list, conformal_quantile, like, to_np, weighted_conformal_quantile

N_MODES = 5


def _mask_of(p, mp):
    if p.get("wheel_mask") is not None:
        return to_np(p["wheel_mask"]).astype(bool)
    return np.isfinite(mp).all(-1)


def _stack(preds, labels):
    """-> P (W,5), y (W,) or None, trial (W,), mask-shaped bookkeeping list."""
    P, Y, T, shapes = [], [], [], []
    labs = as_list(labels) if isinstance(labels, (dict, list, tuple)) else [labels]
    off = 0
    for i, p in enumerate(preds):
        mp = to_np(p.get("mode_prob"))
        if mp is None:
            raise ValueError("Pred has no mode_prob (black-box methods cannot form wheel sets)")
        B, n, _ = mp.shape
        m = _mask_of(p, mp)
        if labels is not None:
            lab = labs[i] if len(labs) == len(preds) else None
            if lab is None:
                raise ValueError("labels must match preds one to one")
            lab = to_np(lab.get("mode") if isinstance(lab, dict) else lab).astype(int)
            if lab.shape != (B, n):
                raise ValueError(f"labels shape {lab.shape} != mode_prob wheels {(B, n)}")
            m = m & (lab >= 0)
            Y.append(lab[m])
        shapes.append((B, n, m))
        P.append(mp[m])
        T.append((np.arange(B)[:, None] + off + np.zeros((1, n), int))[m])
        off += B
    y = np.concatenate(Y) if labels is not None else None
    return np.concatenate(P).astype(float), y, np.concatenate(T), shapes


def _class_scores(P, U):
    """APS score of every class for every wheel -> (W,5)."""
    W = P.shape[0]
    order = np.argsort(-P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, 1)
    cum_excl = np.cumsum(Ps, 1) - Ps
    scored = cum_excl + (U[:, None] * Ps if U is not None else Ps)
    out = np.empty_like(P)
    np.put_along_axis(out, order, scored, 1)
    return out


@dataclass
class SetCal:
    alpha: float
    threshold: float | np.ndarray        # scalar, or (5,) for method="class"
    method: str = "marginal"
    pooling: str = "joint"
    randomized: bool = False
    include_top: bool = False
    seed: int = 0
    n_cal: int = 0
    info: dict = field(default_factory=dict)

    def sets(self, pred, rng=None):
        """-> bool (B,n,5) for a single Pred, or a list of such arrays for a list of Preds."""
        preds = as_list(pred)
        rng = rng if rng is not None else np.random.default_rng(self.seed + 1)
        out = []
        for p in preds:
            mp = to_np(p["mode_prob"]).astype(float)
            B, n, K = mp.shape
            valid = _mask_of(p, mp)
            flat = np.where(valid[..., None], mp, 1.0 / K).reshape(-1, K)
            U = rng.random(flat.shape[0]) if self.randomized else None
            sc = _class_scores(flat, U)
            thr = np.asarray(self.threshold, float)
            S = sc <= (thr[None, :] if thr.ndim else thr)
            if self.include_top:
                S[np.arange(flat.shape[0]), flat.argmax(1)] = True
            S = S.reshape(B, n, K) & valid[..., None]
            out.append(like(p["mode_prob"], S))
        return out[0] if isinstance(pred, dict) else out


def calibrate(pred_cal, labels_cal, alpha, *, method="marginal", pooling="joint", randomized=False,
              include_top=False, seed=0) -> SetCal:
    """labels_cal: dict with "mode" (B,n) i64, or the (B,n) array itself, or a list matching a list of Preds."""
    preds = as_list(pred_cal)
    P, y, trial, _ = _stack(preds, labels_cal)
    rng = np.random.default_rng(seed)
    U = rng.random(P.shape[0]) if randomized else None
    s_all = _class_scores(P, U)[np.arange(P.shape[0]), y]              # true-class score per wheel
    n_trials = int(trial.max()) + 1 if trial.size else 0

    def thr_of(sel):
        s, tr = s_all[sel], trial[sel]
        if pooling == "subsample":
            pick = {}
            for i in rng.permutation(np.flatnonzero(np.ones_like(s, bool))):
                pick.setdefault(tr[i], i)
            return conformal_quantile(s[list(pick.values())], alpha)
        if pooling == "weighted":
            _, inv, cnt = np.unique(tr, return_inverse=True, return_counts=True)
            return weighted_conformal_quantile(s, 1.0 / cnt[inv], alpha, test_weight=1.0)
        if pooling == "joint":
            mx = np.full(n_trials, np.nan)
            np.fmax.at(mx, tr, s)
            return conformal_quantile(mx, alpha)
        raise ValueError(pooling)

    if method == "marginal":
        t = thr_of(np.ones_like(y, bool))
    elif method == "class":
        if pooling == "joint":
            raise ValueError("class-conditional sets are per wheel; use pooling subsample|weighted")
        t = np.array([thr_of(y == c) if np.any(y == c) else np.inf for c in range(N_MODES)])
    else:
        raise ValueError(method)
    return SetCal(alpha=alpha, threshold=t, method=method, pooling=pooling, randomized=randomized,
                  include_top=include_top, seed=seed, n_cal=n_trials,
                  info=dict(n_wheels=int(P.shape[0])))


def coverage(sets, labels, wheel_mask=None):
    """Per-wheel coverage indicators (W,) and per-trial all-covered (B,) for one Pred's sets."""
    S = to_np(sets).astype(bool)
    y = to_np(labels.get("mode") if isinstance(labels, dict) else labels).astype(int)
    m = np.ones(y.shape, bool) if wheel_mask is None else to_np(wheel_mask).astype(bool)
    yy = np.where(m, y, 0)
    hit = np.take_along_axis(S, yy[..., None], -1)[..., 0]
    return hit[m], np.all(hit | ~m, axis=1)
