"""Conformal capability intervals on the 14 capability ratios (CONTRACTS C4).

Score: conformalized quantile regression (CQR; Romano, Patterson, Candes 2019) on the predicted quantiles `y_q`:
    s_k = max(q_lo_k - y_k, y_k - q_hi_k)
If `y_q` is missing, q_lo/q_hi are the alpha/2 and 1-alpha/2 sample quantiles of `y_samples` (fallback).

Two flavours:
  * per-target (joint=False): one threshold t_k per target, additive CQR interval [q_lo_k - t_k, q_hi_k + t_k].
    Guarantee (exchangeable case): P(y_k in C_k) >= 1 - alpha for each k separately (marginal over trials).
  * joint (joint=True): normalized max-over-targets score
        s = max_k (s_k / w_k),  w_k = max(q_hi_k - q_lo_k, width_floor)
    interval [q_lo_k - t w_k, q_hi_k + t w_k]. Guarantee: P(all evaluated targets covered) >= 1 - alpha.
    The width normalisation uses only predictions (not labels), so exchangeability is preserved.

Calibration schemes (`method`):
  * "split": one threshold from all calibration trials. Valid only if calibration and test trials are
    exchangeable. Under leave-one-assembly-type-out this is NOT guaranteed; report its coverage as empirical.
  * "mondrian": one threshold per group (e.g. assembly type); test trials must carry a group. Guarantee is
    group-conditional for groups present in calibration. Unseen test groups fall back to the "group" threshold.
  * "group": two-layer hierarchical conformal (Dunn, Wasserman & Ramdas, JASA 2023) for a NEW, unseen group,
    assuming groups are exchangeable draws and trials are exchangeable within a group.
      - group_variant="ss1": subsample one trial per calibration group, split-CP over the K group draws.
        Coverage >= 1 - alpha for a random trial of a new group. Finite only if K >= ceil((K+1)(1-alpha)),
        i.e. K >= 1/alpha - 1 (K >= 9 at alpha = 0.1).
      - group_variant="pooled": pooled-CDF (each group weight 1, a new group's mass 1/(K+1) at +inf).
        Uses all data; finite also only if K/(K+1) >= 1 - alpha. Its worst-case bound in Dunn et al. is weaker
        than 1 - alpha (verify the exact constant against the paper before citing).
    With the 3 real calibration assembly types the group thresholds are INFINITE at alpha = 0.1; this is
    reported, not hidden. The group methods are meant for the simulation study (hundreds of shape families).
All guarantees are marginal over trials (and over targets only in the joint version); none is per-wheel or
per-fault conditional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ._common import (as_list, concat_field, conformal_quantile, group_codes, like, to_np,
                      weighted_conformal_quantile)

N_TARGETS = 14


def _bounds(preds, alpha, lo_level=None, hi_level=None):
    """-> (q_lo, q_hi) arrays (B,14) from y_q (CQR) or y_samples (fallback)."""
    yq = concat_field(preds, "y_q")
    if yq is not None:
        levels = preds[0].get("q_levels")
        levels = np.asarray(to_np(levels), float) if levels is not None else np.linspace(0.05, 0.95, yq.shape[-1])
        i_lo = int(np.argmin(np.abs(levels - lo_level))) if lo_level is not None else int(np.argmin(levels))
        i_hi = int(np.argmin(np.abs(levels - hi_level))) if hi_level is not None else int(np.argmax(levels))
        return yq[..., i_lo].astype(float), yq[..., i_hi].astype(float)
    ys = concat_field(preds, "y_samples")
    if ys is None:
        raise ValueError("Pred needs y_q or y_samples")
    lo = lo_level if lo_level is not None else alpha / 2
    hi = hi_level if hi_level is not None else 1 - alpha / 2
    return np.nanquantile(ys, lo, axis=1), np.nanquantile(ys, hi, axis=1)


def cqr_scores(q_lo, q_hi, y):
    return np.maximum(q_lo - y, y - q_hi)


def _col_threshold(s, alpha, codes, n_groups, method, group_variant, rng):
    """Threshold for one score column s (N,) (NaN = missing)."""
    if method == "split":
        return conformal_quantile(s, alpha)
    ok = ~np.isnan(s)
    if method == "group":
        present = [g for g in range(n_groups) if np.any(ok & (codes == g))]
        K = len(present)
        if group_variant == "ss1":
            draws = [s[rng.choice(np.flatnonzero(ok & (codes == g)))] for g in present]
            return conformal_quantile(np.asarray(draws), alpha)
        if group_variant == "pooled":
            w = np.zeros_like(s)
            for g in present:
                m = ok & (codes == g)
                w[m] = 1.0 / m.sum()
            return weighted_conformal_quantile(s[ok], w[ok], alpha, test_weight=1.0) if K else math.inf
        raise ValueError(f"unknown group_variant {group_variant}")
    raise ValueError(method)


@dataclass
class CapCal:
    alpha: float
    method: str
    joint: bool
    targets: np.ndarray                 # (m,) indices of the evaluated targets
    threshold: np.ndarray               # (m,) per-target or (1,) joint (pooled/group threshold)
    group_thresholds: dict = field(default_factory=dict)   # mondrian: name -> array like `threshold`
    fallback: np.ndarray | None = None  # mondrian: threshold used for unseen groups (group method)
    lo_level: float | None = None
    hi_level: float | None = None
    width_floor: float = 1e-3
    group_variant: str = "pooled"
    n_cal: int = 0
    n_groups: int = 1
    info: dict = field(default_factory=dict)

    def _thr_for(self, groups, B):
        if self.method != "mondrian":
            return np.broadcast_to(self.threshold, (B, self.threshold.size))
        if groups is None:
            raise ValueError("mondrian intervals need test groups")
        g = [str(x) for x in to_np(groups).ravel()]
        rows = [self.group_thresholds.get(x, self.fallback) for x in g]
        return np.stack(rows, 0)

    def interval(self, pred, groups=None):
        """-> (lo, hi), each (B,14). Non-evaluated targets get the raw predicted quantiles."""
        preds = as_list(pred)
        q_lo, q_hi = _bounds(preds, self.alpha, self.lo_level, self.hi_level)
        B = q_lo.shape[0]
        thr = self._thr_for(groups, B)                     # (B, m) or (B, 1)
        lo, hi = q_lo.copy(), q_hi.copy()
        T = self.targets
        if self.joint:
            w = np.maximum(q_hi[:, T] - q_lo[:, T], self.width_floor)
            pad = thr[:, :1] * w
        else:
            pad = thr
        with np.errstate(invalid="ignore"):
            lo[:, T] = q_lo[:, T] - pad
            hi[:, T] = q_hi[:, T] + pad
        ref = preds[0].get("y_q", preds[0].get("y_samples"))
        return like(ref, lo), like(ref, hi)

    def covered(self, pred, y, groups=None):
        """-> (per-target bool (B,m), joint bool (B,)) over the evaluated targets (NaN targets count as covered
        in the joint indicator and are NaN-masked in the per-target one; callers should exclude NaN trials)."""
        lo, hi = self.interval(pred, groups)
        lo, hi, y = to_np(lo)[:, self.targets], to_np(hi)[:, self.targets], to_np(y, float)[:, self.targets]
        cov = (y >= lo) & (y <= hi)
        joint = np.all(cov | np.isnan(y), axis=1)
        return cov, joint


def calibrate(pred_cal, y_cal, alpha, groups=None, method="split", *, joint=False, targets=None,
              group_variant="pooled", lo_level=None, hi_level=None, width_floor=1e-3, seed=0) -> CapCal:
    """Fit conformal thresholds. `targets`: indices into the 14 TARGET_KEYS to evaluate (default all).

    pred_cal: Pred dict or list of Pred dicts; y_cal: (N,14) array (or list, concatenated in the same order);
    groups: (N,) labels (assembly type, shape family, ...) required for "mondrian" and "group".
    """
    if method not in ("split", "mondrian", "group"):
        raise ValueError(f"method must be split|mondrian|group, got {method}")
    preds = as_list(pred_cal)
    y = np.concatenate([to_np(v, float) for v in y_cal], 0) if isinstance(y_cal, (list, tuple)) else to_np(y_cal, float)
    q_lo, q_hi = _bounds(preds, alpha, lo_level, hi_level)
    T = np.arange(N_TARGETS) if targets is None else np.asarray(targets, int)
    if method in ("mondrian", "group") and groups is None:
        raise ValueError(f"method={method} needs groups")
    codes, names = group_codes(groups, y.shape[0])
    rng = np.random.default_rng(seed)
    s = cqr_scores(q_lo[:, T], q_hi[:, T], y[:, T])                    # (N, m)
    if joint:
        w = np.maximum(q_hi[:, T] - q_lo[:, T], width_floor)
        sn = s / w
        bad = np.all(np.isnan(sn), axis=1)
        S = np.where(bad, np.nan, np.nanmax(np.where(np.isnan(sn), -np.inf, sn), axis=1))[:, None]
    else:
        S = s

    def thr(Sm, cm, meth):
        return np.array([_col_threshold(Sm[:, j], alpha, cm, len(names), meth, group_variant, rng)
                         for j in range(Sm.shape[1])])

    cal = CapCal(alpha=alpha, method=method, joint=joint, targets=T, threshold=None, lo_level=lo_level,
                 hi_level=hi_level, width_floor=width_floor, group_variant=group_variant, n_cal=y.shape[0],
                 n_groups=len(names))
    if method == "split":
        cal.threshold = thr(S, codes, "split")
    elif method == "group":
        cal.threshold = thr(S, codes, "group")
    else:
        cal.group_thresholds = {nm: thr(S[codes == g], codes[codes == g], "split") for g, nm in enumerate(names)}
        cal.fallback = thr(S, codes, "group")
        cal.threshold = cal.fallback
    cal.info = dict(groups=names, finite=bool(np.all(np.isfinite(cal.threshold))))
    return cal
