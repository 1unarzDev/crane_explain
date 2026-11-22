"""Shared helpers for the conformal layer (numpy only; torch inputs are converted)."""
from __future__ import annotations

import math

import numpy as np


def to_np(x, dtype=None):
    """torch tensor / list / array -> numpy array (None passes through)."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


def like(ref, arr):
    """Return `arr` as the same array type as `ref` (torch if ref is a torch tensor)."""
    if hasattr(ref, "detach"):
        import torch
        return torch.as_tensor(arr, device=ref.device)
    return arr


def as_list(pred):
    """Accept one Pred dict or a list of Pred dicts (one per assembly batch)."""
    if pred is None:
        return []
    if isinstance(pred, dict):
        return [pred]
    return list(pred)


def concat_field(preds, key):
    vals = [to_np(p.get(key)) for p in preds]
    if any(v is None for v in vals):
        return None
    return np.concatenate(vals, axis=0)


def conformal_quantile(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest finite-sample score.

    Returns +inf when n is too small for a finite threshold (n < 1/alpha - 1). NaN scores are dropped.
    """
    s = np.asarray(scores, float).ravel()
    s = s[~np.isnan(s)]
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def weighted_conformal_quantile(scores, weights, alpha, test_weight):
    """Smallest t with  sum_i w_i 1{s_i <= t} / (sum_i w_i + test_weight) >= 1 - alpha.

    The test point's mass `test_weight` sits at +inf (the usual conservative split-CP construction).
    With all weights 1 and test_weight 1 this equals `conformal_quantile`.
    """
    s = np.asarray(scores, float).ravel()
    w = np.asarray(weights, float).ravel()
    ok = ~np.isnan(s)
    s, w = s[ok], w[ok]
    total = w.sum() + test_weight
    if s.size == 0 or total <= 0:
        return math.inf
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(w[order]) / total
    idx = np.searchsorted(cum, 1.0 - alpha - 1e-12, side="left")
    if idx >= s.size:
        return math.inf
    return float(s[order][idx])


def group_codes(groups, n):
    """groups -> (codes (n,), names list). None -> all zero."""
    if groups is None:
        return np.zeros(n, int), ["_all"]
    g = np.asarray([str(x) for x in to_np(groups).ravel()])
    if g.size != n:
        raise ValueError(f"groups has {g.size} entries, expected {n}")
    names, codes = np.unique(g, return_inverse=True)
    return codes, [str(x) for x in names]
