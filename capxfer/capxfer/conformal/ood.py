"""Conformal "outside known wheel faults" test on the posterior-predictive discrepancy `ppc_score` (CONTRACTS C4).

Calibrate on in-model trials only (F0-F5; never F6/F7). Flag a test trial if its score exceeds the
ceil((n+1)(1-alpha))-th smallest calibration score. If calibration and test in-model trials are exchangeable,
P(false alarm) <= alpha (marginal). Under assembly-type shift that is an empirical claim, measured per fold.
Detection power on F6/F7 is never guaranteed; it is measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import as_list, concat_field, conformal_quantile, like, to_np


def _scores(x):
    if isinstance(x, (dict, list, tuple)) and (isinstance(x, dict) or (x and isinstance(x[0], dict))):
        return concat_field(as_list(x), "ppc_score").astype(float)
    return to_np(x, float).ravel()


@dataclass
class OODCal:
    alpha: float
    threshold: float
    cal_scores: np.ndarray

    def __float__(self):
        return float(self.threshold)

    def flag(self, pred):
        s = _scores(pred)
        out = s > self.threshold
        ref = pred.get("ppc_score") if isinstance(pred, dict) else None
        return like(ref, out) if ref is not None else out

    def pvalue(self, pred):
        """Conformal p-value (1 + #{cal >= s}) / (n + 1); small = atypical."""
        s = _scores(pred)
        c = np.sort(self.cal_scores)
        ge = c.size - np.searchsorted(c, s, side="left")
        return (1.0 + ge) / (c.size + 1.0)


def calibrate(ppc_cal, alpha) -> OODCal:
    """ppc_cal: array of in-model calibration ppc scores, a Pred, or a list of Preds."""
    s = _scores(ppc_cal)
    s = s[~np.isnan(s)]
    return OODCal(alpha=alpha, threshold=conformal_quantile(s, alpha), cal_scores=s)


def flag(pred, threshold):
    t = float(threshold)
    return _scores(pred) > t
