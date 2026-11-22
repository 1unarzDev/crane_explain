"""WS-E trust layer: evidence-calibrated explanations of wheel capability (CONTRACTS C5)."""
from .contracts import Evidence, load_contracts
from .core import explain, license_violations, plan_trial, point_estimate, uncalibrated
from .realize import render, to_answer_plan
from .score import score


def verify(expl, contracts=None):
    from .realize import verify as _v
    return _v(contracts or load_contracts(), expl)


__all__ = ["Evidence", "explain", "license_violations", "load_contracts", "plan_trial", "point_estimate",
           "render", "score", "to_answer_plan", "uncalibrated", "verify"]
