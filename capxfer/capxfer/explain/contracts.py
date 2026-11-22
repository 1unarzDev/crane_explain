"""Wheel claim-contract catalog: loading, evidence predicates and licensing.

The JSON catalog (configs/wheel_claim_contracts_v1.json) is operative: a claim is licensed only when every predicate
behind its `required_evidence_ids` holds on the calibrated evidence of one trial. Each predicate_id in the catalog
must have an implementation in PREDICATES (checked at load time), so the catalog cannot drift from the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "wheel_claim_contracts_v1.json"
OK = 0


@dataclass
class Evidence:
    """Calibrated evidence for ONE trial. `sets` is None for predictors without a per-wheel latent."""
    lo: np.ndarray | None          # (14,)
    hi: np.ndarray | None          # (14,)
    sets: np.ndarray | None        # (n, 5) bool
    ood: bool
    rung: str
    tau: float
    keys: tuple
    wheels: tuple                  # wheel names (asm.wheel_corner)
    extra: dict = field(default_factory=dict)


def _dir(ev, k):
    return ev.lo is not None and ev.hi is not None and np.isfinite(ev.lo[k]) and np.isfinite(ev.hi[k])


def _set(ev, w):
    return None if ev.sets is None else np.asarray(ev.sets[w], bool)


def _singleton_ok(ev, w):
    s = _set(ev, w)
    return s is not None and s.sum() == 1 and bool(s[OK])


def _singleton_fault(ev, w):
    s = _set(ev, w)
    return s is not None and s.sum() == 1 and not bool(s[OK])


def _unique_account(ev, _=None):
    if ev.sets is None:
        return False
    n = len(ev.sets)
    faults = [w for w in range(n) if _singleton_fault(ev, w)]
    return len(faults) == 1 and all(_singleton_ok(ev, w) for w in range(n) if w != faults[0])


def _deficit(ev, _=None):
    return ev.lo is not None and any(_dir(ev, k) and ev.hi[k] < ev.tau for k in range(len(ev.keys)))


def _rung_permits(ev, _=None):
    if ev.sets is None:
        return False
    if ev.rung == "E0":
        return bool((np.asarray(ev.sets, bool).sum(-1) == 1).all())
    return _unique_account(ev)


PREDICATES = {
    "capability_interval_present": lambda ev, k: _dir(ev, k),
    "capability_upper_below_threshold": lambda ev, k: _dir(ev, k) and ev.hi[k] < ev.tau,
    "capability_lower_at_or_above_threshold": lambda ev, k: _dir(ev, k) and ev.lo[k] >= ev.tau,
    "ood_flag_clear": lambda ev, _: not ev.ood,
    "ood_flag_raised": lambda ev, _: bool(ev.ood),
    "wheel_mode_set_present": lambda ev, w: _set(ev, w) is not None,
    "wheel_mode_set_is_singleton_ok": lambda ev, w: _singleton_ok(ev, w),
    "wheel_mode_set_is_singleton_fault": lambda ev, w: _singleton_fault(ev, w),
    "wheel_mode_set_excludes_ok": lambda ev, w: _set(ev, w) is not None and _set(ev, w).any() and not _set(ev, w)[OK],
    "unique_singleton_fault_account": _unique_account,
    "some_direction_upper_below_threshold": _deficit,
    "rung_permits_cause": _rung_permits,
    "unavailable_in_wheel_domain": lambda ev, _: False,
}


class Contracts:
    def __init__(self, data: dict):
        self.data = data
        self.params = data["parameters"]
        self.tau = study_threshold(float(self.params["capability_threshold"]))
        self.decimals = int(self.params["number_decimals"])
        self.keys = tuple(self.params["target_keys"])
        self.modes = tuple(self.params["modes"])
        self.requirements = {r["requirement_id"]: r for r in data["evidence_requirements"]}
        self.claims = {c["claim_id"]: c for c in data["claim_contracts"]}
        self.non_entailments = {n["non_entailment_id"]: n for n in data["non_entailments"]}
        self.withheld_t = data["withheld_templates"]
        self.baseline_t = data["baseline_templates"]
        self.action_t = data["action_templates"]
        missing = {r["predicate_id"] for r in data["evidence_requirements"]} - set(PREDICATES)
        if missing:
            raise ValueError(f"catalog predicates without implementation: {sorted(missing)}")
        for c in data["claim_contracts"]:
            for rid in c["required_evidence_ids"]:
                if rid not in self.requirements:
                    raise ValueError(f"{c['claim_id']} requires unknown evidence {rid}")

    def licensed(self, claim_id: str, ev: Evidence, subject=None) -> bool:
        c = self.claims[claim_id]
        return all(PREDICATES[self.requirements[r]["predicate_id"]](ev, subject) for r in c["required_evidence_ids"])

    def failed_requirements(self, claim_id, ev, subject=None):
        c = self.claims[claim_id]
        return [r for r in c["required_evidence_ids"]
                if not PREDICATES[self.requirements[r]["predicate_id"]](ev, subject)]

    def direction(self, key):
        return self.params["direction_phrases"][key]

    def mode_phrase(self, mode):
        return self.params["mode_phrases"][mode]

    def fmt(self, x):
        return f"{float(x):.{self.decimals}f}"

    def rnd(self, x):
        return round(float(x), self.decimals)


_CACHE: dict = {}


def study_threshold(fallback):
    """Single source of truth: configs/study_constants.yaml (integrator-owned). CAPXFER_STUDY_CONSTANTS overrides the
    path (tests pin their fixture threshold this way)."""
    import os
    import yaml
    path = Path(os.environ.get("CAPXFER_STUDY_CONSTANTS",
                               Path(__file__).resolve().parents[2] / "configs" / "study_constants.yaml"))
    if path.exists():
        with open(path) as fh:
            return float(yaml.safe_load(fh)["capability_threshold"])
    return fallback


def load_contracts(path=None) -> Contracts:
    p = Path(path) if path else DEFAULT_PATH
    if p not in _CACHE:
        with open(p) as fh:
            _CACHE[p] = Contracts(json.load(fh))
    return _CACHE[p]
