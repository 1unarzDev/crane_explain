"""Fully automatic scoring of explanations against injected ground truth only (no lexical rubric).

Why the score cannot be gamed by wording:
  1. Only structured claims are scored, and truth comes only from `labels` (injected modes, ood) and `y_true`.
  2. An explanation whose text does not verify (realizes exactly its claim list, licensed numbers only) fails
     closed: every supported claim in it counts as incorrect. Rewording therefore can only lose.
  3. The capability threshold and decision-relevant directions come from the scorer's arguments / the catalog,
     never from the explanation.
  4. Duplicate claims are collapsed; unknown vocabulary is incorrect; rates are reported per explanation (macro)
     and per claim (micro), so padding with easy claims does not move the macro rate.
  5. Abstention is never free: informativeness (and correct informativeness) and fault localization are reported
     alongside the error rates, so blanket abstention and blanket assertion each lose on one side.
"""
from __future__ import annotations

import numpy as np

from .contracts import load_contracts
from .realize import verify

MODES = ("ok", "weak", "coast", "brake", "roller_locked")
OOD_CAUSE = {"claim-cause-floor": "low_mu", "claim-cause-payload": "payload"}


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _key(c):
    s = c.get("subject")
    s = tuple(s) if isinstance(s, (list, tuple)) else s
    return (c.get("type"), c.get("contract_id"), s, c.get("value"))


def _claim_truth(c, C, tau, mode, ood, y):
    """True/False for one supported claim against ground truth."""
    t, v = c.get("type"), c.get("value")
    idx = c.get("subject_idx")
    try:
        if t == "capability":
            k = C.keys.index(c["subject"])
            return bool(y[k] >= tau) if v == "reliable" else bool(y[k] < tau) if v == "unreliable" else False
        if t == "wheel_mode":
            ws = idx if isinstance(idx, (list, tuple)) else [idx]
            if v == "ok":
                return bool(all(mode[w] == 0 for w in ws))
            if v == "not_ok":
                return bool(all(mode[w] != 0 for w in ws))
            return bool(all(mode[w] == MODES.index(v) for w in ws))
        if t == "cause":
            if c.get("contract_id") in OOD_CAUSE:
                return ood == OOD_CAUSE[c["contract_id"]]
            faulted = set(np.flatnonzero(np.asarray(mode) != 0).tolist())
            return ood == "none" and faulted == {int(idx)} and int(mode[int(idx)]) == MODES.index(v)
        if t == "ood":
            return ood != "none"
    except (ValueError, TypeError, IndexError, KeyError):
        return False
    return False


def _names_wheel_fault(c):
    return c["type"] == "cause" and c.get("contract_id") == "claim-cause-wheel" or \
        (c["type"] == "wheel_mode" and c.get("value") not in ("ok", None))


def score(explanations, labels, y_true, decision=None, contracts=None):
    """labels: dict(mode (B,n) int, ood str | sequence of B str). y_true (B,14) capability ratios.

    decision: optional dict(required=[...], threshold=...) fixing the decision-relevant directions; default all 14.
    """
    C = contracts or load_contracts()
    y_true = _np(y_true).astype(float)
    mode = _np(labels["mode"]).astype(int)
    B = len(explanations)
    oods = labels.get("ood", "none")
    oods = [oods] * B if isinstance(oods, str) else list(oods)
    tau = float(decision.get("threshold", C.tau)) if decision else C.tau
    relevant = list(decision["required"]) if decision else list(C.keys)

    per = []
    micro_n = micro_ok = 0
    by_type = {t: [0, 0] for t in ("capability", "wheel_mode", "cause", "ood")}
    for i, e in enumerate(explanations):
        ver = verify(C, e)
        seen, sup = set(), []
        for c in e.get("claims", []):
            if c.get("support") == "supported" and _key(c) not in seen:
                seen.add(_key(c))
                sup.append(c)
        truths = [(_claim_truth(c, C, tau, mode[i], oods[i], y_true[i]) and ver["accepted"]) for c in sup]
        for c, ok in zip(sup, truths):
            if c.get("type") in by_type:
                by_type[c["type"]][0] += 1
                by_type[c["type"]][1] += int(ok)
        micro_n += len(sup)
        micro_ok += sum(truths)
        cap_sup = {c["subject"]: ok for c, ok in zip(sup, truths) if c.get("type") == "capability"}
        made = [k for k in relevant if k in cap_sup]
        fault_claims = [(c, ok) for c, ok in zip(sup, truths) if _names_wheel_fault(c) or
                        (c["type"] == "cause")]
        true_faulted = set(np.flatnonzero(mode[i] != 0).tolist())
        named_ok = set()
        for c, ok in zip(sup, truths):
            if ok and c["type"] == "wheel_mode" and c["value"] != "ok":
                named_ok.add(int(c["subject_idx"]))
        cov = [(lo <= y_true[i][C.keys.index(c["subject"])] <= hi)
               for c in sup if c.get("type") == "capability" and {"lo", "hi"} <= set(c.get("numbers", {}))
               for lo, hi in [(c["numbers"]["lo"], c["numbers"]["hi"])]]
        row = dict(
            verified=ver["accepted"],
            n_claims=len(sup),
            correct_rate=(sum(truths) / len(sup)) if sup else np.nan,
            unsupported_cause=any(not ok for _, ok in fault_claims),
            names_wheel_fault=any(_names_wheel_fault(c) for c in sup),
            informativeness=len(made) / len(relevant),
            correct_informativeness=sum(cap_sup[k] for k in made) / len(relevant),
            abstain=not any(c["type"] != "ood" for c in sup),
            ood=oods[i],
            in_model_fault=(oods[i] == "none" and bool(true_faulted)),
            localized=(bool(true_faulted) and true_faulted <= named_ok),
            interval_cover=cov,
        )
        d = e.get("decision")
        if decision and d:
            truth_go = bool(all(y_true[i][C.keys.index(k)] >= tau for k in relevant))
            row.update(outcome=d["outcome"], truth_go=truth_go,
                       decision_error=(d["outcome"] == "go") != truth_go and d["outcome"] != "probe",
                       unsafe_go=d["outcome"] == "go" and not truth_go)
        per.append(row)

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    ood_rows = [r for r in per if r["ood"] != "none"]
    fault_rows = [r for r in per if r["in_model_fault"]]
    out = dict(
        n=B,
        verification_failure_rate=mean([not r["verified"] for r in per]),
        correct_claim_rate=mean([r["correct_rate"] for r in per]),
        correct_claim_rate_micro=(micro_ok / micro_n) if micro_n else float("nan"),
        correct_rate_by_type={t: (v[1] / v[0] if v[0] else float("nan")) for t, v in by_type.items()},
        claims_by_type={t: v[0] for t, v in by_type.items()},
        unsupported_cause_rate=mean([r["unsupported_cause"] for r in per]),
        ood_misattribution=mean([r["names_wheel_fault"] for r in ood_rows]),
        ood_misattribution_by_kind={k: mean([r["names_wheel_fault"] for r in ood_rows if r["ood"] == k])
                                    for k in sorted({r["ood"] for r in ood_rows})},
        informativeness=mean([r["informativeness"] for r in per]),
        correct_informativeness=mean([r["correct_informativeness"] for r in per]),
        abstention_rate=mean([r["abstain"] for r in per]),
        fault_localization_rate=mean([r["localized"] for r in fault_rows]),
        interval_claim_coverage=mean([x for r in per for x in r["interval_cover"]]),
        n_ood=len(ood_rows), n_in_model_fault=len(fault_rows),
    )
    if decision:
        dec = [r for r in per if "outcome" in r]
        out.update(decision_error_rate=mean([r["decision_error"] for r in dec]),
                   unsafe_go_rate=mean([r["unsafe_go"] for r in dec]),
                   deferral_rate=mean([r["outcome"] == "probe" for r in dec]))
    out["per_explanation"] = per
    return out
