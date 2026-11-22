"""Claim planning from calibrated outputs (C5), plus the point-estimate and uncalibrated-uncertainty baselines.

All three explainers emit the SAME structured claim vocabulary and go through the SAME realizer and verifier, so the
scorer (score.py) compares them symmetrically on claims, never on wording.

Decision context (optional): dict(task="sideways dock (strafe left)", required=("P2+",), threshold=0.7).
It restricts the decision-relevant directions, adds a go / no_go / probe decision and, when the decision is not
determined by the calibrated intervals, the next probe that measures the most ambiguous required direction.
"""
from __future__ import annotations

import numpy as np

from .contracts import Evidence, load_contracts
from .realize import render, verify

MODES = ("ok", "weak", "coast", "brake", "roller_locked")


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _wheels(asm, n):
    if asm is not None and getattr(asm, "wheel_corner", None) is not None:
        return tuple(asm.wheel_corner)
    return tuple(f"w{i}" for i in range(n))


def _decision(decision, C):
    if not decision:
        return None
    req = tuple(decision["required"])
    bad = [k for k in req if k not in C.keys]
    if bad:
        raise ValueError(f"unknown required directions {bad}")
    return {"task": decision.get("task", "the task"), "required": req}


class _Builder:
    def __init__(self, C, basis):
        self.C, self.basis, self.claims = C, basis, []

    def add(self, contract_id, ctype, subject, value, support="supported", numbers=None, extra=None, subject_idx=None):
        cid = f"{ctype}:{contract_id}:{len(self.claims)}"
        self.claims.append(dict(id=cid, contract_id=contract_id, type=ctype, subject=subject, value=value,
                                support=support, basis=self.basis, numbers=numbers or {}, extra=extra or {},
                                subject_idx=subject_idx))


def _finish(C, b, ev_dict, wheels, rung, dec, explainer, tau):
    claims = b.claims
    sup = [c for c in claims if c["support"] == "supported"]
    decision, next_probe = None, None
    if dec:
        cap = {c["subject"]: c for c in sup if c["type"] == "capability"}
        req = dec["required"]
        if any(cap.get(k, {}).get("value") == "unreliable" for k in req):
            outcome = "no_go"
        elif all(cap.get(k, {}).get("value") == "reliable" for k in req):
            outcome = "go"
        else:
            outcome = "probe"
            undet = [k for k in req if k not in cap]
            widths = {c["subject"]: c["numbers"].get("hi", np.inf) - c["numbers"].get("lo", -np.inf)
                      for c in claims if c["support"] == "withheld" and c["contract_id"] == "capability_undetermined"}
            key = max(undet, key=lambda k: (widths.get(k, np.inf), -req.index(k)))
            next_probe = {"pid": key.rstrip("+-"), "key": key}
        decision = {"task": dec["task"], "required": list(req), "outcome": outcome}
    non_ood = [c for c in sup if c["type"] != "ood"]
    disposition = "abstain" if not non_ood else ("full" if all(c["support"] == "supported" for c in claims) else "partial")
    expl = dict(claims=claims, decision=decision, next_probe=next_probe, disposition=disposition, rung=rung,
                explainer=explainer, wheels=list(wheels), tau=tau, evidence=ev_dict)
    from .realize import realize_sentences
    _, withheld, _ = realize_sentences(C, expl)
    expl["withheld"] = withheld
    expl["text"] = render(C, expl)
    expl["verification"] = verify(C, expl)
    return expl


def plan_trial(ev: Evidence, C, decision=None, basis="calibrated", explainer="calibrated"):
    """Deterministic claim planning for one trial from (possibly calibrated) evidence via the contract catalog."""
    b = _Builder(C, basis)
    dec = _decision(decision, C)
    relevant = list(dec["required"]) if dec else list(C.keys)
    if C.licensed("claim-ood", ev):
        b.add("claim-ood", "ood", "trial", "outside_known_wheel_faults")
        b.add("capability_not_certified", "capability", "all", "not_certified", support="withheld")
        b.add("cause_ood", "cause", "any", None, support="withheld")
    else:
        for key in relevant:
            k = C.keys.index(key)
            nums = None if ev.lo is None else {"lo": C.rnd(ev.lo[k]), "hi": C.rnd(ev.hi[k])}
            if C.licensed("claim-cap-reliable", ev, k):
                b.add("claim-cap-reliable", "capability", key, "reliable", numbers=nums, subject_idx=k)
            elif C.licensed("claim-cap-unreliable", ev, k):
                b.add("claim-cap-unreliable", "capability", key, "unreliable", numbers=nums, subject_idx=k)
            elif nums is not None and np.isfinite(ev.lo[k]) and np.isfinite(ev.hi[k]):
                b.add("capability_undetermined", "capability", key, "undetermined", support="withheld",
                      numbers=nums, subject_idx=k)
        unique_fault = None
        if ev.sets is None:
            b.add("wheel_no_evidence", "wheel_mode", "all", None, support="withheld")
        else:
            n = len(ev.sets)
            healthy = [w for w in range(n) if C.licensed("claim-wheel-healthy", ev, w)]
            if healthy:
                b.add("claim-wheel-healthy", "wheel_mode", [ev.wheels[w] for w in healthy], "ok", subject_idx=healthy)
            ambiguous = []
            for w in range(n):
                s = np.asarray(ev.sets[w], bool)
                if C.licensed("claim-wheel-mode", ev, w):
                    b.add("claim-wheel-mode", "wheel_mode", ev.wheels[w], MODES[int(np.argmax(s))], subject_idx=w)
                    unique_fault = w
                elif C.licensed("claim-wheel-faulted-unresolved", ev, w):
                    b.add("claim-wheel-faulted-unresolved", "wheel_mode", ev.wheels[w], "not_ok", subject_idx=w,
                          extra={"modes": [MODES[i] for i in np.flatnonzero(s)]})
                elif not s.any():
                    b.add("wheel_no_mode_fits", "wheel_mode", ev.wheels[w], None, support="withheld",
                          extra={"wheel": ev.wheels[w]}, subject_idx=w)
                elif w not in healthy:
                    ambiguous.append(w)
            if ambiguous:
                tmpl = "wheel_ambiguous" if len(ambiguous) > 1 else "wheel_ambiguous_one"
                b.add(tmpl, "wheel_mode", [ev.wheels[w] for w in ambiguous], None, support="withheld",
                      extra={"wheels": [ev.wheels[w] for w in ambiguous]}, subject_idx=ambiguous)
        deficit = [c["subject"] for c in b.claims if c["type"] == "capability" and c["value"] == "unreliable"
                   and c["support"] == "supported"]
        if deficit:
            if unique_fault is not None and C.licensed("claim-cause-wheel", ev, unique_fault):
                w = unique_fault
                b.add("claim-cause-wheel", "cause", ev.wheels[w], MODES[int(np.argmax(ev.sets[w]))], subject_idx=w,
                      extra={"deficit": deficit})
            else:
                b.add("cause_deficit", "cause", "deficit", None, support="withheld", extra={"deficit": deficit})
    ev_dict = dict(lo=None if ev.lo is None else ev.lo.tolist(), hi=None if ev.hi is None else ev.hi.tolist(),
                   sets=None if ev.sets is None else np.asarray(ev.sets, bool).tolist(), ood=bool(ev.ood), basis=basis)
    return _finish(C, b, ev_dict, ev.wheels, ev.rung, dec, explainer, ev.tau)


def explain(pred, intervals, wheel_sets, ood_flag, asm, rung, decision=None, contracts=None):
    """C5 entry point. intervals=(lo, hi) each (B,14) conformal; wheel_sets bool (B,n,5) or None; ood_flag (B,)."""
    C = contracts or load_contracts()
    lo, hi = (_np(intervals[0]).astype(float), _np(intervals[1]).astype(float))
    B = lo.shape[0]
    sets = _np(wheel_sets)
    ood = np.zeros(B, bool) if ood_flag is None else _np(ood_flag).astype(bool).reshape(B)
    n = sets.shape[1] if sets is not None else (asm.n if asm is not None else 0)
    wheels = _wheels(asm, n)
    tau = float(decision.get("threshold", C.tau)) if decision else C.tau
    out = []
    for i in range(B):
        ev = Evidence(lo=lo[i], hi=hi[i], sets=None if sets is None else sets[i].astype(bool), ood=bool(ood[i]),
                      rung=rung, tau=tau, keys=C.keys, wheels=wheels)
        out.append(plan_trial(ev, C, decision, basis="calibrated", explainer="calibrated"))
    return out


def _median(pred):
    yq = _np(pred.get("y_q"))
    if yq is not None:
        return yq[..., yq.shape[-1] // 2].astype(float)
    return np.median(_np(pred["y_samples"]), axis=1).astype(float)


def point_estimate(pred, asm, rung="E0", decision=None, contracts=None):
    """Baseline: always names the argmax wheel + mode and states point capability; never abstains or flags OOD."""
    C = contracts or load_contracts()
    med = _median(pred)
    B = med.shape[0]
    mp = _np(pred.get("mode_prob"))
    dec = _decision(decision, C)
    tau = float(decision.get("threshold", C.tau)) if decision else C.tau
    relevant = list(dec["required"]) if dec else list(C.keys)
    out = []
    for i in range(B):
        b = _Builder(C, "point")
        for key in relevant:
            k = C.keys.index(key)
            v = "reliable" if med[i, k] >= tau else "unreliable"
            b.add(f"claim-cap-{v}", "capability", key, v, numbers={"v": C.rnd(med[i, k])}, subject_idx=k)
        deficit = [c["subject"] for c in b.claims if c["value"] == "unreliable"]
        wheels = _wheels(asm, 0 if mp is None else mp.shape[1])
        if mp is not None:
            am = mp[i].argmax(-1).copy()
            cause = None
            if deficit:  # always name the argmax-fault wheel and its most probable non-ok mode
                faulty = [w for w in range(len(am)) if am[w] != 0]
                pool = faulty if faulty else list(range(len(am)))
                w = max(pool, key=lambda j: 1.0 - mp[i, j, 0])
                if am[w] == 0:
                    am[w] = int(np.argmax(mp[i, w, 1:])) + 1
                cause = w
            healthy = [w for w in range(len(am)) if am[w] == 0]
            if healthy:
                b.add("claim-wheel-healthy", "wheel_mode", [wheels[w] for w in healthy], "ok", subject_idx=healthy)
            for w in range(len(am)):
                if am[w] != 0:
                    b.add("claim-wheel-mode", "wheel_mode", wheels[w], MODES[am[w]], subject_idx=w)
            if cause is not None:
                b.add("claim-cause-wheel", "cause", wheels[cause], MODES[am[cause]], subject_idx=cause,
                      extra={"deficit": deficit})
        ev_dict = dict(point=med[i].tolist(), basis="point")
        out.append(_finish(C, b, ev_dict, wheels, rung, dec, "point_estimate", tau))
    return out


def uncalibrated(pred, asm, rung="E0", decision=None, k_std=2.0, p_min=0.1, ppc_cut=None, contracts=None):
    """Baseline: same contracts and planner, but on UNCALIBRATED uncertainty (posterior/ensemble mean +- k std,
    wheel sets = modes with prob >= p_min, OOD only from a raw ppc cut if given)."""
    C = contracts or load_contracts()
    ys = _np(pred["y_samples"]).astype(float)
    mu, sd = ys.mean(1), ys.std(1)
    lo, hi = mu - k_std * sd, mu + k_std * sd
    mp = _np(pred.get("mode_prob"))
    sets = None
    if mp is not None:
        sets = mp >= p_min
        idx = mp.argmax(-1)
        np.put_along_axis(sets, idx[..., None], True, axis=-1)
    B = lo.shape[0]
    ppc = _np(pred.get("ppc_score"))
    ood = (ppc > ppc_cut) if (ppc_cut is not None and ppc is not None) else np.zeros(B, bool)
    wheels = _wheels(asm, 0 if sets is None else sets.shape[1])
    tau = float(decision.get("threshold", C.tau)) if decision else C.tau
    out = []
    for i in range(B):
        ev = Evidence(lo=lo[i], hi=hi[i], sets=None if sets is None else sets[i], ood=bool(ood[i]), rung=rung,
                      tau=tau, keys=C.keys, wheels=wheels)
        out.append(plan_trial(ev, C, decision, basis="uncalibrated", explainer="uncalibrated"))
    return out


def license_violations(expl, ev: Evidence, contracts=None):
    """Supported claims of ANY explainer that the contracts do not license on calibrated evidence `ev`."""
    C = contracts or load_contracts()
    bad = []
    for c in expl["claims"]:
        if c["support"] != "supported":
            continue
        sidx = c.get("subject_idx")
        if c["contract_id"] == "claim-wheel-healthy":
            ok = all(C.licensed(c["contract_id"], ev, w) for w in sidx)
        elif c["type"] == "cause" and c["contract_id"] == "claim-cause-wheel":
            ok = C.licensed(c["contract_id"], ev, sidx) and ev.sets is not None and \
                bool(np.asarray(ev.sets[sidx], bool)[MODES.index(c["value"])])
        elif c["contract_id"] == "claim-wheel-mode":
            ok = C.licensed(c["contract_id"], ev, sidx) and bool(np.asarray(ev.sets[sidx], bool)[MODES.index(c["value"])])
        else:
            ok = C.licensed(c["contract_id"], ev, sidx)
        if not ok:
            bad.append(c["id"])
    return bad
