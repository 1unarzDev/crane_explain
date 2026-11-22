"""EXPL2 claim planning + truth scoring, vectorised over post-split modules (SIMULATION ONLY).

The claim vocabulary and licensing rules are WS-E's (capxfer/explain/core.py::plan_trial, contracts v1) restricted to
the decision directions P2+ / P2- of one post-split module, rung E1:
  capability  "can reliably strafe <dir>"   iff lo_dir >= tau      (truth: rho_dir >= tau)
              "cannot reliably strafe <dir>" iff hi_dir < tau       (truth: rho_dir < tau)
  decision    go iff both directions 'can'; no_go iff some 'cannot'; otherwise withheld (+ next probe)
  wheel_mode  one grouped 'healthy' claim over singleton-{ok} wheels (true iff all of them are ok);
              singleton non-ok set -> "<wheel> is <mode>" (true iff that mode);
              set excluding ok, >1 mode -> "<wheel> is faulted, mode unresolved" (true iff not ok)
  cause       deficit + unique singleton-fault account (one singleton-fault wheel, all others singleton-ok)
              -> "the deficit is caused by <wheel> <mode>" (true iff that wheel is the module's only faulted wheel,
              in that mode, and no consequential out-of-model condition is present)
The point-estimate explainer instead always names the argmax wheel/mode and forces a cause on a deficit
(WS-E point_estimate). `validate()` in run.py checks these vectorised claims against WS-E's own planner, realizer,
verifier and scorer on a sample, so the two cannot drift.
"""
from __future__ import annotations

import numpy as np

MODES = ("ok", "weak", "coast", "brake", "roller_locked")


def plan(lo2, hi2, tau, sets=None, P=None, pe_cause=False, p_conf=0.0):
    """lo2, hi2 (N,2) per-direction bounds (P2+, P2-). sets (N,4,5) bool OR P (N,4,5) probabilities (point mode).
    Returns dict of claim arrays."""
    N = lo2.shape[0]
    fin_lo, fin_hi = np.isfinite(lo2), np.isfinite(hi2)
    rel = fin_lo & fin_hi & (lo2 >= tau)
    unrel = fin_lo & fin_hi & (hi2 < tau) & ~rel
    nogo = unrel.any(1)
    go = rel.all(1) & ~nogo
    out = dict(rel=rel, unrel=unrel, go=go, nogo=nogo, probe=~go & ~nogo)
    if sets is None and P is None:
        z4 = np.zeros((N, 4), bool)
        out.update(hok=z4, sfault=z4, unres=z4, mode_named=np.full((N, 4), -1), cause_w=np.full(N, -1),
                   cause_m=np.full(N, -1), has_wheel=False)
        return out
    if P is not None:                      # point estimate: singleton argmax set where confident, else ambiguous
        am = P.argmax(-1)
        conf = P.max(-1) >= p_conf
        sets = np.where(conf[..., None], np.eye(5, dtype=bool)[am], True)
    card = sets.sum(-1)
    hok = (card == 1) & sets[..., 0]
    sfault = (card == 1) & ~sets[..., 0]
    unres = (card > 1) & ~sets[..., 0]
    mode_named = np.where(sfault, sets.argmax(-1), -1)
    cause_w = np.full(N, -1)
    cause_m = np.full(N, -1)
    if pe_cause:                           # WS-E point_estimate: on a deficit always blame the most likely faulty wheel
        am = P.argmax(-1)
        pf = 1.0 - P[..., 0]
        faulty = am != 0
        score = np.where(faulty.any(1, keepdims=True), np.where(faulty, pf, -1.0), pf)
        w = score.argmax(1)
        m = np.where(am[np.arange(N), w] != 0, am[np.arange(N), w], P[np.arange(N), w, 1:].argmax(-1) + 1)
        okc = nogo & (P.max(-1)[np.arange(N), w] >= p_conf)
        cause_w = np.where(okc, w, -1)
        cause_m = np.where(okc, m, -1)
        if p_conf == 0.0:                  # the forced cause also forces that wheel's mode claim (WS-E behaviour)
            fm = np.flatnonzero(okc & (am[np.arange(N), w] == 0))
            sfault[fm, w[fm]] = True
            hok[fm, w[fm]] = False
            mode_named[fm, w[fm]] = m[fm]
    else:
        uniq = (sfault.sum(1) == 1) & ((sfault | hok).sum(1) == 4)
        okc = nogo & uniq
        w = sfault.argmax(1)
        cause_w = np.where(okc, w, -1)
        cause_m = np.where(okc, mode_named[np.arange(N), w], -1)
    out.update(hok=hok, sfault=sfault, unres=unres, mode_named=mode_named, cause_w=cause_w, cause_m=cause_m,
               has_wheel=True, sets=sets)
    return out


def truth_claims(cl, y2, z, mode, tau, ood_cons=None):
    """per-module counts of supported claims and correct ones, by type."""
    N = len(z)
    ood_cons = np.zeros(N, bool) if ood_cons is None else ood_cons
    yok = y2 >= tau
    cap_n = cl["rel"].sum(1) + cl["unrel"].sum(1)
    cap_c = (cl["rel"] & yok).sum(1) + (cl["unrel"] & ~yok).sum(1)
    tf = mode != 0
    hk = cl["hok"]
    h_n = hk.any(1).astype(int)
    h_c = (hk.any(1) & ~(hk & tf).any(1)).astype(int)
    m_n = cl["sfault"].sum(1)
    m_c = (cl["sfault"] & (mode == cl["mode_named"])).sum(1)
    u_n = cl["unres"].sum(1)
    u_c = (cl["unres"] & tf).sum(1)
    has_c = cl["cause_w"] >= 0
    w = np.clip(cl["cause_w"], 0, 3)
    only = tf.sum(1) == 1
    c_ok = has_c & only & tf[np.arange(N), w] & (mode[np.arange(N), w] == cl["cause_m"]) & ~ood_cons
    named = cl["sfault"] | cl["unres"]
    return dict(cap_n=cap_n, cap_c=cap_c, wh_n=h_n + m_n + u_n, wh_c=h_c + m_c + u_c, cause_n=has_c.astype(int),
                cause_c=c_ok.astype(int), named=named, named_true=named & tf, hok_false=hk & tf,
                n=cap_n + h_n + m_n + u_n + has_c, c=cap_c + h_c + m_c + u_c + c_ok)


def metrics(cl, y2, z, mode, tau, w=None, ood_cons=None):
    """module-level summary (optionally weighted)."""
    N = len(z)
    w = np.ones(N) if w is None else w
    T = truth_claims(cl, y2, z, mode, tau, ood_cons)
    uns = z < tau
    faulted = (mode != 0).any(1)
    definite = cl["go"] | cl["nogo"]
    correct_def = (cl["go"] & ~uns) | (cl["nogo"] & uns)
    wm = lambda x, s=None: float(np.sum(w * x) / np.sum(w)) if s is None else (
        float(np.sum((w * x)[s]) / np.sum(w[s])) if np.sum(w[s]) > 0 else float("nan"))
    rate = lambda c, n: float(np.sum(w * c) / np.sum(w * n)) if np.sum(w * n) > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        macro = np.where(T["n"] > 0, T["c"] / np.maximum(T["n"], 1), np.nan)
    has = T["n"] > 0
    named_any = T["named"].any(1)
    out = dict(
        n=int(N), n_unsafe=int(uns.sum()), n_faulted=int(faulted.sum()),
        # decision-level (module = unit)
        definite_rate=wm(definite), informativeness=wm(correct_def),
        decision_claim_error=rate((definite & ~correct_def), definite),
        false_go=wm(cl["go"], uns), false_nogo=wm(cl["nogo"], ~uns), unsafe_go=wm(cl["go"] & uns),
        go_rate=wm(cl["go"]), nogo_rate=wm(cl["nogo"]), withheld_rate=wm(cl["probe"]),
        unsafe_informativeness=wm(cl["nogo"], uns), safe_informativeness=wm(cl["go"], ~uns),
        # claims (all types)
        correct_claim_rate_micro=rate(T["c"], T["n"]),
        correct_claim_rate_macro=float(np.sum((w * np.nan_to_num(macro))[has]) / np.sum(w[has])) if has.any() else float("nan"),
        claims_per_module=wm(T["n"]),
        cap_claim_correct=rate(T["cap_c"], T["cap_n"]), wheel_claim_correct=rate(T["wh_c"], T["wh_n"]),
        cause_claim_correct=rate(T["cause_c"], T["cause_n"]), cause_claim_rate=wm(T["cause_n"]),
        # wheel naming
        wheel_named_rate_faulted=wm(named_any, faulted), wheel_named_rate_healthy=wm(named_any, ~faulted),
        named_wheel_precision=rate(T["named_true"].sum(1), T["named"].sum(1)),
        wheel_informativeness=wm(T["named_true"].any(1), faulted),
        localization=wm((T["named_true"].sum(1) == (mode != 0).sum(1)), faulted),
        healthy_claim_false=wm(T["hok_false"].any(1)),
    )
    if cl["has_wheel"]:
        nt = cl["sfault"] & (mode != 0)
        out["named_mode_exact"] = rate((cl["sfault"] & (mode == cl["mode_named"])).sum(1), cl["sfault"].sum(1))
        sets = cl["sets"]
        tm = np.take_along_axis(sets, np.clip(mode, 0, 4)[..., None], -1)[..., 0]
        named_tf = T["named_true"]
        out["mode_in_set_named_faulty"] = rate((named_tf & tm).sum(1), named_tf.sum(1))
        out["mode_in_set_named"] = rate((T["named"] & tm).sum(1), T["named"].sum(1))
        out["set_cov_faulted_wheels"] = rate(((mode != 0) & tm).sum(1), (mode != 0).sum(1))
        out["set_cov_module"] = wm(tm.all(1))
        del nt
    return out
