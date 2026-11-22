"""EXPL2: fair, automatic evaluation of post-split explanations on the SPLIT benchmark (SIMULATION ONLY).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.expl2.run            # needs ood_data.py outputs
Writes ~/.cache/capxfer/EXPL2/{expl2_results.json, expl2_keys.json, expl2_validation.json, rows_id.npz}.

Explainers (see evidence.py): CERT (pre-split certificate, calibrated), PE (point estimate of the same posterior),
UNCAL (uncalibrated uncertainty of the same posterior), PEC (black-box current-body point estimate, carried over).
Scoring uses injected truth only (post-split rho, injected wheel modes, OOD condition); claims are the unit.
Fairness: every arm is swept over its own threshold (CERT: alpha; UNCAL: quantile level l and set threshold p_min;
PE / PEC: abstention margin delta and wheel confidence p_conf) and compared at MATCHED informativeness.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from capxfer.ideas.i1 import stage1 as S1
from capxfer.ideas.split import common as C, evaluate as EV
from capxfer.ideas.expl2 import claims as CL, evidence as EVD

OUT = pathlib.Path.home() / ".cache/capxfer/EXPL2"
P = "P1P5P2"
TAUS = (0.5, 0.8)
DIRS = (2, 3)
OP = dict(cert_alpha=0.10, uncal_level=0.05, uncal_pmin=0.10, pe_delta=0.0, pe_pconf=0.0)
DELTAS = tuple(float(x) for x in np.round(np.arange(0.0, 0.405, 0.01), 2))
PCONF = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)
PMINS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001)
NBOOT = 300


def log(*a):
    print(f"[{time.time() - T0:.0f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- data
def id_rows():
    E = C.episodes(P)
    Gs = C.split_groups(E)
    T = np.load(C.OUT / "truth_singles.npz")
    ep, mod, rho = T["ep"], T["mod"], T["rho"].astype(float)
    test_eps = np.concatenate([Gs[g] for g in ("calB", "heldout", "pair", "L3", "square4")])
    keep = np.flatnonzero(np.isin(ep, test_eps))
    D = S1.load("structured_ensemble", P)
    Pe = C.load_pred("ensemble", P)
    b = np.array([E[k]["b"] for k in ep])
    w4 = (4 * mod)[:, None] + np.arange(4)
    rows = dict(ep=ep[keep], mod=mod[keep], rho=rho[keep],
                mid=np.array([E[k]["modules"][m][0] for k, m in zip(ep, mod)])[keep],
                mode=np.stack([E[k]["mode"][4 * m:4 * m + 4] for k, m in zip(ep, mod)])[keep],
                P4=D["P"][b[:, None], w4][keep], gm4=D["g_mean"][b[:, None], w4][keep],
                pec_med=Pe["yq"][b][:, list(DIRS), 1][keep],
                gi=np.array([E[k]["gi"] for k in ep])[keep], nmod=np.array([E[k]["nmod"] for k in ep])[keep],
                fam=np.array([E[k]["fam"] for k in ep])[keep], row=keep)
    rows["z"] = EV.zmin(rows["rho"])
    rows["ood_cons"] = np.zeros(len(keep), bool)
    return E, Gs, D, rows


class OODD:
    """S1-compatible view of OOD trials (P, n per trial)."""

    def __init__(self, Z):
        self.d = dict(P=Z["t_P"], n=4 * Z["t_nmod"])

    def __getitem__(self, k):
        return self.d[k]


def ood_rows(kind):
    Z = np.load(OUT / f"ood_{kind}.npz")
    t, mod = Z["m_t"], Z["m_mod"]
    w4 = (4 * mod)[:, None] + np.arange(4)
    rows = dict(ep=t, mod=mod, rho=Z["m_rho"], rho_twin=Z["m_rho_twin"], mid=Z["m_mid"].astype(str),
                mode=Z["t_mode"][t[:, None], w4], P4=Z["t_P"][t[:, None], w4], gm4=Z["t_gmean"][t[:, None], w4],
                pec_med=Z["t_bb_yq"][t][:, list(DIRS), 1], gi=Z["t_gi"][t], nmod=Z["t_nmod"][t],
                ood_cons=Z["m_consequential"], payload=Z["m_payload"], mu_min=Z["m_mu_min"])
    rows["z"] = EV.zmin(rows["rho"])
    rows["z_twin"] = EV.zmin(rows["rho_twin"])
    Ep = [dict(b=k) for k in range(len(Z["t_gi"]))]
    return Ep, OODD(Z), rows


# ----------------------------------------------------------------------------- explainers -> claims
def cert_ev(Dm, Em, rows, calA_b, D_cal, alpha, lib, bc):
    """CERT evidence; wheel sets calibrated on devW calA (D_cal), applied to Dm."""
    a1 = alpha / 2
    if a1 not in bc:
        bc[a1] = EVD.Bands(lib, a1)
    bands = bc[a1]
    Tw = S1.calibrate(D_cal, calA_b, a1, K=EVD.K, seed=0)
    N = len(rows["ep"])
    out = dict(lb=np.zeros(N), hi=np.zeros((N, 14)), sets=np.zeros((N, 4, 5), bool), bind=np.full((N, 4), -1),
               nmv=np.zeros(N, int))
    cache = {}
    for r in range(N):
        k = int(rows["ep"][r])
        if k not in cache:
            cache[k] = S1.sets_for(Dm, Em[k]["b"], Tw, "sparse")
        lb, hi, s, bmv, nmv = EVD.cert_module(cache[k], np.arange(4 * rows["mod"][r], 4 * rows["mod"][r] + 4),
                                              rows["mid"][r], bands)
        out["lb"][r], out["hi"][r], out["sets"][r], out["nmv"][r] = lb, hi, s, nmv
        if bmv is not None:
            out["bind"][r] = bmv
    return out


def cert_claims(ce, tau):
    lo2 = np.repeat(ce["lb"][:, None], 2, 1)
    return CL.plan(lo2, ce["hi"][:, list(DIRS)], tau, sets=ce["sets"])


def uncal_claims(Q, P4, tau, level, pmin):
    lo2 = np.repeat(Q[:, 14, EVD.qi(level)][:, None], 2, 1)
    hi2 = Q[:, list(DIRS), EVD.qi(1 - level)]
    sets = P4 >= pmin
    am = P4.argmax(-1)
    np.put_along_axis(sets, am[..., None], True, -1)
    return CL.plan(lo2, hi2, tau, sets=sets)


def pe_claims(Q, P4, tau, delta, pconf):
    med = Q[:, list(DIRS), EVD.qi(0.5)]
    return CL.plan(med - delta, med + delta, tau, P=P4, pe_cause=True, p_conf=pconf)


def pec_claims(med, tau, delta):
    return CL.plan(med - delta, med + delta, tau)


def score_all(rows, cl, tau, sel=None):
    s = np.arange(len(rows["z"])) if sel is None else sel
    cls = {k: (v[s] if isinstance(v, np.ndarray) else v) for k, v in cl.items()}
    return CL.metrics(cls, rows["rho"][s][:, list(DIRS)], rows["z"][s], rows["mode"][s], tau,
                      ood_cons=rows["ood_cons"][s])


def binding(rows, ce, cl, tau, sel):
    """on modules the certificate does not certify and that are truly unsafe: does the binding hypothesis put a fault
    on a truly faulted wheel of the module?"""
    z, mode, bind = rows["z"], rows["mode"], ce["bind"]
    has = (bind >= 0).all(1)
    hit_any = has & ((bind != 0) & (mode != 0)).any(1)
    hit_exact = has & (bind == mode).all(1)
    hit_faultset = has & ((bind != 0) == (mode != 0)).all(1)
    res = {}
    for name, m in (("notcert_unsafe", ~cl["go"] & (z < tau)), ("nogo_unsafe", cl["nogo"] & (z < tau))):
        m = m[sel]
        n = int(m.sum())
        res[name] = dict(n=n, binding_hits_faulted_wheel=float(hit_any[sel][m].mean()) if n else float("nan"),
                         binding_fault_set_exact=float(hit_faultset[sel][m].mean()) if n else float("nan"),
                         binding_mode_vector_exact=float(hit_exact[sel][m].mean()) if n else float("nan"),
                         binding_is_healthy=float((has & (bind == 0).all(1))[sel][m].mean()) if n else float("nan"))
    return res


def boot_ci(rows, fn, sel, nboot=NBOOT, seed=0):
    """cluster bootstrap over assembly groups (gi) of a scalar statistic fn(index array)."""
    gi = rows["gi"][sel]
    ug, inv = np.unique(gi, return_inverse=True)
    members = [sel[inv == j] for j in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        pick = rng.integers(len(ug), size=len(ug))
        idx = np.concatenate([members[j] for j in pick])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [float("nan")] * 2


# ----------------------------------------------------------------------------- curves + matching
def interp_at(curve, x_key, y_key, x0):
    """linear interpolation between adjacent sweep points (= a randomised mix of the two thresholds). Ratios whose
    denominator is the definite rate are interpolated as numerator / denominator so the mix is exact."""
    if y_key == "decision_claim_error" and x_key == "definite_rate":
        num = [dict(definite_rate=c["definite_rate"], e=c["decision_claim_error"] * c["definite_rate"]) for c in curve
               if np.isfinite(c["decision_claim_error"])]
        return interp_at(num, "definite_rate", "e", x0) / x0 if x0 > 0 else float("nan")
    pts = sorted((c[x_key], c[y_key]) for c in curve if np.isfinite(c[x_key]) and np.isfinite(c[y_key]))
    if not pts:
        return float("nan")
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    if x0 < xs[0] or x0 > xs[-1]:
        return float("nan")                    # outside the baseline's achievable range: no extrapolation
    return float(np.interp(x0, xs, ys))


def curves(rows, Q, ces, tau, sel):
    cv = {"CERT": [], "UNCAL": [], "PE": [], "PEC": []}
    for a, ce in ces.items():
        m = score_all(rows, cert_claims(ce, tau), tau, sel)
        cv["CERT"].append(dict(param="alpha", value=a, **m))
    for l in EVD.LEVELS:
        m = score_all(rows, uncal_claims(Q, rows["P4"], tau, l, OP["uncal_pmin"]), tau, sel)
        cv["UNCAL"].append(dict(param="level", value=l, **m))
    for pm in PMINS:
        m = score_all(rows, uncal_claims(Q, rows["P4"], tau, OP["uncal_level"], pm), tau, sel)
        cv["UNCAL"].append(dict(param="p_min", value=pm, **m))
    for d in DELTAS:
        m = score_all(rows, pe_claims(Q, rows["P4"], tau, d, 0.0), tau, sel)
        cv["PE"].append(dict(param="delta", value=d, **m))
        m = score_all(rows, pec_claims(rows["pec_med"], tau, d), tau, sel)
        cv["PEC"].append(dict(param="delta", value=d, **m))
    for pc in PCONF:
        m = score_all(rows, pe_claims(Q, rows["P4"], tau, 0.0, pc), tau, sel)
        cv["PE"].append(dict(param="p_conf", value=pc, **m))
    return cv


def matched(cv):
    """compare at the CERT operating point's informativeness (decision) and wheel informativeness (wheels)."""
    op = next(c for c in cv["CERT"] if abs(c["value"] - OP["cert_alpha"]) < 1e-9)
    out = dict(cert_op=dict(alpha=OP["cert_alpha"]))
    dec_param = dict(CERT="alpha", UNCAL="level", PE="delta", PEC="delta")
    wh_param = dict(CERT="alpha", UNCAL="p_min", PE="p_conf")
    for arm in cv:
        dc = [c for c in cv[arm] if c["param"] == dec_param[arm]]
        out.setdefault("decision", {})[arm] = dict(
            sweep=dec_param[arm],
            at_cert_definite_rate=dict(x=op["definite_rate"], **{y: interp_at(dc, "definite_rate", y, op["definite_rate"])
                                                                for y in ("decision_claim_error", "false_go",
                                                                          "informativeness", "false_nogo")}),
            at_cert_informativeness=dict(x=op["informativeness"], **{
                y: interp_at(dc, "informativeness", y, op["informativeness"])
                for y in ("decision_claim_error", "false_go", "definite_rate")}),
            informativeness_at_cert_false_go=interp_at(dc, "false_go", "informativeness", op["false_go"]),
            max_informativeness=max(c["informativeness"] for c in dc),
            threshold_at_cert_definite_rate=interp_at(dc, "definite_rate", "value", op["definite_rate"]),
            nominal_point=next((dict(value=c["value"], definite_rate=c["definite_rate"],
                                     informativeness=c["informativeness"], false_go=c["false_go"],
                                     decision_claim_error=c["decision_claim_error"]) for c in dc
                                if abs(c["value"] - dict(CERT=OP["cert_alpha"], UNCAL=OP["uncal_level"],
                                                         PE=OP["pe_delta"], PEC=0.0)[arm]) < 1e-9), None))
        if arm in wh_param:
            wc = [c for c in cv[arm] if c["param"] == wh_param[arm]]
            out.setdefault("wheels", {})[arm] = dict(
                sweep=wh_param[arm],
                at_cert_wheel_informativeness=dict(x=op["wheel_informativeness"], **{
                    y: interp_at(wc, "wheel_informativeness", y, op["wheel_informativeness"])
                    for y in ("named_wheel_precision", "wheel_claim_correct", "wheel_named_rate_healthy",
                              "healthy_claim_false")}),
                wheel_informativeness_at_cert_precision=interp_at(wc, "named_wheel_precision", "wheel_informativeness",
                                                                   op["named_wheel_precision"]))
    return out


def slim(cv):
    keep = ("param", "value", "definite_rate", "informativeness", "decision_claim_error", "false_go", "false_nogo",
            "unsafe_go", "withheld_rate", "correct_claim_rate_micro", "correct_claim_rate_macro", "claims_per_module",
            "cap_claim_correct", "wheel_claim_correct", "cause_claim_correct", "cause_claim_rate",
            "wheel_named_rate_faulted", "wheel_named_rate_healthy", "named_wheel_precision", "wheel_informativeness",
            "localization", "healthy_claim_false", "mode_in_set_named_faulty", "set_cov_faulted_wheels")
    return {a: [{k: c[k] for k in keep if k in c} for c in v] for a, v in cv.items()}


# ----------------------------------------------------------------------------- next probe value
def next_probe_value(rows_id, ce, tau):
    """CERT withholds -> next probe = 'P2 on this module after the split' (P2 was already run on the assembly).
    Value on the SPLIT (f) subsample (probe_after.npz, ensemble, P1+P5+P2 per module, re-weighted)."""
    A = np.load(C.OUT / "probe_after.npz")
    r, ncal, w = A["rows"], int(A["ncal"]), A["w"]
    T = np.load(C.OUT / "truth_singles.npz")
    z = EV.zmin(T["rho"].astype(float))[r]
    q = A["ensemble|P1P5P2|q05z"]
    lb = q - EV.cq(q[:ncal] - z[:ncal], EV.ALPHA)
    pos = {int(x): i for i, x in enumerate(rows_id["row"])}
    te = [i for i in range(ncal, len(r)) if int(r[i]) in pos]
    ii = np.array([pos[int(r[i])] for i in te])
    te = np.array(te)
    cl = cert_claims({k: v[ii] for k, v in ce.items()}, tau)
    wd = cl["probe"]
    go_f = lb[te] >= tau
    uz = z[te] < tau
    ww = w[te]
    f = lambda x, s: float((ww * x)[s].sum() / ww[s].sum()) if ww[s].sum() > 0 else float("nan")
    return dict(n_withheld=int(wd.sum()), withheld_rate=f(wd, np.ones(len(te), bool)),
                resolved_go_after_probe=f(go_f, wd), unsafe_among_withheld=f(uz, wd),
                false_go_after_probe_on_withheld=f(go_f, wd & uz),
                correct_go_after_probe_on_withheld=f(go_f & ~uz, wd))


# ----------------------------------------------------------------------------- WS-E pipeline cross-check
def validate(rows, ce, Q, tau, n=1500, seed=0, label="id"):
    """Run WS-E's own planner / realizer / verifier / scorer on a sample and compare with the vectorised claims."""
    from capxfer import assembly as A
    from capxfer.explain import Evidence, load_contracts, plan_trial, point_estimate, score, verify
    from capxfer.explain.core import license_violations
    Cn = load_contracts()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rows["z"]), min(n, len(rows["z"])), replace=False))
    dec = dict(task="sideways dock after the split", required=["P2+", "P2-"], threshold=tau)
    prm = C.params()
    wheels = {mid: tuple(A.build(f"s_{mid}", [(mid, 0, 0, 0)], prm).wheel_corner) for mid in C.MIDS}
    res = {}
    vec = dict(CERT=cert_claims(ce, tau), UNCAL=uncal_claims(Q, rows["P4"], tau, OP["uncal_level"], OP["uncal_pmin"]),
               PE=pe_claims(Q, rows["P4"], tau, 0.0, 0.0))
    expls = {a: [] for a in vec}
    cert_evs = []
    for i in idx:
        nan = np.full(14, np.nan)
        lo, hi = nan.copy(), nan.copy()
        lo[list(DIRS)] = ce["lb"][i]
        hi[list(DIRS)] = ce["hi"][i][list(DIRS)]
        ev = Evidence(lo=lo, hi=hi, sets=ce["sets"][i], ood=False, rung="E1", tau=tau, keys=Cn.keys,
                      wheels=wheels[rows["mid"][i]])
        cert_evs.append(ev)
        expls["CERT"].append(plan_trial(ev, Cn, dec))
        lo, hi = nan.copy(), nan.copy()
        lo[list(DIRS)] = Q[i, 14, EVD.qi(OP["uncal_level"])]
        hi[list(DIRS)] = Q[i, list(DIRS), EVD.qi(1 - OP["uncal_level"])]
        s = rows["P4"][i] >= OP["uncal_pmin"]
        s[np.arange(4), rows["P4"][i].argmax(-1)] = True
        expls["UNCAL"].append(plan_trial(Evidence(lo=lo, hi=hi, sets=s, ood=False, rung="E1", tau=tau, keys=Cn.keys,
                                                  wheels=wheels[rows["mid"][i]]), Cn, dec, basis="uncalibrated",
                                         explainer="uncalibrated"))
        yq = np.full((1, 14, 3), np.nan)
        yq[0, list(DIRS), 1] = Q[i, list(DIRS), EVD.qi(0.5)]
        expls["PE"].append(point_estimate(dict(y_q=yq, mode_prob=rows["P4"][i][None]),
                                          A.build("s", [(rows["mid"][i], 0, 0, 0)], prm), "E1", dec)[0])
    oods = ["none" if not c else "payload" for c in rows["ood_cons"][idx]]
    for a in vec:
        S = score(expls[a], dict(mode=rows["mode"][idx], ood=oods), rows["rho"][idx], decision=dec)
        v = {k: (x[idx] if isinstance(x, np.ndarray) else x) for k, x in vec[a].items()}
        T = CL.truth_claims(v, rows["rho"][idx][:, list(DIRS)], rows["z"][idx], rows["mode"][idx], tau,
                            rows["ood_cons"][idx])
        outc = np.array([e["decision"]["outcome"] for e in expls[a]])
        vo = np.where(v["go"], "go", np.where(v["nogo"], "no_go", "probe"))
        ncl = np.array([r["n_claims"] for r in S["per_explanation"]])
        okr = np.array([r["correct_rate"] for r in S["per_explanation"]])
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = np.where(T["n"] > 0, T["c"] / np.maximum(T["n"], 1), np.nan)
        lic = [len(license_violations(e, ev, Cn)) > 0 for e, ev in zip(expls[a], cert_evs)]
        res[a] = dict(n=int(len(idx)), outcome_agreement=float((outc == vo).mean()),
                      claim_count_agreement=float((ncl == T["n"]).mean()),
                      correct_rate_agreement=float(np.mean(np.isclose(np.nan_to_num(okr, nan=-1),
                                                                      np.nan_to_num(vr, nan=-1)))),
                      wse_verification_failure_rate=S["verification_failure_rate"],
                      wse_correct_claim_rate=S["correct_claim_rate"], wse_correct_claim_rate_micro=S["correct_claim_rate_micro"],
                      wse_unsupported_cause_rate=S["unsupported_cause_rate"],
                      wse_ood_misattribution=S["ood_misattribution"],
                      claims_not_licensed_by_certificate_evidence=float(np.mean(lic)),
                      examples={o: next((e["text"] for e in expls[a] if e["decision"]["outcome"] == o), None)
                                for o in ("go", "no_go", "probe")})
    return res


# ----------------------------------------------------------------------------- main
def run_subset(rows, Q, ces, tau, sel, with_curves=True):
    cl_op = dict(CERT=cert_claims(ces[OP["cert_alpha"]], tau),
                 UNCAL=uncal_claims(Q, rows["P4"], tau, OP["uncal_level"], OP["uncal_pmin"]),
                 PE=pe_claims(Q, rows["P4"], tau, OP["pe_delta"], OP["pe_pconf"]),
                 PEC=pec_claims(rows["pec_med"], tau, 0.0))
    res = dict(operating_point={a: score_all(rows, c, tau, sel) for a, c in cl_op.items()})
    res["binding_CERT"] = binding(rows, ces[OP["cert_alpha"]], cl_op["CERT"], tau, sel)
    if with_curves:
        cv = curves(rows, Q, ces, tau, sel)
        res["matched"] = matched(cv)
        res["curves"] = slim(cv)
    return res, cl_op


def main():
    global T0
    T0 = time.time()
    lib = EV.Library()
    E, Gs, D, rows = id_rows()
    calA_b = np.array([E[k]["b"] for k in Gs["calA"]])
    log("id rows", len(rows["z"]))
    bc = {}
    ces = {}
    for a in EVD.ALPHAS:
        ces[a] = cert_ev(D, E, rows, calA_b, D, a, lib, bc)
        log("cert alpha", a)
    Q = EVD.mixture(lib, rows["P4"], rows["gm4"], rows["mid"])
    log("mixture done")
    res = dict(source="SIMULATION", benchmark="SPLIT all-singles, P1+P5+P2 on the assembly, test episodes",
               operating_points=OP, alphas=EVD.ALPHAS, levels=EVD.LEVELS, deltas=DELTAS, pconfs=PCONF, pmins=PMINS,
               id={}, ood={})
    keyvals = {}
    allsel = np.arange(len(rows["z"]))
    faulted = np.flatnonzero((rows["mode"] != 0).any(1))
    for tau in TAUS:
        r, cl_op = run_subset(rows, Q, ces, tau, allsel)
        r["sanity_vs_SPLIT_T1"] = dict(cert_rate=float(cl_op["CERT"]["go"].mean()),
                                       false_go=float(cl_op["CERT"]["go"][rows["z"] < tau].mean()))
        r["faulted_modules"] = {a: score_all(rows, c, tau, faulted) for a, c in cl_op.items()}
        r["next_probe_CERT"] = next_probe_value(rows, ces[OP["cert_alpha"]], tau)
        # bootstrap CIs (assembly clusters) for the headline numbers
        ci = {}
        for a, c in cl_op.items():
            ci[a] = dict(
                informativeness=boot_ci(rows, lambda s: score_all(rows, c, tau, s)["informativeness"], allsel),
                decision_claim_error=boot_ci(rows, lambda s: score_all(rows, c, tau, s)["decision_claim_error"], allsel),
                correct_claim_rate_micro=boot_ci(rows, lambda s: score_all(rows, c, tau, s)["correct_claim_rate_micro"],
                                                 allsel, nboot=100))
        r["ci95"] = ci
        res["id"][str(tau)] = r
        log("id tau", tau, json.dumps(r["sanity_vs_SPLIT_T1"]))
    np.savez_compressed(OUT / "rows_id.npz", row=rows["row"], z=rows["z"], mode=rows["mode"],
                        cert_lb=ces[OP["cert_alpha"]]["lb"], cert_hi=ces[OP["cert_alpha"]]["hi"][:, list(DIRS)],
                        cert_sets=ces[OP["cert_alpha"]]["sets"], cert_bind=ces[OP["cert_alpha"]]["bind"],
                        q_z=Q[:, 14], q_p2=Q[:, list(DIRS)], qlev=np.array(EVD.QLEV))
    # ---- OOD subsets
    for kind in ("payload", "low_mu"):
        p = OUT / f"ood_{kind}.npz"
        if not p.exists():
            log("missing", p)
            continue
        Eo, Do, ro = ood_rows(kind)
        ceo = {a: cert_ev(Do, Eo, ro, calA_b, D, a, lib, bc) for a in EVD.ALPHAS}
        Qo = EVD.mixture(lib, ro["P4"], ro["gm4"], ro["mid"])
        sub = dict(all=np.arange(len(ro["z"])), consequential=np.flatnonzero(ro["ood_cons"]),
                   consequential_healthy_wheels=np.flatnonzero(ro["ood_cons"] & ~(ro["mode"] != 0).any(1)),
                   inconsequential=np.flatnonzero(~ro["ood_cons"]))
        R = dict(n_modules=int(len(ro["z"])), n_consequential=int(ro["ood_cons"].sum()),
                 n_consequential_healthy=int(len(sub["consequential_healthy_wheels"])), by_tau={})
        for tau in TAUS:
            Rt = {}
            for sname, sel in sub.items():
                if len(sel) == 0:
                    continue
                r, cl_op = run_subset(ro, Qo, ceo, tau, sel, with_curves=(sname != "inconsequential"))
                Rt[sname] = r
            # truth shift: how many modules does the OOD condition make unsafe that the in-model twin would not?
            Rt["ood_made_unsafe"] = float(((ro["z"] < tau) & (ro["z_twin"] >= tau)).mean())
            R["by_tau"][str(tau)] = Rt
            log("ood", kind, tau)
        res["ood"][kind] = R
        if kind == "payload":
            res["validation_ood_payload"] = validate(ro, ceo[OP["cert_alpha"]], Qo, 0.8, n=600, label="ood")
    res["validation_id"] = {str(t): validate(rows, ces[OP["cert_alpha"]], Q, t) for t in TAUS}
    log("validation done")
    (OUT / "expl2_results.json").write_text(json.dumps(res, indent=1, default=float))
    log("wrote expl2_results.json")


if __name__ == "__main__":
    main()
