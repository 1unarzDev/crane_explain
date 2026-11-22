"""Prop. 3 (masking) check.

(a) Rigorous bound, idealised model (c_r = 0): QS re-solved with c_roller = 0 on a subset; every (fault, leg) with
    kappa_i < 1 must satisfy |Delta rho| <= bound (up to solver tolerance).
(b) Measured curve (full simulator, sweeps n = 4..24, nominal c_r): worst-leg loss per single fault vs bound and
    first-order estimate; fitted constant C in median loss = C / (n - 1) and free log-log slope.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.theory.check_masking
"""
from __future__ import annotations

import numpy as np
import torch

from capxfer.ideas.i2 import theory as T
from capxfer.ideas.i2.exp_sweep import configs
from capxfer.ideas.theory.common import (COAST, BRAKE, ROLLER, WEAK, MODE_NAME, masking_quantities, rebuild, params,
                                         load_sweeps, dump)

torch.set_num_threads(4)
GROUPS = dict(passive=(COAST, BRAKE), roller=(ROLLER,), weak=(WEAK,))


def part_a(R, P, per_n=3):
    """Idealised-model verification of the rigorous bound on a subset of sweep assemblies."""
    rows = []
    seen = {}
    for r in R:
        if seen.get(r["n"], 0) >= per_n:
            continue
        seen[r["n"]] = seen.get(r["n"], 0) + 1
        asm = rebuild(r, P)
        modes, gs, tags = configs(asm.n)
        xi, _, du = T.qs_solve(asm, P, modes, gs, T.legs(asm), dom=dict(c_roller=0.0))
        res = T.qs_solve.last_resid.max()
        rho = T.rho_of(xi, T.legs(asm), asm.char_length())
        for c, (i, m) in enumerate(tags):
            if i < 0:
                continue
            mq = masking_quantities(asm, i, m)
            d = np.abs(rho[c] - rho[0])
            oth = [j for j in range(asm.n) if j != i]
            sat = (np.abs(du[c][:, oth]) >= 0.999).any(-1) | (np.abs(du[0]) >= 0.999).any(-1)   # (14,) motor sat
            rr = d / mq["bound"]
            rows.append(dict(n=asm.n, mode=int(m), kappa=mq["kappa"], drho_max=float(d.max()),
                             ratio=float(rr[~sat].max()) if (~sat).any() else 0.0,
                             ratio_incl_sat=float(rr.max()), n_sat_legs=int(sat.sum()), resid=float(res),
                             healthy_dev=float(np.abs(rho[0] - 1).max()),
                             healthy_dev_unsat=float(np.abs(rho[0] - 1)[~(np.abs(du[0]) >= 0.999).any(-1)].max())))
        print("a", asm.name, asm.n, flush=True)
    k = np.array([x["kappa"] for x in rows])
    ratio = np.array([x["ratio"] for x in rows])
    ok = k < 1
    ri = np.array([x["ratio_incl_sat"] for x in rows])
    return dict(n_faults=len(rows), n_kappa_lt_1=int(ok.sum()),
                note="ratio over legs where no healthy motor saturates (Prop. 3 assumption)",
                max_ratio_measured_over_bound=float(ratio[ok].max()),
                median_ratio_measured_over_bound=float(np.median(ratio[ok])),
                frac_bound_holds=float((ratio[ok] <= 1 + 1e-3).mean()),
                incl_saturated_legs_frac_holds=float((ri[ok] <= 1 + 1e-3).mean()),
                incl_saturated_legs_max_ratio=float(ri[ok].max()),
                frac_faults_with_saturated_leg=float(np.mean([x["n_sat_legs"] > 0 for x in rows])),
                healthy_rho_minus_1_max=float(max(x["healthy_dev"] for x in rows)),
                healthy_rho_minus_1_max_unsat_legs=float(max(x["healthy_dev_unsat"] for x in rows)),
                max_qs_resid=float(max(x["resid"] for x in rows)),
                by_n={int(n): dict(kappa_med=float(np.median(k[[x["n"] == n for x in rows]])),
                                   frac_kappa_lt_1=float(ok[[x["n"] == n for x in rows]].mean()))
                      for n in sorted({x["n"] for x in rows})})


def part_b(R, P):
    recs = []
    for r in R:
        asm = rebuild(r, P)
        rs, rq = r["sim"]["rho"], r["qs"]["rho"]
        for c, (i, m) in enumerate(r["tags"]):
            if i < 0:
                continue
            mq = masking_quantities(asm, i, m)
            ls, lq = rs[0] - rs[c], rq[0] - rq[c]
            du = r["sim"]["duty"]
            oth = [j for j in range(r["n"]) if j != i]
            sat = (np.abs(du[c][:, oth]) >= 0.99).any(-1) | (np.abs(du[0]) >= 0.99).any(-1)
            viol = np.abs(ls) - mq["bound"]
            recs.append(dict(n=r["n"], mode=int(m), kappa=mq["kappa"], sim=float(ls.max()), qs=float(lq.max()),
                             bound=float(mq["bound"].max()), first=float(mq["first_order"].max()),
                             legviol=float(np.max(viol)),
                             legviol_unsat=float(np.max(viol[~sat])) if (~sat).any() else -1.0,
                             ratio_unsat=float(np.max((np.abs(ls) / mq["bound"])[~sat])) if (~sat).any() else 0.0))
    out = {}
    for gname, ms in GROUPS.items():
        G = [x for x in recs if x["mode"] in ms]
        ns = sorted({x["n"] for x in G})
        by = {}
        for n in ns:
            g = [x for x in G if x["n"] == n]
            arr = lambda k: np.array([x[k] for x in g])
            fin = np.isfinite(arr("bound"))
            by[n] = dict(count=len(g), sim_med=float(np.median(arr("sim"))), sim_p90=float(np.quantile(arr("sim"), .9)),
                         qs_med=float(np.median(arr("qs"))),
                         first_order_med=float(np.median(arr("first"))),
                         frac_kappa_lt_1=float(fin.mean()),
                         bound_med=float(np.median(arr("bound")[fin])) if fin.any() else None,
                         sim_over_bound_med=float(np.median(arr("sim")[fin] / arr("bound")[fin])) if fin.any() else None,
                         sim_over_bound_max=float(np.max(arr("sim")[fin] / arr("bound")[fin])) if fin.any() else None,
                         frac_sim_le_bound=float((arr("legviol")[fin] <= 0).mean()) if fin.any() else None,
                         frac_sim_le_bound_unsat_legs=float((arr("legviol_unsat")[fin] <= 0).mean()) if fin.any() else None,
                         sim_over_bound_unsat_max=float(arr("ratio_unsat")[fin].max()) if fin.any() else None,
                         sim_over_bound_unsat_med=float(np.median(arr("ratio_unsat")[fin])) if fin.any() else None,
                         sim_over_first_med=float(np.median(arr("sim") / np.maximum(arr("first"), 1e-9))))
        nn = np.array([n for n in ns if n >= 8], float)
        med = np.array([by[int(n)]["sim_med"] for n in nn])
        slope, icpt = np.polyfit(np.log(nn), np.log(med), 1)
        # fixed-slope fits: median = C/(n-1) and C/n (least squares in log space)
        C1 = float(np.exp(np.mean(np.log(med) + np.log(nn - 1))))
        Cn = float(np.exp(np.mean(np.log(med) + np.log(nn))))
        rel = (C1 / (nn - 1) - med) / med
        out[gname] = dict(by_n=by, loglog_slope_n_ge_8=float(slope), C_over_nm1=C1, C_over_n=Cn,
                          fit_rel_err_over_nm1={int(n): float(e) for n, e in zip(nn, rel)})
    return out


def main():
    P = params()
    R = load_sweeps()
    res = dict(idealised=part_a(R, P), measured=part_b(R, P))
    dump("check_masking.json", res)
    print(res["idealised"])
    for g, v in res["measured"].items():
        print(g, "slope", round(v["loglog_slope_n_ge_8"], 3), "C/(n-1)", round(v["C_over_nm1"], 4))
        for n, b in v["by_n"].items():
            print("  ", n, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in b.items()})


if __name__ == "__main__":
    main()
