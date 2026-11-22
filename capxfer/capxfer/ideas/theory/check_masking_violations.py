"""Prop. 2 (masking) outside its assumptions: which simulator (fault, leg) cases exceed the idealised bound, and why.

For every single fault x leg in the I2 sweeps (n = 8..24, kappa_i < 1) classify an exceedance |Delta rho_sim| > bound as
  sat      : some healthy motor saturates (|duty| >= 0.99, sim) in the healthy or faulty run on that leg -> assumption
             'healthy motors unsaturated' broken;
  qs_model : not saturated, and the excess is <= |rho_sim - rho_QS(nominal)| summed over the healthy and faulty runs,
             i.e. within the QS model error (dynamics: finite PI convergence in a 1 s hold, latency);
  roller_drag : not saturated, the nominal QS (c_r = 2.25) itself exceeds the bound (c_r > 0 breaks (I));
  other    : none of the above.
Also reports exceedance rates per n and mode, and the absolute size of the excess.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.theory.check_masking_violations
"""
from __future__ import annotations

import numpy as np

from capxfer.ideas.theory.common import (COAST, BRAKE, ROLLER, WEAK, masking_quantities, rebuild, params,
                                         load_sweeps, dump)

GROUPS = dict(passive=(COAST, BRAKE), roller=(ROLLER,), weak=(WEAK,))


def main():
    P = params()
    R = [r for r in load_sweeps() if r["n"] >= 8]
    recs = []
    for r in R:
        asm = rebuild(r, P)
        rs, rq, du = r["sim"]["rho"], r["qs"]["rho"], r["sim"]["duty"]
        for c, (i, m) in enumerate(r["tags"]):
            if i < 0:
                continue
            mq = masking_quantities(asm, i, m)
            if not np.isfinite(mq["bound"]).all():
                continue
            oth = [j for j in range(r["n"]) if j != i]
            sat = (np.abs(du[c][:, oth]) >= 0.99).any(-1) | (np.abs(du[0]) >= 0.99).any(-1)
            ls = np.abs(rs[0] - rs[c])
            lq = np.abs(rq[0] - rq[c])
            qerr = np.abs(rs[0] - rq[0]) + np.abs(rs[c] - rq[c])
            exc = ls - mq["bound"]
            for leg in range(14):
                cls = "ok"
                if exc[leg] > 0:
                    if sat[leg]:
                        cls = "sat"
                    elif lq[leg] > mq["bound"][leg]:
                        cls = "roller_drag"
                    elif exc[leg] <= qerr[leg]:
                        cls = "qs_model"
                    else:
                        cls = "other"
                recs.append((r["n"], int(m), cls, float(exc[leg]), float(ls[leg] / mq["bound"][leg]), bool(sat[leg])))
    out = {}
    for g, ms in GROUPS.items():
        G = [x for x in recs if x[1] in ms]
        d = {}
        for n in sorted({x[0] for x in G}):
            H = [x for x in G if x[0] == n]
            cnt = {k: sum(x[2] == k for x in H) for k in ("ok", "sat", "qs_model", "roller_drag", "other")}
            ex = [x for x in H if x[2] != "ok"]
            d[n] = dict(fault_legs=len(H), exceed_frac=1 - cnt["ok"] / len(H),
                        exceed_frac_unsat_legs=sum(x[2] != "ok" and not x[5] for x in H) / max(1, sum(not x[5] for x in H)),
                        by_class=cnt,
                        max_abs_excess=max((x[3] for x in ex), default=0.0),
                        max_abs_excess_unsat=max((x[3] for x in ex if not x[5]), default=0.0),
                        max_ratio=max(x[4] for x in H))
        out[g] = d
    allx = [x for x in recs if x[2] != "ok"]
    out["summary"] = dict(fault_legs=len(recs), exceed=len(allx),
                          by_class={k: sum(x[2] == k for x in allx) for k in ("sat", "qs_model", "roller_drag", "other")},
                          max_abs_excess_unsat=max((x[3] for x in allx if not x[5]), default=0.0))
    dump("check_masking_violations.json", out)
    print(out["summary"])
    for g in GROUPS:
        for n, v in out[g].items():
            print(g, n, round(v["exceed_frac"], 3), round(v["exceed_frac_unsat_legs"], 3), v["by_class"],
                  round(v["max_abs_excess_unsat"], 4), round(v["max_ratio"], 2))


if __name__ == "__main__":
    main()
