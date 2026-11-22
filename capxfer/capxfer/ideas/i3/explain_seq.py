"""I3: explanation records for probe-until-certified (WS-E claim vocabulary; does not edit capxfer/explain/).

The sequential certificate replaces WS-E's per-stage 'probe' outcome. At every stage the record states
CERTIFIED / REFUTED / UNDETERMINED for the task, the evidence spent, the guarantee that licenses the verdict and,
if undetermined, the next probe from the pre-registered chain (not the heuristic 'widest required direction').
Feeding WS-E `explain()` the per-stage *sequentially calibrated* bounds below makes its go/no_go licensing identical
to the certificate:  lo_leg = q50 - lam_hat (q50 - q05 + eps),  hi_leg = q50 + mu_hat (q95 - q50 + eps).
"""
from __future__ import annotations

import numpy as np

from .seq import COST, EPS
from .extract import chain_views


def seq_bounds(q05, q50, q95, lam_hat, mu_hat):
    return q50 - lam_hat * (q50 - q05 + EPS), q50 + mu_hat * (q95 - q50 + EPS)


def explain_stage(task, tau, alpha, beta, chain, t, lo, hi, n_cal_inf, n_cal_feas, ood_flag=False):
    """lo/hi: sequentially calibrated bounds on the decision legs (P2+, P2-) at stage t of `chain`."""
    spent = sum(COST[a] for a in chain[:t]) - (COST["P2o"] if "P2o" in chain[:t] and "P2" in chain[:t] else 0)
    seen = chain_views(chain)[t]
    lo_z, hi_z = float(np.min(lo)), float(np.min(hi))
    claims, withheld = [], []
    if lo_z >= tau:
        verdict = "certified"
        claims.append(dict(type="capability", subject="strafe(P2+,P2-)", value="reliable", support="supported",
                           numbers=dict(lo=round(lo_z, 2), tau=tau)))
        text = (f"{task}: CERTIFIED after {spent:.0f} s of probing ({'+'.join(seen) or 'no probes'}). "
                f"Strafe ratio is at least {lo_z:.2f} (needs {tau}); the chance that this certificate is issued for "
                f"an assembly that cannot do it is at most {alpha:.0%} under the calibration assumptions "
                f"({n_cal_inf} infeasible calibration trials), whatever stage the robot stopped at.")
    elif hi_z < tau:
        verdict = "refuted"
        claims.append(dict(type="capability", subject="strafe(P2+,P2-)", value="unreliable", support="supported",
                           numbers=dict(hi=round(hi_z, 2), tau=tau)))
        text = (f"{task}: REFUTED after {spent:.0f} s ({'+'.join(seen)}). Strafe ratio is below {hi_z:.2f} "
                f"(needs {tau}); use the alternative maneuver. False refutation rate <= {beta:.0%}.")
    else:
        verdict = "undetermined"
        withheld.append("capability strafe(P2+,P2-)")
        nxt = chain[t] if t < len(chain) else None
        text = (f"{task}: UNDETERMINED after {spent:.0f} s; strafe ratio in [{lo_z:.2f}, {hi_z:.2f}] straddles {tau}. "
                + (f"Next probe: {nxt} ({COST[nxt]:.0f} s)." if nxt else
                   "Probe budget exhausted: abstain (alternative maneuver or ask an operator)."))
    if ood_flag:
        text += " Evidence also looks unlike the calibration data; the guarantee may not hold."
    withheld.append("cause (wheel/floor/payload) unless licensed by the wheel-set contracts")
    return dict(verdict=verdict, claims=claims, withheld=withheld, text=text, stage=t, seen=list(seen),
                seconds=spent, guarantee=dict(false_go=alpha, false_nogo=beta, kind="stop-aware conformal"))
