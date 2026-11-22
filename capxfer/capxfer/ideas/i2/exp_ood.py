"""I2 Prop. 6 (consistency): payload / friction shifts move capability only through the constraint inconsistency of
faulted wheels. QS prediction of the paired OOD effect dy vs WS-A's paired-seed study (ood_study_v3_trials.npz)."""
import json
import os

import numpy as np

from capxfer import assembly as A
from capxfer.ideas.i2 import theory as T


def main():
    P = A.load_params("params/module_measured.yaml")
    z = np.load(os.path.expanduser("~/.cache/capxfer/A/ood_study_v3_trials.npz"), allow_pickle=True)
    out = {}
    for kind in ("payload_lifter", "payload"):
        for arm in ("healthy", "faults"):
            for typ in ("single", "pair", "L3", "square4"):
                pre = f"{kind}|{arm}|{typ}|"
                if pre + "dy" not in z.files:
                    continue
                asm = A.physical(typ, P)
                mode = z[pre + "mode"]
                keep = ~(mode == 1).any(1)                                   # g of weak wheels not stored
                mode = mode[keep][:120]
                dy = z[pre + "dy"][keep][:120]
                pm, px, py = (z[pre + k][keep][:120] for k in ("payload_mass", "payload_x", "payload_y"))
                legs = T.legs(asm)
                gs = np.ones(mode.shape)
                xi0, _, _ = T.qs_solve(asm, P, mode, gs, legs)
                r0 = T.rho_of(xi0, legs, asm.char_length())
                N = np.stack([T.normal_loads(asm, payload=(float(pm[b]), (float(px[b]), float(py[b]))))
                              for b in range(len(mode))])
                xi1, _, _ = T.qs_solve(asm, P, mode, gs, legs, N=N)
                r1 = T.rho_of(xi1, legs, asm.char_length())
                dq = r1 - r0
                a, b_ = np.abs(dy).max(1), np.abs(dq).max(1)
                faulted = (mode > 0).any(1)
                out[pre[:-1]] = dict(n=int(len(a)), sim_max_abs_dy_p90=float(np.quantile(a, .9)),
                                     qs_max_abs_dy_p90=float(np.quantile(b_, .9)),
                                     corr_signed=float(np.corrcoef(dy.ravel(), dq.ravel())[0, 1]) if a.max() > 1e-3 else None,
                                     sim_healthy_trials_p90=float(np.quantile(a[~faulted], .9)) if (~faulted).any() else None,
                                     frac_gt_0p1_faulted=float((a[faulted] > 0.1).mean()) if faulted.any() else None,
                                     frac_gt_0p1_qs_faulted=float((b_[faulted] > 0.1).mean()) if faulted.any() else None)
                print(pre, out[pre[:-1]], flush=True)
    json.dump(out, open(os.path.expanduser("~/.cache/capxfer/I2/ood_consistency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
