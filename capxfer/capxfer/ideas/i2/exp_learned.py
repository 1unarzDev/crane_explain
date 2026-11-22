"""I2: does the excitation theory predict WHERE the learned diagnoser fails? Uses WS-C's cached per-trial preds
(structured_ensemble, model A, dev_v2) and the geometry extracted by devmeta.py."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from capxfer.ideas.i2 import theory as T

C = os.path.expanduser("~/.cache/capxfer/C")


class Geo(SimpleNamespace):
    @property
    def n(self):
        return len(self.wheel_pos)

    def char_length(self):
        return self._lc


def splits(groups):
    size = lambda G: int(G["family"].split("_")[0][1:]) if G["family"].startswith("k") else None
    for G in groups:
        if G.get("asm_type") not in (None, "random"):
            G["family"] = "phys_" + str(G["asm_type"])
    rnd = [G for G in groups if not G["family"].startswith("phys_")]
    small = [G for G in rnd if (size(G) or 99) <= 3]
    return dict(cal=[G for i, G in enumerate(small) if i % 3], heldout=[G for G in rnd if (size(G) or 0) >= 4],
                physical=[G for G in groups if G["family"].startswith("phys_") and G["ood"] == "none"])


def merged(m):
    m = np.asarray(m).copy()
    m[m == 3] = 2
    return m


def main(method="structured_ensemble", suffix="", dev="dev_v2", eta_list=(0.01, 0.02, 0.03, 0.05)):
    groups = torch.load(os.path.expanduser(f"~/.cache/capxfer/I2/devmeta_{dev}.pt"), weights_only=False)
    sp = splits(groups)
    preds = {tag: torch.load(f"{C}/preds_{method}_{tag}{suffix}.pt", weights_only=False) for tag in ("P1P5", "P1P5P2", "all")}
    p0 = preds["P1P5"]
    # rebuild per-trial excitation in the preds' trial order
    rows = []
    off = 0
    sets = np.asarray(p0["set"])
    for sname in ("cal", "heldout", "physical"):
        idx = np.where(sets == sname)[0]
        k = 0
        for gi, G in enumerate(sp[sname]):
            B, n = G["mode"].shape
            sl = idx[k:k + B]
            k += B
            mt = p0["mode_true"][sl, :n].numpy()
            assert (mt == G["mode"]).all(), (sname, gi)
            geo = Geo(wheel_pos=G["wheel_pos"], w_hat=G["w_hat"], r_hat=G["r_hat"], _lc=G["char_length"])
            E = T.excitation(geo, ("P1", "P5", "P2"))                    # (n,5,3)
            for b in range(B):
                for w in np.where(G["mode"][b] > 0)[0]:
                    m = int(G["mode"][b, w])
                    rows.append(dict(set=sname, fam=G["family"], n=n, trial=int(sl[b]), wheel=int(w), mode=m,
                                     e15=float(E[w, m, :2].max()), e2=float(E[w, m, 2]),
                                     nfault=int((G["mode"][b] > 0).sum()),
                                     drho=float(1 - G["rho"][b].min())))
        assert k == len(idx)
    res = {}
    for tag, p in preds.items():
        am = p["mode_prob"].argmax(-1).numpy()
        for r in rows:
            r[f"ok_{tag}"] = bool(merged(am[r["trial"], r["wheel"]]) == merged(r["mode"]))
    for sname in ("heldout", "physical", "cal"):
        rs = [r for r in rows if r["set"] == sname]
        out = {}
        for eta in eta_list:
            for mode_sel, ms in (("roller", (4,)), ("drive", (1, 2, 3))):
                sel = [r for r in rs if r["mode"] in ms]
                lo = [r for r in sel if r["e15"] < eta]
                hi = [r for r in sel if r["e15"] >= eta]
                f = lambda L, t: (float(np.mean([r[f"ok_{t}"] for r in L])) if L else None, len(L))
                out[f"eta{eta}/{mode_sel}"] = dict(
                    unexcited_P1P5=dict(acc_P1P5=f(lo, "P1P5"), acc_P1P5P2=f(lo, "P1P5P2"), acc_all=f(lo, "all")),
                    excited_P1P5=dict(acc_P1P5=f(hi, "P1P5"), acc_P1P5P2=f(hi, "P1P5P2"), acc_all=f(hi, "all")))
        # share of P1P5 errors that the theory flags as unexcited (eta 0.03), and share fixed by P2
        err = [r for r in rs if not r["ok_P1P5"]]
        out["n_faulty_wheels"] = len(rs)
        out["n_err_P1P5"] = len(err)
        out["err_flagged_unexcited_eta0.03"] = float(np.mean([r["e15"] < 0.03 for r in err])) if err else None
        out["err_roller_share"] = float(np.mean([r["mode"] == 4 for r in err])) if err else None
        fixed = [r for r in err if r["ok_P1P5P2"]]
        out["P2_fixed"] = len(fixed)
        out["P2_fixed_flagged_unexcited_eta0.03"] = float(np.mean([r["e15"] < 0.03 for r in fixed])) if fixed else None
        out["frac_roller_unexcited_eta0.03"] = float(np.mean([r["e15"] < 0.03 for r in rs if r["mode"] == 4]))
        res[sname] = out
    od = os.path.expanduser(f"~/.cache/capxfer/I2/learned_link_{method}{suffix}.json")
    json.dump(res, open(od, "w"), indent=1)
    torch.save(rows, od.replace(".json", "_rows.pt"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
