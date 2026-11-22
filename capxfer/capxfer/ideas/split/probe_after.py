"""(f) 'split first, then probe each module' (SIMULATION ONLY).

Each post-split single module re-runs the probe library alone at the episode's true plant and health (WS-A sim +
sensors, WS-B featurize, rung E1, pids P1+P5 or P1+P5+P2), the W-generation ensemble / structured_ensemble predicts its
capability, and a one-sided conformal lower bound on z = min(rho_P2+-) is calibrated on calA post-split modules
(one random module per calA episode, the same rows as evaluate.py). Writes ~/.cache/capxfer/SPLIT/probe_after.npz.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.split.probe_after [n_test_rows]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from capxfer import featurize, sensors, sim
from capxfer.ideas.split import common as C, evaluate as EV, sims as SM
from capxfer.learn import c2_eval, data as LD
from capxfer.probes import PIDS

VIEWS = {"P1P5": ("P1", "P5"), "P1P5P2": ("P1", "P5", "P2")}
MODELS = ("ensemble", "structured_ensemble")


def rows_to_run(n_test, seed=3):
    E = C.episodes("P1P5")
    Gs = C.split_groups(E)
    T = np.load(C.OUT / "truth_singles.npz")
    ep = T["ep"]
    rowsets = {}
    for r, k in enumerate(ep):
        rowsets.setdefault(int(k), []).append(r)
    calrows = np.array([EV.C_rng_pick(rowsets[k], k) for k in Gs["calA"]])
    test = np.flatnonzero(~np.isin(ep, Gs["calA"]))
    rng = np.random.default_rng(seed)
    # oversample faulted modules (they carry all unsafe cases); weights recorded for re-weighting
    nf = np.array([(E[k]["mode"][4 * m:4 * m + 4] > 0).sum() for k, m in zip(ep, T["mod"])])
    fault = test[nf[test] > 0]
    healthy = test[nf[test] == 0]
    nh = min(len(healthy), n_test // 3)
    nfa = min(len(fault), n_test - nh)
    pick = np.concatenate([rng.choice(fault, nfa, replace=False), rng.choice(healthy, nh, replace=False)])
    w = np.where(nf[pick] > 0, len(fault) / nfa, len(healthy) / nh)
    return E, T, calrows, np.sort(pick), dict(zip(pick.tolist(), w.tolist()))


def main(n_test=9000):
    SM.setup()
    c2_eval.MEM_FRAC = 0.1
    E, T, calrows, test, wmap = rows_to_run(n_test)
    rows = np.concatenate([calrows, test])
    prm = C.params()
    models = {m: c2_eval.load_model(m, device="cuda", tag="W") for m in MODELS}
    out = {f"{m}|{v}|{k}": np.zeros((len(rows),) + s) for m in MODELS for v in VIEWS
           for k, s in (("q05z", ()), ("yq", (14, 3)))}
    rng = np.random.default_rng(11)
    t0 = time.time()
    for mid in C.MIDS:
        sel = np.flatnonzero(np.array([E[T["ep"][r]]["modules"][T["mod"][r]][0] == mid for r in rows]))
        for s in range(0, len(sel), 1024):
            ii = sel[s:s + 1024]
            rr = rows[ii]
            th = [E[T["ep"][r]]["theta"] for r in rr]
            mode = torch.tensor(np.stack([E[T["ep"][r]]["mode"][4 * T["mod"][r]:4 * T["mod"][r] + 4] for r in rr]))
            g = torch.tensor(np.stack([E[T["ep"][r]]["g"][4 * T["mod"][r]:4 * T["mod"][r] + 4] for r in rr]))
            dom = SM.dom_from(th)
            R = sim.simulate_multi([C.single_asm(mid, prm)], prm, [dom], [mode], [g], device="cuda", engine="graph",
                                   store_device="cpu", seed=int(rng.integers(1 << 30)))[0]
            raw = sensors.to_raw(R, prm, dom, rng)
            F = LD.from_featurize(featurize.featurize(raw, "E3", PIDS))
            F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
            for v, pids in VIEWS.items():
                view = LD.apply_view(F, "E1", pids)
                for m, model in models.items():
                    with torch.no_grad():
                        p = model.predict(view, 64)
                    ys = p["y_samples"].float().cpu().numpy()
                    out[f"{m}|{v}|q05z"][ii] = np.quantile(np.minimum(ys[..., 2], ys[..., 3]), 0.05, 1)
                    out[f"{m}|{v}|yq"][ii] = p["y_q"].float().cpu().numpy()
            del R, raw, F
            print(f"[{time.time() - t0:.0f}s] {mid} {s + len(ii)}/{len(sel)}", flush=True)
    np.savez_compressed(C.OUT / "probe_after.npz", rows=rows, ncal=len(calrows),
                        w=np.array([wmap.get(int(r), 0.0) for r in rows]), **out)
    print("wrote probe_after.npz")



def evaluate(p_pre="P1P5"):
    """(f) metrics (weighted back to the test distribution) next to the pre-split methods on the SAME module rows."""
    import json
    from capxfer.probes import Schedule
    A = np.load(C.OUT / "probe_after.npz")
    S = np.load(C.OUT / f"rows_singles_{p_pre}.npz")
    rows, ncal, w = A["rows"], int(A["ncal"]), A["w"]
    z = S["z"][rows]
    cal, te = np.arange(ncal), np.arange(ncal, len(rows))
    prm = C.params()
    Lc1 = C.single_asm("M1", prm).char_length()
    out = dict(pre_pids=p_pre, n_test=int(len(te)), n_cal=ncal, alpha=EV.ALPHA,
               probe_time_s={v: Schedule(Lc1, pids).T for v, pids in VIEWS.items()}, rows={})
    lbs = {f"pa_{m}_{v}": A[f"{m}|{v}|q05z"] - EV.cq(A[f"{m}|{v}|q05z"][cal] - z[cal], EV.ALPHA)
           for m in MODELS for v in VIEWS}
    for m in EV.METHODS:
        lbs[m] = S[f"lb_{m}"][rows]
    for tau in EV.TAUS:
        for m, lb in lbs.items():
            go, uns, ww = lb[te] >= tau, z[te] < tau, w[te]
            out["rows"][f"{tau}|{m}"] = dict(
                unsafe_go=float((ww * (go & uns)).sum() / ww.sum()), false_go=float(go[uns].mean()),
                cert=float((ww * go).sum() / ww.sum()), useful=float((ww * (go & ~uns)).sum() / (ww * ~uns).sum()),
                lb_cov=float((ww * (z[te] >= lb[te])).sum() / ww.sum()), n_unsafe=int(uns.sum()))
    (C.OUT / f"probe_after_eval_{p_pre}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["probe_time_s"]))
    for k, v in out["rows"].items():
        print(k, " ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))


if __name__ == "__main__":
    if sys.argv[1:2] == ["eval"]:
        evaluate(*sys.argv[2:3])
    else:
        main(*(int(a) for a in sys.argv[1:2]))
