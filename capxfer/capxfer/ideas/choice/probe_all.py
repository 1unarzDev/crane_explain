"""CHOICE: 'split, then probe every module' (SIMULATION ONLY) on COMPLETE episodes.

SPLIT's probe_after.npz covers a module sample, not whole episodes, so the split-then-probe-all policy needs every
module of an episode probed. For a size-stratified sample of faulted test episodes, every post-split single module
re-runs P1+P5+P2 alone at the episode's true plant and health (WS-A sim + sensors, WS-B featurize, rung E1) and the
W ensemble predicts q05 of z. The one-sided conformal shift is SPLIT's: fitted on SPLIT's calA post-split rows
(probe_after.npz, first `ncal` rows, ensemble|P1P5P2|q05z). Same model, view and calibration rows as SPLIT arm (f).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.choice.probe_all [n_per_size]
Writes ~/.cache/capxfer/CHOICE/probe_all.npz (rows into truth_singles, q05z).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from capxfer import featurize, sensors, sim
from capxfer.ideas.split import common as C, sims as SM
from capxfer.learn import c2_eval, data as LD
from capxfer.probes import PIDS

OUT = C.OUT.parent / "CHOICE"


def episodes_to_run(n_per_size=600, seed=5):
    E = C.episodes("P1P5P2")
    Gs = C.split_groups(E)
    test = np.setdiff1d(np.arange(len(E)), Gs["calA"])
    faulted = np.array([k for k in test if (E[k]["mode"] > 0).any()])
    rng = np.random.default_rng(seed)
    pick = []
    for nm in range(2, 7):
        ks = faulted[np.array([E[k]["nmod"] for k in faulted]) == nm]
        pick += list(rng.choice(ks, min(n_per_size, len(ks)), replace=False))
    return E, np.sort(np.array(pick))


def main(n_per_size=600):
    SM.setup()
    c2_eval.MEM_FRAC = 0.1
    E, eps = episodes_to_run(n_per_size)
    T = np.load(C.OUT / "truth_singles.npz")
    rows = np.flatnonzero(np.isin(T["ep"], eps))
    print(f"{len(eps)} episodes, {len(rows)} modules", flush=True)
    prm = C.params()
    model = c2_eval.load_model("ensemble", device="cuda", tag="W")
    q05z = np.zeros(len(rows))
    rng = np.random.default_rng(17)
    t0 = time.time()
    mids = np.array([E[T["ep"][r]]["modules"][T["mod"][r]][0] for r in rows])
    for mid in C.MIDS:
        sel = np.flatnonzero(mids == mid)
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
            view = LD.apply_view(F, "E1", ("P1", "P5", "P2"))
            with torch.no_grad():
                p = model.predict(view, 64)
            ys = p["y_samples"].float().cpu().numpy()
            q05z[ii] = np.quantile(np.minimum(ys[..., 2], ys[..., 3]), 0.05, 1)
            del R, raw, F
            print(f"[{time.time() - t0:.0f}s] {mid} {s + len(ii)}/{len(sel)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "probe_all.npz", rows=rows, eps=eps, q05z=q05z)
    print("wrote probe_all.npz")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
