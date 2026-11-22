"""CHOICE: predicted arm outcomes for the physical split-then-dock demo setups (D11) (SIMULATION ONLY).

For each candidate setup (physical pair or L3, a planned injected fault placed on a module drawn uniformly per run,
as a sealed schedule would), B trials at theta ~ DR_WIDE:
  1. the ASSEMBLY runs the probe library (WS-A sim + sensors, WS-B featurize, rung E1, P1+P5+P2 view);
  2. structured_ensemble -> SPLIT certificate (I1 sparse wheel sets calibrated on calA, projected per module, union
     of library stage-2 bands) = per-module lower bound on z = min(rho_P2+, rho_P2-);
     ensemble -> current-body q05 of z, one-sided CQR on calA current-body truth (black-box carry-over);
  3. truth: noise-free z of every module alone at the trial's theta and health (as SPLIT truth_singles).
Arms are scored by choice.evaluate (dock success from the dock-sim model).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.choice.demo [B]
Writes ~/.cache/capxfer/CHOICE/demo_preds.npz.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from capxfer import assembly as A, featurize, sensors, sim
from capxfer.ideas.i1 import stage1 as S1
from capxfer.ideas.split import common as C, evaluate as EV, sims as SM
from capxfer.learn import c2_eval, data as LD
from capxfer.probes import PIDS

OUT = C.OUT.parent / "CHOICE"
COAST, ROLLER, WEAK = 2, 4, 1
# setup name -> (assembly, list of (n_modules_faulted, per-module fault kind))
SETUPS = {
    "pair_F4x2d": ("pair", ["F4x2d"]),
    "pair_F2x2d": ("pair", ["F2x2d"]),
    "pair_F2": ("pair", ["F2"]),
    "pair_F4": ("pair", ["F4"]),
    "pair_F2|F2": ("pair", ["F2", "F2"]),
    "L3_F4x2d": ("L3", ["F4x2d"]),
    "L3_F2x2d": ("L3", ["F2x2d"]),
    "L3_F2|F2": ("L3", ["F2", "F2"]),
    "L3_F4|F4": ("L3", ["F4", "F4"]),
    "L3_F2|F4": ("L3", ["F2", "F4"]),
}


def module_fault(kind, rng):
    m = np.zeros(4, np.int64)
    if kind == "F2":
        m[rng.integers(4)] = COAST
    elif kind == "F4":
        m[rng.integers(4)] = ROLLER
    elif kind == "F2x2d":
        m[[0, 3]] = COAST
    elif kind == "F4x2d":
        m[[0, 3]] = ROLLER
    return m


def calibration(alpha):
    """stage-1 thresholds, the library and the black-box carry-over shift at total level alpha (alpha1 = alpha2)."""
    EV.ALPHA1 = EV.ALPHA2 = alpha / 2
    EV.ALPHA = alpha
    E = C.episodes("P1P5P2")
    Gs = C.split_groups(E)
    calA = Gs["calA"]
    D = S1.load("structured_ensemble", "P1P5P2")
    Tw = S1.calibrate(D, np.array([E[k]["b"] for k in calA]), alpha / 2, K=EV.K, seed=0)
    Pe = C.load_pred("ensemble", "P1P5P2")
    qz = np.quantile(EV.zmin(Pe["ys"]), 0.05, 1)
    pre_b = np.array([E[k]["b"] for k in calA])
    pre_z = np.array([EV.zmin(E[k]["rho"]) for k in calA])
    t_bb = EV.cq(qz[pre_b] - pre_z, alpha)
    return Tw, EV.Library(), t_bb


def main(B=256):
    SM.setup()
    c2_eval.MEM_FRAC = 0.1
    prm = C.params()
    cal = {a: calibration(a) for a in (0.10, 0.05)}
    models = {m: c2_eval.load_model(m, device="cuda", tag="W") for m in ("structured_ensemble", "ensemble")}
    rng = np.random.default_rng(20261121)
    out = {}
    t0 = time.time()
    for name, (aname, faults) in SETUPS.items():
        asm = A.physical(aname, prm)
        nm = len(asm.modules)
        mids = [m[0] for m in asm.modules]
        dom = sim.sample_domain(B, rng, sim.DR_WIDE)
        mode = np.zeros((B, 4 * nm), np.int64)
        fmod = np.zeros((B, nm), bool)
        for b in range(B):
            ks = rng.permutation(nm)[:len(faults)]
            for k, f in zip(ks, faults):
                mode[b, 4 * k:4 * k + 4] = module_fault(f, rng)
                fmod[b, k] = True
        g = np.ones((B, 4 * nm), np.float32)
        R = sim.simulate_multi([asm], prm, [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in dom.items()}],
                               [torch.tensor(mode)], [torch.tensor(g)], device="cuda", engine="graph",
                               store_device="cpu", seed=int(rng.integers(1 << 30)))[0]
        raw = sensors.to_raw(R, prm, dom, rng)
        F = LD.from_featurize(featurize.featurize(raw, "E3", PIDS))
        F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
        view = LD.apply_view(F, "E1", ("P1", "P5", "P2"))
        with torch.no_grad():
            pse = models["structured_ensemble"].predict(view, 64)
            pe = models["ensemble"].predict(view, 64)
        P = np.nan_to_num(pse["mode_prob"].float().cpu().numpy().astype(float))
        ys = pe["y_samples"].float().cpu().numpy()
        qz_body = np.quantile(np.minimum(ys[..., 2], ys[..., 3]), 0.05, 1)
        del R, raw, F
        # truth: each module alone at the trial's theta / health
        z = np.zeros((B, nm))
        for k, mid in enumerate(mids):
            d = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in dom.items()}
            rho = SM.run_multi([C.single_asm(mid, prm)], [d], [torch.tensor(mode[:, 4 * k:4 * k + 4])],
                               [torch.tensor(g[:, 4 * k:4 * k + 4])], seed=int(rng.integers(1 << 30)))[0]
            z[:, k] = EV.zmin(rho)
        rec = dict(z=z, fmod=fmod, mode=mode, qz_body=qz_body, ypos=np.asarray(asm.module_pos)[:, 1],
                   mids=np.array(mids))
        Dn = dict(n=np.full(B, asm.n), P=P)
        for a, (Tw, lib, t_bb) in cal.items():
            lb = np.zeros((B, nm))
            for b in range(B):
                S = S1.sets_for(Dn, b, Tw, "sparse")
                for k, mid in enumerate(mids):
                    EV.ALPHA1 = EV.ALPHA2 = a / 2
                    lb[b, k] = lib.comp(mid, EV.projected_mvs(S, np.arange(4 * k, 4 * k + 4)))[0]
            rec[f"lb_cert_{a}"] = lb
            rec[f"lb_bb_{a}"] = qz_body - t_bb
        for k, v in rec.items():
            out[f"{name}|{k}"] = v
        print(f"[{time.time() - t0:.0f}s] {name}: z<0.5 frac {np.mean(z < .5):.2f}  cert(0.10)>=0.5 "
              f"{np.mean(rec['lb_cert_0.1'] >= .5):.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "demo_preds.npz", setups=np.array(list(SETUPS)), **out)
    print("wrote demo_preds.npz")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
