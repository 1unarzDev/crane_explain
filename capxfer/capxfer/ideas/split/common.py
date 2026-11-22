"""Episodes and splits for the SPLIT benchmark (SIMULATION ONLY).

An episode is one devW trial of a >= 2-module assembly (pair / L3 / square4 physical types, random polyomino families
of 2-6 modules with random module yaws) with its in-model injected faults (healthy / single / double; modes weak, coast,
brake, roller-locked) and its true plant theta. The robot ran the probe library on the ASSEMBLY (the WS-C W-model
predictions on devW are exactly that pre-split evidence). The assembly then splits:
  singles : every module drives alone (in its own module frame, yaw 0, its own 4 wheels and faults)
  off1    : one leaf module (its removal leaves the rest connected; chosen per assembly with a fixed seed) splits off;
            sub-assemblies = that module alone + the remainder (original relative layout and yaws).
Truth = noise-free capability ratios of each sub-assembly simulated at the episode's true plant and health.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from capxfer import assembly as A, datagen
from capxfer.ideas.i1 import mapping

OUT = pathlib.Path.home() / ".cache/capxfer/SPLIT"
PRED = "~/.cache/capxfer/C/preds_{m}_{p}_W_devW.pt"
STRAFE = (2, 3)                          # TARGET_KEYS P2+, P2-
MIDS = ("M1", "M2", "M3", "M4")
PLANT_KEYS = ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale", "mass_scale",
              "izz_scale", "radius_scale", "kp_scale", "ki_scale", "v_batt", "cg_dx", "cg_dy")


def params():
    return A.load_params(datagen.PARAMS)


def load_pred(m, p):
    d = torch.load(PRED.format(m=m, p=p), weights_only=False)
    P = d["mode_prob"]
    return dict(P=None if P is None else np.nan_to_num(P.numpy().astype(float)), yq=d["y_q"].numpy().astype(float),
                ys=d["y_samples"].numpy().astype(np.float32), g_mean=d["g_mean"].numpy(),
                tid=list(d["trial_id"]))


def connected(cells):
    cells = set(cells)
    if not cells:
        return False
    st = [next(iter(cells))]
    seen = {st[0]}
    while st:
        x, y = st.pop()
        for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if c in cells and c not in seen:
                seen.add(c)
                st.append(c)
    return len(seen) == len(cells)


def episodes(p="P1P5"):
    """Episode table (all multi-module devW trials) in the WS-C prediction row order."""
    d = torch.load(PRED.format(m="structured_ensemble", p=p), weights_only=False)
    G, gmap = mapping.build(d)
    gid = np.array(d["group_id"])
    within = np.zeros(len(gid), int)
    for u in np.unique(gid):
        idx = np.flatnonzero(gid == u)
        within[idx] = np.arange(len(idx))
    rng = np.random.default_rng(12345)
    leaf = {}
    for gi, g in enumerate(G):
        mods = g["assembly"]["modules"]
        if len(mods) < 2:
            continue
        cells = [(m[1], m[2]) for m in mods]
        ok = [k for k in range(len(mods)) if connected(cells[:k] + cells[k + 1:])]
        leaf[gi] = int(rng.choice(ok))
    rows = []
    for b in range(len(gid)):
        g = G[gmap[b]]
        mods = g["assembly"]["modules"]
        if len(mods) < 2:
            continue
        i = within[b]
        n = 4 * len(mods)
        rows.append(dict(b=b, gi=int(gmap[b]), i=int(i), set=d["set"][b], fam=d["family"][b], nmod=len(mods),
                         n=n, modules=[tuple(m) for m in mods], mode=g["mode"][i].astype(np.int64),
                         g=g["g"][i].astype(np.float32), rho=g["rho"][i].astype(float),
                         theta={k: float(g["domain"][k][i]) for k in PLANT_KEYS},
                         yaw_std=float(g["domain"]["wheel_yaw_err_std"][i]), off1=leaf[gmap[b]]))
    return rows


def split_groups(E, seed=1):
    """cal = devW 'cal' (random sizes 2-3, never trained on) halves: calA calibrates, calB is the in-dist test.
    Test groups: calB, heldout (random 4-6), physical pair / L3 / square4."""
    rng = np.random.default_rng(seed)
    cal = np.array([k for k, e in enumerate(E) if e["set"] == "cal"])
    perm = rng.permutation(cal)
    calA, calB = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    out = dict(calA=calA, calB=calB,
               heldout=np.array([k for k, e in enumerate(E) if e["set"] == "heldout"]))
    for ph in ("pair", "L3", "square4"):
        out[ph] = np.array([k for k, e in enumerate(E) if e["fam"] == f"phys_{ph}"])
    return out


def single_asm(mid, prm):
    return A.build(f"single_{mid}", [(mid, 0, 0, 0)], prm)


def remainder_asm(e, prm):
    mods = [m for k, m in enumerate(e["modules"]) if k != e["off1"]]
    return A.build("rem", mods, prm), [k for k in range(e["nmod"]) if k != e["off1"]]


def subs(e, kind):
    """list of (sub_id, kind, module indices) for an episode."""
    if kind == "singles":
        return [(k, "single", [k]) for k in range(e["nmod"])]
    rest = [k for k in range(e["nmod"]) if k != e["off1"]]
    return [(0, "single", [e["off1"]]), (1, "rem", rest)]


def wheels_of(mods):
    return np.concatenate([np.arange(4 * k, 4 * k + 4) for k in mods])
