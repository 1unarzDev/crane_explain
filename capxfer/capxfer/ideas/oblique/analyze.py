"""WS-OBL analysis: excitation / observability / decision-relevant coverage of P1+P5, P1+P5+P2, P1+P5+O22, P1+P5+O26.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.oblique.analyze [~/.cache/capxfer/OBL/obl_v1.pt]

Operational excitation criterion (I2 `analyze.signals_z`): a fault is observed under a probe if, on either leg, the
max over wheel speeds, duties and body twist of |fault - healthy twin| / max(healthy DR std, sensor floor) >= 3.
Geometric excitation (THEORY Prop. 4): |g_i . xi| >= 0.01 m/s (drive faults) or |h_i . xi| >= 0.03 m/s (roller).
Decision: strafe (P2 both legs) at tau = 0.8; relevant iff sim rho_min(P2+-) < 0.8 <= healthy.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

from capxfer import probes
from capxfer.sim import WEAK, COAST, BRAKE, ROLLER

FLOOR = dict(om=0.02, duty=0.01, tw=0.005)
Z_OBS = 3.0
ETA = dict(drive=0.01, roller=0.03)
TAU = 0.8
LEG_T = 2 * (probes.RAMP + 1.0 + probes.RAMP + probes.REST)                 # 4.0 s per probe (two legs)
SETS = {"P1+P5": ["P1", "P5"], "P1+P5+P2": ["P1", "P5", "P2"], "P1+P5+O22": ["P1", "P5", "O22"],
        "P1+P5+O26": ["P1", "P5", "O26"], "O22 alone": ["O22"], "O26 alone": ["O26"]}
MODE = {WEAK: "weak", COAST: "coast", BRAKE: "brake", ROLLER: "roller"}


def signals_z(r, floor_only=False):
    """floor_only: normalise by the sensor floor alone (noise-free paired test, closest to Prop. 4(d))."""
    S, nc = r["sim"], r["nc"]
    k = 0.0 if floor_only else 1.0
    out = []
    for key, fl in (("om", FLOOR["om"]), ("duty", FLOOR["duty"])):
        x = S[key]
        sd = np.maximum(k * x[nc:].std(0), fl)
        out.append((np.abs(x[:nc] - x[0:1]) / sd).max(-1))
    tw = S["twist"]
    sd = np.maximum(k * tw[nc:].std(0), FLOOR["tw"])
    out.append((np.abs(tw[:nc] - tw[0:1]) / sd).max(-1))
    return np.maximum.reduce(out)                                         # (nc, 2*n_probes)


def main(path=os.path.expanduser("~/.cache/capxfer/OBL/obl_v1.pt")):
    R = torch.load(path, weights_only=False)["results"]
    pids = R[0]["pids"]
    leg_pid = np.repeat(np.arange(len(pids)), 2)
    p2legs = np.where(leg_pid == pids.index("P2"))[0]
    rows = []                                                             # one per (asm, fault)
    emin = {k: [] for k in pids}
    for a, r in enumerate(R):
        z = signals_z(r)
        zp = np.stack([z[:, leg_pid == j].max(1) for j in range(len(pids))], 1)    # (nc, P)
        zf = signals_z(r, floor_only=True)
        zfp = np.stack([zf[:, leg_pid == j].max(1) for j in range(len(pids))], 1)
        rho = r["sim"]["rho"]
        healthy_ok = rho[0, p2legs].min() >= TAU
        pw = np.abs((r["wheel_pos"] * r["w_hat"]).sum(1))
        even = np.abs(r["w_hat"][:, 0]) > 0.5
        for j, k in enumerate(pids):                                     # min normalised excitation (all channels)
            emin[k].append(min(r["E"][:, WEAK, j].min(), r["E"][:, ROLLER, j].min()) / probes.V_NOM)
        for c, (i, m) in enumerate(r["tags"]):
            if i < 0:
                continue
            e = r["E"][i, m]                                              # (P,)
            eta = ETA["roller"] if m == ROLLER else ETA["drive"]
            rows.append(dict(a=a, n=r["n"], phys=str(r["family"]).startswith("phys"), m=m, z=zp[c], zf=zfp[c], e=e, eta=eta,
                             exc=e >= eta, unexc0=e < 1e-9,
                             blind=bool(m == ROLLER and even[i] and pw[i] * abs(r["p5_wz"]) < ETA["roller"]),
                             rel=bool(healthy_ok and rho[c, p2legs].min() < TAU),
                             drho=float(np.abs(rho[c] - rho[0]).max())))
    F = len(rows)
    Z = np.stack([x["z"] for x in rows])
    ZF = np.stack([x["zf"] for x in rows])
    EX = np.stack([x["exc"] for x in rows])
    U0 = np.stack([x["unexc0"] for x in rows])
    M = np.array([x["m"] for x in rows])
    REL = np.array([x["rel"] for x in rows])
    BL = np.array([x["blind"] for x in rows])
    NN = np.array([x["n"] for x in rows])
    out = dict(n_asm=len(R), n_faults=F, n_by_wheels={int(k): int((NN == k).sum()) for k in np.unique(NN)},
               n_asm_by_modules={int(k): int(sum(r["n"] == 4 * k for r in R)) for k in range(1, 7)},
               n_relevant_strafe_0p8=int(REL.sum()), n_blind_P1P5_closed_form=int(BL.sum()),
               probe_twists={k: [float(v) for v in R[0]["probe_twists"][k]] for k in ("O22", "O26")},
               heading_deg={k: math.degrees(math.atan2(R[0]["probe_twists"][k][1], R[0]["probe_twists"][k][0]))
                            for k in ("O22", "O26")})
    # theory check: min normalised excitation of each probe over all assemblies and channels
    out["min_norm_excitation"] = {k: float(np.min(v)) for k, v in emin.items()}
    out["closed_form_m"] = {"O22": math.sin(math.radians(22.5)),
                            "O26": min(abs(math.sin(math.radians(out["heading_deg"]["O26"]))),
                                       abs(math.cos(math.radians(out["heading_deg"]["O26"] + 45))))}
    res = {}
    for name, ps in SETS.items():
        J = [pids.index(p) for p in ps]
        geo = EX[:, J].any(1)
        obs = (Z[:, J] >= Z_OBS).max(1)
        obf = (ZF[:, J] >= Z_OBS).max(1)
        d = dict(time_s=LEG_T * len(J),
                 geo_excited=float(geo.mean()), observed_z3=float(obs.mean()), signal_above_floor=float(obf.mean()),
                 above_floor_given_geo=float(obf[geo].mean()) if geo.any() else None,
                 observed_given_geo=float(obs[geo].mean()) if geo.any() else None,
                 by_mode={MODE[m]: dict(geo=float(geo[M == m].mean()), floor=float(obf[M == m].mean()),
                                        obs=float(obs[M == m].mean())) for m in MODE},
                 blind_spot=dict(count=int(BL.sum()), geo=float(geo[BL].mean()) if BL.any() else None,
                                 floor=float(obf[BL].mean()) if BL.any() else None,
                                 obs=float(obs[BL].mean()) if BL.any() else None),
                 relevant_strafe=dict(count=int(REL.sum()), geo=int(geo[REL].sum()), obs=int(obs[REL].sum()),
                                      floor=int(obf[REL].sum()), missed_obs=int((REL & ~obs).sum())),
                 by_n={int(k): dict(geo=float(geo[NN == k].mean()), obs=float(obs[NN == k].mean()))
                       for k in np.unique(NN)})
        res[name] = d
    out["sets"] = res
    # excitation lemma in the full sim, per probe: unexcited channels (e = 0 exactly) -> z < 3 ?
    lem = {}
    for j, k in enumerate(pids):
        u = U0[:, j]
        lem[k] = dict(n_unexcited=int(u.sum()), frac_z_lt3=float((Z[u, j] < Z_OBS).mean()) if u.any() else None,
                      frac_floor_lt3=float((ZF[u, j] < Z_OBS).mean()) if u.any() else None,
                      excited_frac_z_ge3=float((Z[EX[:, j], j] >= Z_OBS).mean()))
    out["excitation_lemma_sim"] = lem
    # where the oblique is excited but not observed: what are they?
    j = pids.index("O22")
    miss = EX[:, j] & (Z[:, j] < Z_OBS)
    j5 = [pids.index(p) for p in ("P1", "P5", "O22")]
    miss3 = ~(Z[:, j5] >= Z_OBS).max(1)
    out["O22_excited_not_observed"] = dict(
        count=int(miss.sum()), by_mode={MODE[m]: int((miss & (M == m)).sum()) for m in MODE},
        by_n={int(k): int((miss & (NN == k)).sum()) for k in np.unique(NN)},
        max_drho=float(max([rows[i]["drho"] for i in np.where(miss)[0]], default=0.0)),
        relevant=int((miss & REL).sum()))
    out["P1P5O22_not_observed"] = dict(
        count=int(miss3.sum()), by_mode={MODE[m]: int((miss3 & (M == m)).sum()) for m in MODE},
        by_n={int(k): int((miss3 & (NN == k)).sum()) for k in np.unique(NN)},
        max_drho=float(max([rows[i]["drho"] for i in np.where(miss3)[0]], default=0.0)),
        relevant=int((miss3 & REL).sum()))
    # z of the closed-form blind spot under each probe
    out["blind_spot_z_median"] = {k: float(np.median(Z[BL, pids.index(k)])) if BL.any() else None for k in pids}
    op = path.replace(".pt", "_analysis.json")
    json.dump(out, open(op, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
