"""I2: danger sets for DOUBLE faults (QS consequence model, validated on single faults vs the simulator; excitation
observability, validated by the sweep's z-by-excitation curve). Compares fixed probe sets, the planner and an oblique
single probe.  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i2.exp_double
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from capxfer import assembly as A, probes
from capxfer.sim import COAST, ROLLER
from capxfer.ideas.i2 import theory as T

PIDS = probes.PIDS
XI_OBL = np.array([0.27, 0.13, 0.0])
ETA = {COAST: 0.01, ROLLER: 0.03}
TAUS = (0.5, 0.7, 0.9)
LEGS = {"strafe": [probes.TARGET_KEYS.index("P2+"), probes.TARGET_KEYS.index("P2-")], "any14": list(range(14))}


def assemblies(P, per_size=4, seed=0):
    rng = np.random.default_rng(seed)
    out = [A.physical(k, P) for k in ("single", "pair", "L3", "square4")]
    cnt = {}
    for j in range(180):
        a = A.random_assembly(rng, P, sizes=((j % 6) + 1,))
        k = (j % 6) + 1
        if cnt.get(k, 0) < per_size:
            out.append(a)
            cnt[k] = cnt.get(k, 0) + 1
    return out


def fault_pairs(n, rng, max_pairs=24):
    pairs = list(itertools.combinations(range(n), 2))
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    cfg = []
    for (i, j) in pairs:
        for m in (COAST, ROLLER):
            cfg.append(((i, m), (j, m)))
        cfg.append(((i, COAST), (j, ROLLER)))
    return cfg


def exc_of(asm, xi, i, m):
    g, h = T.rows(asm)
    return abs((g if m == COAST else h)[i] @ xi)


def main():
    P = A.load_params("params/module_measured.yaml")
    rng = np.random.default_rng(5)
    tw_all = None
    times = {p.pid: 2 * probes.leg_duration(p) for p in probes.LIBRARY}
    sets = {"P1": ["P1"], "P5": ["P5"], "P1P5": ["P1", "P5"], "P1P5P2": ["P1", "P5", "P2"], "P1P2": ["P1", "P2"],
            "all": list(PIDS)}
    res = {f"{d}@{t}": {k: [0, 0, 0.0] for k in list(sets) + ["oblique", "plan"]} for d in LEGS for t in TAUS}
    per_asm = []
    for asm in assemblies(P):
        n = asm.n
        cfg = fault_pairs(n, rng)
        modes = np.zeros((len(cfg) + 1, n), np.int64)
        for b, fs in enumerate(cfg):
            for (i, m) in fs:
                modes[b + 1, i] = m
        gs = np.ones(modes.shape)
        legs = T.legs(asm)
        xi, _, _ = T.qs_solve(asm, P, modes, gs, legs)
        rho = T.rho_of(xi, legs, asm.char_length())
        tw = T.probe_twists(asm)
        # excitation of each double fault under each probe: observable if ANY component channel >= eta
        def seen_by(xi_p):
            return np.array([max(exc_of(asm, xi_p, i, m) / ETA[m] for (i, m) in fs) >= 1.0 for fs in cfg])
        seen = {p: seen_by(tw[p]) for p in PIDS}
        seen["oblique"] = seen_by(XI_OBL)
        for d, L in LEGS.items():
            for tau in TAUS:
                key = f"{d}@{tau}"
                rel = (rho[1:, L].min(1) < tau) & (rho[0, L].min() >= tau)
                for k, S in sets.items():
                    s = np.any([seen[p] for p in S], 0)
                    res[key][k][0] += int(rel.sum()); res[key][k][1] += int((rel & ~s).sum())
                    res[key][k][2] += sum(times[p] for p in S)
                s = seen["oblique"]
                res[key]["oblique"][0] += int(rel.sum()); res[key]["oblique"][1] += int((rel & ~s).sum())
                res[key]["oblique"][2] += 4.0
                # planner: min-time library subset covering every QS-relevant (+0.05 margin) fault
                relp = rho[1:, L].min(1) < tau + 0.05
                best, bt = list(PIDS), 1e9
                for r_ in range(0, 8):
                    for S in itertools.combinations(PIDS, r_):
                        t = sum(times[p] for p in S)
                        if t >= bt:
                            continue
                        cov = np.any([seen[p] for p in S], 0) if S else np.zeros(len(cfg), bool)
                        if (cov | ~relp).all():
                            best, bt = list(S), t
                s = np.any([seen[p] for p in best], 0) if best else np.zeros(len(cfg), bool)
                res[key]["plan"][0] += int(rel.sum()); res[key]["plan"][1] += int((rel & ~s).sum())
                res[key]["plan"][2] += bt
                if d == "strafe" and tau == 0.9:
                    per_asm.append(dict(name=asm.name, n=n, plan=best, rel=int(rel.sum()),
                                        miss_P1P5=int((rel & ~(seen["P1"] | seen["P5"])).sum())))
        print(asm.name, n, per_asm[-1], flush=True)
    na = len(per_asm)
    out = {k: {s: dict(relevant=v[0], missed=v[1], danger_frac=(v[1] / v[0]) if v[0] else 0.0,
                       mean_time_s=v[2] / na) for s, v in d.items()} for k, d in res.items()}
    out["per_asm_strafe@0.9"] = per_asm
    json.dump(out, open(os.path.expanduser("~/.cache/capxfer/I2/double_danger.json"), "w"), indent=1)
    for k, d in out.items():
        if k.startswith("per"):
            continue
        print(k, {s: (round(v["danger_frac"], 3), v["missed"], v["relevant"], round(v["mean_time_s"], 1))
                  for s, v in d.items()})


if __name__ == "__main__":
    main()
