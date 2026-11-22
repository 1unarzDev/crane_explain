"""I2 analysis of the sweep: QS agreement, excitation lemma, masking law, danger sets, probe planning.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i2.analyze [sweep.pt]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch

from capxfer import probes
from capxfer.sim import WEAK, COAST, BRAKE, ROLLER

PIDS = probes.PIDS
LEG_PID = np.array([PIDS.index(k[:-1]) for k in probes.TARGET_KEYS])     # (14,) probe index of each leg
FLOOR = dict(om=0.02, duty=0.01, tw=0.005)                                # sensor-level floors (rad/s, -, m/s)
Z_OBS = 3.0
ETA = dict(drive=0.01, roller=0.03)                                       # excitation thresholds, m/s (Prop. 3)


def signals_z(r):
    """-> z (nc, 14): nuisance-normalised max signal change vs the healthy twin, per leg."""
    S = r["sim"]
    nc = r["nc"]
    dr = slice(nc, None)
    out = []
    for key, fl in (("om", FLOOR["om"]), ("duty", FLOOR["duty"])):
        x = S[key]
        sd = np.maximum(x[dr].std(0), fl)                                 # (14,n)
        out.append((np.abs(x[:nc] - x[0:1]) / sd).max(-1))
    tw = S["twist"].copy()
    sd = np.maximum(tw[dr].std(0), FLOOR["tw"])
    out.append((np.abs(tw[:nc] - tw[0:1]) / sd).max(-1))
    return np.maximum.reduce(out)


def qs_z(r):
    """QS-predicted signature change normalised by the SIMULATED healthy DR spread -> (nc,14). Planner-side."""
    S, Q = r["sim"], r["qs"]
    nc = r["nc"]
    out = []
    for key, qk, fl in (("om", "om", FLOOR["om"]), ("duty", "duty", FLOOR["duty"]), ("twist", "twist", FLOOR["tw"])):
        sd = np.maximum(S[key][nc:].std(0), fl)
        out.append((np.abs(Q[qk] - Q[qk][0:1]) / sd).max(-1))
    return np.maximum.reduce(out)


def fault_rows(r):
    tags = r["tags"]
    E = r["E"]                                                            # (n,5,7)
    rows = []
    for c, (i, m) in enumerate(tags):
        if i < 0:
            continue
        rows.append((c, i, m, E[i, m]))
    return rows


def cover_ok(exc_f, rel, chosen, eta):
    """every relevant fault excited >= eta by some chosen probe."""
    if not rel.any():
        return True
    return bool((exc_f[rel][:, chosen].max(1) >= eta[rel]).all()) if len(chosen) else False


def plan(exc_f, rel, eta, times):
    """Minimum-time subset of the library covering every relevant fault (exhaustive over 2^7)."""
    best, bt = None, 1e9
    for k in range(0, len(PIDS) + 1):
        for S in itertools.combinations(range(len(PIDS)), k):
            t = sum(times[j] for j in S)
            if t < bt and cover_ok(exc_f, rel, list(S), eta):
                best, bt = list(S), t
    return best, bt


def main(path=os.path.expanduser("~/.cache/capxfer/I2/sweep_v1.pt")):
    d = torch.load(path, weights_only=False)
    R = [r for r in d["results"] if r is not None]
    times = [2 * probes.leg_duration(p) for p in probes.LIBRARY]
    out = {}

    # ---------------- 1. QS vs sim
    ds, dq, ns = [], [], []
    for r in R:
        nc = r["nc"]
        ds.append(r["sim"]["rho"][:nc].ravel())
        dq.append(r["qs"]["rho"].ravel())
        ns.append(np.full(nc * 14, r["n"]))
    ds, dq, ns = map(np.concatenate, (ds, dq, ns))
    resid_bad = float(np.mean(np.concatenate([r["qs"]["resid"].ravel() for r in R]) > 1e-4))
    out["qs_vs_sim"] = dict(
        n_cases=int(len(ds)), mae=float(np.abs(ds - dq).mean()), p95=float(np.quantile(np.abs(ds - dq), .95)),
        r2=float(1 - ((ds - dq) ** 2).sum() / ((ds - ds.mean()) ** 2).sum()), qs_unconverged=resid_bad,
        decision_agree={str(t): float(((ds < t) == (dq < t)).mean()) for t in (0.5, 0.7, 0.9)},
        decision_agree_on_sim_positive={str(t): float(((dq < t)[ds < t]).mean()) if (ds < t).any() else None
                                        for t in (0.5, 0.7, 0.9)})

    # ---------------- 2. excitation lemma (per fault x leg)
    un_d, un_z, ex_d, ex_z, ex_e, ex_m = [], [], [], [], [], []
    for r in R:
        z = signals_z(r)
        rho = r["sim"]["rho"]
        for c, i, m, e in fault_rows(r):
            el = e[LEG_PID]                                               # (14,)
            dr = np.abs(rho[c] - rho[0])
            u = el < 1e-9
            un_d.append(dr[u]); un_z.append(z[c][u])
            ex_d.append(dr[~u]); ex_z.append(z[c][~u]); ex_e.append(el[~u]); ex_m.append(np.full((~u).sum(), m))
    un_d, un_z, ex_d, ex_z, ex_e, ex_m = map(np.concatenate, (un_d, un_z, ex_d, ex_z, ex_e, ex_m))
    out["excitation_lemma"] = dict(
        n_unexcited=int(len(un_d)), n_excited=int(len(ex_d)),
        unexcited_drho_p50_p99_max=[float(np.quantile(un_d, q)) for q in (.5, .99, 1.0)],
        unexcited_frac_drho_lt_0p01=float((un_d < 0.01).mean()),
        unexcited_frac_z_lt_3=float((un_z < Z_OBS).mean()),
        unexcited_z_p50_p99=[float(np.quantile(un_z, q)) for q in (.5, .99)],
        consequential_legs_drho_gt_0p05=int((np.concatenate([un_d, ex_d]) > 0.05).sum()),
        consequential_legs_that_are_excited=float((ex_d > 0.05).sum() / max(1, (np.concatenate([un_d, ex_d]) > 0.05).sum())),
        excited_frac_z_ge_3_by_mode={int(m): float((ex_z[ex_m == m] >= Z_OBS).mean()) for m in (1, 2, 3, 4)},
        roller_excited_z_ge3_by_e=[(lo, hi, float((ex_z[(ex_m == 4) & (ex_e >= lo) & (ex_e < hi)] >= Z_OBS).mean())
                                    if ((ex_m == 4) & (ex_e >= lo) & (ex_e < hi)).any() else None,
                                    int(((ex_m == 4) & (ex_e >= lo) & (ex_e < hi)).sum()))
                                   for lo, hi in ((0, .01), (.01, .02), (.02, .03), (.03, .05), (.05, .1), (.1, 1))])

    # ---------------- 3. masking law: worst-leg loss of a single fault vs n
    mask = {}
    for mname, ms in (("passive", (COAST, BRAKE)), ("roller", (ROLLER,)), ("weak0.5", (WEAK,))):
        per_n = {}
        for r in R:
            rho_s, rho_q = r["sim"]["rho"], r["qs"]["rho"]
            for c, i, m, e in fault_rows(r):
                if m in ms:
                    per_n.setdefault(r["n"], []).append((np.max(rho_s[0] - rho_s[c]), np.max(rho_q[0] - rho_q[c])))
        rows = {}
        for n in sorted(per_n):
            a = np.array(per_n[n])
            rows[n] = dict(count=len(a), sim_med=float(np.median(a[:, 0])), sim_p90=float(np.quantile(a[:, 0], .9)),
                           qs_med=float(np.median(a[:, 1])), frac_lt_0p9=float((a[:, 0] > 0.1).mean()),
                           frac_lt_0p5=float((a[:, 0] > 0.5).mean()))
        ns_ = np.array([n for n in rows if n >= 8], float)
        med = np.array([rows[int(n)]["sim_med"] for n in ns_])
        slope = float(np.polyfit(np.log(ns_), np.log(np.maximum(med, 1e-6)), 1)[0]) if len(ns_) > 2 else None
        mask[mname] = dict(by_n=rows, loglog_slope_n_ge_8=slope)
    out["masking"] = mask

    # ---------------- 4. danger sets: fixed probe sets vs plan
    fixed = {"P1P5": ["P1", "P5"], "P1P5P2": ["P1", "P5", "P2"], "P1P2": ["P1", "P2"], "P1-P5": list(PIDS[:5]),
             "all": list(PIDS)}
    decisions = {"strafe": [probes.TARGET_KEYS.index("P2+"), probes.TARGET_KEYS.index("P2-")],
                 "any14": list(range(14))}
    dang = {}
    for dname, legs in decisions.items():
        for tau in (0.5, 0.9):
            key = f"{dname}@{tau}"
            acc = {k: dict(rel=0, missed=0, t=0.0, asm_with_miss=0) for k in list(fixed) + ["plan", "plan_qsz", "plan_oracle"]}
            by_size = {}
            n_asm = 0
            for r in R:
                z = signals_z(r)                                          # (nc,14)
                rho_s, rho_q = r["sim"]["rho"], r["qs"]["rho"]
                fr = fault_rows(r)
                if not fr:
                    continue
                n_asm += 1
                cs = np.array([c for c, *_ in fr])
                ms = np.array([m for _, _, m, _ in fr])
                exc = np.stack([e for *_, e in fr])                       # (F,7)
                rel_sim = (rho_s[cs][:, legs].min(1) < tau) & (rho_s[0, legs].min() >= tau)
                rel_qs = (rho_q[cs][:, legs].min(1) < tau + 0.05)
                obs_pid = np.stack([z[cs][:, LEG_PID == j].max(1) >= Z_OBS for j in range(len(PIDS))], 1)  # (F,7)
                eta = np.where(ms == ROLLER, ETA["roller"], ETA["drive"])
                sel_plan, t_plan = plan(exc, rel_qs, eta, times)
                zq = qs_z(r)[cs]
                zq_pid = np.stack([zq[:, LEG_PID == j].max(1) for j in range(len(PIDS))], 1)
                exc2 = np.where(zq_pid >= 2 * Z_OBS, exc, 0.0)                 # excitation AND QS-predicted z >= 6
                sel_plan2, _ = plan(exc2, rel_qs, eta, times)
                # oracle: min-time subset whose SIM observability covers every SIM-relevant fault
                best, bt = None, 1e9
                for k in range(0, 8):
                    for S in itertools.combinations(range(7), k):
                        t = sum(times[j] for j in S)
                        if t < bt and (not rel_sim.any() or (len(S) and obs_pid[rel_sim][:, list(S)].any(1).all())):
                            best, bt = list(S), t
                sets = {k: [PIDS.index(p) for p in v] for k, v in fixed.items()}
                sets["plan"] = sel_plan if sel_plan is not None else list(range(7))
                sets["plan_qsz"] = sel_plan2 if sel_plan2 is not None else list(range(7))
                sets["plan_oracle"] = best if best is not None else list(range(7))
                for k, S in sets.items():
                    seen = obs_pid[:, S].any(1) if len(S) else np.zeros(len(cs), bool)
                    miss = rel_sim & ~seen
                    acc[k]["rel"] += int(rel_sim.sum())
                    acc[k]["missed"] += int(miss.sum())
                    acc[k]["t"] += sum(times[j] for j in S)
                    acc[k]["asm_with_miss"] += int(miss.any())
                    bs = by_size.setdefault(r["n"], {}).setdefault(k, [0, 0])
                    bs[0] += int(rel_sim.sum())
                    bs[1] += int(miss.sum())
            dang[key] = dict(n_asm=n_asm, **{k: dict(danger_frac=(v["missed"] / v["rel"]) if v["rel"] else 0.0,
                                                     missed=v["missed"], relevant=v["rel"],
                                                     mean_probe_time_s=v["t"] / n_asm,
                                                     asm_with_danger=v["asm_with_miss"]) for k, v in acc.items()},
                             by_n={int(n): {k: v for k, v in s.items()} for n, s in sorted(by_size.items())})
    out["danger"] = dang
    op = path.replace(".pt", "_analysis.json")
    json.dump(out, open(op, "w"), indent=1)
    print(json.dumps(out, indent=1)[:12000])


if __name__ == "__main__":
    main(*sys.argv[1:])
