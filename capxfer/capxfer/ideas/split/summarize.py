"""SPLIT paper tables (SIMULATION ONLY) -> ~/.cache/capxfer/SPLIT/tables.md (+ tables.json).

Every number is read from the frozen outputs results_singles_<p>.json, rows_singles_<p>.npz,
probe_after_eval_<p>.json, rem_comp_<p>.json. Bootstrap CIs resample devW ASSEMBLIES (each holds 64 trials).
"""
from __future__ import annotations

import json

import numpy as np

from capxfer.ideas.split import common as C, evaluate as EV

PIDS = ("P1P5", "P1P5P2")
MAIN = ("comp_se", "post_se", "comp_ab", "post_ab", "bb_carry", "bb_post", "healthy", "orc_h", "oracle")
LAB = dict(comp_se="(a) COMP, structured_ens wheel sets", post_se="(a') structured_ens mixture, post-split-calibrated",
           comp_ab="(d) COMP, analytic_bayes wheel sets", post_ab="(d') analytic_bayes mixture, post-split-calibrated",
           bb_carry="(b) black-box ensemble, current-body bound carried over",
           bb_post="(b') black-box ensemble, recalibrated on post-split outcomes", healthy="(c) assume healthy",
           orc_h="oracle health, unknown plant", oracle="(e) oracle")


def boot(ep, gi_of_ep, stat, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    gi = gi_of_ep[ep]
    ug = np.unique(gi)
    rows_of = {g: np.flatnonzero(gi == g) for g in ug}
    vals = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([rows_of[g] for g in pick])
        vals.append(stat(idx))
    return np.nanquantile(vals, [0.025, 0.975])


def main():
    E = C.episodes("P1P5")
    Gs = C.split_groups(E)
    gi_of_ep = np.array([e["gi"] for e in E])
    test_eps = np.concatenate([Gs[g] for g in ("calB", "heldout", "pair", "L3", "square4")])
    md, J = [], {}
    md.append("# SPLIT benchmark tables (SIMULATION ONLY; devW episodes, W models, alpha = 0.10)\n")
    for p in PIDS:
        S = np.load(C.OUT / f"rows_singles_{p}.npz")
        R = json.loads((C.OUT / f"results_singles_{p}.json").read_text())
        te = np.flatnonzero(np.isin(S["ep"], test_eps))
        z = S["z"]
        md.append(f"\n## T1 ({p} on the assembly): every module alone after the split; test modules n = {len(te)}\n")
        md.append("Cells: false-go P(go | z < tau) [95% CI, assembly bootstrap] / certification P(go) / "
                  "P(go | z >= tau) / unsafe-go P(go and z < tau). z = min(rho_P2+, rho_P2-).\n")
        md.append("| method | tau=0.5 false-go | cert | useful | unsafe-go | tau=0.8 false-go | cert | useful | unsafe-go | "
                  "lb cov(z) | strafe-pair cov / median width |")
        md.append("|---|" + "---|" * 10)
        for m in MAIN:
            cells = []
            for tau in EV.TAUS:
                lb = S[f"lb_{m}"]
                go, uns = lb >= tau, z < tau
                fg = lambda idx: go[te[idx]][uns[te[idx]]].mean() if uns[te[idx]].any() else np.nan
                ci = boot(S["ep"][te], gi_of_ep, fg)
                t = R["tables"][f"{tau}|test_all|{m}"]
                J[f"{p}|{tau}|{m}"] = dict(t, false_go_ci=ci.tolist())
                cells += [f"{t['false_go']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", f"{t['cert']:.3f}", f"{t['useful']:.3f}",
                          f"{t['unsafe_go']:.4f}"]
            t = R["tables"][f"0.5|test_all|{m}"]
            md.append(f"| {LAB[m]} | " + " | ".join(cells) + f" | {t['lb_cov']:.3f} | {t['cov_strafe']:.3f} / "
                      f"{t['width']:.3f} |")
        n_uns = {tau: R["tables"][f"{tau}|test_all|oracle"]["n_unsafe"] for tau in EV.TAUS}
        md.append(f"\nUnsafe modules in test: {n_uns}. Stage-1 projected-set coverage of the true module health "
                  f"(structured_ens): {R['stage1_module_cov']['comp_se']['test_all']:.3f}; "
                  f"analytic_bayes: {R['stage1_module_cov']['comp_ab']['test_all']:.3f}.\n")
        # size / masking
        md.append(f"\n## T2 ({p}): masking vs assembly size (faulted test episodes)\n")
        md.append("| modules | tau | n eps | current body z<tau | black box certifies body | some module z<tau after split | "
                  "(b) false-go | (c) false-go | (a) false-go | (a) cert | oracle cert |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for tau in EV.TAUS:
            for nm in range(2, 7):
                pr = R["pre"][f"{tau}|nmod{nm}"]
                T = {m: R["tables"][f"{tau}|nmod{nm}|{m}"] for m in ("bb_carry", "healthy", "comp_se", "oracle")}
                md.append(f"| {nm} | {tau} | {pr['n_faulted_eps']} | {pr['pre_body_unsafe']:.3f} | "
                          f"{pr['bb_certifies_body']:.3f} | {pr['post_any_module_unsafe']:.3f} | "
                          f"{T['bb_carry']['false_go']:.3f} | {T['healthy']['false_go']:.3f} | "
                          f"{T['comp_se']['false_go']:.3f} | {T['comp_se']['cert']:.3f} | {T['oracle']['cert']:.3f} |")
        # off1: the split-off module
        off = np.array([S["mod"][r] == E[S["ep"][r]]["off1"] for r in range(len(S["ep"]))])
        to = te[off[te]]
        md.append(f"\n## T3 ({p}): 'split off one module' variant, the module that leaves (n = {len(to)})\n")
        md.append("| method | tau=0.5 false-go | cert | tau=0.8 false-go | cert |")
        md.append("|---|---|---|---|---|")
        for m in MAIN:
            c = []
            for tau in EV.TAUS:
                go, uns = S[f"lb_{m}"][to] >= tau, z[to] < tau
                c += [f"{go[uns].mean():.3f}", f"{go.mean():.3f}"]
                J[f"{p}|off1|{tau}|{m}"] = dict(false_go=float(go[uns].mean()), cert=float(go.mean()),
                                               n_unsafe=int(uns.sum()))
            md.append(f"| {LAB[m]} | " + " | ".join(c) + " |")
        rp = C.OUT / f"rem_comp_{p}.json"
        if rp.exists():
            rows = json.loads(rp.read_text())["rows"]
            zz = np.array([r["z"] for r in rows])
            md.append(f"\nRemainder sub-assembly (>= 2 modules, {len(rows)} episodes, 2/3 with a fault on the remainder; "
                      f"M = {json.loads(rp.read_text())['M']} plant draws):")
            for tau in EV.TAUS:
                for m in ("lb_comp", "lb_orc"):
                    lb = np.array([r[m] for r in rows])
                    go = lb >= tau
                    md.append(f"- tau={tau} {m}: unsafe {int((zz < tau).sum())}, cert {go.mean():.3f}, "
                              f"unsafe-go {(go & (zz < tau)).mean():.4f}, lb coverage {(zz >= lb).mean():.3f}; "
                              f"(b)/(c) certify {np.mean(zz >= -9):.3f} with unsafe-go {(zz < tau).mean():.4f}")
                    J[f"{p}|rem|{tau}|{m}"] = dict(cert=float(go.mean()), unsafe_go=float((go & (zz < tau)).mean()),
                                                  n_unsafe=int((zz < tau).sum()), n=len(rows))
        # probe after split
        pa = json.loads((C.OUT / f"probe_after_eval_{p}.json").read_text())
        md.append(f"\n## T4 (pre-split evidence {p}) vs (f) split first, then probe each module "
                  f"(same {pa['n_test']} test modules, faulted oversampled and re-weighted; calibrated on calA modules)\n")
        md.append(f"Probe time per module: P1+P5 {pa['probe_time_s']['P1P5']} s, P1+P5+P2 {pa['probe_time_s']['P1P5P2']} s.\n")
        md.append("| method | extra probe time after split | tau=0.5 false-go | cert | tau=0.8 false-go | cert |")
        md.append("|---|---|---|---|---|---|")
        for m, cost in (("comp_se", "0"), ("post_se", "0"), ("bb_carry", "0"), ("healthy", "0"),
                        ("pa_ensemble_P1P5", "8.7 s x modules"), ("pa_ensemble_P1P5P2", "12.7 s x modules"),
                        ("pa_structured_ensemble_P1P5", "8.7 s x modules"),
                        ("pa_structured_ensemble_P1P5P2", "12.7 s x modules")):
            r5, r8 = pa["rows"][f"0.5|{m}"], pa["rows"][f"0.8|{m}"]
            md.append(f"| {LAB.get(m, m)} | {cost} | {r5['false_go']:.3f} | {r5['cert']:.3f} | {r8['false_go']:.3f} | "
                      f"{r8['cert']:.3f} |")
            J[f"{p}|pa|{m}"] = dict(r5=r5, r8=r8)
        md.append("\nEpisode task ('send one module to strafe-dock'; faulted test episodes; pick the certified module "
                  "with the largest lower bound): ok / unsafe pick / abstain")
        for tau in EV.TAUS:
            md.append(f"- tau={tau}: " + "; ".join(
                f"{m} {R['task'][f'{tau}|{m}']['ok']:.3f}/{R['task'][f'{tau}|{m}']['unsafe']:.3f}/"
                f"{R['task'][f'{tau}|{m}']['abstain']:.3f}" for m in ("comp_se", "post_se", "comp_ab", "bb_carry",
                                                                       "healthy", "oracle")))
    (C.OUT / "tables.md").write_text("\n".join(md) + "\n")
    (C.OUT / "tables.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
