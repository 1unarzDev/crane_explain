"""Markdown tables for docs/ideas/I4.md from the frozen I4 result files (numbers are only read, never typed).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.tables > ~/.cache/capxfer/I4/tables.md
"""
from __future__ import annotations

import json

from .common import I4, TYPES

f3 = lambda x: "–" if x is None else f"{x:.3f}"
f2 = lambda x: "–" if x is None else f"{x:.2f}"


def _load(name):
    p = I4 / name
    return json.loads(p.read_text()) if p.exists() else None


def lofo():
    r = {}
    for p in sorted(I4.glob("lofo_results_*.json")):
        r.update(json.loads(p.read_text()))
    if not r:
        return "(lofo_results.json missing)\n"
    out = ["### T1. In-family vs leave-family-out (devW physical trials, E2/P1+P5, alpha 0.1, untested 10 legs)",
           "", "Cells: raw joint / split-joint (calibrated on devW random groups minus the type's family) joint / "
           "group-ss1-joint joint / median joint width / diagnosed (all wheels correct, coast+brake merged, faulted "
           "trials) / APS trial cover.", "",
           "| type | method | generation | raw joint | split-joint | group-ss1 joint | width | diag | APS |",
           "|---|---|---|---|---|---|---|---|---|"]
    meths = ("structured_ensemble", "structured_npe(m0)", "ensemble", "blackbox(m0)")
    for t in TYPES:
        lofo_types = ("single", "pair", "L3")
        if t != "square4":
            gens = [("W", "W: 5 nets x full steps, in-family")]
            gens += [(f"I4_lofo_{o}", f"control: I4_lofo_{o} (same reduced recipe, {t} family IN training)")
                     for o in lofo_types if o != t]
            gens.append((f"I4_lofo_{t}", f"**LOFO: I4_lofo_{t} ({t} family removed)**"))
        else:
            gens = [("W", "W (square never in training)")] + [(f"I4_lofo_{o}", f"I4_lofo_{o} (square never in training)")
                                                               for o in lofo_types]
        for m in meths:
            for g, gl in gens:
                x = r.get(f"{g}:{m}", {}).get(t)
                if not x:
                    continue
                out.append(f"| {t} | {m} | {gl} | {f3(x['raw']['joint'])} | {f3(x['split_joint']['joint'])} | "
                           f"{f3(x['group_ss1_joint']['joint'])} | {f3(x['split_joint']['width'])} | "
                           f"{f3(x['diag'].get('diag_trial'))} | {f3((x.get('aps') or {}).get('trial_cover'))} |")
        x = r.get("W:analytic_bayes", {}).get(t)
        if x:
            out.append(f"| {t} | analytic_bayes | (no training) | {f3(x['raw']['joint'])} | "
                       f"{f3(x['split_joint']['joint'])} | {f3(x['group_ss1_joint']['joint'])} | "
                       f"{f3(x['split_joint']['width'])} | {f3(x['diag'].get('diag_trial'))} | "
                       f"{f3((x.get('aps') or {}).get('trial_cover'))} |")
    return "\n".join(out) + "\n"


def bench():
    out = ["### T2. Unseen-morphology benchmark (68 free-polyomino families: all 5 tetrominoes, 12 pentominoes, 35 "
           "hexominoes, 16 sampled heptominoes; 4 assemblies x 48 trials per family of size 4-6, 2 x 48 for size 7; "
           "models/W trained on sizes 1-3 only)", "",
           "Joint-10 coverage (trial-averaged) / family-minimum / median joint width. raw = model quantiles; "
           "split-indist = split CQR on training-family trials (sizes 1-3); LOFO-group = Dunn et al. ss1 over the other "
           "families (3 subsample seeds averaged); leave-size-out = ss1 calibrated only on families of other sizes.", "",
           "| method | size | raw | split-indist | LOFO split | LOFO-group ss1 | leave-size-out ss1 | per-key marg (LOFO-group) | diag | APS LOFO |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("structured_ensemble", "ensemble", "structured_npe", "blackbox", "analytic_bayes", "nominal_sim"):
        r = _load(f"bench_v1_W_{m}.json")
        if not r:
            continue
        rows = [("all", r["overall"])] + [(k, v) for k, v in r["by_size"].items()]
        for k, v in rows:
            c = lambda vv: (f"{f3(v[vv]['joint'])} / {f2(v[vv]['fam_min_joint'])} / {f3(v[vv]['width'])}"
                            if vv in v else "–")
            dg = r["diag"].get("overall" if k == "all" else str(k)) or {}
            al = (r.get("aps_lofo") or {}).get("overall" if k == "all" else str(k)) or {}
            out.append(f"| {m} | {k} | {c('raw')} | {c('split_indist_joint')} | {c('lofo_split_joint')} | "
                       f"{c('lofo_group_ss1_joint')} | {c('lsizeo_group_ss1_joint')} | "
                       f"{f3(v.get('lofo_group_ss1', {}).get('marg'))} | {f3(dg.get('diag_trial'))} | "
                       f"{f3(al.get('trial_cover'))} ({f2(al.get('set_size'))}) |")
        ind = r["indist"]
        out.append(f"| {m} | in-dist 1/2/3 raw joint | " + " / ".join(f3(ind[s]["raw"]["joint"]) for s in sorted(ind))
                   + f" | | | | | | {f3((r['diag'].get('indist') or {}).get('diag_trial'))} | |")
        out.append(f"| {m} | predict ms/trial | {r['predict_ms_per_trial']:.1f} | | | | | | | |")
    return "\n".join(out) + "\n"


def stress():
    r = _load("stress_v1_results.json")
    d = _load("stress_v1_drho.json") or {}
    if not r:
        return "(stress results missing)\n"
    conds = ("nominal", "yaw_mis", "latency_hi", "roller_aniso", "compliant", "friction_form", "combined")
    out = ["### T3. Sim-to-sim stress (4 physical types x 256 paired trials per condition; W models; split-joint "
           "calibrated on in-model devW sim)", "",
           "Plant shift size: share of trials whose max_k |rho_stress - rho_nominal| > 0.1 (single/pair/L3/square4).", "",
           "| condition | consequential share | " + " | ".join(r) + " |", "|---|---|" + "---|" * len(r)]
    for c in conds:
        cs = " / ".join(f2(d.get(f"{c}/{t}", {}).get("frac_trial_drho_gt_0p1")) for t in TYPES)
        cells = []
        for m, rr in r.items():
            x = rr.get(f"{c}/all")
            cells.append("–" if not x else f"{f3(x['split_joint']['joint'])} ({f3(x['raw']['joint'])}) "
                                            f"dg {f2(x['diag'].get('diag_trial'))}")
        out.append(f"| {c} | {cs} | " + " | ".join(cells) + " |")
    out += ["", "Cells: split-joint joint-10 coverage (raw joint) dg = diagnosed rate on faulted trials.", ""]
    out += ["Per type, split-joint joint coverage (combined / roller_aniso):", "",
            "| method | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
    for m, rr in r.items():
        out.append(f"| {m} | " + " | ".join(
            f"{f3(rr.get(f'combined/phys_{t}', {}).get('split_joint', {}).get('joint'))} / "
            f"{f3(rr.get(f'roller_aniso/phys_{t}', {}).get('split_joint', {}).get('joint'))}" for t in TYPES) + " |")
    return "\n".join(out) + "\n"


def l1():
    r = _load("l1_swap.json")
    if not r:
        return "(l1_swap.json missing)\n"
    out = ["### T4. Fleet L1 truth as the plant (single module, fl faulted; target swap)", "",
           "split-joint joint-10 coverage of capxfer truth / of L1-matched truth / of L1-native truth (marginal in "
           "parentheses for L1-matched).", "",
           "| method | " + " | ".join(r["cases"]) + " |", "|---|" + "---|" * len(r["cases"])]
    for m, rr in r["methods"].items():
        cells = []
        for c in r["cases"]:
            x = rr[c]["split_joint"]
            cells.append(f"{f2(x['capxfer_truth']['joint'])} / {f2(x['L1_matched']['joint'])} / "
                         f"{f2(x['L1_native']['joint'])} ({f2(x['L1_matched']['marg'])})")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(lofo())
    print(bench())
    print(stress())
    print(l1())
