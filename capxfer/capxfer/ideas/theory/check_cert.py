"""Theorem 6 (compositional certificate) empirical check: read the SPLIT and I1 outputs and compare the realised
lower-bound miscoverage P(z < lb) with alpha1 + alpha2, per group, plus stage-1 projected-set coverage and the
monotonicity-excursion tolerance (I1 monotone.log). No new simulation.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.theory.check_cert
"""
from __future__ import annotations

import json
import os
import re

from capxfer.ideas.theory.common import dump

C = os.path.expanduser("~/.cache/capxfer")


def main():
    out = {}
    for probes_ in ("P1P5P2", "P1P5"):
        d = json.load(open(f"{C}/SPLIT/results_singles_{probes_}.json"))
        a = d["alpha1"] + d["alpha2"]
        rows = {}
        for key, v in d["tables"].items():
            tau, grp, meth = key.split("|")
            if meth != "comp_se":
                continue
            rows[f"{tau}|{grp}"] = dict(n=v["n"], miscov=1 - v["lb_cov"], unsafe_go=v["unsafe_go"],
                                        false_go=v["false_go"], cert=v["cert"], le_alpha=(1 - v["lb_cov"]) <= a)
        out[probes_] = dict(alpha1_plus_alpha2=a, K=d["K"], groups=rows,
                            n_groups_le_alpha=sum(r["le_alpha"] for r in rows.values()), n_groups=len(rows),
                            max_miscov=max(r["miscov"] for r in rows.values()),
                            stage1_module_cov=d["stage1_module_cov"]["comp_se"])
    txt = open(f"{C}/I1/monotone.log").read()
    m = re.search(r"ALL: P\(excursion>0.005\)=([\d.]+)\s+max=([\d.]+)", txt)
    out["monotone"] = dict(p_excursion_gt_0p005=float(m.group(1)), max_excursion=float(m.group(2)))
    dump("check_cert.json", out)
    for k in ("P1P5P2", "P1P5"):
        print(k, out[k]["n_groups_le_alpha"], "/", out[k]["n_groups"], "max miscov", round(out[k]["max_miscov"], 4))
    print(out["monotone"])


if __name__ == "__main__":
    main()
