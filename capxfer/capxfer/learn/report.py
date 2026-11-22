"""Markdown tables from ~/.cache/capxfer/C/results_*.json (numbers only from those files).
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.report [--rung E1]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/.cache/capxfer/C"))
ORDER = ("structured_npe", "structured_ensemble", "blackbox", "ensemble", "nominal_sim", "analytic_bayes")
COLS = ("mode_nll", "mode_acc", "fault_recall", "fault_recall_cb", "y_nll", "y_mae", "cover90", "width90",
        "drift_mae", "drift_cover_up95")


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def table(res, keys, rows):
    out = ["| method | group | " + " | ".join(COLS) + " | n |", "|" + "---|" * (len(COLS) + 3)]
    for meth, r in res.items():
        for k, label in rows:
            if k in r["results"]:
                m = r["results"][k]
                out.append(f"| {meth} | {label} | " + " | ".join(fmt(m.get(c)) for c in COLS) + f" | {m['n_trials']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="E1")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = {}
    for m in ORDER:
        p = ROOT / f"results_{m}{a.tag}.json"
        if p.exists():
            res[m] = json.loads(p.read_text())
    R = a.rung
    print(f"### Sim, rung {R}, probe P1+P5 (raw, uncalibrated quantiles)\n")
    rows = [(f"val/{R}/default/all", "in-dist (sizes 1-3)")]
    rows += [(f"val/{R}/default/k{k}", f"in-dist k{k}") for k in (1, 2, 3)]
    rows += [(f"test_heldout/{R}/default/all", "held-out sizes 4-6")]
    rows += [(f"test_heldout/{R}/default/k{k}", f"held-out k{k}") for k in (4, 5, 6)]
    for sub in ("consequential", "benign"):
        rows += [(f"val/{R}/default/all/{sub}", f"in-dist, {sub}"), (f"test_heldout/{R}/default/all/{sub}",
                                                                      f"held-out, {sub}")]
    print(table(res, None, rows))
    print(f"\n### Physical types, rung {R}, P1+P5\n")
    print(table(res, None, [(f"test_physical/{R}/default/phys_{t}{sub}", t + sub.replace("/", ", "))
                            for t in ("single", "pair", "L3", "square4") for sub in ("", "/consequential")]))
    print("\n### Ladder (held-out sizes 4-6) and OOD AUROC of ppc_score (physical types, payload / low-mu; healthy-vs-healthy when available)\n")
    print("| method | rung | pids | mode_acc | fault_recall | y_mae | cover90 | width90 | AUROC payload | AUROC low-mu |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for meth, r in res.items():
        for rung in ("E0", "E1", "E2"):
            for pt in ("default", "all"):
                k = f"test_heldout/{rung}/{pt}/all"
                if k not in r["results"]:
                    continue
                m = r["results"][k]
                ap_ = r["results"].get(f"test_physical/{rung}/{pt}/ood_auroc_payload_healthy",
                                       r["results"].get(f"test_physical/{rung}/{pt}/ood_auroc_payload"))
                al = r["results"].get(f"test_physical/{rung}/{pt}/ood_auroc_low_mu_healthy",
                                      r["results"].get(f"test_physical/{rung}/{pt}/ood_auroc_low_mu"))
                print(f"| {meth} | {rung} | {pt} | {fmt(m.get('mode_acc'))} | {fmt(m.get('fault_recall'))} | "
                      f"{fmt(m.get('y_mae'))} | {fmt(m.get('cover90'))} | {fmt(m.get('width90'))} | {fmt(ap_)} | "
                      f"{fmt(al)} |")
    print("\n### Timing\n")
    for meth, r in res.items():
        print(f"- {meth}: fit wall {r['fit'].get('wall_s', 0):.0f} s, predict "
              f"{r['results'].get('predict_ms_per_trial', 0):.2f} ms/trial (256 samples), eval wall "
              f"{r.get('eval_wall_s', 0):.0f} s")


if __name__ == "__main__":
    main()
