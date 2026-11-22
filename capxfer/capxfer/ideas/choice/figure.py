"""CHOICE figure (Fig. C1) from ~/.cache/capxfer/CHOICE/results.json (SIMULATION ONLY).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.choice.figure [alpha] [tau]
Writes ~/.cache/capxfer/CHOICE/fig_choice.{png,pdf}.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path.home() / ".cache/capxfer/CHOICE"
LAB = dict(cert="certificate (pre-split)", bb="black-box carry-over", healthy="assume healthy", random="random",
           probe_all="split, then probe all", oracle="oracle")
COL = dict(cert="#1b6ca8", bb="#d1495b", healthy="#edae49", random="#8d8d8d", probe_all="#00798c", oracle="#222222")


def main(alpha="0.1", tau="0.8"):
    R = json.loads((OUT / "results.json").read_text())["tables"]
    pols = ("cert", "bb", "healthy", "random", "probe_all", "oracle")
    sizes = range(2, 7)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for p in pols:
        T = [R[f"{alpha}|{tau}|{p}"][f"nmod{n}"] for n in sizes]
        ax[0].plot(sizes, [t["single_shot"]["success"] for t in T], "o-", color=COL[p], label=LAB[p])
        ax[1].plot(sizes, [t["single_shot"]["unsafe"] for t in T], "o-", color=COL[p])
        ax[2].plot(sizes, [t["mission"]["time_per_success"] for t in T], "o-", color=COL[p])
    ax[0].set_ylabel("dock success (first dispatch)")
    ax[1].set_ylabel(f"unsafe dispatch P(sent, z < {tau})")
    ax[2].set_ylabel("mission time per successful dock (s)")
    for a in ax:
        a.set_xlabel("modules in the faulted assembly")
        a.set_xticks(list(sizes))
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Which module to send? (simulation; alpha = {alpha}, tau = {tau})", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_choice_{alpha}_{tau}.{ext}", dpi=160)
    print("wrote", OUT / f"fig_choice_{alpha}_{tau}.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
