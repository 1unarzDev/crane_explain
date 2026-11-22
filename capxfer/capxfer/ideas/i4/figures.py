"""I4 figures from frozen I4 result files (no recomputation).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.figures   # -> ~/.cache/capxfer/I4/fig_*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import I4

# reference categorical palette, fixed order (dataviz skill references/palette.md, light mode)
COL = {"structured_ensemble": "#2a78d6", "ensemble": "#eb6834", "structured_npe": "#1baf7a", "blackbox": "#eda100",
       "analytic_bayes": "#e87ba4", "nominal_sim": "#6250d6"}
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def degradation():
    files = {m: I4 / f"bench_v1_W_{m}.json" for m in COL}
    res = {m: json.loads(p.read_text()) for m, p in files.items() if p.exists()}
    panels = [("raw", "joint", "raw quantiles, joint-10"), ("split_indist_joint", "joint", "split CQR calibrated on sizes 1-3"),
              ("lofo_group_ss1_joint", "joint", "group conformal (ss1), other families"),
              ("raw", "width", "raw median width (marginal 5-95 %)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (v, key, title) in zip(axs, panels):
        for m, r in res.items():
            xs = ([1, 2, 3] if v == "raw" else []) + sorted(int(k) for k in r["by_size"])
            ys = []
            for k in xs:
                src = r["indist"].get(str(k)) if k <= 3 else r["by_size"][str(k)]
                vv = "raw" if k <= 3 and v != "raw" else v      # sizes 1-3: in-distribution raw reference
                ys.append(src[vv][key] if src and vv in src else np.nan)
            ax.plot(xs, ys, "-o", color=COL[m], lw=2, ms=5, label=m.replace("_", " "))
        if key == "joint":
            ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
            ax.set_ylim((0.35, 1.02) if v == "raw" else (0.75, 1.02))
        ax.axvspan(0.6, 3.4, color="#f0efec", zorder=0)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("modules (sizes 1-3 = training families)", fontsize=8, color=INK2)
        _style(ax)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(I4 / "fig_degradation.png", dpi=160)


def per_family():
    res = {m: json.loads((I4 / f"bench_v1_W_{m}.json").read_text()) for m in ("structured_ensemble", "ensemble")
           if (I4 / f"bench_v1_W_{m}.json").exists()}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, v in zip(axs, ("raw", "lofo_group_ss1_joint")):
        for i, (m, r) in enumerate(res.items()):
            fams = sorted(r["per_family"], key=lambda f: (int(f.split("_")[0][1:]), f))
            ys = [r["per_family"][f][v]["joint"] for f in fams]
            xs = np.arange(len(fams)) + (i - 0.5) * 0.35
            ax.bar(xs, ys, width=0.35, color=COL[m], label=m.replace("_", " "))
        ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"per-family joint-10 coverage: {v}", fontsize=9)
        ax.set_xlabel("free-polyomino family (sorted by size)", fontsize=8, color=INK2)
        _style(ax)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(I4 / "fig_per_family.png", dpi=160)


if __name__ == "__main__":
    for f in (degradation, per_family):
        try:
            f()
        except Exception as e:      # missing inputs
            print(f.__name__, "skipped:", e)
