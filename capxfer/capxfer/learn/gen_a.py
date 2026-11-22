"""Generate WS-C training sets with WS-A's datagen (full sensor model) into ~/.cache/capxfer/C/A_gen/<name>.
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.gen_a --name trainA --n-trials 60000
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.gen_a --name trainA_nominal --n-trials 60000 --nominal
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.gen_a --name trainW_nominal --n-trials 60000 --nominal --seed 21 \
      --engine compile --wheel-budget 8192 --gpu-frac 0.36     # W generation's H5 ablation set (trainW recipe, NOMINAL_DR)
Roller-locked gets a wider prior share (WS-A: least-validated mode)."""
from __future__ import annotations

import argparse

import torch

from capxfer import datagen
from . import _stub

OUT = _stub.ROOT / "A_gen"
FAULT_KW = dict(mode_probs=(0.3, 0.15, 0.15, 0.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-trials", type=int, default=60000)
    ap.add_argument("--sizes", default="1,2,3")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--nominal", action="store_true")
    ap.add_argument("--gpu-frac", type=float, default=0.06)
    ap.add_argument("--wheel-budget", type=int, default=2048)
    ap.add_argument("--engine", default="graph", help="datagen engine (trainW used 'compile')")
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    dr = dict(_stub.NOMINAL_DR) if a.nominal else None
    datagen.generate(a.name, a.n_trials, {"random": 1.0}, seed=a.seed, sizes=tuple(int(s) for s in a.sizes.split(",")),
                     wheel_budget=a.wheel_budget, engine=a.engine, out_root=OUT, fault_kw=FAULT_KW, dr=dr,
                     log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
