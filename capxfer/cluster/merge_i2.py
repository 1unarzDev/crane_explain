"""Merge fetched i2_sweep_cell outputs into one exp_sweep-format file (analyze.py reads it).

  PYTHONPATH=. .venv/bin/python -m cluster.merge_i2 FETCH_DIR --out ~/.cache/capxfer/I2/sweep_n20_24_cluster.pt
"""
import argparse
import os
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_idx = {}
    for f in sorted(Path(a.src).expanduser().rglob("i2_asm*.pt")):
        r = torch.load(f, weights_only=False)
        by_idx.setdefault(r["idx"], r)
    res = [by_idx[k] for k in sorted(by_idx)]
    if not res:
        raise SystemExit("no i2_asm*.pt under " + a.src)
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(results=res, g_weak=res[0]["g_weak"], n_dr=res[0]["n_dr"], params_sha=None, source="SIMULATION",
                    origin="research-cluster i2_sweep_cell", idx=sorted(by_idx)), out)
    ns = sorted({r["n"] for r in res})
    print(f"wrote {out}: {len(res)} morphologies, n in {ns}")


if __name__ == "__main__":
    main()
