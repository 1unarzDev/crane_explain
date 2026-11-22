"""I2: extract per-group geometry + labels from WS-A dev_v2 raw shards in the same order as learn.c2_eval splits."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main(name="dev_v2", out=os.path.expanduser("~/.cache/capxfer/I2/devmeta_dev_v2.pt")):
    root = Path(os.path.expanduser(f"~/.cache/capxfer/A/{name}"))
    m = json.loads((root / "manifest.json").read_text())
    groups = []
    for s in m["shards"]:
        sh = torch.load(root / s["file"], weights_only=False, mmap=True)
        for g in sh:
            a = g["meta"]["assembly"]
            a = a if isinstance(a, dict) else vars(a)
            groups.append(dict(family=str(g["labels"]["family"]), asm_type=g["labels"].get("asm_type"),
                               ood=g["labels"].get("ood", "none"),
                               wheel_pos=np.asarray(a["wheel_pos"]), w_hat=np.asarray(a["w_hat"]),
                               r_hat=np.asarray(a["r_hat"]), modules=a["modules"],
                               char_length=float(g["meta"]["char_length"]),
                               mode=g["labels"]["mode"].cpu().numpy(), g=g["labels"]["g"].cpu().numpy(),
                               rho=g["labels"]["rho_twist"].cpu().numpy()))
        del sh
        print(s["file"], len(groups), flush=True)
    torch.save(groups, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
