"""Extract per-trial metadata (assembly dict, true health, domain, noise-free rho) from devW shards -> small cache."""
import pathlib
import pickle

import torch

SRC = pathlib.Path.home() / ".cache/capxfer/A/devW"
OUT = pathlib.Path.home() / ".cache/capxfer/I1/devW_meta.pkl"

meta = {}
for p in sorted(SRC.glob("shard_*.pt")):
    for c in torch.load(p, weights_only=False):
        m, lab = c["meta"], c["labels"]
        meta_g = dict(assembly=m["assembly"], char_length=m["char_length"], trial_ids=list(m["trial_ids"]),
                      mode=lab["mode"].numpy(), g=lab["g"].numpy(), rho=lab["rho_twist"].float().numpy(),
                      domain={k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in lab["domain"].items()},
                      family=lab.get("family"), asm_type=lab.get("asm_type"), ood=lab["ood"])
        for i, t in enumerate(meta_g["trial_ids"]):
            meta[t] = (len(meta), i)
        meta.setdefault("_groups", []).append(meta_g)
    print(p.name, len(meta["_groups"]), flush=True)
pickle.dump(meta, open(OUT, "wb"))
print("wrote", OUT)
