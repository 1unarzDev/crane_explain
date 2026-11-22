"""Task 1: leave-family-out retraining FROM SCRATCH (no warm start: a warm start from models/W would leak the
held-out family through the base weights).

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.lofo_train --type single [--methods structured_ensemble,ensemble]

Recipe = the W generation (trainW: 60k random assemblies of sizes 1-3, DR_WIDE, fault prior 0.3/0.15/0.15/0.4) with
every group of the held-out type's shape family removed (assembly.shape_family labels all rotations/reflections of a
shape with the same (size, bbox) label; for sizes <= 3 each label is exactly one free polyomino). Same architectures,
seeds and steps as models/W (structured_ensemble 5 x 6000, ensemble 5 x 3000). Early stopping on every 10th
remaining group (W used devW sizes 1-3, which contain the held-out family, so it cannot be used here).
--type none trains the matched in-family control with the identical recipe (no family removed).
blackbox / structured_npe are evaluated as member 0 of the corresponding ensemble (same architecture, steps, data).
Models: ~/.cache/capxfer/I4/models/I4_lofo_<type>/<method>.pt + manifest.json (capxfer.learn.registry, root=I4).
"""
from __future__ import annotations

import argparse
import json
import time

from capxfer.learn import data as D
from capxfer.learn import registry as R
from capxfer.learn.train import CONFIGS

from .common import A_ROOT, I4, TYPE_FAMILY, family_of_compact, load_feats_dir

STEPS = dict(structured_ensemble=6000, ensemble=3000, structured_npe=6000, blackbox=6000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default=None, choices=tuple(TYPE_FAMILY) + ("none",))
    ap.add_argument("--types", default=None, help="comma list: load trainW ONCE, then fork one trainer per type "
                    "(copy-on-write shared features; CUDA is initialised only in the children)")
    ap.add_argument("--methods", default="structured_ensemble,ensemble")
    ap.add_argument("--mem-frac", type=float, default=0.07)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", default=None, help="per-method steps, e.g. structured_ensemble=2000,ensemble=1500")
    ap.add_argument("--build-cache", action="store_true",
                    help="featurize trainW once into I4/feats_trainW.pt; trainers then torch.load(mmap=True) it so the "
                         "page cache is shared between parallel trainers (fork after torch CPU work deadlocks)")
    a = ap.parse_args()
    import torch
    cache = I4 / "feats_trainW.pt"
    if a.build_cache:
        feats = load_feats_dir(A_ROOT / "trainW", max_shards=a.max_shards, log=lambda s: print(s, flush=True))
        torch.save(feats, cache)
        print("cache written", cache, flush=True)
        return
    all_feats = torch.load(cache, mmap=True, weights_only=False) if cache.exists() else None
    run(a, all_feats)


def run(a, all_feats):
    fam = TYPE_FAMILY.get(a.type)
    tag = f"I4_lofo_{a.type}"
    t0 = time.time()
    if all_feats is None:
        keep = (lambda c: family_of_compact(c) != fam) if fam else None
        feats = load_feats_dir(A_ROOT / "trainW", keep=keep, max_shards=a.max_shards, log=lambda s: print(s, flush=True))
    else:
        feats = [F for F in all_feats if str(F["labels"]["family"]) != fam]
    fams = sorted({str(F["labels"]["family"]) for F in feats})
    assert fam not in fams
    tr = [F for i, F in enumerate(feats) if i % 10]
    va = [F for i, F in enumerate(feats) if i % 10 == 0]
    n_tr = sum(F["wheel_x"].shape[0] for F in tr)
    print(f"{tag}: excluded {fam}; families {fams}; train {len(tr)} groups / {n_tr} trials; val {len(va)}; "
          f"load {time.time() - t0:.0f}s", flush=True)
    for meth in a.methods.split(","):
        klass, cfg, _ = CONFIGS[meth]
        over = dict(kv.split("=") for kv in a.steps.split(",")) if a.steps else {}
        steps = int(int(over.get(meth, STEPS[meth])) * a.steps_scale)
        cfg = {**cfg, "steps": steps, "mem_frac": a.mem_frac}
        if "n_members" in cfg:
            cfg["n_members"] = a.members
        model = klass(cfg)
        t1 = time.time()
        fit = model.fit(D.batch_iter(tr, 64, 16, seed=1), D.batch_iter(va, 64, 16, seed=2), cfg)
        out = R.model_path(tag, meth, root=I4)
        model.save(out)
        wall = time.time() - t1
        R.record(tag, meth, [A_ROOT / "trainW" / "manifest.json"], root=I4,
                 extra=dict(kind="I4_lofo_from_scratch", excluded_family=fam, held_out_type=a.type,
                            train_families=fams, n_train_trials=n_tr, steps=steps, n_members=cfg.get("n_members"), wall_s=wall,
                            fit=json.loads(json.dumps(fit, default=float))))
        print(f"{tag} {meth}: {wall:.0f}s -> {out}", flush=True)
    print(f"{tag} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
