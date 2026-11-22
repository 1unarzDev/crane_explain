"""Retrain after sys-ID: warm-start fine-tuning of trained WS-C methods on fresh sim data drawn from the
sys-ID-centred domain-randomisation spec (study plan: retrain AFTER sys-ID, before the freeze).

  from capxfer.learn.retrain import retrain_after_sysid
  info = retrain_after_sysid(dr_spec, "~/.cache/capxfer/C/models/A", methods=("structured_ensemble",), tag="S1")

  PYTHONPATH=. .venv/bin/python -m capxfer.learn.retrain --dr-spec spec.json --base ~/.cache/capxfer/C/models/A \
      --methods structured_ensemble --tag S1

`dr_spec`: a `sim.sample_domain` DR dict (key -> (lo, hi); missing keys keep sim.DR_DEFAULT), a sys-ID fit dict
(with "best"; converted by `sysid.dr_around(fit, rel_width)`), or a path to a JSON file holding either.
Output: models/<tag>/<method>.pt + manifest.json (registry M2), data in A_gen/<tag>_data, and a timing record
models/<tag>/retrain_info.json. analytic_bayes has no learned weights (nominal-parameter physics) and is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import registry as R
from .base import Predictor, cap_memory

FAULT_KW = dict(mode_probs=(0.3, 0.15, 0.15, 0.4))          # same fault prior as trainA (gen_a.py)
DEFAULTS = dict(n_trials=20000, sizes=(1, 2, 3), steps=1000, lr=1e-4, mem_frac=0.25, gen_mem_frac=None, seed=101,
                rel_width=0.15, val_every=10, batch_size=64, per_asm=16)


def resolve_dr_spec(dr_spec, rel_width=0.15):
    if isinstance(dr_spec, (str, Path)):
        dr_spec = json.loads(Path(os.path.expanduser(str(dr_spec))).read_text())
    if isinstance(dr_spec, dict) and "best" in dr_spec:
        from capxfer import sysid
        dr_spec = sysid.dr_around(dr_spec, rel_width)
    return {k: (tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v)) for k, v in dr_spec.items()}


def generate(dr, name, n_trials, sizes, seed, gen_mem_frac, log=print):
    """WS-A datagen with the given DR into A_gen/<name> (reused if a complete manifest exists)."""
    from capxfer import datagen
    from .gen_a import OUT
    mp = OUT / name / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()).get("complete"):
        m = json.loads(mp.read_text())
        if m["args"].get("dr") == {k: list(v) if isinstance(v, tuple) else v for k, v in dr.items()}:
            log(f"reusing {mp}")
            return mp
    cap_memory(gen_mem_frac)
    datagen.generate(name, n_trials, {"random": 1.0}, seed=seed, sizes=tuple(sizes), wheel_budget=2048,
                     engine="graph", out_root=OUT, fault_kw=FAULT_KW, dr=dr, log=lambda s: log(s))
    return mp


def _base_path(base_model_dir, meth):
    d = Path(os.path.expanduser(str(base_model_dir)))
    for p in (d / f"{meth}.pt", d.parent / f"{meth}_{d.name}.pt"):     # versioned dir, then legacy flat file
        if p.exists():
            return p
    raise FileNotFoundError(f"no base checkpoint for {meth} in {d}")


def retrain_after_sysid(dr_spec, base_model_dir, methods=("structured_ensemble",), tag=None, data_name=None,
                        log=print, feats=None, root=R.ROOT, **kw):
    """Fine-tune `methods` (warm start from base_model_dir) on fresh sim data from `dr_spec`. Returns timing/info.
    feats: optional pre-loaded list of Feat groups (skips generation; used by tests)."""
    cfg = {**DEFAULTS, **kw}
    t_all = time.time()
    tag = tag or time.strftime("sysid_%Y%m%d_%H%M%S")
    dr = resolve_dr_spec(dr_spec, cfg["rel_width"])
    info = dict(tag=tag, dr_spec=dr, cfg={k: v for k, v in cfg.items()}, base_model_dir=str(base_model_dir),
                methods={})
    data_manifests = []
    t0 = time.time()
    if feats is None:
        data_name = data_name or f"{R.tag_name(tag)}_data"
        # one process-wide cap (loading/generation may only lower it, M1): generate under the training cap
        mp = generate(dr, data_name, cfg["n_trials"], cfg["sizes"], cfg["seed"], cfg["gen_mem_frac"] or cfg["mem_frac"],
                      log)
        data_manifests.append(mp)
        info["gen_s"] = time.time() - t0
        t0 = time.time()
        src = D.SOURCE
        D.SOURCE = "A_featurize"
        try:
            feats = D.load_feats(data_name)
        finally:
            D.SOURCE = src
        info["load_s"] = time.time() - t0
    tr = [F for i, F in enumerate(feats) if i % cfg["val_every"]] or feats
    va = [F for i, F in enumerate(feats) if i % cfg["val_every"] == 0] or feats
    info["n_train_trials"] = int(sum(F["wheel_x"].shape[0] for F in tr))
    for meth in methods:
        if meth == "analytic_bayes":
            info["methods"][meth] = "skipped (no learned weights)"
            continue
        t0 = time.time()
        bp = _base_path(base_model_dir, meth)
        model = Predictor.load(bp, mem_frac=cfg["mem_frac"])
        fit_cfg = dict(steps=cfg["steps"], lr=cfg["lr"], val_every=max(100, cfg["steps"] // 4),
                       log_every=max(50, cfg["steps"] // 10))
        fit = model.fit(D.batch_iter(tr, cfg["batch_size"], cfg["per_asm"], seed=cfg["seed"]),
                        D.batch_iter(va, cfg["batch_size"], cfg["per_asm"], seed=cfg["seed"] + 1), fit_cfg)
        out = R.model_path(tag, meth, root)
        model.save(out)
        wall = time.time() - t0
        R.record(tag, meth, data_manifests, root=root, extra=dict(kind="retrain_after_sysid", base_checkpoint=str(bp),
                                                       base_sha256=R.sha256_file(bp), dr_spec=dr, fit_cfg=fit_cfg,
                                                       wall_s=wall))
        info["methods"][meth] = dict(path=str(out), wall_s=wall, fit=json.loads(json.dumps(fit, default=float)))
        log(f"retrain {meth}: {wall:.0f} s -> {out}")
    info["total_s"] = time.time() - t_all
    (R.model_dir(tag, root) / "retrain_info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr-spec", required=True, help="JSON file: DR dict or sys-ID fit (with 'best')")
    ap.add_argument("--base", default=str(R.ROOT / "models" / "A"))
    ap.add_argument("--methods", default="structured_ensemble")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-trials", type=int, default=DEFAULTS["n_trials"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--mem-frac", type=float, default=DEFAULTS["mem_frac"])
    a = ap.parse_args()
    info = retrain_after_sysid(a.dr_spec, a.base, a.methods.split(","), a.tag, n_trials=a.n_trials, steps=a.steps,
                               mem_frac=a.mem_frac, log=lambda s: print(s, flush=True))
    print(json.dumps({k: v for k, v in info.items() if k != "dr_spec"}, indent=1, default=str))


if __name__ == "__main__":
    main()
