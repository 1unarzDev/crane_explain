"""Train and evaluate WS-C methods on the sim held-out-family protocol.

  PYTHONPATH=. .venv/bin/python -m capxfer.learn.train --methods structured_npe,blackbox --steps 6000
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.train --methods analytic_bayes --eval-only
Train families: assembly sizes 1-3 ("train"). Evaluation: held-out families sizes 4-6 ("test_heldout"), the four
physical types ("test_physical": single/pair/L3 are train families, square4 is held out), in-distribution "val".
Outputs: ~/.cache/capxfer/C/models/<method>.pt, ~/.cache/capxfer/C/results_<method>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import metrics as Mx
from . import registry as R
from .analytic_bayes import AnalyticBayes
from .blackbox import Blackbox
from .ensemble import Ensemble, StructuredEnsemble
from .structured_npe import NominalSim, StructuredNPE

ROOT = Path(os.path.expanduser("~/.cache/capxfer/C"))
CONFIGS = {
    "structured_npe": (StructuredNPE, dict(), "train"),
    "blackbox": (Blackbox, dict(), "train"),
    "ensemble": (Ensemble, dict(n_members=5, member="blackbox"), "train"),
    "structured_ensemble": (StructuredEnsemble, dict(n_members=5, member="structured_npe"), "train"),
    "nominal_sim": (NominalSim, dict(), "train_nominal"),
    "analytic_bayes": (AnalyticBayes, dict(cap=1024), None),
}
EVAL_RUNGS = ("E0", "E1", "E2")
EVAL_SETS = ("val", "test_heldout", "test_physical")


def a_shards_available():
    return sorted(glob.glob(os.path.expanduser("~/.cache/capxfer/A/*/manifest.json")))


def _size(F):
    fam = F["labels"]["family"] or ""
    return int(fam.split("_")[0][1:]) if fam.startswith("k") else None


def a_splits(name, ood_names=(), val_every=10, train_name=None):
    """Family split from a mixed WS-A dataset: train = random assemblies of size <= 3 (every val_every-th group -> val),
    test_heldout = random sizes 4-6, test_physical = the four physical types (+ OOD datasets' groups)."""
    allf = D.load_feats(name)
    rnd = [F for F in allf if not str(F["labels"]["family"]).startswith("phys_")]
    small = [F for F in rnd if (_size(F) or 99) <= 3]
    sets = dict(train=[F for i, F in enumerate(small) if i % val_every],
                val=[F for i, F in enumerate(small) if i % val_every == 0],
                test_heldout=[F for F in rnd if (_size(F) or 0) >= 4],
                test_physical=[F for F in allf if str(F["labels"]["family"]).startswith("phys_")])
    if train_name:                      # separate WS-C-generated training set; dev set's small families -> val
        sets["train"] = D.load_feats(train_name)
        sets["val"] = small[::3]
    for o in ood_names:
        sets["test_physical"] += [F for F in D.load_feats(o) if str(F["labels"]["family"]).startswith("phys_")]
    print({k: (len(v), sum(F["wheel_x"].shape[0] for F in v)) for k, v in sets.items()}, flush=True)
    return sets


def evaluate(model, feats_by_set, rungs=EVAL_RUNGS, pids=D.DEFAULT_PIDS, n_samples=256, extra_pids=True):
    res = {}
    t_pred, n_pred = 0.0, 0
    for set_name, feats in feats_by_set.items():
        for rung in rungs:
            pid_opts = [("default", pids)] + ([("all", D.PIDS)] if extra_pids and rung == "E2" else [])
            for ptag, pl in pid_opts:
                groups = {}
                for F in feats:
                    v = D.apply_view(F, rung, pl)
                    t0 = time.time()
                    p = model.predict(v, n_samples)
                    t_pred += time.time() - t0
                    n_pred += v["wheel_x"].shape[0]
                    tm = Mx.trial_metrics(p, v, model.cfg["q_levels"])
                    ood = F["labels"].get("ood", "none")
                    fam = F["labels"]["family"]
                    key = fam if set_name == "test_physical" else "all"
                    groups.setdefault((key, ood), []).append(tm)
                    if set_name != "test_physical" and str(fam).startswith("k"):
                        groups.setdefault((fam.split("_")[0], ood), []).append(tm)
                for (key, ood), rows in groups.items():
                    if ood != "none":
                        continue
                    pre = f"{set_name}/{rung}/{ptag}/{key}"
                    res[pre] = Mx.aggregate(rows)
                    if "consequential" in rows[0]:
                        res[pre + "/consequential"] = Mx.aggregate(rows, ("consequential", True))
                        res[pre + "/benign"] = Mx.aggregate(rows, ("consequential", False))
                if set_name == "test_physical":       # OOD (payload / low-mu) separability of ppc_score
                    pin = np.concatenate([r["ppc"] for (k, o), rs in groups.items() if o == "none" for r in rs])
                    pin_h = np.concatenate([r["ppc"][r["healthy_trial"]] for (k, o), rs in groups.items() if o == "none"
                                            for r in rs])
                    for o in ("payload", "low_mu"):
                        rows_o = [r for (k, oo), rs in groups.items() if oo == o for r in rs]
                        if rows_o:
                            pout = np.concatenate([r["ppc"] for r in rows_o])
                            res[f"{set_name}/{rung}/{ptag}/ood_auroc_{o}"] = Mx.auroc(pin, pout)
                            # healthy-vs-healthy (OOD sets may carry no wheel faults): removes the fault confound
                            pout_h = np.concatenate([r["ppc"][r["healthy_trial"]] for r in rows_o])
                            res[f"{set_name}/{rung}/{ptag}/ood_auroc_{o}_healthy"] = Mx.auroc(pin_h, pout_h)
                    pooled = [r for (k, o), rs in groups.items() if o == "none" for r in rs]
                    pre = f"{set_name}/{rung}/{ptag}/all"
                    res[pre] = Mx.aggregate(pooled)
                    res[pre + "/consequential"] = Mx.aggregate(pooled, ("consequential", True))
                    res[pre + "/benign"] = Mx.aggregate(pooled, ("consequential", False))
    res["predict_ms_per_trial"] = 1000 * t_pred / max(n_pred, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="structured_npe,blackbox")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-train-shards", type=int, default=None)
    ap.add_argument("--max-eval-shards", type=int, default=None)
    ap.add_argument("--source", default=None, help="override data.SOURCE (C_stub | A_featurize)")
    ap.add_argument("--mem-frac", type=float, default=0.2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--a-data", default=None, help="WS-A dataset name (implies --source A_featurize)")
    ap.add_argument("--a-ood", default="", help="comma list of WS-A OOD dataset names")
    ap.add_argument("--a-train", default=None, help="separate training set (e.g. trainA from learn.gen_a)")
    ap.add_argument("--a-train-nominal", default=None, help="nominal-domain training set for nominal_sim")
    ap.add_argument("--ab-table-dr", default="nominal", choices=("nominal", "wide_centre"),
                    help="analytic_bayes hypothesis-table plant: params nominal or the centre of sim.DR_WIDE (gen W)")
    ap.add_argument("--no-eval", action="store_true", help="fit + save + record only (skip train.evaluate tables)")
    a = ap.parse_args()
    if a.source:
        D.SOURCE = a.source
    if a_shards_available():
        print("NOTE: WS-A shards present:", a_shards_available(), flush=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    feats_cache = {}

    def feats(name, mx=None):
        if (name, mx) not in feats_cache:
            feats_cache[(name, mx)] = D.load_feats(name, mx)
        return feats_cache[(name, mx)]

    if a.a_data:
        D.SOURCE = "A_featurize"
        sp = a_splits(a.a_data, [o for o in a.a_ood.split(",") if o], train_name=a.a_train)
        feats_cache.update({(k, a.max_train_shards): v for k, v in sp.items()})
        if a.a_train_nominal and "nominal_sim" in a.methods:
            feats_cache[("train_nominal", a.max_train_shards)] = D.load_feats(a.a_train_nominal)
        eval_sets = {s: sp[s] for s in EVAL_SETS}
    else:
        eval_sets = {s: feats(s, a.max_eval_shards) for s in EVAL_SETS}
    for meth in a.methods.split(","):
        klass, cfg, train_name = CONFIGS[meth]
        cfg = {**cfg, "steps": a.steps, "mem_frac": a.mem_frac}
        if meth == "analytic_bayes" and a.ab_table_dr == "wide_centre":
            from capxfer import sim as _sim
            from .analytic_bayes import dr_centre
            cfg["table_dr"] = dr_centre(_sim.DR_WIDE)
        path = R.model_path(a.tag, meth)                     # versioned generation dir models/<tag>/ (M2)
        t0 = time.time()
        old = R.resolve(a.tag, meth)
        if a.eval_only and old is not None:
            path = old
            model = klass.load(path, mem_frac=a.mem_frac)
            fit_info = {"loaded": str(path)}
        else:
            model = klass(cfg)
            if train_name is None:                               # analytic Bayes: tune on val, one asm per batch
                vit = (D.apply_view(F, r, D.DEFAULT_PIDS, torch.arange(64))
                       for F in eval_sets["val"] for r in ("E1",))
                fit_info = model.fit(None, vit, cfg)
            else:
                if a.a_data and train_name == "train_nominal" and not a.a_train_nominal:
                    print("nominal_sim needs a nominal-domain WS-A dataset; skipping", flush=True)
                    continue
                tr = feats(train_name, a.max_train_shards)
                fit_info = model.fit(D.batch_iter(tr, 64, 16, seed=1), D.batch_iter(eval_sets["val"], 64, 16, seed=2),
                                     cfg)
            model.save(path)
            dname = {"train": a.a_train, "train_nominal": a.a_train_nominal}.get(train_name) if a.a_data else train_name
            extra = dict(steps=a.steps, dataset=dname, fit={k: v for k, v in fit_info.items()
                                                            if isinstance(v, (int, float, str))})
            if train_name is None:                               # AB: record its tuning set and table plant
                extra.update(tuning_data=a.a_data, table_dr=model.cfg.get("table_dr") or "nominal")
            R.record(a.tag, meth, [D.manifest_path(dname)] if dname else [], extra=extra)
        fit_info["wall_s"] = time.time() - t0
        print(meth, "fit", fit_info, flush=True)
        if a.no_eval:
            continue
        res = evaluate(model, eval_sets)
        out = dict(method=meth, source=D.SOURCE, fit=fit_info, cfg={k: v for k, v in model.cfg.items()
                                                                     if isinstance(v, (int, float, str, list, tuple))},
                   results=res, eval_wall_s=time.time() - t0 - fit_info["wall_s"])
        (ROOT / f"results_{meth}{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
        print(meth, "done", json.dumps({k: v for k, v in res.items() if k.endswith("/all")}, default=float)[:3000],
              flush=True)


if __name__ == "__main__":
    main()
