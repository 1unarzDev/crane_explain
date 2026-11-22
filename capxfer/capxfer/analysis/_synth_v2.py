"""Fully synthetic inputs for the PROTOCOL_V2 dry run of `run_realv2` (tests only).

Numbers produced here are meaningless and must never appear in the paper. `source` is "SYNTHETIC" unless a test
deliberately sets "FIELD" to exercise the provenance gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import realv2_design as D
from .realv2 import KEYS, RHO_COLS, _sha

LOSS = {"pair": 0.049, "L3": 0.025, "square4": 0.019}     # sim masking medians (I2), used as synthetic truth


def _rho(rng, base, n, sd):
    return base + rng.normal(0, sd, size=(n, len(KEYS)))


def build_v2(outdir, seed=b"\x07" * 32, sd=0.008, reveal=True, cert_fg=0.0, source="FIELD", lone_strafe=0.65):
    """Write mask/pred/r3/demo CSVs + a freeze record + manifest into `outdir`; return the manifest path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    sched = D.expand_schedule(seed)
    k2 = [KEYS.index("P2+"), KEYS.index("P2-")]
    healthy = np.full(len(KEYS), 0.97)
    rows, preds = [], []
    worst = {}
    for t in sched["mask_trials"]:
        b = t["build"]
        at = "single" if b.startswith("single_") else b
        eff = np.zeros(len(KEYS))
        if t["module"]:
            mw = f"{t['module']}:{t['wheel']}"
            if mw not in worst:
                worst[mw] = int(rng.integers(len(KEYS)))
            if at == "single":
                eff[:] = 0.2
                eff[k2] = (healthy[0] - lone_strafe) if reveal else 0.05
            else:
                eff[worst[mw]] = LOSS[at]
        r = _rho(rng, healthy - eff, 1, sd)[0]
        rows.append(dict(trial_id=f"T{t['seq']:04d}", session="S1", build=b, assembly_type=at, module=t["module"],
                         wheel=t["wheel"], firmware=t["firmware"], excluded=False, source=source,
                         session_status="FIELD_OK", start_utc="2026-12-01T10:00:00Z", echo_ok=True,
                         **dict(zip(RHO_COLS, r))))
    mask = pd.DataFrame(rows)
    for b, conds in D.mask_builds().items():
        at = "single" if b.startswith("single_") else b
        for m, w in conds:
            mw = f"{m}:{w}"
            zp = lone_strafe if at == "single" else 0.97
            preds.append(dict(build=b, module=m, wheel=w, L_pred=(0.77 if at == "single" else LOSS[at]), z_pred=zp,
                              k_star=KEYS[worst.get(mw, 0)]))
    pred = pd.DataFrame(preds)
    # R3 units: one row per (assembly trial, module)
    units = []
    for r in mask.itertuples():
        if r.assembly_type not in D.MULTI or not isinstance(r.module, str):
            continue
        for m in D.TYPE_MODULES[r.assembly_type]:
            faulted = m == r.module
            sc = (0.35 + rng.normal(0, 0.1)) if faulted else (0.9 + rng.normal(0, 0.03))
            split = sched["r3_split"][f"{r.module}:{r.wheel}"]
            if faulted and split == "TEST" and rng.random() < cert_fg:   # non-exchangeable breakage
                sc = 0.99
            units.append(dict(trial_id=r.trial_id, build=r.build, assembly_type=r.assembly_type, unit_module=m,
                              fault_mw=f"{r.module}:{r.wheel}", r3_split=split,
                              score=sc, sim_lb=sc + 0.1, carry_over_lb=0.95, excluded=False))
    r3 = pd.DataFrame(units)
    eps = []
    for e in sched["demo"]:
        f = set(e["faulted_modules"])
        row = dict(episode=e["episode"], faulted_modules="|".join(sorted(f)), carry_over_lb=0.95,
                   nearest_goal_module=D.DEMO_MODULES[1], random_module=e["random_module"],
                   cert_tie_coin=e["cert_tie_coin"], oracle_tie_coin=e["oracle_tie_coin"],
                   carry_over_tie_coin=e["carry_over_tie_coin"], excluded=False)
        for m in D.DEMO_MODULES:                       # all three docked here; real data dock only chosen modules
            row[f"y_{m}"] = float(rng.random() < (0.05 if m in f else 0.97))
            row[f"s_{m}"] = (0.35 if m in f else 0.9) + rng.normal(0, 0.02)
            row[f"z_alone_{m}"] = lone_strafe if m in f else 0.97
        eps.append(row)
    demo = pd.DataFrame(eps)
    prho = pred[["build", "module", "wheel"]].copy()
    for k in RHO_COLS:
        prho[k] = 0.95
    for name, df in (("mask.csv", mask), ("pred.csv", pred), ("r3.csv", r3), ("demo.csv", demo), ("prho.csv", prho)):
        df.to_csv(out / name, index=False)
    freeze = dict(root=str(out), files={"pred.csv": _sha(out / "pred.csv")}, seed_commitment=D.commitment(seed),
                  freeze_time_utc="2026-11-30T00:00:00Z", git_commit="synthetic")
    (out / "freeze_v2.json").write_text(json.dumps(freeze))
    man = dict(output_dir="out", freeze_record="freeze_v2.json", seed_hex=seed.hex(), mask_trials="mask.csv",
               predictions="pred.csv", r3_units="r3.csv", demo="demo.csv", predicted_rho="prho.csv",
               config=dict(n_boot=500))
    mp = out / "manifest.json"
    mp.write_text(json.dumps(man))
    return mp
