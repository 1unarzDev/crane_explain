"""Fully synthetic study inputs for the end-to-end dry run of `run_frozen` (no upstream code needed).

Numbers produced here are meaningless and must never appear in the paper.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import table as tb

TYPES = {"single": 4, "pair": 8, "L3": 12, "square4": 16}
FAULTS = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
_Z = 1.6448536269514722
PIDS_EXTRA = ("P2", "P3", "P4", "P6x", "P6y")


def _trials(rng, per_cell):
    rows = []
    for at, n in TYPES.items():
        for f in FAULTS:
            for i in range(per_cell):
                sess = f"S{at}{rng.integers(3)}"
                mods = "+".join(sorted(rng.choice(["M1", "M2", "M3", "M4"], size=n // 4, replace=False)))
                rows.append(dict(trial_id=f"{at}-{f}-{i:03d}", assembly_type=at, session=sess, module_set=mods,
                                 fault=f, ood={"F6": "payload", "F7": "low_mu"}.get(f, "none"), source="SYNTHETIC",
                                 n=n))
    return pd.DataFrame(rows)


def _truth(rng, df, ood_with_fault=False):
    N = len(df)
    modes = []
    effect = np.zeros((N, 14))
    for i, r in enumerate(df.itertuples()):
        m = np.zeros(r.n, int)
        k = {"F1": 1, "F2": 2, "F3": 3, "F4": 4}.get(r.fault)
        if ood_with_fault and r.fault in ("F6", "F7") and i % 2 == 0:
            k = int(rng.integers(1, 5))                         # OOD condition + one in-model wheel fault
        if k:
            m[rng.integers(r.n)] = k
        if r.fault == "F5":
            w = rng.choice(r.n, 2, replace=False)
            m[w] = rng.integers(1, 5, 2)
        modes.append(m)
        effect[i] = (np.array([0, .25, .4, .5, .3])[m].sum() / r.n) * (0.5 + rng.random(14))
    type_off = df.assembly_type.map({"single": 0.0, "pair": -0.02, "L3": 0.01, "square4": -0.03}).fillna(0.0).to_numpy()
    mean = 1.0 - 2.0 * effect
    y = mean + type_off[:, None] + 0.05 * rng.standard_normal((N, 14))
    y -= ((df.ood != "none").to_numpy()[:, None]) * 0.12
    return modes, mean, y


def _pred_table(rng, df, modes, mean, y, *, err, spread, sharp, mode_prob=True, source="SYNTHETIC", bias=0.0):
    N = len(df)
    mhat = mean + bias + err * rng.standard_normal((N, 14))
    s = np.sqrt(0.05 ** 2 + err ** 2) * spread
    y_q = np.stack([mhat - _Z * s, mhat, mhat + _Z * s], -1)
    t = {k: df[k].to_numpy().astype(str) for k in tb.STR_KEYS}
    t["source"] = np.full(N, source)
    t.update(y=y, y_q=y_q, q_levels=np.array([0.05, 0.5, 0.95]),
             ppc_score=rng.chisquare(6, N) * (1 + 2.0 * (df.ood != "none").to_numpy()))
    W = max(tb.MAX_WHEELS, int(df.n.max()))
    mask = np.zeros((N, W), bool)
    mt = np.full((N, W), -1, np.int64)
    mp = np.full((N, W, 5), np.nan)
    for i, m in enumerate(modes):
        n = m.size
        mask[i, :n], mt[i, :n] = True, m
        lg = rng.standard_normal((n, 5))
        lg[np.arange(n), m] += sharp
        lg[:, 0] += 0.5
        e = np.exp(lg - lg.max(1, keepdims=True))
        mp[i, :n] = e / e.sum(1, keepdims=True)
    t.update(wheel_mask=mask, mode_true=mt, n_wheels=df.n.to_numpy())
    if mode_prob:
        t["mode_prob"] = mp
    return t


def _probe_table(rng, df):
    info = dict(P2=0.6, P3=0.4, P4=0.4, P6x=0.15, P6y=0.5)
    rows = []
    for r in df.itertuples():
        base = rng.uniform(0.2, 1.0) * (r.fault not in ("F0",))
        for k in range(0, 3):
            for extra in itertools.combinations(PIDS_EXTRA, k):
                seen = ("P1", "P5") + extra
                for c in PIDS_EXTRA:
                    if c in extra:
                        continue
                    after = extra + (c,)
                    real = base * np.exp(-sum(info[p] * rng.uniform(0.5, 1.5) for p in after))
                    rows.append(dict(trial_id=r.trial_id, session=r.session, seen="+".join(sorted(seen)), candidate=c,
                                     exp_regret=real + 0.05 * rng.standard_normal(),
                                     info_gain=info[c] + 0.2 * rng.standard_normal(),
                                     realized_regret_next=real))
    return pd.DataFrame(rows)


def build(outdir, seed=0, per_cell=20, ood_with_fault=False, sizes=None, config=None, redesign=False):
    """Write a complete synthetic manifest + inputs under `outdir`; returns the manifest path.
    ood_with_fault: half of the F6/F7 trials also carry one in-model wheel fault (H3 redesign condition).
    sizes: sim held-out family sizes in modules (default 1..4; e.g. (1, ..., 6) gives 4..24-wheel families).
    redesign: also write structured_ensemble / analytic_bayes tables and a 4-rung probe ladder (H1-H3 redesign)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = _trials(rng, per_cell)
    modes, mean, y = _truth(rng, df, ood_with_fault)
    specs = {
        "structured_npe": {"E0": dict(err=0.10, spread=1.0, sharp=1.0), "E1": dict(err=0.06, spread=1.0, sharp=2.0),
                           "E2": dict(err=0.03, spread=1.0, sharp=3.0)},
        "blackbox": {"E2": dict(err=0.09, spread=0.7, sharp=0.0, mode_prob=False)},
        "ensemble": {"E2": dict(err=0.06, spread=0.6, sharp=2.0)},
        "nominal_sim": {"E2": dict(err=0.06, spread=1.0, sharp=2.0, bias=0.06)},
    }
    if redesign:
        specs["structured_ensemble"] = {"E2": dict(err=0.03, spread=1.0, sharp=3.5)}
        specs["analytic_bayes"] = {"E2": dict(err=0.05, spread=1.0, sharp=0.5)}
    man = dict(output_dir="out", pred_tables={}, sim_tables={}, frozen=False,
               config=dict(dict(n_boot=300, seed=seed), **(config or {})))
    for m, rungs in specs.items():
        for r, kw in rungs.items():
            p = out / f"pred_{m}_{r}.npz"
            tb.save(p, _pred_table(rng, df, modes, mean, y, **kw))
            man["pred_tables"].setdefault(m, {})[r] = p.name
    if redesign:
        man["probe_ladder"] = []
        for label, pids, err in (("P1P5", ["P1", "P5"], 0.08), ("P1P5P2", ["P1", "P5", "P2"], 0.05),
                                 ("P1-P5", ["P1", "P2", "P3", "P4", "P5"], 0.03), ("all", [], 0.015)):
            p = out / f"ladder_{label}.npz"
            tb.save(p, _pred_table(rng, df, modes, mean, y, err=err, spread=1.0, sharp=3.0))
            man["probe_ladder"].append(dict(label=label, pids=pids, table=p.name))
    sim_df = df[df.ood == "none"].copy()
    sim_df["trial_id"] = "sim-" + sim_df.trial_id
    fam = rng.integers(0, 40, len(sim_df))                     # 40 synthetic shape families
    sim_df["assembly_type"] = [f"fam{k:02d}" for k in fam]
    sz = np.asarray(sizes if sizes is not None else (1, 2, 3, 4))
    sim_df["n"] = 4 * sz[fam % sz.size]
    smodes, smean, sy = _truth(rng, sim_df)
    for m, kw in (("structured_npe", dict(err=0.06, spread=1.0, sharp=2.0)),
                  ("nominal_sim", dict(err=0.06, spread=1.0, sharp=2.0))):
        p = out / f"sim_{m}.npz"
        tb.save(p, _pred_table(rng, sim_df, smodes, smean, sy, source="SIMULATION", **kw))
        man["sim_tables"][m] = p.name
    ex = []
    for r in df.itertuples():
        o = r.ood != "none"
        ex.append(dict(trial_id=r.trial_id, explainer="calibrated", n_unsupported_cause=int(rng.random() < 0.03),
                       wheel_cause_asserted=int(o and rng.random() < 0.04), correct_claim_rate=0.95,
                       informativeness=0.5, text="SYNTHETIC calibrated explanation"))
        ex.append(dict(trial_id=r.trial_id, explainer="point_estimate", n_unsupported_cause=int(rng.random() < 0.3),
                       wheel_cause_asserted=int((not o and r.fault != "F0") or (o and rng.random() < 0.8)),
                       correct_claim_rate=0.7, informativeness=1.0,
                       text="SYNTHETIC point-estimate explanation"))
    pd.DataFrame(ex).to_csv(out / "explain.csv", index=False)
    man["explain_scores"] = "explain.csv"
    dock = [dict(arm=a, run=i, success=int(rng.random() < p), final_pose_error=float(rng.gamma(2, 0.01))) for a, p in
            (("proposed", 0.85), ("omni", 0.3), ("oracle", 0.95), ("analytic_bayes", 0.6)) for i in range(20)]
    pd.DataFrame(dock).sample(frac=1, random_state=seed).to_csv(out / "docking.csv", index=False)
    man["docking"] = "docking.csv"
    _probe_table(rng, df.sample(n=min(120, len(df)), random_state=seed)).to_csv(out / "probe.csv", index=False)
    man["probe_policy"] = "probe.csv"
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(man, indent=1))
    return mpath
