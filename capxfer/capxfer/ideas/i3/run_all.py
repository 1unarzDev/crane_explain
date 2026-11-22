"""I3 experiments -> ~/.cache/capxfer/I3/results_i3.json (+ printed tables). SIMULATION only.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i3.run_all
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import seq as S

import os
ALPHA, BETA = float(os.environ.get("I3_ALPHA", "0.05")), 0.10
CH = os.environ.get("I3_CHAINS", "").split(",") if os.environ.get("I3_CHAINS") else None
TAG = os.environ.get("I3_TAG", "")
TAUS = (0.5, 0.8)
KEYS = ("false_go_given_inf", "certify_given_feas", "refute_given_inf", "false_nogo_given_feas", "abstain",
        "decision_error", "time_mean")


def agg(rs):
    """Average numeric leaves of a list of nested result dicts (also q95 for false-go rates)."""
    r0 = rs[0]
    if isinstance(r0, dict):
        keys = set(r0).intersection(*[set(r) for r in rs[1:]])
        return {k: agg([r[k] for r in rs]) for k in keys}
    if isinstance(r0, (int, float, np.floating, np.integer)) and not isinstance(r0, bool):
        v = np.array(rs, float)
        return dict(mean=float(v.mean()), q95=float(np.quantile(v, 0.95)), sd=float(v.std()))
    return r0


def group_split(gids, rng, frac=0.5):
    u = np.unique(gids)
    sel = rng.permutation(u)[: int(len(u) * frac)]
    m = np.isin(gids, sel)
    return np.flatnonzero(m), np.flatnonzero(~m)


def exp_exchangeable(d, tau, R=50, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(R):
        c, t = group_split(d["gid"], rng)
        res.append(S.evaluate(d, c, t, tau, ALPHA, BETA, chains=CH))
    return agg(res)


def exp_trial_level(d, tau, R=50, seed=2):
    """Exact-exchangeability check: random TRIAL-level 50/50 splits (the theorem's assumption holds exactly)."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(R):
        p = rng.permutation(len(d["z"]))
        res.append(S.evaluate(d, p[: len(p) // 2], p[len(p) // 2:], tau, ALPHA, BETA, chains=CH))
    return agg(res)


def exp_loao(d, tau):
    fam = d["family"]
    out = {}
    for ty in ("single", "pair", "L3", "square4"):
        t = np.flatnonzero(fam == f"phys_{ty}")
        c = np.flatnonzero(fam != f"phys_{ty}")
        out[ty] = S.evaluate(d, c, t, tau, ALPHA, BETA, chains=CH)
    size = np.array([int(f.split("_")[0][1:]) if f.startswith("k") else 0 for f in fam])
    c = np.flatnonzero((size >= 1) & (size <= 3) | np.char.startswith(fam.astype(str), "phys_"))
    t = np.flatnonzero(size >= 4)
    out["random_4to6_cal_1to3+phys"] = S.evaluate(d, c, t, tau, ALPHA, BETA, chains=CH)
    return out


def exp_ood(d, dood, tau, name):
    c = np.arange(len(d["z"]))
    ch = [c for c in (CH or ["C1_strafe_first", "C4_P2_first"]) if c != "C2_default_first" and c != "C3_diag_first"]
    out = {"all": S.evaluate(d, c, np.arange(len(dood["z"])), tau, ALPHA, BETA, dtest=dood, chains=ch)}
    oc = np.flatnonzero(dood["oodc"])
    if len(oc):
        out["ood_consequential"] = S.evaluate(d, c, oc, tau, ALPHA, BETA, dtest=dood, chains=ch)
    return out


def exp_shift(dA_tgt, dA_src, tau, R=50, seed=1):
    """Score function = A models (narrow-DR 'wrong simulator'). Target domain = devW plant. Calibrate either on
    target-domain data (the real-calibration analogue) or on the source-sim domain (dev_v2), test on target."""
    rng = np.random.default_rng(seed)
    ch = [c for c in (CH or ["C1_strafe_first", "C4_P2_first"]) if c != "C2_default_first" and c != "C3_diag_first"]
    tgt, src = [], []
    for _ in range(R):
        c, t = group_split(dA_tgt["gid"], rng)
        tgt.append(S.evaluate(dA_tgt, c, t, tau, ALPHA, BETA, chains=ch, extra=False))
        src.append(S.evaluate(dA_src, np.arange(len(dA_src["z"])), t, tau, ALPHA, BETA, chains=ch, dtest=dA_tgt,
                              extra=False))
    return dict(calibrated_on_target=agg(tgt), calibrated_on_source_sim=agg(src))


def table(title, res, keys=KEYS, mean=False, group=None):
    if group is not None:
        res = {p: r["by_group"][group] for p, r in res.items() if isinstance(r, dict) and group in r.get("by_group", {})}
        title += f" [module-count group {group}]"
    print(f"\n### {title}")
    print("| policy | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for p, r in sorted(res.items()):
        if not isinstance(r, dict) or "false_go_given_inf" not in r:
            continue
        vals = [(r[k]["mean"] if mean else r[k]) for k in keys]
        print(f"| {p} | " + " | ".join(f"{v:.3f}" if k != "time_mean" else f"{v:.1f}" for k, v in zip(keys, vals)) + " |")


def main():
    W = S.load("views_devW_ensemble_W.pt")
    print("devW trials", len(W["z"]), {t: float(np.mean(W["z"] < t)) for t in TAUS})
    out = {}
    for tau in TAUS:
        o = out[f"tau{tau}"] = {}
        o["exchangeable"] = exp_exchangeable(W, tau)
        for g in (None, 1, 2, 3):
            table(f"tau={tau} exchangeable (50 assembly-level 50/50 splits; means)", o["exchangeable"], mean=True,
                  group=g)
        o["trial_level"] = exp_trial_level(W, tau)
        for g in (None, 1):
            table(f"tau={tau} TRIAL-level exchangeable splits (exactness check; means)", o["trial_level"], mean=True,
                  group=g)
        o["loao"] = exp_loao(W, tau)
        for ty, r in o["loao"].items():
            table(f"tau={tau} held-out {ty} (n_inf={next(iter(r.values()))['n_inf']})", r)
        for nm in ("dev_ood_lowmu_v3", "dev_ood_payload_v3"):
            try:
                dd = S.load(f"views_{nm}_ensemble_W_min.pt")
            except FileNotFoundError:
                continue
            o[nm] = exp_ood(W, dd, tau, nm)
            for k, r in o[nm].items():
                table(f"tau={tau} OOD {nm} / {k}", r)
        try:
            A_t, A_s = S.load("views_devW_ensemble_A_g90_min.pt"), S.load("views_dev_v2_ensemble_A_g90_min.pt")
            o["shift"] = exp_shift(A_t, A_s, tau)
            for k, r in o["shift"].items():
                table(f"tau={tau} shift (A score fn) {k}", r, mean=True)
        except FileNotFoundError as e:
            print("shift skipped", e)
    (S.ROOT / f"results_i3{TAG}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
