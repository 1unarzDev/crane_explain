"""Offline evaluation of probe-selection policies over logged full-library trials (secondary, exploratory).

Every real trial runs the full probe library, so any probe policy can be replayed offline on identical data.
Input CSV (one row per trial x seen-set x candidate primitive), produced upstream (WS-C/WS-F):
  trial_id, session, seen ("+"-joined sorted pids, e.g. "P1+P5"), candidate (pid),
  exp_regret   : model's ANTICIPATED decision regret after adding `candidate`, computed from the data in `seen` ONLY
                 (pre-posterior: decode posterior samples through physics to simulate the candidate, re-infer, and
                 average the docking-decision regret). Must not use the candidate's logged data (leakage).
  info_gain    : model's anticipated mutual information (wheel modes; candidate outcome) from `seen` ONLY.
  realized_regret_next : TRUE decision regret (evaluator side, oracle capability) after observing seen + candidate.
Policies: greedy expected-regret (argmin exp_regret), info-max (argmax info_gain), fixed pre-specified order,
uniform random (seeded), and the realized-regret oracle (a ceiling, not a policy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S

FIXED_ORDER = ("P2", "P3", "P4", "P6y", "P6x")
START = ("P1", "P5")


def _key(pids):
    return "+".join(sorted(pids))


def rollout(df, budget=3, seed=0, start=START, fixed=FIXED_ORDER):
    """-> DataFrame trial_id, session, policy, step (1..budget), chosen, realized_regret."""
    rng = np.random.default_rng(seed)
    idx = {(r.trial_id, r.seen, r.candidate): r for r in df.itertuples(index=False)}
    by_state = df.groupby(["trial_id", "seen"])
    cands = {k: list(g["candidate"]) for k, g in by_state}
    rows = []
    for tid, sess in df.groupby("trial_id")["session"].first().items():
        for pol in ("greedy_regret", "info_max", "fixed", "random", "oracle"):
            seen = list(start)
            for step in range(1, budget + 1):
                c = cands.get((tid, _key(seen)))
                if not c:
                    break
                rs = [idx[(tid, _key(seen), x)] for x in c]
                if pol == "greedy_regret":
                    pick = min(rs, key=lambda r: (r.exp_regret, r.candidate))
                elif pol == "info_max":
                    pick = max(rs, key=lambda r: (r.info_gain, r.candidate))
                elif pol == "fixed":
                    pick = next((r for x in fixed for r in rs if r.candidate == x), rs[0])
                elif pol == "random":
                    pick = rs[int(rng.integers(len(rs)))]
                else:
                    pick = min(rs, key=lambda r: (r.realized_regret_next, r.candidate))
                rows.append(dict(trial_id=tid, session=sess, policy=pol, step=step, chosen=pick.candidate,
                                 realized_regret=float(pick.realized_regret_next)))
                seen.append(pick.candidate)
    return pd.DataFrame(rows)


def evaluate(df, budget=3, seed=0, n_boot=2000, conf=0.95):
    ro = rollout(df, budget, seed)
    summary, tests = [], []
    for (pol, step), g in ro.groupby(["policy", "step"]):
        W = S.bootstrap_weights(g["session"].to_numpy(), n_boot, seed)
        ci = S.boot_ci(g["realized_regret"].to_numpy(), W, "mean", conf)
        summary.append(dict(policy=pol, step=step, n=len(g), mean_regret=float(g["realized_regret"].mean()),
                            ci_lo=ci["lo"], ci_hi=ci["hi"]))
    wide = ro.pivot_table(index=["trial_id", "step"], columns="policy", values="realized_regret")
    for other in ("info_max", "fixed", "random"):
        for step in range(1, budget + 1):
            sub = wide.xs(step, level="step")[["greedy_regret", other]].dropna()
            p, pos, neg = S.sign_test_greater((sub[other] - sub["greedy_regret"]).to_numpy())
            tests.append(dict(comparison=f"greedy_regret<{other}", step=step, p_one_sided=p, n_better=pos,
                              n_worse=neg, status="EXPLORATORY"))
    return pd.DataFrame(summary), pd.DataFrame(tests), ro
