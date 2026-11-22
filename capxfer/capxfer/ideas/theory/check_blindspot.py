"""Prop. 5 (blind spot of P1+P5, minimal probe cover) check on random 90-deg-grid X-pattern assemblies.

Checks, per assembly (4 physical types + N random 1..6-module assemblies with random module yaws):
  1. Closed form: channels unexcited by {P1, P5} == roller locks on wheels with w = +-x and p.w = 0 (exact, eta -> 0),
     and == {w = +-x, |p.w| * |wz_P5| < eta} for the thresholded version.
  2. No single library probe covers every channel unless all modules share the yaw parity (then P2 or P1 alone does).
  3. A single translation at heading phi covers every channel iff phi is not a multiple of 45 deg; the minimum
     normalised excitation is min(|sin phi|, |cos phi|, |cos(phi +- 45)|), maximised (= sin 22.5 = 0.383) at 22.5 deg.
  4. Minimum-cardinality / minimum-time library cover (exhaustive over 2^7) vs the closed form.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.theory.check_blindspot
"""
from __future__ import annotations

import itertools

import numpy as np

from capxfer import assembly as A, probes
from capxfer.ideas.i2 import theory as T
from capxfer.ideas.theory.common import ROLLER, WEAK, params, dump

ETA = 0.03
PIDS = probes.PIDS


def channel_exc(asm, xis):
    g, h = T.rows(asm)
    return np.concatenate([np.abs(g @ np.asarray(xis).T), np.abs(h @ np.asarray(xis).T)], 0)   # (2n, P)


def main(n_rand=2000, seed=7):
    P = params()
    rng = np.random.default_rng(seed)
    asms = [A.physical(k, P) for k in ("single", "pair", "L3", "square4")]
    asms += [A.random_assembly(rng, P) for _ in range(n_rand)]
    times = np.array([2 * probes.leg_duration(p) for p in probes.LIBRARY])
    res = dict(n_asm=len(asms), closed_form_exact=0, closed_form_eta=0, oblique={})
    phis = np.deg2rad([22.5, 25.7, 30.0, 10.0, 45.0, 0.0])
    min_exc = {float(np.rad2deg(p)): [] for p in phis}
    for asm in asms:
        tw = T.probe_twists(asm)
        X = np.array([tw[p] for p in PIDS])                                # (7,3)
        E = channel_exc(asm, X)                                            # (2n,7)
        n = asm.n
        # 1. closed form
        even = np.abs(asm.w_hat[:, 0]) > 0.5
        pw = (asm.wheel_pos * asm.w_hat).sum(1)
        unex = (E[:, [PIDS.index("P1"), PIDS.index("P5")]].max(1) < 1e-12)
        pred = np.concatenate([np.zeros(n, bool), even & (np.abs(pw) < 1e-12)])
        res["closed_form_exact"] += int(np.array_equal(unex, pred))
        unex_eta = np.concatenate([E[:n, [0, 4]].max(1) < 0.01, E[n:, [0, 4]].max(1) < ETA])
        pred_eta = np.concatenate([np.zeros(n, bool), even & (np.abs(pw) * abs(tw["P5"][2]) < ETA)])
        res["closed_form_eta"] += int(np.array_equal(unex_eta, pred_eta))
        # 2./4. library covers: exact (eta -> 0) and thresholded (eta_drive = 0.01, eta_roller = 0.03 m/s)
        thr = np.concatenate([np.full(n, 0.01), np.full(n, ETA)])[:, None]
        parity = {int(m[3]) % 2 for m in asm.modules}
        for tag, Cov in (("exact", E > 1e-12), ("eta", E >= thr)):
            cov1 = Cov.all(0)
            ok_trans = cov1[[0, 1, 2, 3, 5, 6]].any()                     # translation probes only
            res.setdefault(f"single_trans_cover_{tag}", 0)
            res.setdefault(f"single_trans_cover_pred_{tag}", 0)
            res[f"single_trans_cover_{tag}"] += int(ok_trans)
            res[f"single_trans_cover_pred_{tag}"] += int(ok_trans == (len(parity) == 1))
            best_k, best_t, best_set = None, 1e9, None
            for k in range(1, 8):
                for S in itertools.combinations(range(7), k):
                    if Cov[:, list(S)].any(1).all():
                        if best_k is None:
                            best_k = k
                        if times[list(S)].sum() < best_t:
                            best_t, best_set = times[list(S)].sum(), "+".join(PIDS[j] for j in S)
            for key, val in ((f"min_card_hist_{tag}", str(best_k)), (f"min_time_cover_hist_{tag}", best_set)):
                res.setdefault(key, {})
                res[key][val] = res[key].get(val, 0) + 1
            if tag == "eta":
                k_sz = f"min_card_by_parity_eta_{'mixed' if len(parity) > 1 else 'same'}"
                res.setdefault(k_sz, {})
                res[k_sz][str(best_k)] = res[k_sz].get(str(best_k), 0) + 1
        # 3. oblique translations
        for p in phis:
            u = np.array([np.cos(p), np.sin(p), 0.0])
            min_exc[float(np.rad2deg(p))].append(channel_exc(asm, u[None])[:, 0].min())
    pred_val = lambda ph: min(abs(np.sin(ph)), abs(np.cos(ph)), abs(np.cos(ph - np.pi / 4)), abs(np.cos(ph + np.pi / 4)))
    for p in phis:
        v = np.array(min_exc[float(np.rad2deg(p))])
        res["oblique"][f"{np.rad2deg(p):.1f}"] = dict(min_over_asm=float(v.min()), max_over_asm=float(v.max()),
                                                      closed_form=float(pred_val(p)))
    grid = np.deg2rad(np.linspace(0, 90, 9001))
    vals = np.array([pred_val(p) for p in grid])
    res["oblique_argmax_deg"] = float(np.rad2deg(grid[vals.argmax()]))
    res["oblique_max_val"] = float(vals.max())
    dump("check_blindspot.json", res)
    print(res)


if __name__ == "__main__":
    main()
