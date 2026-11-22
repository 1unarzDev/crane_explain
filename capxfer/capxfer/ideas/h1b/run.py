"""H1B (SIMULATION ONLY): uncalibrated posterior-ranking baseline (critic W1), tau=0.5 vacuity numbers (W2),
full I3 sequential grid (W6). No retraining: reuses cached predictions only.

Inputs (read-only):
  EXPL2 rows_id.npz   per test module: structured_ensemble posterior-predictive mixture quantiles of z at 27 levels
                      (the SAME posterior + library as the certificate, no conformal step)
  SPLIT rows_singles_P1P5P2.npz (cert / carry-over / healthy lower bounds at alpha 0.10), truth
  CHOICE rows_alpha0.050.npz (cert at alpha 0.05), probe_all.npz, dock_rows.jsonl (dock model, mission model)
  results_sim/i3/i3_a010_summary.json, ~/.cache/capxfer/I3/results_i3_v4.json (alpha 0.05)
Uncalibrated arms (structured posterior used directly):
  U-mean    rank by posterior mean of z (trapezoid over the 27-level quantile function), go iff mean >= tau
  U-med     rank by posterior median, go iff median >= tau
  U-q05     default one-sided 95 % posterior lower quantile (l = 0.05), rank by it, go iff >= tau
  U-nom     nominal l = alpha (= U-q05 at alpha 0.05)
  U-tunedFG ORACLE-TUNED: largest l with test false-go <= alpha (found with truth on the test set)
  U-tunedY  ORACLE-TUNED: l matching the certificate's certification yield on the test set
  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.h1b.run
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from capxfer.ideas.split import common as C
from capxfer.ideas.choice import evaluate as CH

HOME = pathlib.Path.home() / ".cache/capxfer"
OUT = HOME / "H1B"
REPO = pathlib.Path(__file__).resolve().parents[3]
TAUS = (0.5, 0.8)
ALPHAS = (0.10, 0.05)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def qinterp(Q, lev, l):
    """per-row linear interpolation of the quantile function at level l."""
    i = int(np.clip(np.searchsorted(lev, l) - 1, 0, len(lev) - 2))
    w = (l - lev[i]) / (lev[i + 1] - lev[i])
    return (1 - w) * Q[:, i] + w * Q[:, i + 1]


def boot(vals_fn, groups, B=300, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    m = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        v = vals_fn(ii)
        if np.isfinite(v):
            m.append(v)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def mod_metrics(lb, z, tau, groups=None):
    go, uns = lb >= tau, z < tau
    d = dict(false_go=float(go[uns].mean()), cert=float(go.mean()), unsafe_go=float((go & uns).mean()),
             lb_cov=float((z >= lb).mean()), n=int(len(z)), n_unsafe=int(uns.sum()))
    if groups is not None:
        d["false_go_ci"] = boot(lambda ii: go[ii][uns[ii]].mean() if uns[ii].any() else np.nan, groups)
    return d


def main():
    X = np.load(HOME / "EXPL2/rows_id.npz")
    S = np.load(C.OUT / "rows_singles_P1P5P2.npz")
    A05 = np.load(CH.OUT / "rows_alpha0.050.npz")
    lev, Qz, row = X["qlev"], X["q_z"], X["row"]
    z = S["z"][row]
    assert np.allclose(z, X["z"])
    N_all = len(S["z"])
    EP = CH.Episodes()
    gi = np.array([EP.E[int(k)]["gi"] for k in S["ep"][row]])
    nmod = S["nmod"][row]
    # posterior mean via the quantile function (levels 0.001..0.999; tails clipped)
    mean = np.trapezoid(Qz, lev, axis=1) / (lev[-1] - lev[0])
    full = lambda v: (lambda a: (a.__setitem__(row, v), a)[1])(np.full(N_all, np.nan))
    src = {k: str(p) for k, p in dict(rows_id=HOME / "EXPL2/rows_id.npz", split_rows=C.OUT / "rows_singles_P1P5P2.npz",
                                     rows_a05=CH.OUT / "rows_alpha0.050.npz").items()}
    res = dict(sources={k: dict(path=v, sha256=sha(v)) for k, v in src.items()}, split={}, choice={}, tuned={},
               vacuity={})
    # ------------------------------------------------------------------ split benchmark (47,285 test modules)
    lbs_by_alpha = {}
    for alpha in ALPHAS:
        cert = (S["lb_comp_se"] if alpha == 0.10 else A05["cert"])[row]
        arms = {"cert": cert, "carry": (S["lb_bb_carry"] if alpha == 0.10 else A05["bb"])[row],
                "healthy": S["lb_healthy"][row], "U-mean": mean, "U-med": qinterp(Qz, lev, 0.5),
                "U-q05": qinterp(Qz, lev, 0.05), "U-nom": qinterp(Qz, lev, alpha)}
        for tau in TAUS:
            uns = z < tau
            # oracle-tuned levels (grid of 0.0005 steps over [0.001, 0.5])
            grid = np.arange(0.001, 0.5, 0.0005)
            fg = np.array([(qinterp(Qz, lev, l)[uns] >= tau).mean() for l in grid])
            ce = np.array([(qinterp(Qz, lev, l) >= tau).mean() for l in grid])
            ok = np.flatnonzero(fg <= alpha)
            l_fg = float(grid[ok.max()]) if len(ok) else float("nan")
            ccert = float((cert >= tau).mean())
            l_y = float(grid[np.argmin(np.abs(ce - ccert))])
            res["tuned"][f"{alpha}|{tau}"] = dict(l_tunedFG=l_fg, l_tunedY=l_y)
            a2 = dict(arms, **{"U-tunedFG": qinterp(Qz, lev, l_fg), "U-tunedY": qinterp(Qz, lev, l_y)})
            for m, lb in a2.items():
                res["split"][f"{alpha}|{tau}|{m}"] = mod_metrics(lb, z, tau, gi)
                for nm in range(2, 7):
                    s = nmod == nm
                    res["split"][f"{alpha}|{tau}|{m}|nmod{nm}"] = mod_metrics(lb[s], z[s], tau)
            lbs_by_alpha[(alpha, tau)] = a2
    # ------------------------------------------------------------------ module choice (3,000 faulted episodes)
    dock = CH.load_dock()
    dm = CH.DockModel([r for r in dock if r["set"] in ("ep", "zbin")])
    for alpha in ALPHAS:
        pa, sub = EP.probe_all_lb(alpha)
        groups_sub = np.array([EP.E[k]["gi"] for k in sub])
        for tau in TAUS:
            for m, lb in lbs_by_alpha[(alpha, tau)].items():
                if m in ("carry", "healthy"):
                    continue          # unchanged from CHOICE (nearest-goal rule); quoted from CHOICE results.json
                lbs = {"cert": full(lb), "pa": pa}
                R = CH.single_shot("cert", sub, EP, lbs, tau, dm, np.random.default_rng(0))
                M = CH.run_missions("cert", sub, EP, lbs, tau, dm)
                d = dict(single_shot=CH.summarize_ss(R, groups_sub), mission=CH.summarize_mission(M))
                for nm in range(2, 7):
                    sel = np.array([EP.E[k]["nmod"] == nm for k in sub])
                    d[f"nmod{nm}"] = CH.summarize_ss(R[sel])
                res["choice"][f"{alpha}|{tau}|{m}"] = d
                print(alpha, tau, m, json.dumps(d["single_shot"]), json.dumps(d["mission"]), flush=True)
    # ------------------------------------------------------------------ tau=0.5 vacuity (all 47,285 modules)
    for tau in TAUS:
        uns = z < tau
        v = dict(n_modules=int(len(z)), n_unsafe=int(uns.sum()), unsafe_rate=float(uns.mean()),
                 unsafe_rate_faulted_modules=float(uns[S["nf"][row] > 0].mean()),
                 healthy_unsafe_go=float(uns.mean()),                       # assume-healthy says go on every module
                 const_lb_eq_tau_coverage=float((z >= tau).mean()),
                 by_nmod={f"nmod{nm}": float(uns[nmod == nm].mean()) for nm in range(2, 7)})
        for alpha in ALPHAS:
            v[f"healthy_unsafe_go_le_alpha{alpha}"] = bool(uns.mean() <= alpha)
            v[f"healthy_unsafe_go_ci"] = boot(lambda ii: uns[ii].mean(), gi)
        res["vacuity"][str(tau)] = v
    ch = json.loads((CH.OUT / "results.json").read_text())
    for alpha in ALPHAS:
        for tau in TAUS:
            for pol in ("cert", "bb", "healthy", "random", "probe_all", "oracle"):
                t = ch["tables"][f"{alpha}|{tau}|{pol}"]
                res["choice"][f"{alpha}|{tau}|CHOICE:{pol}"] = dict(single_shot=t["single_shot"], mission=t["mission"])
    res["sources"]["choice_results"] = dict(path=str(CH.OUT / "results.json"), sha256=sha(CH.OUT / "results.json"))
    # ------------------------------------------------------------------ I3 full grid
    res["i3"] = i3_grid()
    p = OUT / "h1b_results.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p, sha(p))


def i3_grid():
    out = {}
    for tag, path in (("a010", REPO / "results_sim/i3/i3_a010_summary.json"),
                      ("a005", REPO / "results_sim/i3/i3_v4_summary.json")):
        d = json.loads(path.read_text())
        out[f"source_{tag}"] = dict(path=str(path), sha256=sha(path), alpha=d.get("alpha"))
        for tau in ("tau0.5", "tau0.8"):
            for split in ("exchangeable", "trial_level"):
                for pol, v in d["results"][tau][split].items():
                    for g, gv in v.get("by_group", {}).items():
                        out[f"{tag}|{tau}|{split}|{pol}|g{g}"] = dict(
                            fg=gv["false_go_given_inf"]["mean"], fg_marg=gv["false_go_marginal"]["mean"],
                            cert=gv["certify_given_feas"]["mean"], abstain=gv["abstain"]["mean"],
                            time=gv["time_mean"]["mean"], n_inf=gv["n_inf"]["mean"],
                            cond=v.get("calibration_conditional", {}).get(g))
                    out[f"{tag}|{tau}|{split}|{pol}|all"] = dict(
                        fg=v["false_go_given_inf"]["mean"], fg_marg=v["false_go_marginal"]["mean"],
                        cert=v["certify_given_feas"]["mean"], abstain=v["abstain"]["mean"], time=v["time_mean"]["mean"])
    return out


if __name__ == "__main__":
    main()
