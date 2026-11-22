"""Statistics helpers for the frozen analysis.

Reuses (imports, never edits):
  * ~/crane_explain/analysis/hexar_external/confirmatory_v1/statistics.py : exact binomial tails, Clopper-Pearson
    bounds (`cp_bounds`), exact one-sided sign / McNemar test (`paired_p`).
  * ~/crane_explain/eval/statistics/paired_v43.py : linear-interpolated empirical `quantile` for percentile CIs.
Both are loaded by file path under private module names (the first one would shadow the stdlib `statistics`).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[3]          # ~/crane_explain


def _load(name, rel):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conf = _load("_capxfer_confirmatory_stats", "analysis/hexar_external/confirmatory_v1/statistics.py")
_pv43 = _load("_capxfer_paired_v43", "eval/statistics/paired_v43.py")
REUSED = {"confirmatory_statistics": str(ROOT / "analysis/hexar_external/confirmatory_v1/statistics.py"),
          "paired_v43": str(ROOT / "eval/statistics/paired_v43.py")}


# ---------------------------------------------------------------- binomial
def clopper_pearson(k, n, conf=0.95):
    """Two-sided exact CI with (1-conf)/2 in each tail. n = 0 -> (nan, nan)."""
    if n == 0:
        return (math.nan, math.nan)
    return tuple(float(x) for x in _conf.cp_bounds(int(k), int(n), (1 - conf) / 2))


def cp_lower(k, n, conf=0.95):
    """One-sided lower bound at level conf."""
    return math.nan if n == 0 else float(_conf.cp_bounds(int(k), int(n), 1 - conf)[0])


def cp_upper(k, n, conf=0.95):
    return math.nan if n == 0 else float(_conf.cp_bounds(int(k), int(n), 1 - conf)[1])


def sign_test_greater(diffs):
    """Exact one-sided sign test H1: median(diff) > 0. Ties (diff == 0) dropped. -> (p, n_pos, n_neg)."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return float(_conf.paired_p(pos, neg)), pos, neg


def mcnemar_greater(a, b):
    """Exact one-sided McNemar on paired binary outcomes, H1: P(a=1) > P(b=1). -> (p, n10, n01)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return float(_conf.paired_p(n10, n01)), n10, n01


def fisher_greater(k1, n1, k0, n0):
    """One-sided Fisher exact, H1: p1 > p0."""
    return float(sps.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1])


def boschloo_greater(k1, n1, k0, n0):
    return float(sps.boschloo_exact([[k1, k0], [n1 - k1, n0 - k0]], alternative="greater").pvalue)


def _wilson(k, n, z):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k0, n0, conf=0.90):
    """Newcombe hybrid-score CI (method 10) for p1 - p0, two-sided at `conf`."""
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = _wilson(k1, n1, z)
    l0, u0 = _wilson(k0, n0, z)
    d = p1 - p0
    lo = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lo, hi


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- cluster bootstrap
def bootstrap_weights(clusters, n_boot, seed):
    """Trial weight matrix (n_boot, N).

    `clusters`: one (N,) label array -> one-way cluster bootstrap (resample clusters with replacement);
    a list of label arrays -> crossed product weights (pigeonhole bootstrap, as in paired_v43's crossed scheme;
    a sensitivity analysis, not an exact guarantee). Fewer than 2 clusters on any axis -> None (CI withheld).
    """
    axes = clusters if isinstance(clusters, (list, tuple)) else [clusters]
    rng = np.random.default_rng(seed)
    W = None
    for lab in axes:
        _, inv = np.unique(np.asarray([str(x) for x in lab]), return_inverse=True)
        C = int(inv.max()) + 1 if inv.size else 0
        if C < 2:
            return None
        draws = rng.integers(0, C, size=(n_boot, C))
        mult = np.zeros((n_boot, C))
        np.add.at(mult, (np.repeat(np.arange(n_boot), C), draws.ravel()), 1.0)
        w = mult[:, inv]
        W = w if W is None else W * w
    return W


def _wmedian(v, w):
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return math.nan
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def boot_ci(values, W, stat="mean", conf=0.95):
    """Percentile CI of a weighted statistic over bootstrap weights W (n_boot, N). NaN values are dropped."""
    v = np.asarray(values, float)
    if W is None:
        return dict(lo=math.nan, hi=math.nan, status="WITHHELD_FEWER_THAN_2_CLUSTERS")
    ok = ~np.isnan(v)
    v, Wk = v[ok], W[:, ok]
    if stat == "mean":
        den = Wk.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = (Wk * np.where(np.isfinite(v), v, 0.0)).sum(1) / den
        reps = reps[np.isfinite(reps)]
    elif stat == "median":
        reps = np.array([_wmedian(v, w) for w in Wk])
    else:
        raise ValueError(stat)
    reps = [float(x) for x in reps if not math.isnan(x)]
    if len(reps) < 2:
        return dict(lo=math.nan, hi=math.nan, status="WITHHELD_UNDEFINED_DRAWS")
    a = (1 - conf) / 2
    return dict(lo=float(_pv43.quantile(reps, a)), hi=float(_pv43.quantile(reps, 1 - a)),
                lower_one_sided=float(_pv43.quantile(reps, 1 - conf)), n_rep=len(reps), status="AVAILABLE")


def coverage_summary(hit, clusters_oneway, clusters_crossed, n_boot, seed, conf=0.95):
    """Point estimate + CP CI (iid) + one-way and crossed cluster-bootstrap CIs for a binary indicator."""
    h = np.asarray(hit, bool)
    n, k = int(h.size), int(h.sum())
    cp = clopper_pearson(k, n, conf)
    out = dict(n=n, k=k, coverage=k / n if n else math.nan, cp_lo=cp[0], cp_hi=cp[1], cp_lower_one_sided=cp_lower(k, n, conf))
    for name, cl in (("boot_oneway", clusters_oneway), ("boot_crossed", clusters_crossed)):
        if cl is None or n == 0:
            out[name] = dict(status="NOT_REQUESTED")
            continue
        out[name] = boot_ci(h.astype(float), bootstrap_weights(cl, n_boot, seed), "mean", conf)
    lows = [out["cp_lower_one_sided"]] + [out[b].get("lower_one_sided", math.nan) for b in ("boot_oneway", "boot_crossed")]
    lows = [x for x in lows if not math.isnan(x)]
    out["conservative_lower"] = min(lows) if lows else math.nan
    return out
