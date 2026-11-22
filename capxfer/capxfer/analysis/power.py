"""Design-stage power / precision calculations (exact enumeration; no data). Run:
    PYTHONPATH=. .venv/bin/python -m capxfer.analysis.power
The printed tables are quoted in docs/BENCHMARK_DESIGN.md. They are design numbers, not results.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from . import stats as S


def _cp(k, n, conf=0.95):
    """Fast two-sided CP via the beta quantile (equals stats.clopper_pearson; checked in tests)."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else sps.beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


def _cp_upper(k, n, conf=0.95):
    return 1.0 if k == n else float(sps.beta.ppf(conf, k + 1, n - k))


def _cp_lower(k, n, conf=0.95):
    return 0.0 if k == 0 else float(sps.beta.ppf(1 - conf, k, n - k + 1))


def cp_halfwidth(n, p=0.90, conf=0.95, deff=1.0):
    """Expected half-width of the two-sided CP interval, at effective sample size n / deff."""
    ne = max(int(round(n / deff)), 1)
    pmf = sps.binom.pmf(np.arange(ne + 1), ne, p)
    hw = np.array([(lambda b: (b[1] - b[0]) / 2)(_cp(k, ne, conf)) for k in range(ne + 1)])
    return float(pmf @ hw)


def p_fold_pass(n, cov, thr=0.85, deff=1.0):
    """P(observed coverage >= thr) at true coverage `cov` with effective n."""
    ne = max(int(round(n / deff)), 1)
    k = math.ceil(thr * ne - 1e-9)
    return float(sps.binom.sf(k - 1, ne, cov))


def p_at_least(m, folds, p):
    return float(sps.binom.sf(m - 1, folds, p))


def fisher_power(n1, p1, n0, p0, alpha=0.05):
    k = np.arange(n1 + 1)
    j = np.arange(n0 + 1)
    w1, w0 = sps.binom.pmf(k, n1, p1), sps.binom.pmf(j, n0, p0)
    tot = 0.0
    for a in k:
        for b in j:
            if w1[a] * w0[b] < 1e-12:
                continue
            if S.fisher_greater(int(a), n1, int(b), n0) < alpha:
                tot += w1[a] * w0[b]
    return tot


def ni_power(n1, p1, n0, p0, margin=0.20, conf=0.90):
    """P(Newcombe two-sided `conf` lower bound of p1 - p0 > -margin)."""
    k = np.arange(n1 + 1)
    j = np.arange(n0 + 1)
    w1, w0 = sps.binom.pmf(k, n1, p1), sps.binom.pmf(j, n0, p0)
    tot = 0.0
    for a in k:
        for b in j:
            if w1[a] * w0[b] < 1e-12:
                continue
            if S.newcombe_diff(int(a), n1, int(b), n0, conf)[0] > -margin:
                tot += w1[a] * w0[b]
    return tot


def p_upper_below(n, rate, alpha=0.10, conf=0.95):
    """P(one-sided CP upper bound <= alpha) when the true rate is `rate`."""
    tot = 0.0
    for k in range(n + 1):
        if _cp_upper(k, n, conf) <= alpha:
            tot += sps.binom.pmf(k, n, rate)
        else:
            break
    return tot


def p_at_least_hetero(m, ps):
    """P(at least m of len(ps) independent folds pass) with fold-specific pass probabilities (Poisson-binomial)."""
    dist = np.array([1.0])
    for q in ps:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[m:].sum())


def p_lower_pass(n, cov, thr=0.85, conf=0.95, deff=1.0):
    """P(one-sided `conf` CP lower bound >= thr) at true coverage `cov`, effective n = n / deff. This is an upper bound
    on the power of the H1 'strong' rule, which also needs both cluster-bootstrap lower bounds to pass."""
    ne = max(int(round(n / deff)), 1)
    k = np.arange(ne + 1)
    ok = np.array([_cp_lower(int(x), ne, conf) >= thr for x in k])
    return float(sps.binom.pmf(k, ne, cov) @ ok)


def mcnemar_power(n, p10, p01, alpha=0.05, deff=1.0):
    """Power of the exact one-sided McNemar test (H1(b): method diagnosed & baseline not = p10, reverse = p01) on n
    paired trials (effective n / deff). Discordant D ~ Bin(n, p10 + p01); n10 | D ~ Bin(D, p10 / (p10 + p01))."""
    ne = max(int(round(n / deff)), 1)
    pd_ = p10 + p01
    q = p10 / pd_
    tot = 0.0
    for D in range(ne + 1):
        wD = sps.binom.pmf(D, ne, pd_)
        if wD < 1e-14 or D == 0:
            continue
        k = np.arange(D + 1)
        pv = sps.binom.sf(k - 1, D, 0.5)
        tot += wD * float(sps.binom.pmf(k, D, q) @ (pv < alpha))
    return tot


def p_point_at_most(n, rate, thr=0.10):
    """P(observed rate <= thr) at true `rate` (H3 point-estimate conditions (i), (iv))."""
    return float(sps.binom.cdf(math.floor(thr * n + 1e-9), n, rate))


# sim-informed H1(b) scenarios: (label, p10, p01) on in-model trials (diagnosed = all wheels' argmax correct)
H1B_SCENARIOS = (
    ("sim W/devW-v2 physical types, P1P5, coast+brake merged", 0.366, 0.008),
    ("sim physical types, P1P5, unmerged", 0.372, 0.033),
    ("sim physical types, P1P5P2, merged", 0.391, 0.020),
    ("dry run pseudo-real E2 (DX dryW), unmerged, 45/136 vs 13/136", 0.331, 0.096),
    ("stress: 0.15 vs 0.08", 0.15, 0.08),
)
# sim-informed H1(a) scenarios for the black-box ensemble: true joint coverage per fold
H1A_SCENARIOS = (
    ("all folds 0.90 (DX dryW pooled 0.904; sim LOFO 0.909)", (0.90, 0.90, 0.90, 0.90)),
    ("heterogeneous 0.86/0.88/0.90/0.95 (DX dryW fold pattern)", (0.86, 0.88, 0.90, 0.95)),
    ("all folds 0.88", (0.88, 0.88, 0.88, 0.88)),
    ("one bad fold 0.80/0.90/0.90/0.92", (0.80, 0.90, 0.90, 0.92)),
)


def main():
    print("## Coverage precision: expected two-sided 95% CP half-width at true coverage 0.90")
    print("n_test | deff 1.0 | deff 1.5 | deff 2.0")
    for n in (100, 150, 200, 250, 300, 400, 870):
        print(n, *[f"{cp_halfwidth(n, deff=d):.3f}" for d in (1.0, 1.5, 2.0)], sep=" | ")
    print("\n## H1 fold pass P(cov_hat >= 0.85) with n_test = 290 in-model trials per fold")
    print("true cov | deff 1 | deff 2 | P(>=3 of 4 folds), deff 2")
    for c in (0.80, 0.85, 0.88, 0.90, 0.92):
        p1, p2 = p_fold_pass(290, c), p_fold_pass(290, c, deff=2.0)
        print(c, f"{p1:.3f}", f"{p2:.3f}", f"{p_at_least(3, 4, p2):.3f}", sep=" | ")
    print("\n## H1(a) black-box ensemble, joint coverage over the 10 untested legs, n_test = 290 per fold")
    print("scenario | P(>=3 of 4 pass) deff 1 | deff 2 | P(strong: CP lower >= 0.85 in >=3) deff 1 | deff 2")
    for lbl, cs in H1A_SCENARIOS:
        r = [p_at_least_hetero(3, [p_fold_pass(290, c, deff=d) for c in cs]) for d in (1.0, 2.0)]
        st = [p_at_least_hetero(3, [p_lower_pass(290, c, deff=d) for c in cs]) for d in (1.0, 2.0)]
        print(lbl, *[f"{x:.3f}" for x in r + st], sep=" | ")
    for n in (200, 250):
        cs = H1A_SCENARIOS[1][1]
        print(f"  n_test={n}, heterogeneous: P(>=3 of 4) deff 2 =",
              f"{p_at_least_hetero(3, [p_fold_pass(n, c, deff=2.0) for c in cs]):.3f}")
    print("\n## H1(b) exact one-sided McNemar power (in-model trials, pooled over folds), alpha 0.05")
    print("scenario | n=50 | n=100 | n=290 | n=1160 | n=1160 deff 2")
    for lbl, p10, p01 in H1B_SCENARIOS:
        print(lbl, *[f"{mcnemar_power(n, p10, p01):.3f}" for n in (50, 100, 290, 1160)],
              f"{mcnemar_power(1160, p10, p01, deff=2.0):.3f}", sep=" | ")
    print("\n## H1(b) APS all-wheels coverage: P(pooled in-model estimate >= 0.85), n = 1160")
    print("true | deff 1 | deff 2 | deff 4")
    for c in (0.86, 0.88, 0.90):
        print(c, *[f"{p_fold_pass(1160, c, deff=d):.3f}" for d in (1.0, 2.0, 4.0)], sep=" | ")
    print("faulted-trial diagnosed-rate precision (n = 960 faulted in-model): CP half-width at 0.88 =",
          f"{cp_halfwidth(960, 0.88):.3f} (deff 2: {cp_halfwidth(960, 0.88, deff=2.0):.3f})")
    print("\n## H3 OOD + wheel fault (n = 120: 15 F6 + 15 F7 per type with one wheel fault) and consequential OOD")
    print("P(calibrated misattribution point estimate <= 0.10), n=120: rate 0.03 / 0.05 / 0.07 =",
          *[f"{p_point_at_most(120, r):.3f}" for r in (0.03, 0.05, 0.07)])
    for n in (120, 180, 240):
        k = np.arange(n + 1)
        ok = np.array([_cp_lower(int(x), n) > 0.10 for x in k])
        print(f"P(point overclaim CP lower > 0.10), n_conseq={n}: rate 0.2 / 0.3 / 0.5 =",
              *[f"{float(sps.binom.pmf(k, n, r) @ ok):.3f}" for r in (0.2, 0.3, 0.5)])
    print("\n## H4 docking, n = 20 per arm, one-sided Fisher exact alpha = 0.05: power")
    print("p_proposed \\ p_omni | " + " | ".join(str(p) for p in (0.1, 0.3, 0.5)))
    for pp in (0.6, 0.7, 0.8, 0.9, 0.95):
        print(pp, *[f"{fisher_power(20, pp, 20, po):.3f}" for po in (0.1, 0.3, 0.5)], sep=" | ")
    print("\n## H4 non-inferiority vs oracle (margin 0.20, Newcombe 90% two-sided lower bound), n = 20 per arm")
    print("p_proposed \\ p_oracle | 0.90 | 0.95 | 1.00")
    for pp in (0.80, 0.85, 0.90, 0.95, 1.00):
        print(pp, *[f"{ni_power(20, pp, 20, po):.3f}" for po in (0.90, 0.95, 1.00)], sep=" | ")
    print("\n## H4 non-inferiority with n = 40 per arm (proposed/oracle only)")
    for pp in (0.85, 0.90, 0.95):
        print(pp, *[f"{ni_power(40, pp, 40, po):.3f}" for po in (0.90, 0.95, 1.00)], sep=" | ")
    print("\n## H3 P(CP 95% upper bound of unsupported-cause rate <= 0.10)")
    print("n | rate 0.03 | 0.05 | 0.07")
    for n in (240, 600, 1400):
        print(n, *[f"{p_upper_below(n, r):.3f}" for r in (0.03, 0.05, 0.07)], sep=" | ")
    print("\n## H3 point-estimate misattribution: P(CP 95% lower bound > 0.10) with n_ood = 240 (F6+F7 pooled)")
    for r in (0.15, 0.2, 0.3):
        k = np.arange(241)
        ok = np.array([_cp_lower(int(x), 240) > 0.10 for x in k])
        print(r, f"{float(sps.binom.pmf(k, 240, r) @ ok):.3f}")


if __name__ == "__main__":
    main()
