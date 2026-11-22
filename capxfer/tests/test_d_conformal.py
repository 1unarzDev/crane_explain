"""WS-D conformal tests on synthetic Preds (no upstream dependency).

Binomial tolerance: for n test trials, |coverage - (1-alpha)| is allowed up to 4 * sqrt(a(1-a)/n) below target
(one-sided; finite-sample split CP is conservative, so we only bound under-coverage tightly).
"""
import math

import numpy as np
import pytest
import torch

from capxfer.conformal import capability as cap
from capxfer.conformal import ood, wheel_sets as ws
from capxfer.conformal._common import conformal_quantile, weighted_conformal_quantile
from capxfer.conformal._synth import make_batch, make_groups

ALPHA = 0.1


def _split(d, a, b, N):
    return {k: (v[a:b] if hasattr(v, "shape") and v.shape[:1] == (N,) else v) for k, v in d.items()}


def _tol(n, a=ALPHA, z=4.0):
    return z * math.sqrt(a * (1 - a) / n)


@pytest.fixture(scope="module")
def exch():
    rng = np.random.default_rng(0)
    N = 6000
    p, l, y = make_batch(rng, N, n=4)
    return _split(p, 0, 3000, N), _split(p, 3000, N, N), l, y


def test_quantile_helpers():
    s = np.arange(1, 20, dtype=float)       # n = 19 -> k = ceil(20*0.9) = 18
    assert conformal_quantile(s, 0.1) == 18.0
    assert math.isinf(conformal_quantile(s[:5], 0.1))
    assert weighted_conformal_quantile(s, np.ones_like(s), 0.1, 1.0) == 18.0


@pytest.mark.parametrize("joint", [False, True])
def test_capability_exchangeable_coverage(exch, joint):
    pc, pt, _, y = exch
    c = cap.calibrate(pc, y[:3000], ALPHA, joint=joint)
    cov, j = c.covered(pt, y[3000:])
    n = cov.shape[0]
    if joint:
        assert j.mean() >= 1 - ALPHA - _tol(n)
        assert cov.mean() >= j.mean()
    else:
        assert np.all(cov.mean(0) >= 1 - ALPHA - _tol(n))
        assert abs(cov.mean() - (1 - ALPHA)) < 0.02          # not grossly conservative either


def test_capability_sample_fallback_and_torch(exch):
    pc, pt, _, y = exch
    pc2 = {k: v for k, v in pc.items() if k != "y_q"}
    pt2 = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) and v.dtype != object else v
           for k, v in pt.items() if k != "y_q"}
    c = cap.calibrate(pc2, y[:3000], ALPHA, joint=True)
    lo, hi = c.interval(pt2)
    assert isinstance(lo, torch.Tensor) and lo.shape == (3000, 14)
    _, j = c.covered(pt2, y[3000:])
    assert j.mean() >= 1 - ALPHA - _tol(3000)


def test_mondrian_group_conditional():
    rng = np.random.default_rng(3)
    G = make_groups(rng, 4, 1500, tau=0.2)
    preds = [g[0] for g in G]
    grp = np.repeat(np.arange(4), 1500)
    half = lambda d: _split(d, 0, 750, 1500)
    rest = lambda d: _split(d, 750, 1500, 1500)
    c = cap.calibrate([half(p) for p in preds], [g[2][:750] for g in G], ALPHA,
                      groups=np.repeat(np.arange(4), 750), method="mondrian", joint=True)
    for k, g in enumerate(G):
        _, j = c.covered(rest(g[0]), g[2][750:], groups=np.full(750, k))
        assert j.mean() >= 1 - ALPHA - _tol(750)
    # unseen group falls back to the group threshold (infinite with K=4 at alpha=0.1 -> full coverage)
    _, j = c.covered(rest(G[0][0]), G[0][2][750:], groups=np.full(750, "unseen"))
    assert j.mean() == 1.0
    assert grp.size == 6000


def _group_shift_experiment(rng, K, tau, reps, per_group=30):
    res = {"split": [], "ss1": [], "pooled": []}
    for r in range(reps):
        G = make_groups(rng, K + 1, per_group, tau=tau)
        preds, ys = [g[0] for g in G[:K]], [g[2] for g in G[:K]]
        grp = np.repeat(np.arange(K), per_group)
        te = G[K]
        for name, kw in [("split", dict(method="split")), ("ss1", dict(method="group", group_variant="ss1")),
                         ("pooled", dict(method="group", group_variant="pooled"))]:
            c = cap.calibrate(preds, ys, ALPHA, groups=grp, joint=True, seed=r, **kw)
            res[name].append(c.covered(te[0], te[2])[1].mean())
    return {k: float(np.mean(v)) for k, v in res.items()}


def test_group_shift_documented_undercoverage():
    """DOCUMENTED BEHAVIOUR (feeds BENCHMARK_DESIGN.md section 'Why not plain split CP').

    Exchangeable groups with strong group offsets (tau = 0.3 >> within-group noise), held-out NEW group:
      * K = 3 calibration groups (the real-data situation): split CP under-covers (about 0.84-0.85 at target 0.90);
        group conformal has an infinite threshold (K < 1/alpha - 1), i.e. it refuses to give a finite interval.
      * K = 19: group ss1 reaches >= 1 - alpha (within Monte-Carlo tolerance); split still sits below target.
    """
    rng = np.random.default_rng(7)
    r3 = _group_shift_experiment(rng, 3, 0.3, reps=200)
    assert r3["split"] < 1 - ALPHA - 0.03
    assert r3["ss1"] == 1.0 and r3["pooled"] == 1.0
    r19 = _group_shift_experiment(rng, 19, 0.3, reps=200)
    assert r19["ss1"] >= 1 - ALPHA - 0.04          # MC s.e. ~0.015 at 200 reps
    assert r19["split"] < r19["ss1"] + 0.02


def test_nonexchangeable_shift_breaks_everything():
    """A systematically shifted test group (offset far outside the calibration groups) is under-covered by every
    method: no conformal variant protects against non-exchangeable shift. This is why real-data LOAO coverage is
    reported as EMPIRICAL coverage under shift, and why the OOD test exists."""
    rng = np.random.default_rng(11)
    G = make_groups(rng, 20, 50, tau=0.05)
    preds, ys = [g[0] for g in G], [g[2] for g in G]
    grp = np.repeat(np.arange(20), 50)
    tp, _, ty = make_batch(rng, 500, n=8, group_offset=-0.4)
    for kw in (dict(method="split"), dict(method="group", group_variant="ss1"),
               dict(method="group", group_variant="pooled")):
        c = cap.calibrate(preds, ys, ALPHA, groups=grp, joint=True, **kw)
        assert c.covered(tp, ty)[1].mean() < 1 - ALPHA - 0.2


@pytest.mark.parametrize("kw", [dict(), dict(pooling="subsample"), dict(randomized=True, pooling="subsample"),
                                dict(pooling="weighted"), dict(method="class", pooling="subsample")])
def test_aps_exchangeable_coverage(exch, kw):
    pc, pt, l, _ = exch
    c = ws.calibrate(pc, l["mode"][:3000], ALPHA, **kw)
    S = c.sets(pt)
    assert S.shape == (3000, 4, 5) and S.dtype == bool
    hit, _ = ws.coverage(S, l["mode"][3000:])
    assert hit.mean() >= 1 - ALPHA - _tol(3000)
    if kw.get("method") == "class":
        yt = l["mode"][3000:].ravel()
        for k in range(5):
            m = yt == k
            assert hit[m].mean() >= 1 - ALPHA - _tol(int(m.sum()))


def test_aps_randomized_is_tight_and_joint(exch):
    pc, pt, l, _ = exch
    hit, _ = ws.coverage(ws.calibrate(pc, l["mode"][:3000], ALPHA, randomized=True, pooling="subsample").sets(pt),
                         l["mode"][3000:])
    assert abs(hit.mean() - (1 - ALPHA)) < 0.02
    cj = ws.calibrate(pc, l["mode"][:3000], ALPHA, pooling="joint")
    _, alltrial = ws.coverage(cj.sets(pt), l["mode"][3000:])
    assert alltrial.mean() >= 1 - ALPHA - _tol(3000)


def test_aps_padded_lists():
    rng = np.random.default_rng(5)
    batches = [make_batch(rng, 800, n=n) for n in (4, 8, 16)]
    preds, labs = [b[0] for b in batches], [b[1] for b in batches]
    c = ws.calibrate(preds, labs, ALPHA, pooling="subsample")
    test = [make_batch(rng, 800, n=n) for n in (4, 8, 16)]
    S = c.sets([t[0] for t in test])
    hits = np.concatenate([ws.coverage(s, t[1])[0] for s, t in zip(S, test)])
    assert hits.mean() >= 1 - ALPHA - _tol(2400, z=4)
    with pytest.raises(ValueError):
        ws.calibrate({"mode_prob": None}, np.zeros((1, 4), int), ALPHA)


def test_ood_false_alarm_and_power():
    rng = np.random.default_rng(9)
    p, _, _ = make_batch(rng, 4000)
    o = ood.calibrate(p["ppc_score"][:2000], ALPHA)
    fa = o.flag({"ppc_score": p["ppc_score"][2000:]}).mean()
    assert fa <= ALPHA + _tol(2000)
    q, _, _ = make_batch(rng, 1000, ood_frac=1.0)
    assert o.flag(q).mean() > 0.5
    pv = o.pvalue(q)
    assert np.all((pv > 0) & (pv <= 1))
    assert np.array_equal(ood.flag(q, o), o.flag(q))
