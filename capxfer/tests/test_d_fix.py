"""WS-FIX tests for WS-D: arbitrary wheel counts (M10) and the H3 redesign metrics (OOD conditions, overclaim)."""
import json

import numpy as np
import pytest

from capxfer.analysis import run_frozen, table as tb
from capxfer.analysis import frozen as F
from capxfer.analysis._synth import build
from capxfer.conformal import wheel_sets as ws


def _batch(rng, B, n):
    lg = rng.standard_normal((B, n, 5))
    mp = np.exp(lg) / np.exp(lg).sum(-1, keepdims=True)
    mode = rng.integers(0, 5, (B, n))
    pred = dict(y_q=np.sort(rng.random((B, 14, 3)), -1), ppc_score=rng.random(B), mode_prob=mp,
                q_levels=np.array([0.05, 0.5, 0.95]))
    meta = {k: np.array([f"{k}{n}_{i}" for i in range(B)]) for k in tb.STR_KEYS}
    return pred, rng.random((B, 14)), dict(mode=mode), meta


def test_from_preds_any_wheel_count(tmp_path):
    rng = np.random.default_rng(0)
    bs = [_batch(rng, 5, n) for n in (4, 24, 20)]
    t = tb.from_preds(*zip(*bs))
    assert t["mode_prob"].shape == (15, 24, 5) and t["wheel_mask"].sum(1).tolist() == [4] * 5 + [24] * 5 + [20] * 5
    assert np.allclose(t["mode_prob"][5:10], bs[1][0]["mode_prob"])
    tb.save(tmp_path / "t.npz", t)
    t2 = tb.load(tmp_path / "t.npz")
    assert t2["mode_true"].shape == (15, 24)
    # default minimum width stays 16 for small assemblies; pad_to=None gives exactly max n
    small = tb.from_preds(*zip(*[_batch(rng, 3, 8)]))
    assert small["wheel_mask"].shape[1] == tb.MAX_WHEELS
    assert tb.from_preds(*zip(*[_batch(rng, 3, 8)]), pad_to=None)["wheel_mask"].shape[1] == 8
    c = tb.concat([small, t])
    assert c["wheel_mask"].shape == (18, 24) and not c["wheel_mask"][:3, 8:].any()
    assert tb.crop_wheels(tb.subset(c, np.arange(3)))["mode_prob"].shape == (3, 8, 5)
    with pytest.raises(ValueError):
        tb.crop_wheels(c, 8)


def test_wheel_sets_padding_without_mask_equals_masked():
    rng = np.random.default_rng(1)
    p, _, lab, _ = _batch(rng, 400, 24)
    m = np.zeros((400, 24), bool)
    m[:200, :12] = True
    m[200:] = True
    mp = np.where(m[..., None], p["mode_prob"], np.nan)
    y = np.where(m, lab["mode"], -1)
    a = ws.calibrate(dict(mode_prob=mp, wheel_mask=m), y, 0.1)
    b = ws.calibrate(dict(mode_prob=mp), y, 0.1)                    # mask inferred from NaN rows / -1 labels
    assert a.threshold == b.threshold and np.isfinite(a.threshold)
    sa, sb = a.sets(dict(mode_prob=mp, wheel_mask=m)), b.sets(dict(mode_prob=mp))
    assert np.array_equal(sa, sb) and not sa[:200, 12:].any()


def test_overclaim_rule():
    y = np.array([[0.2, 0.9], [0.8, 0.9], [0.2, 0.2]])
    lo = np.array([[0.6, 0.6], [0.6, 0.1], [0.1, 0.1]])
    wrong, ncert, anyc = F.overclaim(lo, y, np.array([0, 1]), 0.5)
    assert wrong.tolist() == [True, False, False] and ncert.tolist() == [2, 1, 0]
    wrong_g, _, _ = F.overclaim(lo, y, np.array([0, 1]), 0.5, gate=np.array([True, False, False]))
    assert wrong_g.tolist() == [False, False, False]


@pytest.fixture(scope="module")
def dry24(tmp_path_factory):
    d = tmp_path_factory.mktemp("dry24")
    m = build(d, seed=2, per_cell=10, ood_with_fault=True, sizes=(1, 2, 3, 4, 5, 6),
              config=dict(tau=0.9))          # synthetic ratios sit near 1: a high tau makes OOD trials consequential
    return d, run_frozen(m)


def test_run_frozen_with_24_wheel_sim_and_h3_redesign(dry24):
    d, r = dry24
    st = tb.load(d / "sim_structured_npe.npz")
    assert st["n_wheels"].max() == 24 and st["wheel_mask"].shape[1] == 24
    assert "H5" in r and r["H5"]["per_method"]
    h = r["H3_ood"]
    assert h["n"]["ood_fault"] > 0 and h["n"]["ood_consequential"] > 0
    e = h["per_method"]["structured_npe"]["E2"]
    assert set(e["coverage"]) >= {"inmodel", "ood_fault", "ood_consequential"}
    assert "conformal_gated.decision" in e["overclaim"]["ood_consequential"]
    rd = r["H3"]["redesign"]
    assert rd["status"] == F.STATUS and rd["n_ood_fault"] > 0
    assert rd["supported"] in (True, False)
    # the approved redesign H3 verdict is binding by default; the original one is kept as original_supported
    assert r["H3"]["rule"] == F.RULE_TAG and r["H3"]["supported"] == rd["supported"]
    assert "original_supported" in r["H3"]
    js = json.loads((d / "out" / "results.json").read_text())
    assert "H3_ood" in js
    for f in ("fig_ood_coverage.csv", "fig_overclaim.csv", "fig_explain_h3_redesign.csv"):
        assert (d / "out" / f).exists()


def test_rule_set_redesign_switches_verdicts(tmp_path):
    m = build(tmp_path, seed=3, per_cell=8, ood_with_fault=True, redesign=True,
              config=dict(tau=0.9, rule_set="redesign", n_boot=100))
    r = run_frozen(m)
    h1, h2, h3 = r["H1"], r["H2"], r["H3"]
    assert h1["redesign"]["cap_method"] == "ensemble" and h1["redesign"]["diag_method"] == "structured_ensemble"
    assert "capability" in h1["redesign"] and h1["method"] == "ensemble"
    assert "diagnosis" in h1["redesign"] and h1["rule"].startswith("redesign")
    assert [x["label"] for x in h2["probe_ladder"]["rungs"]] == ["P1P5", "P1P5P2", "P1-P5", "all"]
    assert len(h2["probe_ladder"]["steps"]) == 3 and h2["supported"] == h2["probe_ladder"]["supported"]
    assert h3["supported"] == h3["redesign"]["supported"] and "original_supported" in h3
    assert r["hypotheses"]["H1"]["verdict"] in ("supported", "supported (strong criterion met)", "not supported")
