"""WS-R (D2 wiring): capability claims from the capability table (ensemble, no mode_prob), wheel claims from the
diagnosis table (structured_ensemble); capability_overclaim column; trial alignment is enforced."""
import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.analysis import _synth as SY
from capxfer.pipeline import stages as ST


def _tables():
    rng = np.random.default_rng(0)
    df = SY._trials(rng, 6)
    modes, mean, y = SY._truth(rng, df)
    diag = SY._pred_table(rng, df, modes, mean, y, err=0.03, spread=1.0, sharp=3.0, source="SIMULATION")
    cap = SY._pred_table(rng, df, modes, mean, y, err=0.02, spread=1.0, sharp=3.0, source="SIMULATION")
    cap = {k: v for k, v in cap.items() if k not in ("mode_prob",)}       # black-box capability method
    return df, cap, diag


def test_split_wiring_and_overclaim():
    df, cap, diag = _tables()
    prm = ST.params()
    loaded = {k: ({"meta": {"assembly": A.physical(k, prm)}}, None) for k in SY.TYPES}
    ex, info = ST.explain_stage(cap, loaded, dict(alpha=0.1, seed=0), wheel_table=diag)
    assert len(ex) == 2 * len(df) and ex.verified.all()
    assert {"capability_overclaim", "ood_flag"} <= set(ex.columns)
    cal = ex[ex.explainer == "calibrated"]
    assert cal.wheel_cause_asserted.sum() > 0                             # wheel claims came from the diagnosis table
    assert cal.capability_overclaim.mean() <= ex[ex.explainer == "point_estimate"].capability_overclaim.mean() + 1e-9
    for f in info["folds"].values():
        assert "capability_overclaim_rate" in f["calibrated"]
    # capability-only table without a diagnosis table: wheel claims are withheld, never invented
    ex2, _ = ST.explain_stage(cap, loaded, dict(alpha=0.1, seed=0))
    assert ex2[ex2.explainer == "calibrated"].wheel_cause_asserted.sum() == 0
    bad = dict(diag, trial_id=diag["trial_id"][::-1].copy())
    with pytest.raises(ValueError, match="different trials"):
        ST.explain_stage(cap, loaded, dict(alpha=0.1, seed=0), wheel_table=bad)


def test_capability_overclaim_rule():
    dec = dict(required=["P2+", "P2-"], threshold=0.5)
    y = np.full((2, 14), 0.9)
    y[1, ST.TARGET_KEYS.index("P2+")] = 0.2
    e = dict(claims=[dict(type="capability", support="supported", value="reliable", subject="P2+")])
    assert ST.capability_overclaim([e, e], y, dec).tolist() == [False, True]
    w = dict(claims=[dict(type="capability", support="withheld", value="reliable", subject="P2+")])
    assert ST.capability_overclaim([w], y[1:], dec).tolist() == [False]
