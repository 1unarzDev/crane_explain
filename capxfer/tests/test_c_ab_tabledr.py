"""WS-R: analytic_bayes hypothesis tables at a configurable plant (generation W: centre of sim.DR_WIDE)."""
from capxfer import sim as S
from capxfer.learn import _stub
from capxfer.learn.analytic_bayes import AnalyticBayes, dr_centre
from capxfer.learn.base import Predictor


def test_dr_centre_and_roundtrip(tmp_path):
    c = dr_centre(S.DR_WIDE)
    assert c["mu"] == (0.75, 0.75) and c["cg_off"] == 0.0 and c["wheel_yaw_err"] == 0.0
    assert set(c) == set(S.DR_WIDE)
    m = AnalyticBayes(dict(device="cpu", table_dr=c))
    assert m.table_dr()["mu"] == (0.75, 0.75)
    assert AnalyticBayes(dict(device="cpu")).table_dr() is _stub.NOMINAL_DR
    p = tmp_path / "ab.pt"
    m.save(p)
    m2 = Predictor.load(p, device="cpu")
    assert m2.table_dr()["v_batt"] == c["v_batt"]
