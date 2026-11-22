"""WS-I3F: the one lambda_v2.json schema (capxfer.lambda_v2/v1) shared by choose_module, builders and realv2."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import yaml

from capxfer.analysis import realv2 as RV
from capxfer.analysis._synth_v2 import build_v2
from capxfer.field import bundle as BU
from capxfer.field import choose_module as CM


def test_n_needed_matches_realv2_rule():
    for a in (0.10, 0.05, 0.2):
        n = BU.n_needed(a)
        assert math.isinf(RV.classcond_threshold(list(range(n - 1)), a))
        assert math.isfinite(RV.classcond_threshold(list(range(n)), a))
    assert (BU.n_needed(0.10), BU.n_needed(0.05)) == (9, 19)


def test_record_roundtrip_and_null_is_inf(tmp_path):
    rec = BU.make_lambda_record(0.41, 0.55, 30, 0.02, 40, status="FINAL", bundle_sha256="a" * 64)
    p = BU.write_lambda(tmp_path / "lambda_v2.json", rec)
    r = BU.read_lambda(p)
    assert (r["lambda_hat"], r["lambda_hat_alpha05"], r["carry_over_offset"], r["status"]) == (0.41, 0.55, 0.02, "FINAL")
    lam, lam05, sha, rr = CM.load_lambda_record(p)                       # choose_module reads the same file
    assert (lam, lam05, sha) == (0.41, 0.55, BU.sha256_file(p)) and rr["bundle_sha256"] == "a" * 64
    assert CM.load_lambda(p)[:2] == (0.41, 0.55)                          # old reader agrees
    inf = BU.make_lambda_record(math.inf, None, 3, math.inf, 3, status="SYNTHETIC")
    assert inf["lambda_hat"] is None and inf["carry_over_offset"] is None
    r = BU.read_lambda(BU.write_lambda(tmp_path / "inf.json", inf))
    assert r["lambda_hat"] == math.inf and r["carry_over_offset"] == math.inf


@pytest.mark.parametrize("kw, msg", [
    (dict(lambda_hat=0.6, lambda_hat_alpha05=0.5, n_null_cal=30), "smaller alpha"),
    (dict(lambda_hat=0.6, lambda_hat_alpha05=0.7, n_null_cal=5), "n_null_cal 5 < 9"),
    (dict(lambda_hat=0.6, lambda_hat_alpha05=0.7, n_null_cal=12), "n_null_cal 12 < 19"),
    (dict(lambda_hat=None, lambda_hat_alpha05=0.7, n_null_cal=30), "+inf but lambda_hat_alpha05 is finite"),
])
def test_invariants(kw, msg):
    with pytest.raises(BU.LambdaFormatError, match=msg.replace("+", r"\+")):
        BU.make_lambda_record(kw["lambda_hat"], kw["lambda_hat_alpha05"], kw["n_null_cal"], status="FINAL",
                              bundle_sha256="b" * 64)


def test_final_needs_bundle_and_placeholder_only_nulls(tmp_path):
    with pytest.raises(BU.LambdaFormatError, match="bundle_sha256"):
        BU.make_lambda_record(0.4, 0.5, 30, status="FINAL")
    ph = BU.placeholder_lambda()
    assert BU.validate_lambda(ph) == [] and ph["status"] == "PLACEHOLDER"
    bad = dict(ph, lambda_hat=0.3, n_null_cal=30)
    assert any("only nulls" in e for e in BU.validate_lambda(bad))
    assert BU.validate_lambda(dict(ph, schema="other/v9"))
    assert BU.validate_lambda(dict(ph, lambda_hat="0.3"))                 # strings are not numbers
    (tmp_path / "x.json").write_text("{not json")
    with pytest.raises(SystemExit, match="bad lambda file"):
        CM.load_lambda_record(tmp_path / "x.json")


def test_legacy_file_reads_as_legacy(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"lambda_hat": 0.5, "lambda_hat_alpha05": 0.7, "note": "SYNTHETIC dry run"}))
    r = BU.read_lambda(p)
    assert r["status"] == "LEGACY" and r["lambda_hat"] == 0.5 and r["carry_over_offset"] is None
    p.write_text(json.dumps({"lambda_hat": None}))
    assert BU.read_lambda(p)["lambda_hat_alpha05"] == math.inf


def test_cal_calibration_equals_r3_lambda_and_uses_cal_only(tmp_path):
    """The field-day lambda (CAL units only, before DEMO) is exactly the lambda_hat R3 recomputes after collection,
    and TEST units cannot move it."""
    mp = build_v2(tmp_path)
    man = yaml.safe_load(mp.read_text())
    trials = pd.read_csv(tmp_path / man["mask_trials"])
    units = pd.read_csv(tmp_path / man["r3_units"])
    res = RV.run_realv2(mp)
    c = RV.cal_calibration(units, trials)
    assert c["lambda_hat"] == res["R3"]["lambda_hat"] and c["n_null_cal"] == res["R3"]["n_null_cal"]
    assert c["lambda_hat_alpha05"] == res["R3_alpha_sensitivity"]["lambda_hat"]
    assert c["carry_over_offset"] is None                                  # no bb_q05 / z_assembly in these units
    u2 = units.copy()
    u2.loc[u2["r3_split"] == "TEST", "score"] = 99.0
    assert RV.cal_calibration(u2, trials)["lambda_hat"] == c["lambda_hat"]
    t2 = trials.copy()                                                     # nor can the TEST assembly truth
    t2.loc[~t2["build"].str.startswith("single_"), [f"rho_{k}" for k in RV.KEYS]] = 0.0
    assert RV.cal_calibration(units, t2)["lambda_hat"] == c["lambda_hat"]
    rec = BU.make_lambda_record(c["lambda_hat"], c["lambda_hat_alpha05"], c["n_null_cal"], status="SYNTHETIC")
    BU.write_lambda(tmp_path / "lambda_v2.json", rec)
    man["lambda_v2"] = "lambda_v2.json"
    mp.write_text(yaml.safe_dump(man))
    assert RV.run_realv2(mp)["lambda_v2_check"]["status"] == "MATCH"
    BU.write_lambda(tmp_path / "lambda_v2.json", BU.make_lambda_record(
        (c["lambda_hat"] or 0) + 1.0, (c["lambda_hat_alpha05"] or 0) + 2.0, 40, status="SYNTHETIC"))
    assert RV.run_realv2(mp)["lambda_v2_check"]["status"] == "MISMATCH"


def test_carry_over_offset_rule():
    units = pd.DataFrame(dict(trial_id=[f"t{i}" for i in range(20) for _ in (0, 1)],
                              unit_module=["M1", "M3"] * 20, r3_split="CAL", fault_mw="none", score=0.9,
                              excluded=False, bb_q05=[0.9 + 0.01 * i for i in range(20) for _ in (0, 1)],
                              z_assembly=0.9, build="pair", assembly_type="pair"))
    trials = pd.DataFrame(columns=list(RV.MASK_COLS) + list(RV.RHO_COLS))
    c = RV.cal_calibration(units, trials)
    d = sorted(0.01 * i for i in range(20))                             # one value per trial, not per module
    assert c["n_carry_cal"] == 20 and c["carry_over_offset"] == pytest.approx(d[math.ceil(0.9 * 21) - 1])
