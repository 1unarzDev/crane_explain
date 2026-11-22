"""WS-D2 tests: the approved H1 split (capability = black-box ensemble, diagnosis = structured ensemble vs analytic
Bayes), the default rule_set, joint coverage as the primary metric, and the coast+brake-merged diagnosis definition."""
import numpy as np
import pytest

from capxfer.analysis import run_frozen
from capxfer.analysis import frozen as F
from capxfer.analysis._synth import build


def test_defaults_are_the_approved_redesign():
    d = F.DEFAULTS
    assert d["rule_set"] == "redesign"
    assert (d["h1r_cap_method"], d["h1r_diag_method"], d["h1r_diag_baseline"]) == (
        "ensemble", "structured_ensemble", "analytic_bayes")
    assert d["h1r_diag_merge_cb"] is True and d["h1r_aps_min"] == 0.85 and d["h1r_diag_rate_min"] is None
    assert d["h3_cap_method"] == "ensemble"
    assert "h1r_width_baseline" not in d and "h1r_method" not in d       # the directional width claim is gone


def test_diagnosed_coast_brake_merge():
    # wheel 0 true BRAKE predicted COAST-ish; wheel 1 true OK
    mp = np.array([[[0.1, 0.0, 0.5, 0.4, 0.0], [0.9, 0.0, 0.05, 0.05, 0.0]],
                   [[0.4, 0.0, 0.3, 0.3, 0.0], [0.9, 0.0, 0.05, 0.05, 0.0]]])   # trial 1: merged 0.6 beats OK 0.4
    t = dict(mode_prob=mp, mode_true=np.array([[3, 0], [2, 0]]), wheel_mask=np.ones((2, 2), bool))
    assert F._diagnosed(t, False).tolist() == [False, False]
    assert F._diagnosed(t, True).tolist() == [True, True]


@pytest.fixture(scope="module")
def red(tmp_path_factory):
    d = tmp_path_factory.mktemp("h1r")
    m = build(d, seed=5, per_cell=8, ood_with_fault=True, redesign=True, config=dict(tau=0.9, n_boot=100))
    return d, m, run_frozen(m)


def test_h1_split_verdict_and_primary_metric(red):
    _, _, r = red
    h1 = r["H1"]
    rd = h1["redesign"]
    assert h1["rule"] == F.RULE_TAG and rd["status"] == F.STATUS
    # top-level H1 now reports the deployed capability method's joint coverage
    assert h1["method"] == "ensemble" and h1["per_fold"] == rd["capability"]["per_fold"]
    assert set(h1["per_fold"]) == {"single", "pair", "L3", "square4"}
    assert rd["primary_coverage_metric"] == F.JOINT_METRIC and "10 untested" in F.JOINT_METRIC
    for v in h1["per_fold"].values():                          # per-key reported, and never below joint coverage
        assert v["per_key_coverage"] >= v["coverage"] - 1e-12
    # the superseded earlier-draft H1 is kept, descriptive
    assert h1["original"]["method"] == "structured_npe" and h1["original"]["status"] == "SUPERSEDED_DESCRIPTIVE"
    # (b) diagnosis: both definitions, primary = merged
    dg = rd["diagnosis"]
    assert dg["primary_definition"] == "coast_brake_merged"
    assert set(dg["definitions"]) == {"coast_brake_merged", "unmerged"}
    assert dg["mcnemar_p"] == dg["definitions"]["coast_brake_merged"]["mcnemar_p"]
    assert "structured_ensemble" in dg["wheel_sets"]
    cap = rd["capability"]
    exp_b = bool(dg["mcnemar_p"] < 0.05 and dg["wheel_sets"]["structured_ensemble"]["all_wheels_covered"]["est"] >= 0.85)
    assert dg["passes"] == exp_b
    assert rd["supported"] == bool(cap["passes"] and exp_b) and h1["supported"] == rd["supported"]
    assert rd["strongly_supported"] == bool(rd["supported"] and cap["strong"])
    # secondary: descriptive only, includes the width comparison and structured methods
    sec = rd["secondary"]
    assert sec["status"] == "DESCRIPTIVE_NO_VERDICT"
    assert {"structured_ensemble", "structured_npe", "blackbox"} <= set(sec["capability"])
    assert all("supported" not in v for v in sec["capability"].values())
    assert sec["width_comparison"]["methods"] == ["structured_ensemble", "ensemble"]
    assert "supported" not in sec["width_comparison"] and "sign_p" not in sec["width_comparison"]
    assert r["hypotheses"]["H1"]["supported"] == rd["supported"]


def test_h3_and_h5_use_joint_primary(red):
    _, _, r = red
    assert r["H3"]["rule"] == F.RULE_TAG and r["H3"]["redesign"]["cap_method"] == "ensemble"
    assert r["H3_ood"]["definitions"]["coverage_primary"] == F.JOINT_METRIC
    assert "10 untested" in r["H5"]["coverage_metric"]
    row = r["H5"]["per_method"][0]
    assert row["per_key_cov_real"] >= row["cov_real"] - 1e-12


def test_config_keys_switch_methods_and_original_rule(red, tmp_path):
    import json
    d, m, _ = red
    man = json.loads(m.read_text())
    man["output_dir"] = str(tmp_path / "o")
    man["pred_tables"] = {k: {rr: str(d / p) for rr, p in v.items()} for k, v in man["pred_tables"].items()}
    for k in ("sim_tables", "explain_scores", "docking", "probe_policy", "probe_ladder"):
        man.pop(k)
    man["config"].update(h1r_cap_method="structured_npe", h1r_aps_min=1.01)   # unreachable APS floor -> (b) fails
    (tmp_path / "m.json").write_text(json.dumps(man))
    r = run_frozen(tmp_path / "m.json")
    rd = r["H1"]["redesign"]
    assert rd["cap_method"] == "structured_npe" and r["H1"]["method"] == "structured_npe"
    assert rd["diagnosis"]["aps_passes"] is False and rd["supported"] is False
    man["config"].update(rule_set="original", h1r_aps_min=0.85)
    (tmp_path / "m2.json").write_text(json.dumps(man))
    r2 = run_frozen(tmp_path / "m2.json")
    assert r2["H1"]["rule"].startswith("original") and "redesign" in r2["H1"]
    assert r2["H1"]["supported"] == (r2["H1"]["folds_passing_point"] >= 3)


def test_missing_diag_tables_not_evaluated(tmp_path):
    m = build(tmp_path, seed=6, per_cell=6, config=dict(n_boot=50))     # no structured_ensemble / analytic_bayes
    r = run_frozen(m)
    rd = r["H1"]["redesign"]
    assert rd["supported"] is None and any(n.startswith("(b) not evaluated") for n in rd["notes"])
    assert "capability" in rd and r["hypotheses"]["H1"]["verdict"] == "not evaluated (input missing)"
