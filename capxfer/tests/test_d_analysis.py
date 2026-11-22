"""WS-D analysis tests: statistics helpers and an end-to-end dry run of run_frozen on fully synthetic inputs."""
import json

import numpy as np
import pytest
from scipy import stats as sps

from capxfer.analysis import run_frozen, stats as S, table as tb
from capxfer.analysis._synth import build


def test_clopper_pearson_matches_beta():
    for k, n in [(0, 20), (17, 20), (270, 300), (20, 20)]:
        lo, hi = S.clopper_pearson(k, n, 0.95)
        elo = 0.0 if k == 0 else sps.beta.ppf(0.025, k, n - k + 1)
        ehi = 1.0 if k == n else sps.beta.ppf(0.975, k + 1, n - k)
        assert abs(lo - elo) < 1e-6 and abs(hi - ehi) < 1e-6


def test_newcombe_reference_example():
    # Newcombe (1998) Stat Med 17:873, example (e): 56/70 vs 48/80, method 10 95% CI = (0.0524, 0.3339)
    lo, hi = S.newcombe_diff(56, 70, 48, 80, conf=0.95)
    assert abs(lo - 0.0524) < 5e-4 and abs(hi - 0.3339) < 5e-4


def test_tests_and_holm():
    p, pos, neg = S.sign_test_greater(np.r_[np.ones(15), -np.ones(5), np.zeros(3)])
    assert (pos, neg) == (15, 5) and abs(p - sps.binomtest(15, 20, 0.5, "greater").pvalue) < 1e-9
    p, n10, n01 = S.mcnemar_greater(np.r_[np.ones(10, bool), np.zeros(2, bool)], np.zeros(12, bool))
    assert (n10, n01) == (10, 0) and p < 0.001
    assert S.fisher_greater(18, 20, 6, 20) < 0.001
    assert np.allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cluster_bootstrap_wider_than_iid_under_clustering():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(10), 50)
    p = np.repeat(rng.uniform(0.6, 1.0, 10), 50)
    hit = rng.random(500) < p
    r = S.coverage_summary(hit, cl, [cl, np.tile(np.arange(5), 100)], 500, 0)
    assert r["boot_oneway"]["hi"] - r["boot_oneway"]["lo"] > (r["cp_hi"] - r["cp_lo"]) * 0.9
    assert r["conservative_lower"] <= r["cp_lower_one_sided"]
    assert S.boot_ci(hit, S.bootstrap_weights(np.zeros(500), 10, 0))["status"].startswith("WITHHELD")


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    d = tmp_path_factory.mktemp("dry")
    m = build(d, seed=1, per_cell=12)
    return d, m, run_frozen(m)


def test_end_to_end_outputs(dry):
    d, _, r = dry
    out = d / "out"
    for f in ("results.json", "fig_coverage_by_fold.csv", "fig_ladder.csv", "fig_struct_vs_bb.csv",
              "fig_wheel_sets.csv", "fig_ood_flag.csv", "fig_explain_ood.csv", "fig_docking.csv",
              "fig_sim_real.csv", "fig_probe_policy.csv"):
        assert (out / f).exists(), f
    js = json.loads((out / "results.json").read_text())
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert h in js and "supported" in js[h], h
    assert js["status"] == "DRY_RUN_NOT_FROZEN" and js["source"] == "SYNTHETIC"
    # paper/results_map.json paths (WS-G) resolve, except the sysid/provenance ones (supplied via `provenance`)
    import pathlib
    # run_frozen serves the superseded v1 paper map (paper/archive_v1); the current map binds PROTOCOL_V2 output
    # (results_v2.json) and is checked in tests/test_d_realv2.py.
    rm = pathlib.Path(__file__).resolve().parents[1] / "paper" / "archive_v1" / "results_map.json"
    if rm.exists():
        missing = []
        for name, spec in json.loads(rm.read_text())["map"].items():
            node = js
            for part in spec["path"].split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                missing.append(spec["path"])
        allowed = ("sysid.", "counts.n_sysid", "counts.n_sim_train", "counts.n_sim_families",
                   "width.loao.structured_npe.E3", "coverage.loao.analytic_bayes")
        assert all(p.startswith(allowed) for p in missing), missing
    assert set(js["H1"]["per_fold"]) == {"single", "pair", "L3", "square4"}
    assert len(js["code_sha256"]) == 9 and all(len(v["sha256"]) == 64 for v in js["inputs"].values())
    assert js["eval_targets"] == ["P2+", "P2-", "P3+", "P3-", "P4+", "P4-", "P6x+", "P6x-", "P6y+", "P6y-"]


def test_end_to_end_sanity(dry):
    _, _, r = dry
    # synthetic structured model is well specified -> split coverage near target in every fold
    for f, v in r["H1"]["per_fold"].items():
        assert 0.75 <= v["coverage"] <= 1.0, (f, v)
    # group conformal over 3 calibration types is infinite at alpha = 0.1 (documented design consequence)
    g = [x for x in r["per_fold"] if x["scheme"] == "group_ss1_type"]
    assert g and not any(x["threshold_finite"] for x in g)
    # ladder: synthetic errors shrink E0 > E1 > E2
    widths = {x["rung"]: x["median_width"] for x in r["ladder"] if x["fold"] == "all"}
    assert widths["E0"] > widths["E1"] > widths["E2"]
    assert r["H3"]["point_misattribution"] > r["H3"]["calibrated_misattribution"]
    assert r["H4"]["fisher_p_proposed_gt_omni"] < 0.05
    assert r["H5"]["per_method"][1]["gap"] > r["H5"]["per_method"][0]["gap"]   # nominal_sim gap larger


def test_leakage_guards(dry, tmp_path):
    d, m, _ = dry
    man = json.loads(m.read_text())
    t = tb.load(d / man["pred_tables"]["blackbox"]["E2"])
    tb.save(tmp_path / "bad.npz", tb.subset(t, np.arange(1, t["y"].shape[0])))
    man2 = dict(man, output_dir=str(tmp_path / "o"),
                pred_tables={k: {r: str(d / p) for r, p in v.items()} for k, v in man["pred_tables"].items()})
    man2["pred_tables"]["blackbox"]["E2"] = str(tmp_path / "bad.npz")
    for k in ("sim_tables", "explain_scores", "docking", "probe_policy"):
        man2.pop(k)
    (tmp_path / "m.json").write_text(json.dumps(man2))
    with pytest.raises(ValueError, match="exactly the reference"):
        run_frozen(tmp_path / "m.json")
    t2 = dict(t)
    t2["source"] = np.full(t["y"].shape[0], "SYNTHETIC")
    tb.save(tmp_path / "syn.npz", t2)
    man3 = dict(man2, frozen=True, pred_tables={"structured_npe": {"E2": str(tmp_path / "syn.npz")}})
    (tmp_path / "m3.json").write_text(json.dumps(man3))
    with pytest.raises(ValueError, match="FIELD"):
        run_frozen(tmp_path / "m3.json")
