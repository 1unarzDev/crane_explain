"""WS-D3 pre-freeze changes (PROTOCOL section 4 'Pre-freeze changes', 2026-11-20): H2 = diagnosis ladder, H5 = under-coverage
shortfall, condition-based OOD fault fallback, H3 per-type evaluability."""
import numpy as np
import pandas as pd

from capxfer.analysis import frozen as F, run_frozen
from capxfer.analysis._synth import build


def _ladder_tab(N, W, diag_rates, rng, labels):
    """PredTables whose argmax is correct on all wheels for a nested set of trials growing with diag_rates."""
    mode = np.where(rng.random((N, W)) < 0.3, rng.integers(1, 5, (N, W)), 0)
    mode[:, 0] = rng.integers(1, 5, N)                                     # every trial faulted
    u = rng.random(N)
    out = []
    for r in diag_rates:
        ok = u < r
        am = np.where(ok[:, None], mode, (mode + 1) % 5)
        mp = np.full((N, W, 5), 0.05)
        np.put_along_axis(mp, am[..., None], 0.8, -1)
        out.append(dict(mode_prob=mp, mode_true=mode, wheel_mask=np.ones((N, W), bool)))
    return out


def test_diagnosed_mcnemar_ladder_rule():
    rng = np.random.default_rng(0)
    tabs = _ladder_tab(400, 4, [0.6, 0.9, 0.9, 0.9], rng, None)
    d = [F._diagnosed(t, True) for t in tabs]
    assert d[0].mean() < d[1].mean() and np.array_equal(d[1], d[2])
    p, n10, n01 = F.S.mcnemar_greater(d[1], d[0])
    assert p < 1e-6 and n01 == 0


def test_h5_shortfall_and_signed_gap(tmp_path):
    r = run_frozen(build(tmp_path, seed=4, per_cell=8, config=dict(n_boot=200)))
    h5 = r["H5"]
    assert "shortfall" in h5["metric"]
    for row in h5["per_method"]:
        assert row["shortfall"] == max(0.0, row["gap"])
    assert set(h5["signed_gap"]) == {"structured_npe", "nominal_sim"}
    assert "abs_gap_one_sided_lower" in h5 and h5["supported"] == (h5["one_sided_lower"] > 0)


def test_condition_fallback_counts_f6w_as_faulted():
    t = dict(ood=np.array(["none", "payload", "payload", "low_mu"]), fault=np.array(["F1", "F6", "F6", "F7"]),
             condition=np.array(["F1", "F6", "F6w", "F7w"]), y=np.ones((4, 14)))
    c = F.trial_conditions(t, F.DEFAULTS, F.eval_targets(["P1", "P5"]))
    assert c["ood_fault"].tolist() == [False, False, True, True]
    assert c["inmodel_fault"].tolist() == [True, False, False, False]
    del t["condition"]
    assert F.trial_conditions(t, F.DEFAULTS, F.eval_targets(["P1", "P5"]))["ood_fault"].sum() == 0


def test_h2_diagnosis_primary_and_h3_by_type(tmp_path):
    m = build(tmp_path, seed=3, per_cell=8, ood_with_fault=True, redesign=True,
              config=dict(tau=0.9, n_boot=100))
    r = run_frozen(m)
    pl = r["H2"]["probe_ladder"]
    assert "diagnosed" in pl["primary_metric"] and pl["width_steps"]["status"] == "DESCRIPTIVE_NO_VERDICT"
    # owner-approved: confirmatory set = in-model double-fault (F5) trials; all faulted trials descriptive
    from capxfer.analysis import table as tb
    import json as _json
    man = _json.load(open(m))
    ref = tb.load(tmp_path / man["pred_tables"]["structured_npe"]["E2"]) if not str(
        man["pred_tables"]["structured_npe"]["E2"]).startswith("/") else tb.load(man["pred_tables"]["structured_npe"]["E2"])
    nfw = ((ref["mode_true"] > 0) & ref["wheel_mask"]).sum(1)
    im = ref["ood"] == "none"
    assert pl["confirmatory_trials"] == "double_fault" and "double-fault" in pl["primary_metric"]
    assert pl["n_faulted"] == int((im & (nfw == 2)).sum())
    if pl["supported"] is not None:
        assert pl["all_faulted"]["n"] == int((im & (nfw > 0)).sum())
    for s in pl["steps"]:
        assert {"p_increase", "p_increase_holm", "p_decrease", "n_gained", "n_lost"} <= set(s)
    if pl["supported"] is not None:
        assert pl["supported"] == (pl["first_step_increase"] and pl["no_later_significant_decrease"])
    assert all("aps_mean_set_size" in x and "faulted_diagnosed_rate" in x for x in pl["rungs"])
    rd = r["H3"]["redesign"]
    assert set(rd["by_type"]) <= set(F.DEFAULTS["folds"])
    for ty, e in rd["by_type"].items():
        assert e["evaluable"] == (e["n_consequential"] > 0)
        if not e["evaluable"]:
            assert "not a failure" in e["status"] and ty in rd["types_not_evaluable"]
    # tau = 0 -> no consequential OOD anywhere: every type and the pooled H3 are 'not evaluable', never 'not supported'
    m0 = build(tmp_path / "t0", seed=3, per_cell=8, ood_with_fault=True, redesign=True, config=dict(tau=0.0, n_boot=50))
    r0 = run_frozen(m0)
    rd0 = r0["H3"]["redesign"]
    assert rd0["supported"] is None and all(not e["evaluable"] for e in rd0["by_type"].values())
    assert r0["hypotheses"]["H3"]["verdict"].startswith("not evaluable")
