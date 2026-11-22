"""WS-PR2 tests: PROTOCOL_V2 analysis (capxfer/analysis/realv2.py) and design (realv2_design.py) on synthetic data."""
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from capxfer.analysis import realv2 as R, realv2_design as D
from capxfer.analysis._synth_v2 import build_v2

SEED = bytes(range(32))


# ------------------------------------------------------------------------------------------------------ design
def test_type_modules_match_assembly():
    from capxfer.assembly import PHYSICAL_TYPES
    for t, mods in D.TYPE_MODULES.items():
        assert tuple(m for m, *_ in PHYSICAL_TYPES[t]) == mods


def test_schedule_deterministic_balanced_and_committed():
    a, b = D.expand_schedule(SEED), D.expand_schedule(SEED)
    assert D.schedule_sha256(a) == D.schedule_sha256(b)
    assert D.schedule_sha256(a) != D.schedule_sha256(D.expand_schedule(bytes(32)))
    assert a["commitment"] == D.commitment(SEED) and len(a["commitment"]) == 64
    assert len(a["mask_trials"]) == D.mask_counts()["trials"] == 198
    assert a["mask_order"][0] == "square4" and all(x.startswith("single_") for x in a["mask_order"][1:5])
    for m in D.MODULES:                                   # 2 CAL + 2 TEST wheels per module
        assert sum(a["r3_split"][f"{m}:{w}"] == "CAL" for w in D.WHEELS) == 2
    fm = [tuple(e["faulted_modules"]) for e in a["demo"]]
    assert len(fm) == 20 and sorted(fm.count(x) for x in set(fm)) == [6, 7, 7]
    assert all(e["assembly"] == "L3" and e["fault"] == "coast" and e["bay"] == "narrow" for e in a["demo"])
    assert all(set(e["faulted_modules"]) <= set(D.DEMO_MODULES) and len(set(e["faulted_modules"])) == 2
               for e in a["demo"])
    assert set(a["mode"].values()) <= {"coast", "brake"}
    # every faulted MASK trial uses the sealed mode of its (module, wheel)
    for t in a["mask_trials"]:
        if t["module"]:
            assert t["firmware"] == a["mode"][f"{t['module']}:{t['wheel']}"]
    with pytest.raises(ValueError):
        D.expand_schedule(b"short")


def test_hours_fit_the_budget():
    h = D.hours_table()
    assert 10.0 <= h["total"][0] and h["total"][1] <= 16.5  # D16: all three modules dock per DEMO episode (owner-accepted budget)
    assert h["mask_trials"] == 198


# --------------------------------------------------------------------------------------------------------- R1
def _brute_trend(blocks, types, vals, order):
    rank = {t: i for i, t in enumerate(order)}
    groups = {}
    for b, t, v in zip(blocks, types, vals):
        groups.setdefault(b, []).append((rank[t], v))
    gs = [sorted(g) for g in groups.values() if len(g) >= 2]
    obs = sum(R._conc([v for _, v in g]) for g in gs)
    tot, hit = 0, 0
    for combo in itertools.product(*[list(itertools.permutations([v for _, v in g])) for g in gs]):
        tot += 1
        hit += sum(R._conc(list(p)) for p in combo) >= obs - 1e-12
    return hit / tot


def test_stratified_trend_exact_matches_brute_force():
    rng = np.random.default_rng(3)
    blocks = ["a", "a", "a", "b", "b", "c", "c", "d", "d", "d"]
    types = ["pair", "L3", "square4", "pair", "square4", "L3", "square4", "pair", "L3", "square4"]
    vals = rng.normal(size=10)
    r = R.stratified_trend_test(blocks, types, vals)
    assert abs(r["p"] - _brute_trend(blocks, types, vals, D.MULTI)) < 1e-12
    vals = [0.3, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]   # perfectly ordered
    r = R.stratified_trend_test(blocks, types, vals)
    assert r["T"] == r["T_max"] and r["p"] == pytest.approx(1 / (6 * 2 * 2 * 6))


def test_stratified_trend_valid_under_null():
    rng = np.random.default_rng(0)
    rows = [(f"{m}:{w}", t) for t in D.MULTI for (m, w) in D.mw_conditions() if m in D.TYPE_MODULES[t]]
    rej = np.mean([R.stratified_trend_test([b for b, _ in rows], [t for _, t in rows], rng.normal(size=len(rows)))["p"]
                   < 0.05 for _ in range(400)])
    assert rej <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 400)


def test_equivalence_bootstrap():
    rng = np.random.default_rng(0)
    assert R.equivalence_bootstrap(rng.normal(0, 0.01, 12), 0.05)["pass"]
    assert not R.equivalence_bootstrap(rng.normal(0.06, 0.01, 12), 0.05)["pass"]
    assert R.equivalence_bootstrap([0.01], 0.05)["status"] == "TOO_FEW"


def test_worst_direction_loss():
    h = np.full(14, 0.97)
    f = h.copy()
    f[3] = 0.5
    f[5] = np.nan
    assert R.worst_direction_loss(h, f) == pytest.approx(0.47)


# --------------------------------------------------------------------------------------------------------- R3
def test_classcond_threshold_finite_rule_and_validity():
    assert R.classcond_threshold(np.arange(8.0), 0.10) == math.inf           # n0 = 8 < 9
    assert np.isfinite(R.classcond_threshold(np.arange(9.0), 0.10))
    assert R.classcond_threshold(np.arange(18.0), 0.05) == math.inf
    rng = np.random.default_rng(1)
    fg = []
    for _ in range(3000):
        s = rng.normal(size=31)
        lam = R.classcond_threshold(s[:30], 0.10)
        fg.append(s[30] > lam)
    assert np.mean(fg) <= 0.10 + 3 * math.sqrt(0.09 / 3000)


def test_nhg_validity_p_is_a_distribution_and_matches_mc():
    n0, n1, a = 40, 30, 0.10
    probs = [R.nhg_validity_p(n0, n1, k, a) - (R.nhg_validity_p(n0, n1, k + 1, a) if k < n1 else 0) for k in range(n1 + 1)]
    assert R.nhg_validity_p(n0, n1, 0, a) == pytest.approx(1.0)
    assert min(probs) >= -1e-12
    rng = np.random.default_rng(2)
    ks = []
    for _ in range(4000):
        s = rng.normal(size=n0 + n1)
        lam = R.classcond_threshold(s[:n0], a)
        ks.append(int((s[n0:] > lam).sum()))
    ks = np.array(ks)
    for k in (3, 5, 7):
        assert abs(R.nhg_validity_p(n0, n1, k, a) - (ks >= k).mean()) < 0.03


def test_r3_verdict_reading():
    assert R.r3_verdict(0, 0, 0.1).startswith("NOT_EVALUABLE")
    assert R.r3_verdict(20, 46, 0.1) == "REFUTED"
    assert R.r3_verdict(2, 46, 0.1) == "NOT_REFUTED_STRONG"
    assert R.r3_verdict(6, 46, 0.1) == "NOT_REFUTED"


# ------------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return build_v2(tmp_path_factory.mktemp("rv2"))


def test_run_realv2_end_to_end(synth):
    r = R.run_realv2(synth)
    assert r["status"] == R.STATUS_OK and r["provenance"]["status"] == "PASS"
    assert r["R1"]["verdict"] == "SUPPORTED" and r["R1"]["trend"]["n_blocks"] == 16
    assert r["R2"]["verdict"] == "SUPPORTED" and r["R2"]["n_masked"] == 16
    assert r["R2_tau_secondary"]["verdict"] == "NOT_SUPPORTED"               # lone strafe 0.65 > 0.5
    assert r["R3"]["verdict"].startswith("NOT_REFUTED") and r["R3"]["n_null_cal"] >= 9
    assert r["R3"]["vs_carry_over"]["carry_over_false_go"]["est"] == 1.0
    assert r["R3_tau_secondary"]["verdict"] == "NOT_EVALUABLE_TOO_FEW_CAL_NULLS"
    assert r["R4"]["verdict"] == "SUPPORTED" and r["R4"]["dispatch"]["cert"]["unsafe_dispatch"]["est"] == 0.0
    assert json.loads((synth.parent / "out" / "results_v2.json").read_text())["status"] == R.STATUS_OK


def test_r3_refutes_a_broken_certificate(tmp_path):
    r = R.run_realv2(build_v2(tmp_path, cert_fg=0.5))
    assert r["R3"]["verdict"] == "REFUTED"


def test_r2_fails_without_reveal(tmp_path):
    r = R.run_realv2(build_v2(tmp_path, lone_strafe=0.92))
    assert r["R2"]["verdict"] == "NOT_SUPPORTED"


def test_tau_gate_blocks_r2_r3(tmp_path):
    mp = build_v2(tmp_path)
    m = pd.read_csv(tmp_path / "mask.csv")
    sel = m["build"].str.startswith("single_") & m["module"].isna()
    # v4 (D17, relative capability): the gate is the ABSOLUTE healthy-tracking floor (0.75); 0.85 now passes
    m.loc[sel, ["rho_P2+", "rho_P2-"]] = 0.70                                  # healthy lone strafe below the floor
    m.to_csv(tmp_path / "mask.csv", index=False)
    r = R.run_realv2(mp)
    assert r["R2"]["verdict"] == "NOT_EVALUABLE_TAU_GATE" and r["R3"]["verdict"] == "NOT_EVALUABLE_TAU_GATE"


def test_provenance_gate_failures(tmp_path):
    mp = build_v2(tmp_path, source="SYNTHETIC")
    r = R.run_realv2(mp)
    assert r["status"] == R.STATUS_BAD and any("non-FIELD" in f for f in r["provenance"]["failures"])
    mp = build_v2(tmp_path / "b")
    (tmp_path / "b" / "pred.csv").write_text((tmp_path / "b" / "pred.csv").read_text() + "\n")    # tamper
    man = json.loads(mp.read_text())
    man["seed_hex"] = (b"\x00" * 32).hex()
    mp.write_text(json.dumps(man))
    f = R.run_realv2(mp)["provenance"]["failures"]
    assert any("hash mismatch" in x for x in f) and any("commitment" in x for x in f)
    freeze = json.loads((tmp_path / "b" / "freeze_v2.json").read_text())
    trials = pd.DataFrame(dict(source=["FIELD"], session_status=["FIELD_OK"], start_utc=["2026-11-01T00:00:00Z"],
                               echo_ok=[False]))
    g = R.provenance_gate(dict(freeze, files={}), tmp_path, trials)
    assert any("before the freeze" in x for x in g["failures"]) and any("echo" in x for x in g["failures"])
    assert R.provenance_gate(None, tmp_path)["status"] == "FAIL"


def _ep(**kw):
    r = dict(episode=0, faulted_modules="M2|M4", carry_over_lb=0.95, nearest_goal_module="M2", random_module="M4",
             cert_tie_coin="M2", oracle_tie_coin="M2", carry_over_tie_coin="M2", excluded=False)
    for m, (y, sc, z) in dict(M2=(0.0, 0.3, 0.6), M3=(1.0, 0.9, 0.97), M4=(0.0, 0.35, 0.62)).items():
        r.update({f"y_{m}": y, f"s_{m}": sc, f"z_alone_{m}": z})
    r.update(kw)
    return r


def test_r4_choice_rules():
    c = R.demo_choices(_ep(), lam=0.5, tau=0.8)
    assert c == dict(cert="M3", carry_over="M2", random="M4", oracle="M3", cert_ranking="M3")
    c = R.demo_choices(_ep(carry_over_lb=0.5), lam=0.95, tau=0.8)          # nothing certified; body not certified
    assert c["cert"] is None and c["carry_over"] is None and c["cert_ranking"] == "M3"
    c = R.demo_choices(_ep(z_alone_M3=0.7), lam=0.5, tau=0.8)               # no capable module -> oracle abstains
    assert c["oracle"] is None


def test_r4_paired_dedup_and_abstain_as_failure():
    eps = pd.DataFrame([_ep(episode=i) for i in range(12)])
    r = R.r4(eps, lam=0.5, cfg=R.DEFAULTS)
    assert r["n_episodes"] == 12 and r["docks_per_episode"]["mean"] == 3
    assert r["success"]["cert"]["est"] == 1.0 and r["success"]["carry_over"]["est"] == 0.0
    assert r["verdict"] == "SUPPORTED" and r["dispatch"]["carry_over"]["unsafe_dispatch"]["est"] == 1.0
    bad = pd.DataFrame([_ep(y_M2=float("nan"))])                           # a chosen module was not docked
    assert R.r4(bad, lam=0.5, cfg=R.DEFAULTS)["invalid_episodes"] == [0]
    ok = pd.DataFrame([_ep(y_M2=float("nan"), nearest_goal_module="M3", random_module="M3")] * 3)
    assert R.r4(ok, lam=0.5, cfg=R.DEFAULTS)["docks_per_episode"]["mean"] == 1   # deduplicated: only M3 docked
    r = R.r4(pd.DataFrame([_ep(episode=i) for i in range(12)]), lam=0.99, cfg=R.DEFAULTS)   # cert abstains
    assert r["success"]["cert"]["est"] == 0.0 and r["success"]["cert_ranking"]["est"] == 1.0


def test_design_power_functions():
    p, crit = D.power_r2(16, 0.85)
    assert crit == 8 and p > 0.99
    nr, st = D.power_r3(46, 0.30)
    assert nr < 0.1
    both, c1, c2 = D.power_r4(20, n_rep=300)
    assert both > 0.9 and c1 > 0.9 and c2 > 0.9
    ed = D.expected_docks(n_rep=4000)
    assert 1.5 < ed["mean"] < 2.3 and ed["max"] == 3
    pc, pf, q = D._choice_model()
    assert q["cert"] == 1.0 and 0.3 < q["random"] < 0.5


def test_paper_results_map_resolves(synth):
    from pathlib import Path
    r = R.run_realv2(synth)
    rm = Path(__file__).resolve().parents[1] / "paper" / "results_map.json"
    if not rm.exists():
        pytest.skip("no paper map")
    missing = []
    for name, spec in json.loads(rm.read_text())["map"].items():
        node = r
        for part in spec["path"].split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            missing.append(spec["path"])
    assert not missing, missing
    assert set(r["docking"]["certificate"]) == {"successes", "runs"}
