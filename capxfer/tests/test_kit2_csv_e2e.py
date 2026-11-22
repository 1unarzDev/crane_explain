"""WS-KIT2 end-to-end on SIMULATION sessions in the minimal binned format (capxfer/field_kit_v2/CSV_FORMAT.md), CPU
(~1 min): simulator -> the team's 20 Hz pre-processing -> binned.csv on disk -> csv_adapter -> builders (mask, r3
CAL-only, lambda, r3, carry-over, offline DEMO choices, demo.csv with all three docks) -> predict.write_pred ->
realv2.run_realv2 -> R1-R4 verdict JSON, including the posterior-ranking arm (PROTOCOL_V2 change log v3)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capxfer import assembly as A
from capxfer.analysis import realv2 as RV
from capxfer.analysis import realv2_design as D
from capxfer.featurize import featurize
from capxfer.field import builders as BL
from capxfer.field import choose_module as CM
from capxfer.field import csv_adapter as CA
from capxfer.field import predict as PR

SEED = bytes.fromhex("5e" * 32)                 # PUBLIC test seed (never the sealed one)


def _spec(build):
    return [(build.split("_", 1)[1], 0, 0, 0)] if build.startswith("single_") else list(A.PHYSICAL_TYPES[build])


def _fault_str(f):
    return ";".join(f"{m}:{w}:{md}" for (m, w), (md, _) in sorted(f.items())) or "ok"


def _session(out, name, rows, seed, docks=None):
    """rows: [(trial dict without data, spec, faults {(m, w): (mode, g)})] -> one SIMULATION binned session."""
    trials, binned = [], {}
    by = {}
    for k, (t, spec, f) in enumerate(rows):
        by.setdefault(CA.layout_str(spec), []).append((k, t, spec, f))
    for j, (lay, items) in enumerate(by.items()):
        spec = items[0][2]
        asm = A.build(CA.asm_name(spec), spec, A.load_params(CA.PARAMS_PATH))
        modes = np.zeros((len(items), asm.n), np.int64)
        gs = np.ones((len(items), asm.n), np.float32)
        for i, (_, _, _, f) in enumerate(items):
            modes[i], gs[i], _ = CA.fault_arrays(f, asm)
        _, bn, _ = CA.simulate_trials(spec, modes, gs, seed=seed + j)
        for i, (k, t, _, _) in enumerate(items):
            binned[t["trial_id"]] = bn[i]
    trials = [t for t, _, _ in rows]
    return CA.write_binned_session(out, name, trials, binned, docks=docks)


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    w = tmp_path_factory.mktemp("kit2")
    sched = D.expand_schedule(SEED)
    mt = sched["mask_trials"]
    pick = lambda pred: next(t for t in mt if pred(t))                                        # noqa: E731
    cal = pick(lambda t: t["build"] == "pair" and t["module"] == "M3" and t["r3_split"] == "CAL")
    test = pick(lambda t: t["build"] == "pair" and t["module"] == "M1" and t["r3_split"] == "TEST")
    sel = [pick(lambda t: t["build"] == "pair" and t["module"] is None), cal, test]
    for m, c in (("M1", test), ("M3", cal)):
        sel.append(pick(lambda t, m=m: t["build"] == f"single_{m}" and t["module"] is None))
        sel.append(pick(lambda t, m=m, c=c: t["build"] == f"single_{m}" and t["module"] == m and t["wheel"] == c["wheel"]))
    for m in ("M2", "M4"):          # v4: the DEMO solo probes are relative to every L3 module's singles baseline
        sel.append(pick(lambda t, m=m: t["build"] == f"single_{m}" and t["module"] is None))
    rows = []
    for k, t in enumerate(sel):
        spec = _spec(t["build"])
        f = {(t["module"], t["wheel"]): (t["firmware"], 1.0)} if t["module"] else {}
        rows.append((dict(trial_id=f"MASK_T{k:03d}", seq=t["seq"], block="MASK", layout=CA.layout_str(spec),
                          fault=_fault_str(f), imu_module=spec[0][0], truth_tag=spec[0][0], aborted=0), spec, f))
    ms = _session(w, "SIMULATION_MASK", rows, seed=11)
    ep = sched["demo"][0]
    fw = {(m, wh): ("coast", 1.0) for m, wh in ep["faulted_wheels"].items()}
    spec = list(A.PHYSICAL_TYPES["L3"])
    rows = [(dict(trial_id="DEMO_T000", seq=0, block="DEMO_PROBE", layout=CA.layout_str(spec), fault=_fault_str(fw),
                  imu_module="M2", truth_tag="M2", aborted=0), spec, fw)]
    for k, m in enumerate(ep["dock_order"]):
        f = {kk: v for kk, v in fw.items() if kk[0] == m}
        rows.append((dict(trial_id=f"DEMO_T{k + 1:03d}", seq=0, block="DEMO_SOLO", layout=f"{m}@0,0,0",
                          fault=_fault_str(f), imu_module=m, truth_tag=m, aborted=0), [(m, 0, 0, 0)], f))
    docks = [dict(episode=0, module=m, dock_order=k + 1, success=m not in ep["faulted_wheels"])
             for k, m in enumerate(ep["dock_order"])]
    ds = _session(w, "SIMULATION_DEMO", rows, seed=21, docks=docks)
    return dict(w=w, sched=sched, ms=ms, ds=ds, ep=ep)


def test_verify_load_and_labels(day):
    assert CA.verify_binned_session(day["ms"])["verdict"] == "SIMULATION"
    with pytest.raises(CA.SessionRefused):
        CA.load_binned_session(day["ms"], day["sched"])                    # SIMULATION refused by default
    ld = CA.load_binned_sessions([day["ms"]], day["sched"], allow_simulation=True)
    assert sum(len(r["meta"]["trial_ids"]) for _, r in ld) == 9
    for _, r in ld:
        assert all(r["meta"]["labels"]["schedule_match"]) and not r["meta"]["qc"]
        assert all(g <= 0.1 and min(c.values()) > 0.95 for g, c in r["meta"]["csv_checks"].values())
    # a declared fault contradicting the sealed schedule is flagged (exclusion 7 (b)), never silently relabelled
    import shutil
    bad = day["w"] / "SIMULATION_bad"
    shutil.copytree(day["ms"], bad)
    tr = pd.read_csv(bad / "trials.csv", dtype=str, keep_default_na=False)
    tr.loc[1, "fault"] = "ok"
    tr.to_csv(bad / "trials.csv", index=False)
    lb = CA.load_binned_sessions([bad], day["sched"], allow_simulation=True)
    assert [ok for _, r in lb for ok in r["meta"]["labels"]["schedule_match"]].count(False) == 1


def test_binned_reconstruction_is_exact_for_featurize():
    """featurize on the reconstructed RawTrial returns the team's bin values exactly (speed, duty, gz)."""
    spec = list(A.PHYSICAL_TYPES["pair"])
    m = np.zeros((2, 8), np.int64)
    m[1, 3] = 2
    raw, bn, asm = CA.simulate_trials(spec, m, np.ones((2, 8), np.float32), seed=5)
    import tempfile
    s = CA.write_binned_session(tempfile.mkdtemp(), "SIMULATION_x",
                                [dict(trial_id=f"T{b}", seq=0, block="SYSID", layout=CA.layout_str(spec), fault="ok",
                                      imu_module="M1", truth_tag="M1", aborted=0) for b in range(2)],
                                {f"T{b}": bn[b] for b in range(2)})
    rb = CA.load_binned_session(s, None, allow_unverified=True)[0]
    F = featurize(rb, rung="E1", pids=("P1", "P5", "P2"))
    seg = [x for x in rb["meta"]["schedule"].segments if x.pid in ("P1", "P2", "P5")]
    k0 = int(round(seg[0].t0 / 0.05))                                       # first featurize bin = team bin k0
    assert np.allclose(F["wheel_x"][0, 0, 0, 1].item(), bn[0]["M1_fl_speed"][k0], atol=1e-3)
    assert np.allclose(F["wheel_x"][0, 0, 0, 2].item(), bn[0]["M1_fl_duty"][k0], atol=1e-5)
    assert np.allclose(F["body_x"][0, 0, 3].item(), bn[0]["gz"][k0], atol=1e-5)
    full = featurize(raw, rung="E1", pids=("P1", "P5", "P2"))
    assert float((full["y"] - F["y"]).abs().max()) < 0.03                  # targets agree within noise


def test_pipeline_to_realv2(day):
    w, out = day["w"], day["w"] / "analysis"
    lm = CA.load_binned_sessions([day["ms"]], day["sched"], blocks=("MASK",), allow_simulation=True)
    ldm = CA.load_binned_sessions([day["ds"]], day["sched"], blocks=("DEMO_PROBE", "DEMO_SOLO"), allow_simulation=True)
    mask = BL.build_mask(out / "mask.csv", [], None, allow_simulation=True, loaded=lm)
    m = pd.read_csv(mask)
    cond = RV.condition_table(m)                                           # v4: RELATIVE capability (D17)
    z = {(r.build, r.mw): r.z for r in cond.itertuples()}
    zc = {(b, mw.split(":")[0]): v for (b, mw), v in z.items()}
    assert zc[("pair", "M3")] > 0.85 and zc[("single_M3", "M3")] < 0.8    # masked in the pair, revealed alone
    assert set(m["r3_split"]) == {"CAL", "TEST", "NONE"} and not m["excluded"].any()
    stub = CM.synthetic_bundle(w / "stub")
    post = BL.synthetic_posterior_table(w / "posterior.npz", stub)
    r3c = BL.build_r3(out / "r3_cal.csv", [], None, stub, splits=("CAL",), allow_simulation=True, loaded=lm,
                      posterior=post)
    assert set(pd.read_csv(r3c)["r3_split"]) == {"CAL"}
    lam = BL.build_lambda(out / "lambda_v2.json", r3c, mask, stub, status="SYNTHETIC")
    r3 = BL.build_r3(out / "r3.csv", [], None, stub, allow_simulation=True, loaded=lm, posterior=post)
    BL.apply_carry_over(r3, lam)
    u = pd.read_csv(r3)
    assert np.isfinite(u["pm"]).all() and np.isfinite(u["u_q05"]).all()
    ch = BL.build_demo_choices(out / "demo_choices.csv", ldm, day["sched"], stub, lam, post,
                               mask_csv=mask)                                                     # BEFORE the docks
    assert Path(str(ch) + ".sha256").is_file()
    c = pd.read_csv(ch, keep_default_na=False)
    e0 = c[c["episode"] == 0].iloc[0]
    capable = next(mm for mm in D.DEMO_MODULES if mm not in day["ep"]["faulted_wheels"])
    assert e0["probe_exclusion"] == "" and e0["choice_oracle"] == capable and e0["choice_post_rank"] == capable
    # v4 (D17): z_alone is RELATIVE to the module's MASK singles healthy baseline (absolute kept alongside)
    assert e0["capability"] == "relative" and float(e0[f"z_alone_{capable}"]) > 0.9
    assert float(e0[f"z_alone_abs_{capable}"]) > 0.0
    demo = BL.build_demo_v3(out / "demo.csv", ch, CA.dock_outcomes(day["ds"] / "docks.csv"))
    dm = pd.read_csv(demo)
    assert set(RV.DEMO_COLS) <= set(dm.columns) and (dm["dock_design"] == "all3").all() and len(dm) == 20
    assert (~dm["excluded"].astype(bool)).sum() == 1
    conds = sorted({(r.build, r.module, r.wheel, r.firmware) for r in m.itertuples() if isinstance(r.module, str)})
    pred = PR.write_pred(out / "pred.csv", conds=conds, n_fault=1, n_healthy=2, qs=False)
    man = BL.write_manifest(out / "manifest.yaml", mask, pred, r3, demo, lam, output_dir="res",
                            extra={"predicted_rho": str(out / "pred_rho.csv")})
    res = RV.run_realv2(man)
    assert res["status"] == RV.STATUS_BAD                     # SIMULATION: the provenance gate fails, as it must
    for k in ("R1", "R2", "R3", "R4"):
        assert isinstance(res[k]["verdict"], str), k
    r4 = res["R4"]
    assert r4["dock_rule"] == "all3" and r4["n_episodes"] == 1 and r4["docked_per_episode"] == 3
    e = r4["episodes"][0]
    assert e["choice_post_rank"] == capable and e["y_post_rank"] == 1.0 and e["choice_oracle"] == capable
    assert set(r4["secondary_contrasts"]) == {"post_rank_vs_carry_over", "cert_vs_post_rank"}
    assert "vs_uncalibrated_q05" in res["R3"]
    assert json.loads((out / "res" / "results_v2.json").read_text())["R4"]["dock_rule"] == "all3"


def test_all3_rule_is_blind_to_choices():
    """v3: with dock_design all3 an episode missing ANY dock is invalid, whichever module the arms chose."""
    base = dict(episode=0, faulted_modules="M2|M4", carry_over_lb=0.95, nearest_goal_module="M4", random_module="M3",
                cert_tie_coin="M2", oracle_tie_coin="M2", carry_over_tie_coin="M2", excluded=False, dock_design="all3")
    for mm, (y, s, z, pm) in dict(M2=(0.0, 0.3, 0.6, 0.6), M3=(1.0, 0.9, 0.97, 0.95), M4=(math.nan, 0.35, 0.62, 0.6)).items():
        base.update({f"y_{mm}": y, f"s_{mm}": s, f"z_alone_{mm}": z, f"pm_{mm}": pm})
    r = RV.r4(pd.DataFrame([base]), lam=0.5, cfg=RV.DEFAULTS)
    assert r["invalid_episodes"] == [0]                       # M4 undocked; the carry-over arm chose M4 anyway
    base["y_M4"] = 0.0
    r = RV.r4(pd.DataFrame([dict(base, episode=i) for i in range(12)]), lam=0.5, cfg=RV.DEFAULTS)
    assert r["n_episodes"] == 12 and r["success"]["post_rank"]["est"] == 1.0
    assert r["secondary_contrasts"]["post_rank_vs_carry_over"]["n10"] == 12


def test_hours_and_power_under_d16():
    h = D.hours_table()
    assert h["demo_expected_docks"] == 3.0 and h["rows"]["demo"] == (4.9, 7.33)
    assert D.simulate_demo(5, np.random.default_rng(0))["docked"].tolist() == [3] * 5


def test_relative_capability_d17():
    """v4: rho / the build's healthy median; tau applies to it; the G-tau floor is on absolute healthy strafe."""
    keys = list(RV.RHO_COLS)
    rows = []
    for k in range(4):
        rows.append(dict(trial_id=f"H{k}", build="single_M1", assembly_type="single", module=None, wheel=None,
                         firmware="ok", excluded=False, **{c: 0.85 for c in keys}))
    rows.append(dict(trial_id="F0", build="single_M1", assembly_type="single", module="M1", wheel="fl",
                     firmware="coast", excluded=False, **{c: (0.60 if c in ("rho_P2+", "rho_P2-") else 0.85) for c in keys}))
    t = pd.DataFrame(rows)
    rel = RV.to_relative(t)
    assert np.allclose(rel.loc[rel["trial_id"] == "H0", keys].to_numpy(float), 1.0)
    assert abs(RV.relative_trial_strafe(t)["F0"] - 0.60 / 0.85) < 1e-9
    cond = RV.condition_table(t)
    zf, zh = RV.lone_truth(cond)
    assert abs(zf["M1:fl"] - 0.60 / 0.85) < 1e-9 and abs(zh["M1"] - 1.0) < 1e-9
    assert abs(RV.condition_table(t, relative=False).iloc[0]["z"] - 0.60) < 1e-9
    g = RV.tau_gate(zh, 0.8, 0.1, RV.lone_healthy_abs(cond), RV.DEFAULTS["tau_gate_abs_min"])
    assert g["pass_"]                                           # 0.85 absolute tracking passes the 0.75 floor
    t2 = t.copy()
    t2[keys] = t2[keys] * 0.8
    c2 = RV.condition_table(t2)
    assert not RV.tau_gate(RV.lone_truth(c2)[1], 0.8, 0.1, RV.lone_healthy_abs(c2), 0.75)["pass_"]
    pred = pd.DataFrame([dict(build="single_M1", module="", wheel="", L_pred=0.0, z_pred=0.85, k_star="",
                              **{c: 0.85 for c in keys}),
                         dict(build="single_M1", module="M1", wheel="fl", L_pred=0.25, z_pred=0.6, k_star="P2+",
                              **{c: (0.60 if c in ("rho_P2+", "rho_P2-") else 0.85) for c in keys})])
    pr = RV.relative_predictions(pred)
    assert abs(pr.iloc[1]["z_pred"] - 0.60 / 0.85) < 1e-9 and abs(pr.iloc[1]["L_pred"] - (1 - 0.60 / 0.85)) < 1e-9


def test_free_speed_measurements(tmp_path):
    p = tmp_path / "measurements.csv"
    p.write_text("item,module,value,unit,method\n"
                 "free_speed_full_duty_fl,M1,10.0,rad/s,hook\nfree_speed_half_duty_fl,M1,5.1,rad/s,hook\n"
                 "free_speed_full_duty_M2_rr,,0.48,m/s,hook\nfree_speed_duty05_M2_rr,,0.24,m/s,hook\n"
                 "mass,M1,4.5,kg,scale\n")
    r = CA.free_speeds(CA.read_measurements(p), radius=0.048)
    assert r["n_wheels"] == 2 and abs(r["wheels"]["M2:rr"]["full"] - 10.0) < 1e-9
    assert abs(r["no_load_surface_mps"] - 0.48) < 1e-9 and abs(r["ff_ref_over_no_load"] - 0.65 / 0.48) < 1e-9
    assert 0.5 <= r["half_over_full"] <= 0.51 and not r["problems"]
