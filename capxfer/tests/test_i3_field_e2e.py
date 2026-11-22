"""WS-I3F end-to-end on a synthetic SIMULATION field day (CPU, no ROS, ~3-4 min):
sealed MASK + DEMO sheets (one PROTOCOL_V2 seed) -> FieldLog SIMULATION sessions of a MASK subset and two DEMO episodes
-> builders (mask.csv, r3.csv CAL-only, lambda_v2.json, r3.csv full, carry-over, demo.csv) -> realv2.run_realv2 ->
R1-R4 verdict JSON; and the frozen REAL-model bundle loaded offline on CPU by choose_module on a DEMO L3 probe."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capxfer.analysis import realv2 as RV
from capxfer.field import builders as BL
from capxfer.field import bundle as BU
from capxfer.field import choose_module as CM
from capxfer.field import schedule as SCH
from capxfer.field import synth_session as SS

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "capxfer" / "field_packet" / "tools" / "make_run_sheets.py"
SEED = "5e" * 32
WS_CACHE = Path.home() / ".cache" / "capxfer" / "I3F" / "wheelsets_W_P1P5P2.npz"


def _sheets(kind, out, seed_file):
    r = subprocess.run([sys.executable, str(TOOL), "extra", kind, str(out), "--seed-file", str(seed_file)],
                       capture_output=True, text=True, cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-2000:]
    return SCH.load(Path(out) / "schedule.json")


def _by_layout(sched, nums):
    """Group schedule trial numbers by their schedule session entry (one synthetic FieldLog session per layout)."""
    ent = {}
    for s, _, t in SCH.iter_trials(sched):
        if int(t["trial"]) in nums:
            ent.setdefault(id(s), []).append(int(t["trial"]))
    return list(ent.values())


@pytest.fixture(scope="module")
def fieldday(tmp_path_factory):
    w = tmp_path_factory.mktemp("i3f")
    (w / "sealed").mkdir()
    sf = w / "sealed" / "seed_v2.txt"
    sf.write_text(SEED + "\n")
    mask = _sheets("masking", w / "mask_sheets", sf)
    demo = _sheets("choice", w / "demo_sheets", sf)
    # MASK subset: one CAL wheel of M1 and one TEST wheel of M3, one repeat in every part (square4, pair, lone),
    # plus one healthy trial per build used
    eps = {e["episode"]: e for e in mask["episodes"]}
    split = {f"{t['episode']}": t.get("r3_split") for _, _, t in SCH.iter_trials(mask)}
    pick = [next(k for k in eps if k.startswith("M1:") and split.get(k) == "CAL"),
            next(k for k in eps if k.startswith("M3:") and split.get(k) == "TEST")]
    nums, builds = set(), set()
    for k in pick:
        for b, tl in eps[k]["parts"].items():
            if b in ("square4", "pair", "single_M1", "single_M3"):
                nums.add(int(tl[0]))
                builds.add(b)
    for b in builds:
        nums.add(next(int(t["trial"]) for _, _, t in SCH.iter_trials(mask) if t.get("episode") == f"H_{b}"))
    mask_sessions = []
    for i, tn in enumerate(_by_layout(mask, nums)):
        d, _ = SS.make_session(w / "mask_sessions", n_trials=len(tn), seed=100 + i, schedule=mask, trial_numbers=tn,
                               codebook_dir=w / "_cb", device="cpu")
        mask_sessions.append(d)
    # DEMO (D15: L3, coast on 2 modules): two episodes (L3 probe + the three lone probes), each its own session
    demo_sessions = []
    for ep in demo["choice"]["episodes"][:2]:
        groups = [[ep["assembly_trial"]]] + [[ep["lone_trials"][m]] for m in ep["dock_order"]]
        for j, tn in enumerate(groups):
            d, _ = SS.make_session(w / "demo_sessions", n_trials=1, seed=200 + 10 * ep["choice_block"] + j, schedule=demo,
                                   trial_numbers=tn, codebook_dir=w / "_cb", device="cpu")
            demo_sessions.append((ep, j, d))
    stub = CM.synthetic_bundle(w / "stub")
    return dict(w=w, mask=mask, demo=demo, mask_sessions=mask_sessions, demo_sessions=demo_sessions, stub=stub,
                sheets=w / "demo_sheets", mask_sheets=w / "mask_sheets", picks=pick)


def _pred(mask_csv, out):
    """Stand-in for the frozen prediction file (integrator's generator): L_pred = 0.03, z_pred from the lone truth."""
    t = pd.read_csv(mask_csv)
    rows = []
    for (b, m, wh), g in t.groupby(["build", t["module"].fillna(""), t["wheel"].fillna("")]):
        rows.append(dict(build=b, module=m, wheel=wh, L_pred=0.03, z_pred=float(g["z"].mean()), k_star="P2+"))
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_builders_through_realv2(fieldday):
    f, w = fieldday, fieldday["w"]
    out = w / "analysis"
    mask_csv = BL.build_mask(out / "mask.csv", f["mask_sessions"], f["mask_sheets"] / "schedule.json",
                             allow_simulation=True)
    m = pd.read_csv(mask_csv)
    assert set(RV.MASK_COLS) <= set(m.columns) and set(RV.RHO_COLS) <= set(m.columns)
    assert (m["source"] == "SIMULATION").all() and not m["excluded"].any()
    assert set(m["build"]) == {"square4", "pair", "single_M1", "single_M3"}
    assert set(m["r3_split"]) == {"CAL", "TEST", "NONE"} and m["start_utc"].astype(bool).all()
    healthy_lone = m[(m["build"] == "single_M1") & (m["firmware"] == "ok")]
    # sim healthy lone module strafes. WS-MR (D17): the measured robot's confirmed firmware ends the holds at ~0.8-0.9 of
    # command, so tau 0.8 applies to RELATIVE capability (rho / the build's healthy MASK median, realv2.to_relative):
    # the healthy trial is ~1 relative (gate 1.0 +- 0.06) and its absolute z is the measured-robot level (> 0.7)
    rel = RV.to_relative(m)
    hl_rel = rel[(rel["build"] == "single_M1") & (rel["firmware"] == "ok")]
    assert abs(float(hl_rel[list(RV.STRAFE)].min(axis=1).iloc[0]) - 1) < 0.06
    assert float(healthy_lone["z"].iloc[0]) > 0.7
    # between MASK and DEMO: CAL units only (PROTOCOL_V2 §9)
    r3_cal = BL.build_r3(out / "r3_cal.csv", f["mask_sessions"], f["mask_sheets"] / "schedule.json", f["stub"],
                         splits=("CAL",), allow_simulation=True)
    rc = pd.read_csv(r3_cal)
    assert set(rc["r3_split"]) == {"CAL"} and set(RV.R3_COLS) <= set(rc.columns)
    lam = BL.build_lambda(out / "lambda_v2.json", r3_cal, mask_csv, f["stub"], status="SYNTHETIC")
    rec = BU.read_lambda(lam)
    assert rec["status"] == "SYNTHETIC" and rec["lambda_hat"] == math.inf      # n0 < 9: nothing certified (by rule)
    assert rec["carry_over_offset"] == math.inf and rec["raw"]["n_carry_cal"] < 9   # too few CAL trials: never go
    assert rec["bundle_sha256"] == BU.sha256_file(Path(f["stub"]) / "bundle.yaml")
    # after collection: all units + carry-over
    r3 = BL.build_r3(out / "r3.csv", f["mask_sessions"], f["mask_sheets"] / "schedule.json", f["stub"],
                     allow_simulation=True)
    BL.apply_carry_over(r3, lam)
    r = pd.read_csv(r3)
    assert set(r["r3_split"]) == {"CAL", "TEST"} and (r["carry_over_lb"] == -math.inf).all()
    sq = r[r["assembly_type"] == "square4"]
    assert sq.groupby("trial_id")["unit_module"].nunique().eq(4).all()
    # the stub certificate (echo posterior) scores the faulted module lower than the others
    for tid, g in r[r["fault_mw"] != "none"].groupby("trial_id"):
        fm = g["fault_mw"].iloc[0].split(":")[0]
        assert g.set_index("unit_module")["score"][fm] < g["score"].max()
    # DEMO: choose_module on each L3 probe (stub bundle + the lambda file), then the operator's dock sheet
    log = w / "choices" / "choices.csv"
    docks = {}
    for ep, j, d in f["demo_sessions"]:
        if j == 0:
            CM.choose(d, ep["choice_block"], f["sheets"], f["stub"], lam, log=log)
            lone = [d_ for e_, j_, d_ in f["demo_sessions"] if e_["episode"] == ep["episode"] and j_ > 0]
            docks[ep["episode"]] = CM.finalize(lone, ep["choice_block"], f["sheets"], log)["docks"]
    runs = w / "manual" / "choice_runs.csv"
    runs.parent.mkdir()
    with open(runs, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["fieldlog_dir", "episode", "dock_in_episode", "module", "run_global", "start_pose_id", "t_start_local",
                     "executor_status", "operator_outcome", "contact_seen", "fault_in_place_before", "fault_in_place_after",
                     "choice_logged", "invalid_reason", "repeat_of", "v_batt_min_v", "notes"])
        g = 0
        for ep in f["demo"]["choice"]["episodes"][:2]:
            for k, mod in enumerate(docks[ep["episode"]], start=1):
                g += 1
                ok = mod not in ep["faulted_modules"]
                wr.writerow(["x", ep["episode"], k, mod, g, 3, "10:00:00", "done" if ok else "infeasible",
                             "success" if ok else "collision", 0 if ok else 1, 1, 1, 1, "none", "", 12.0, ""])
    demo_csv = BL.build_demo(out / "demo.csv", f["sheets"], log, runs, [d for _, _, d in f["demo_sessions"]],
                             allow_simulation=True)
    dm = pd.read_csv(demo_csv)
    assert set(RV.DEMO_COLS) <= set(dm.columns) and len(dm) == 20
    done = dm[~dm["excluded"].astype(bool)]
    mods = ("M2", "M3", "M4")
    assert len(done) == 2 and np.isfinite(done[[f"{c}_{m}" for c in ("s", "z_alone") for m in mods]].to_numpy()).all()
    assert (done["carry_over_lb"] == -math.inf).all() and set(done["episode"]) == {0, 1}
    assert (dm[dm["excluded"].astype(bool)]["exclusion"].str.contains("no_choice_logged")).all()
    for _, row in done.iterrows():           # healthy lone module strafes; coast ones lose strafe (P(z<0.8)=0.85 in sim)
        fm = row["faulted_modules"].split("|")
        cap = next(m for m in mods if m not in fm)
        # WS-MR (D17): build_demo's z_alone is ABSOLUTE; healthy measured-robot level ~0.77-0.9 (relative ~1) -> > 0.7
        assert row[f"z_alone_{cap}"] > 0.7 and min(row[f"z_alone_{m}"] for m in fm) < row[f"z_alone_{cap}"]
        assert all(np.isfinite(row[f"y_{m}"]) == (m in row["dock_list"].split("+")) for m in mods)
    pred = _pred(mask_csv, out / "pred.csv")
    man = BL.write_manifest(out / "manifest.yaml", mask_csv, pred, r3, demo_csv, lam, output_dir="res")
    res = RV.run_realv2(man)
    assert res["status"] == RV.STATUS_BAD                             # SIMULATION data: provenance gate fails, as it must
    assert any("non-FIELD" in x for x in res["provenance"]["failures"]) or res["provenance"]["status"] == "FAIL"
    for k in ("R1", "R2", "R3", "R4"):
        assert isinstance(res[k]["verdict"], str), k
    assert res["R3"]["verdict"] == "NOT_EVALUABLE_TOO_FEW_CAL_NULLS"
    assert res["R4"]["n_episodes"] == 2 and not res["R4"]["invalid_episodes"]
    for e in res["R4"]["episodes"]:                                  # echo-stub certificate ranks the capable module first
        assert e["choice_cert_ranking"] not in e["faulted"] and e["choice_cert"] is None   # lambda = inf: abstains
    assert res["lambda_v2_check"]["status"] == "MATCH"
    assert json.loads((out / "res" / "results_v2.json").read_text())["R4"]["verdict"]


def test_real_bundle_offline_cpu(fieldday, tmp_path, monkeypatch):
    """The frozen REAL-model bundle (W structured_ensemble + ensemble) built by bundle.py, verified, then loaded by
    choose_module on CPU on a SIMULATION DEMO L3 probe; a PLACEHOLDER lambda is refused on a non-SIMULATION session."""
    if not (BU.MODELS_W / "structured_ensemble.pt").is_file() or not BU.SPLIT_LIBRARY.is_file():
        pytest.skip("W models / SPLIT library not in ~/.cache")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    f = fieldday
    if WS_CACHE.is_file():
        ws = WS_CACHE
    else:
        rng = np.random.default_rng(0)
        ws = tmp_path / "ws.npz"
        np.savez(ws, s_ok=rng.uniform(0, 0.3, 1000), s_f=rng.uniform(0, 0.6, 1000))
    bd = BU.build_field_bundle(tmp_path / "bundle", wheelsets=ws, bb_offset=-0.08, note="test")
    v = BU.verify_bundle(bd, bd / "lambda_v2.PLACEHOLDER.json")
    assert v["ok"], v
    b = CM.load_bundle(bd)
    assert not b["synthetic"] and b["provenance"]["certificate_code_version"] == v["code_version"]
    assert b["calibration"]["status"] == "PLACEHOLDER" and (bd / "MANIFEST.sha256").is_file()
    ep, _, d = next(x for x in f["demo_sessions"] if x[1] == 0)
    res = CM.choose(d, ep["choice_block"], f["sheets"], bd, bd / "lambda_v2.PLACEHOLDER.json",
                    log=tmp_path / "choices.csv", device="cpu")
    assert set(res["scores"]) == set(ep["modules"]) and len(res["rows"]) == 6 and res["random"] in ep["modules"]
    assert all(np.isfinite(v_) or v_ == -math.inf for v_ in res["scores"].values())
    # an OPEN session (no session.yaml yet, as on the field) is not SIMULATION: PLACEHOLDER refused
    open_d = tmp_path / "open_session"
    shutil.copytree(d, open_d)
    (open_d / "session.yaml").unlink()
    with pytest.raises(SystemExit, match="PLACEHOLDER"):
        CM.choose(open_d, ep["choice_block"], f["sheets"], bd, bd / "lambda_v2.PLACEHOLDER.json", device="cpu")
    # a lambda file calibrated with another bundle is refused
    other = BU.make_lambda_record(0.3, 0.4, 30, 0.01, 40, status="FINAL", bundle_sha256="0" * 64)
    BU.write_lambda(tmp_path / "lam_other.json", other)
    with pytest.raises(SystemExit, match="another bundle"):
        CM.choose(d, ep["choice_block"], f["sheets"], bd, tmp_path / "lam_other.json", device="cpu")
    # tampering with the frozen bundle is detected
    (bd / "code" / "capxfer" / "probes.py").write_text("# altered\n")
    assert not BU.verify_bundle(bd, load_models=False)["ok"]
