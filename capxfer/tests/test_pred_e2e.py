"""WS-PRED end-to-end on a synthetic SIMULATION field day (CPU, no ROS):
sealed MASK + DEMO sheets -> SIMULATION FieldLog sessions (a MASK subset + one DEMO episode) -> builders (mask, r3,
lambda, carry, demo) -> predict (pred.csv + pred_rho.csv from the simulator at theta-hat with the sealed modes;
pred_units.csv from robot-visible streams only, checked against r3.csv and choices.csv) -> realv2 -> R1-R4 verdicts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from capxfer.analysis import realv2 as RV
from capxfer.field import builders as BL
from capxfer.field import choose_module as CM
from capxfer.field import predict as PR
from capxfer.field import schedule as SCH
from capxfer.field import synth_session as SS

from test_i3_field_e2e import SEED, _by_layout, _sheets


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    torch.set_num_threads(4)
    w = tmp_path_factory.mktemp("pred")
    (w / "sealed").mkdir()
    sf = w / "sealed" / "seed_v2.txt"
    sf.write_text(SEED + "\n")
    mask = _sheets("masking", w / "mask_sheets", sf)
    demo = _sheets("choice", w / "demo_sheets", sf)
    eps = {e["episode"]: e for e in mask["episodes"]}
    split = {f"{t['episode']}": t.get("r3_split") for _, _, t in SCH.iter_trials(mask)}
    pick = [next(k for k in eps if k.startswith("M1:") and split.get(k) == "CAL"),
            next(k for k in eps if k.startswith("M1:") and split.get(k) == "TEST")]
    nums, builds = set(), set()
    for k in pick:
        for b, tl in eps[k]["parts"].items():
            if b in ("pair", "single_M1"):
                nums.update(int(x) for x in tl[:2])
                builds.add(b)
    for b in builds:
        nums.add(next(int(t["trial"]) for _, _, t in SCH.iter_trials(mask) if t.get("episode") == f"H_{b}"))
    ms = [SS.make_session(w / "mask_sessions", n_trials=len(tn), seed=300 + i, schedule=mask, trial_numbers=tn,
                          codebook_dir=w / "_cb", device="cpu")[0] for i, tn in enumerate(_by_layout(mask, nums))]
    ep = demo["choice"]["episodes"][0]
    ds = []
    for j, tn in enumerate([[ep["assembly_trial"]]] + [[ep["lone_trials"][m]] for m in ep["dock_order"]]):
        ds.append(SS.make_session(w / "demo_sessions", n_trials=1, seed=400 + j, schedule=demo, trial_numbers=tn,
                                  codebook_dir=w / "_cb", device="cpu")[0])
    return dict(w=w, mask=mask, demo=demo, ms=ms, ds=ds, ep=ep, stub=CM.synthetic_bundle(w / "stub"), builds=builds)


def _guard(monkeypatch):
    """Any evaluator-only access during the unit predictions fails the test."""
    def boom(*a, **k):
        raise AssertionError("predict read truth")
    import capxfer.featurize as F
    monkeypatch.setattr(F, "targets", boom)
    monkeypatch.setattr(BL, "truth_rho", boom)
    monkeypatch.setattr(BL, "file_checks", boom)
    orig = CM.Models.predict

    def checked(self, raw):
        assert "truth_pose" not in raw
        assert set(raw["meta"]["labels"]) <= set(PR.SAFE_LABELS) | {"mode_echo"}
        return orig(self, raw)
    monkeypatch.setattr(CM.Models, "predict", checked)


def test_sessions_builders_predict_realv2(day, monkeypatch):
    f, w = day, day["w"]
    out = w / "analysis"
    ms_sched = w / "mask_sheets" / "schedule.json"
    ds_sched = w / "demo_sheets" / "schedule.json"
    # ---- builders (evaluator side): mask, CAL-only r3 -> lambda, full r3 + carry-over
    mask_csv = BL.build_mask(out / "mask.csv", f["ms"], ms_sched, allow_simulation=True)
    r3_cal = BL.build_r3(out / "r3_cal.csv", f["ms"], ms_sched, f["stub"], splits=("CAL",), allow_simulation=True)
    lam = BL.build_lambda(out / "lambda_v2.json", r3_cal, mask_csv, f["stub"], status="SYNTHETIC")
    r3 = BL.build_r3(out / "r3.csv", f["ms"], ms_sched, f["stub"], allow_simulation=True)
    BL.apply_carry_over(r3, lam)
    # ---- DEMO on the field: choose_module + finalize, then the operator's dock sheet -> demo.csv
    log = w / "choices" / "choices.csv"
    ep = f["ep"]
    CM.choose(f["ds"][0], ep["choice_block"], w / "demo_sheets", f["stub"], lam, log=log)
    docks = CM.finalize(f["ds"][1:], ep["choice_block"], w / "demo_sheets", log)["docks"]
    runs = w / "manual" / "choice_runs.csv"
    runs.parent.mkdir(exist_ok=True)
    with open(runs, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["fieldlog_dir", "episode", "dock_in_episode", "module", "run_global", "start_pose_id",
                     "t_start_local", "executor_status", "operator_outcome", "contact_seen", "fault_in_place_before",
                     "fault_in_place_after", "choice_logged", "invalid_reason", "repeat_of", "v_batt_min_v", "notes"])
        for k, mod in enumerate(docks, start=1):
            ok = mod not in ep["faulted_modules"]
            wr.writerow(["x", ep["episode"], k, mod, k, 3, "10:00:00", "done", "success" if ok else "collision",
                         0, 1, 1, 1, "none", "", 12.0, ""])
    demo_csv = BL.build_demo(out / "demo.csv", w / "demo_sheets", log, runs, f["ds"], allow_simulation=True)
    # ---- predict: frozen per-condition simulator predictions (sealed modes of this schedule; builds of the subset)
    sched = SCH.load(ms_sched)
    conds = PR.conditions(sched, builds=f["builds"])
    pred = PR.write_pred(out / "pred.csv", conds, sched, bundle_dir=f["stub"], seed=11, dt=0.01)
    p = pd.read_csv(pred, keep_default_na=False)
    assert list(p.columns[:6]) == list(RV.PRED_COLS)
    assert p.groupby(["build", "module", "wheel"]).size().eq(1).all()
    m = pd.read_csv(mask_csv, keep_default_na=False)
    fw = {(r.build, r.module, r.wheel): r.firmware for r in m.itertuples() if r.module}
    pf = {(r.build, r.module, r.wheel): r.firmware for r in p.itertuples() if r.module}
    assert all(pf[k] == v for k, v in fw.items())                   # predicted the injected (sealed) mode
    assert (p["bundle_sha256"] == PR.sha256_file(Path(f["stub"]) / "bundle.yaml")).all()
    # ---- predict: robot-side units, with every evaluator-only path booby-trapped
    loaded = BL.load(f["ms"], ms_sched, allow_simulation=True) + BL.load(f["ds"], SCH.load(ds_sched), allow_simulation=True)
    with monkeypatch.context() as mp:
        _guard(mp)
        units = PR.build_units(out / "pred_units.csv", None, ms_sched, f["stub"], lam, ds_sched, loaded=loaded)
        for _, raw in loaded:                                        # truth replaced by garbage: same predictions
            raw["truth_pose"] = torch.full_like(raw["truth_pose"], float("nan"))
        units2 = PR.build_units(out / "pred_units2.csv", None, ms_sched, f["stub"], lam, ds_sched, loaded=loaded)
    a, b = pd.read_csv(units), pd.read_csv(units2)
    assert a.equals(b)
    assert set(a.columns) == set(PR.UNIT_COLS) and set(a["part"]) == {"MASK", "DEMO"}
    rep = PR.check_units(units, r3, log)
    assert rep["r3"]["ok"] and rep["r3"]["carry_over_ok"], rep
    assert rep["choices"]["ok"] and rep["choices"]["n_episodes"] == 1, rep
    assert (a["lambda_status"] == "SYNTHETIC").all() and not a["go_a10"].astype(bool).any()    # lambda = inf
    # ---- realv2 on predict's prediction files
    man = BL.write_manifest(out / "manifest.yaml", mask_csv, pred, r3, demo_csv, lam, output_dir="res",
                            extra={"predicted_rho": str(Path(pred).with_name("pred_rho.csv"))})
    res = RV.run_realv2(man)
    assert res["status"] == RV.STATUS_BAD                            # SIMULATION data never confirmatory
    for k in ("R1", "R2", "R3", "R4"):
        assert isinstance(res[k]["verdict"], str), k
    pair = res["R1"]["by_type"]["pair"]
    assert pair["n_conditions"] == 2 and math.isfinite(pair["L_pred_median"])
    assert pair["equivalence"]["n"] == 2                             # every multi-module condition got an L_pred
    r2rows = res["R2"].get("rows") or []
    assert all(r["z_pred_alone"] is not None for r in r2rows)
    fid = res["descriptive"]["fidelity"]
    assert "status" not in fid and any(k.startswith("single|") for k in fid)
    assert "predicted_rho" in res["inputs"] and "predictions" in res["inputs"]
    assert res["R4"]["n_episodes"] == 1
    assert json.loads((out / "res" / "results_v2.json").read_text())["R1"]["verdict"]
