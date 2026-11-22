"""WS-P2: PROTOCOL_V2 MASK / DEMO / SIMVAL-day-2 designs in the packet, probe subsets, link fields, field-side module choice
(CPU, no ROS)."""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.analysis import realv2_design as RD
from capxfer.field import choose_module as CM
from capxfer.field import probe_runner as PR
from capxfer.field import schedule as SCH

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "capxfer" / "field_packet" / "tools"
sys.path.insert(0, str(TOOLS))
P = A.load_params(REPO / "params" / "module_measured.yaml")


def _cfg():
    import make_run_sheets as MR
    return MR, MR.load_config()


def test_probe_subset_schedule():
    asm = A.physical("pair", P)
    full = SCH.trial_schedule(asm)
    short = SCH.trial_schedule(asm, {"probes": ["P1", "P5", "P2"]})
    assert [s.pid for s in short.segments[::2]] == ["P1", "P2", "P5"]
    assert abs(short.T - 12.7) < 1e-6 and full.T > 29
    assert SCH.trial_schedule(asm, None, duration_s=12.75).T == short.T            # loader fallback without a schedule
    assert SCH.trial_schedule(asm, None, duration_s=29.4).T == full.T
    with pytest.raises(ValueError):
        SCH.trial_pids({"probes": ["P9"]})


def test_realv2_loader_without_scipy_matches():
    import packet_lib as PL
    seed = bytes.fromhex("9a" * 32)
    mod = PL.load_realv2_design()
    assert mod.schedule_sha256(mod.expand_schedule(seed)) == RD.schedule_sha256(RD.expand_schedule(seed))
    import shutil
    import subprocess
    py = shutil.which("python3", path="/usr/bin:/bin")                              # the robot laptop: no scipy/pandas
    if py:
        code = ("import sys; sys.path.insert(0, %r); import packet_lib as PL; m = PL.load_realv2_design(); "
                "print(m.schedule_sha256(m.expand_schedule(bytes.fromhex('9a' * 32))))" % str(TOOLS))
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == RD.schedule_sha256(RD.expand_schedule(seed))


def test_masking_follows_realv2_design():
    MR, cfg = _cfg()
    seed = "ab" * 32
    s = MR.build_masking(cfg, seed, P)
    X = RD.expand_schedule(bytes.fromhex(seed))
    assert s == MR.build_masking(cfg, seed, P)                                       # deterministic from the seed
    assert s["realv2"]["commitment"] == RD.commitment(bytes.fromhex(seed))
    assert s["realv2"]["realv2_schedule_sha256"] == RD.schedule_sha256(X)
    assert s["n_trials"] == RD.mask_counts()["trials"] == 198
    byb = {b["block"]: b for x in s["sessions"] for b in x["blocks"]}
    run = [t for fs in s["field_sessions"] for r in fs["blocks"] for t in byb[r["block"]]["trials"]]
    assert [t["realv2_seq"] for t in run] == list(range(len(X["mask_trials"])))     # the sealed PROTOCOL_V2 order
    for t, x in zip(run, X["mask_trials"]):
        assert t["round"] == x["build"] and t["r3_split"] == x["r3_split"]
        assert t["faults"] == ([] if x["module"] is None else [{"wheel": f"{x['module']}:{x['wheel']}", "mode": x["firmware"]}])
        assert t.get("probes") is None                                              # full probe library (PROTOCOL_V2 §2.1)
    assert [r["round"] for fs in s["field_sessions"] for r in fs["blocks"]] == X["mask_order"]
    assert len(s["episodes"]) == 16
    for e in s["episodes"]:                         # each condition: every parent assembly AND the module alone
        m = e["fault_carrier"]
        want = [b for b in X["mask_order"] if b == f"single_{m}" or m in RD.TYPE_MODULES.get(b, ())]
        assert sorted(e["parts"]) == sorted(want) and all(len(v) == RD.MASK_REPEATS for v in e["parts"].values())
    for fs in s["field_sessions"]:
        for blk in PR.session_plan(s, fs["id"]):
            mods = sorted(m[0] for m in blk["session"]["modules"])
            assert blk["reassemble"]
            for tr in blk["trials"]:
                assert all(f["wheel"].split(":")[0] in mods for f in tr["faults"])
            if blk["assembly"] == "single":
                assert "ONLY " + mods[0] in blk["instruction"]


def test_demo_follows_realv2_design():
    MR, cfg = _cfg()
    seed = "ef" * 32
    s, rows = MR.build_choice(cfg, seed, P)
    X = RD.expand_schedule(bytes.fromhex(seed))
    eps = s["choice"]["episodes"]
    assert len(eps) == RD.DEMO_EPISODES == 20 and s["choice"]["demo_source"] == "realv2_design"
    assert sum(1 for r in rows if r["step"].startswith("PROBE")) == 20 * 4                 # L3 + 3 lone probes
    assert sum(1 for r in rows if r["step"] == "DOCK") == 20 * 3                           # dock slots (dedup list <= 3)
    assert s["choice"]["bay"]["width_m"] == pytest.approx(0.44)
    for e, x in zip(eps, X["demo"]):
        assert e["assembly"] == "L3" and e["modules"] == list(RD.DEMO_MODULES)
        assert e["faulted_modules"] == list(x["faulted_modules"]) and e["dock_order"] == list(x["dock_order"])
        assert e["faults"] == [{"wheel": f"{m}:{x['faulted_wheels'][m]}", "mode": "coast"} for m in x["faulted_modules"]]
        assert e["capable_module"] not in e["faulted_modules"]
        assert e["coins"]["random_coin"] == x["random_module"] and e["bb_nearest_goal"] == ["M4"]
        lone = {m: t for m, t in e["lone_trials"].items()}
        by = {t["trial"]: t for _, _, t in SCH.iter_trials(s)}
        for m, t in lone.items():                                     # the module alone keeps its own coast wheel
            assert by[t]["faults"] == [f for f in e["faults"] if f["wheel"].startswith(m + ":")]
        assert all(f["mode"] != "roller_locked" for f in by[e["assembly_trial"]]["faults"])      # no zip ties
    blob = json.dumps(rows)
    assert "certificate" not in blob and "oracle" not in blob and "coin" not in blob  # arm identities never on the sheet


def test_dock_list_dedup():
    assert CM.dock_list({"certificate": "M3", "carry_over": "M4", "random": "M3", "oracle": "M3"},
                        ["M4", "M2", "M3"]) == ["M4", "M3"]
    assert CM.dock_list({"certificate": "M2", "carry_over": None, "random": "M2", "oracle": "M2"},
                        ["M4", "M2", "M3"]) == ["M2"]


def test_simval2_and_tagstatic():
    MR, cfg = _cfg()
    seed = "12" * 32
    X = RD.expand_schedule(bytes.fromhex(seed))
    s2 = MR.build_simval2(cfg, seed, P)
    tr = [t for _, _, t in SCH.iter_trials(s2)]
    assert len(tr) == 10 and [t["realv2_seq"] for t in tr] == list(range(10))
    assert sum(1 for t in tr if t["faults"]) == 5
    st = MR.build_tagstatic(cfg, seed, P)
    assert [t["condition"] for _, _, t in SCH.iter_trials(st)] == [f"STATIC_POSE{k + 1}" for k in X["apriltag_pose_order"]]


def test_certificate_core():
    rng = np.random.default_rng(0)
    ws = {"s_ok": rng.uniform(0, 0.1, 500), "s_f": rng.uniform(0, 0.1, 500), "alpha1": 0.05, "K": 2}
    asm = A.physical("pair", P)
    mids = [m for m, *_ in asm.modules]
    P5 = np.full((asm.n, 5), 0.02)
    P5[:, 0] = 0.92
    P5[[0, 3]] = 0.02                                   # first module: fl and rr roller-locked (mode 4)
    P5[[0, 3], 4] = 0.92
    bands = {}
    for m in mids:
        for mv in itertools.product(range(5), repeat=4):
            nf = sum(1 for v in mv if v)
            if nf <= 2:
                bands[(m, CM.mode_key(mv))] = 0.95 if nf == 0 else (0.2 if mv.count(4) == 2 else 0.7)
    lb = CM.certificate_bounds(P5, asm.wheel_module, mids, ws, bands)
    assert lb[mids[0]] == pytest.approx(0.2) and lb[mids[1]] == pytest.approx(0.95)
    assert CM.realv2_choice(lb, 0.5, "M1") == (mids[1], {mids[0]: False, mids[1]: True})
    assert CM.realv2_choice({"M1": 0.3, "M3": 0.4}, math.inf, "M1")[0] is None         # none certified: abstain
    assert CM.realv2_choice({"M1": 0.9, "M3": 0.9}, 0.5, "M3")[0] == "M3"              # exact tie: sealed coin
    assert CM.realv2_choice({"M2": 0.85, "M3": 0.95, "M4": 0.85}, 0.5, "M2")[0] == "M3"
    assert CM.cq([1, 2, 3], 0.01) == math.inf


def test_choose_module_dry_run(tmp_path):
    res = CM.dry_run(tmp_path / "w")                    # DEMO sheets -> SIMULATION L3-probe sessions -> choice
    assert len(res) == 3 and all(r["certificate"] not in r["faulted"] and r["certificate"] in r["docks"] for r in res)
    assert all(r["oracle"] == max(r["z_alone"], key=r["z_alone"].get) for r in res)   # oracle from the lone-probe truth
    log = list(csv.DictReader(open(tmp_path / "w" / "choices" / "choices.csv")))
    assert {r["arm"] for r in log} >= {"certificate", "carry_over", "random", "oracle", "dock_list", "-"}
    assert all(r["bundle_sha256"] and r["lambda_sha256"] for r in log if r["arm"] == "-")


def test_link_rows_checked(tmp_path):
    import packet_lib as PL
    import validate_csv_session as V
    MR, cfg = _cfg()
    s, _ = MR.build_choice(cfg, "ab" * 32, P)
    u = next(x for x in s["link_units"] if x["phase"] == "assembly")
    md = tmp_path / "manual"
    md.mkdir()

    def check(row):
        with open(md / "episode_log.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(PL.header("episode_log"))
            w.writerow(row)
        rep = V.Report()
        V.check_link_rows(md, rep, [s])
        return [i for i in rep.items if i[1] == "M4"][0][0]
    row = ["gnd_x", "DEMO", u["episode"], "assembly", "all", "S", 1, u["trials"][0], u["trials"][-1], u["fault_carrier"],
           ";".join(u["zip_wheels"]), 1, "assembly", 1, ""]
    assert check(row) == "PASS"
    row[9] = "M4"                                                                    # wrong physical carrier
    assert check(row) == "FAIL"
