"""WS-P: field packet tools and the probe runner's session mode (CPU, no ROS; < 60 s)."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.field import fieldlog_config as FC
from capxfer.field import probe_runner as PR
from capxfer.field import schedule as SCH

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "capxfer" / "field_packet" / "tools"
sys.path.insert(0, str(TOOLS))
P = A.load_params(REPO / "params" / "module_measured.yaml")


class RecPub:
    def __init__(self):
        self.ev = []

    def marker(self, d):
        self.ev.append(("marker", None, d))

    def assembly(self, m, d):
        self.ev.append(("assembly", m, d))

    def fault(self, m, d):
        self.ev.append(("fault", m, d))

    def twist(self, mods, xi):
        self.ev.append(("twist", tuple(mods), xi))

    def sensors(self, imu, lidar):
        self.ev.append(("sensors", None, (imu, lidar)))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += max(d, 1e-4)


def test_session_mode_markers_and_relay():
    s = SCH.generate("5e" * 16)
    plan = PR.session_plan(s, "S01")
    assert [b["pos"] for b in plan] == list(range(1, len(plan) + 1))
    plan = [dict(b, trials=b["trials"][:1]) for b in plan[:3]]                     # 1 trial of the first 3 blocks
    pub, clk = RecPub(), FakeClock()
    n = PR.Runner(pub, P, clock=clk, sleep=clk.sleep).run_session(plan, gap=0.0)
    assert n == 3
    marks = [e[2] for e in pub.ev if e[0] == "marker"]
    starts = [m for m in marks if m["event"] == "start"]
    assert [m["block"] for m in starts] == [b["pos"] for b in plan] and all(m["session"] == 1 for m in starts)
    for b, m in zip(plan, starts):
        imu, lid = SCH.sensor_modules(A.build(b["assembly"], [tuple(x) for x in b["session"]["modules"]], P))
        assert (m["imu"], m["lidar"]) == (PR.slot_code(imu), PR.slot_code(lid))
    assert marks[0]["event"] == "assembly" and marks[1]["event"] == "block"         # first block = first build
    sens = [i for i, e in enumerate(pub.ev) if e[0] == "sensors"]
    first_start = [i for i, e in enumerate(pub.ev) if e[0] == "marker" and e[2]["event"] in ("assembly", "block", "start")]
    assert sens[0] < first_start[0]                                                   # relay switches before the markers
    assert set(FC.TRIAL_EVENT_CODES) >= {"start", "end", "abort", "assembly", "block"}
    assert {"block_pos", "sess_num", "imu_slot", "lidar_slot"} <= set(FC.TRIAL_COLS)


def test_session_resume_and_dry_run(tmp_path):
    r = SCH.write(tmp_path / "s", seed_hex="77" * 16)
    s = json.loads(Path(r["schedule"]).read_text())
    plan = PR.session_plan(s, "S02", from_block=3)
    assert plan[0]["pos"] == 3 and plan[0]["reassemble"]
    out = subprocess.run([sys.executable, "-m", "capxfer.field.probe_runner", str(r["schedule"]), "--session", "S02",
                          "--dry-run", "--to-block", "2"], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "REBUILD" in out.stdout and "block pos 2" in out.stdout and "block pos 3" not in out.stdout


def test_make_run_sheets_main_matches_schedule(tmp_path):
    seed = tmp_path / "seed.txt"
    seed.write_text("ab" * 32)
    out = tmp_path / "main"
    r = subprocess.run([sys.executable, str(TOOLS / "make_run_sheets.py"), "main", str(out), "--seed-file", str(seed)],
                       capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    s = SCH.load(out / "schedule.json")
    assert s["n_trials"] == 1420 and len(s["field_sessions"]) == SCH.N_SESSIONS
    assert s == SCH.generate("ab" * 32, session_order=s["session_order"])              # schedule.py DEFAULTS, no overrides lost
    with open(out / "fieldlog_runsheets.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == SCH.N_SESSIONS
    for row in rows:
        want = SCH.session_runsheet_rows(s, row["sched_session"])
        with open(out / row["dir"] / "runsheet.csv") as f:
            got = list(csv.DictReader(f))
        assert [int(g["arm_code"]) for g in got] == [w[2]["arm_code"] for w in want]
        assert [int(g["block_code"]) for g in got] == [w[2]["block_code"] for w in want]
        order = [int(x) for x in row["trial_order"].split()]
        byb = {(x["assembly"], b["block"]): b for x in s["sessions"] for b in x["blocks"]}
        fs = next(f for f in SCH.field_sessions(s) if f["id"] == row["sched_session"])
        assert order == [t["trial"] for r_ in fs["blocks"] for t in byb[(r_["assembly"], r_["block"])]["trials"]]
        assert len(order) == len(want)
    assert (out / "S01" / "blocks_S01.csv").is_file() and "PENDING OWNER DECISION" in (out / "SUMMARY.md").read_text()


@pytest.mark.parametrize("kind", ["shakedown", "hooktest", "pilot", "sysid", "simval", "apriltag", "natural", "floor2"])
def test_extra_kinds_build(kind):
    import make_run_sheets as MR
    cfg = MR.load_config()
    s = MR.build_extra(kind, cfg, "cd" * 32, P, module="M2", day=2)
    nums = [t["trial"] for x in s["sessions"] for b in x["blocks"] for t in b["trials"]]
    assert len(nums) == len(set(nums)) == s["n_trials"] > 0
    assert all(n > 2000 for n in nums)                                                 # never collides with main trial numbers
    for x in s["sessions"]:
        asm = A.build(x["assembly"], [tuple(m) for m in x["modules"]], P)
        for b in x["blocks"]:
            for t in b["trials"]:
                for f in t["faults"]:
                    assert f["wheel"] == "none" or f["wheel"] in asm.wheel_corner
                    assert f["mode"] != "payload" or f["mass_kg"] <= 6.0
    if kind == "sysid":
        assert s["sessions"][0]["modules"] == [["M2", 0, 0, 0]]


def test_templates_match_schema():
    import packet_lib as PL
    for name in PL.SCHEMAS:
        hdr, rows = PL.read_csv(REPO / "capxfer" / "field_packet" / "templates" / f"{name}.csv")
        assert hdr == PL.header(name) and rows == []
    hdr, rows = PL.read_csv(REPO / "capxfer" / "field_packet" / "MEASUREMENTS_SHEET.csv")
    assert hdr == PL.header("measurements") and len(rows) >= 100
    assert PL.check_cell(PL.SCHEMAS["block_log"][18], "7.5") is not None              # payload > 6 kg refused
