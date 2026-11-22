"""WS-I3F dock executor (capxfer.field.dock_executor) and the tag-based dock scorer (builders.score_dock). No ROS."""
from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.field import builders as BL
from capxfer.field import dock_executor as DX
from capxfer.plan import dock as DK

REPO = Path(__file__).resolve().parents[1]
P = A.load_params(REPO / "params" / "module_measured.yaml")


def _single(m="M1"):
    return A.build("single", [(m, 0, 0, 0)], P)


def test_imports_without_rclpy():
    code = ("import sys; sys.modules['rclpy'] = None; sys.modules['geometry_msgs'] = None; "
            "import capxfer.field.dock_executor as D; assert 'rclpy' not in [k for k, v in sys.modules.items() if v]; "
            "print(D.ARENA_SCHEMA)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO),
                       env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "capxfer.field.dock_executor", "--module", "M1", "--log-dir", "/tmp/x"],
                       capture_output=True, text=True, cwd=str(REPO),
                       env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
    assert r.returncode != 0 and "rclpy not found" in (r.stderr + r.stdout)


def test_dry_run_prints_timeline(capsys):
    assert DX.main(["--module", "M3", "--dry-run", "--start-offset", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "plan: ok route sideways" in out and "first chunk" in out and "50 Hz" in out and "e-stop" in out


def test_arena_file_and_scene(tmp_path):
    asm = _single()
    for variant in ("narrow", "wide"):                                   # same geometry as the WS-F / CHOICE sims
        ar = DX.default_arena(asm, variant=variant)
        sc = DX.arena_scene(asm, ar)
        ref = DK.make_scene(asm, 0, variant)
        assert sc.meta["ws"] == pytest.approx(ref.meta["ws"]) and sc.goal[1] == pytest.approx(ref.goal[1])
        assert sc.goal[0] == pytest.approx(ar["bay_x"]) and sc.variant == variant
    # DEMO default: narrow bay (D15) = module footprint length + 0.12 m. With the MEASURED 0.24 m chassis (params
    # MEASURED_V1; was 0.32 m ESTIMATE -> 0.44) this is 0.36 m (WS-MR, 2026-11-21: field kit arena sheet must follow)
    assert asm.side == pytest.approx(0.24)
    assert DX.default_arena(asm)["bay_width"] == pytest.approx(0.36)
    s = DX.start_pose(asm, ar, -0.10)
    assert s[0] == pytest.approx(ar["bay_x"] - 0.10) and s[1] == pytest.approx(2.6 - asm.side / 2 - 0.9)
    (tmp_path / "a.yaml").write_text("schema: capxfer.dock_arena/v1\ny_face: 2.5\nbay_x: 1.8\nbay_width: 0.62\n"
                                     "bay_depth: 0.62\n")
    a2 = DX.load_arena(tmp_path / "a.yaml", asm)
    assert DX.arena_scene(asm, a2).goal == pytest.approx((1.8, 2.5 + 0.62 - 0.06 - asm.side / 2, 0.0))
    (tmp_path / "b.yaml").write_text("schema: capxfer.dock_arena/v1\ny_face: 2.5\nbay_x: 1.8\nbay_width: 0.2\n"
                                     "bay_depth: 0.62\n")
    with pytest.raises(ValueError, match="narrower"):
        DX.load_arena(tmp_path / "b.yaml", asm)


def test_truth_topic_refused():
    with pytest.raises(SystemExit, match="evaluator-only"):
        DX.check_pose_topic("/fleet/truth/capxfer_asm/pose")
    assert DX.check_pose_topic("/m1/dock/pose") == "/m1/dock/pose"


class FakeIO:
    """Robot at a fixed pose; virtual clock; records every command."""

    def __init__(self, pose, stop_after=None, ex=None):
        self.pose, self.t, self.tw, self.ev = pose, 0.0, [], []
        self.stop_after, self.ex = stop_after, ex

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt

    def twist(self, xi):
        self.tw.append(tuple(xi))
        if self.stop_after and len(self.tw) == self.stop_after:
            self.ex.request_stop()                                          # Ctrl-C mid-chunk

    def event(self, d):
        self.ev.append(d)

    def still(self):
        return True

    def fix(self, dur):
        self.t += dur
        return self.pose


def _ex(io, asm, scene, **kw):
    return DX.DockExecutor(io, asm, scene, DX.capset_from("omni", asm), clock=io.clock, sleep=io.sleep, **kw)


def test_estop_zero_twist_and_abort(tmp_path):
    asm = _single()
    ar = DX.default_arena(asm)
    sc = DX.arena_scene(asm, ar)
    io = FakeIO(sc.start, stop_after=30)
    log = DX.DockLog(tmp_path / "dock.csv", dict(run=1, episode="D01", module="M1"))
    ex = _ex(io, asm, sc, log=log)
    io.ex = ex
    res = ex.run()
    log.close()
    assert res["status"] == "abort"
    assert io.tw[-10:] == [(0.0, 0.0, 0.0)] * 10 and io.ev[-1]["event"] == "abort"
    assert any(abs(v) > 0.1 for v in io.tw[20])                           # it was moving when stopped
    rows = list(csv.DictReader(open(tmp_path / "dock.csv")))
    assert tuple(rows[0].keys()) == DX.LOG_COLS
    kinds = {r["kind"] for r in rows}
    assert kinds == {"EV", "FIX", "PLAN", "CMD"} and rows[-1]["event"] == "abort"


def test_timer_driven_samples_and_done_at_goal():
    asm = _single()
    sc = DX.arena_scene(asm, DX.default_arena(asm))
    io = FakeIO(sc.start)
    ex = _ex(io, asm, sc, max_iters=1)
    res = ex.run()
    assert res["status"] == "timeout" and res["replans"] == 1             # robot never moves: 1 plan, then timeout
    n = int(math.ceil(DK.CommandSchedule(DK._prefix(DK.plan_dock(ex.capset, asm, sc))).T * 50))
    assert len(io.tw) == n + 1 + 1                                        # samples + zero after chunk + final zero
    assert io.t >= n / 50.0 - 1e-6                                        # published on 50 Hz deadlines
    io2 = FakeIO(sc.goal)
    assert _ex(io2, asm, sc).run()["status"] == "done" and io2.ev[-1]["event"] == "done"
    io3 = FakeIO(None)
    io3.fix = lambda d: None
    assert _ex(io3, asm, sc).run()["status"] == "fix_lost"
    io4 = FakeIO(None)
    io4.fix = lambda d: None
    r4 = _ex(io4, asm, sc, loop="open").run(start=sc.start)               # open loop without localisation
    assert r4["status"] == "done" and r4["replans"] == 1


def test_sim_closed_loop_healthy_docks_and_scores():
    asm = _single("M3")
    ar = DX.default_arena(asm)
    res, out = DX.simulate_dock(asm, P, ar, fault="F0", offset_m=-0.05, seed=1)
    assert res["status"] == "done" and out["success"] and not out["collision"], (res, out)
    assert res["events"][0] == "run_start" and res["events"][-1] == "done"


def _traj(start, end, n=200, hold=60):
    a = np.linspace(0, 1, n)[:, None]
    p = (1 - a) * np.asarray(start) + a * np.asarray(end)
    p = np.vstack([p, np.repeat(p[-1:], hold, 0)])
    return np.arange(len(p)) / 30.0, p


def test_score_dock_rules():
    asm = _single()
    sc = DX.arena_scene(asm, DX.default_arena(asm))
    t, p = _traj(sc.start, sc.goal)
    assert BL.score_dock(t, p, asm, sc)["label"] == "success"
    g = np.asarray(sc.goal) + [0.08, 0, 0]
    t, p = _traj(sc.start, g)
    s = BL.score_dock(t, p, asm, sc)
    assert s["label"] in ("collision", "pose_error") and not s["success"]
    t, p = _traj(sc.start, np.asarray(sc.goal) + [0.0, -0.03, 0.0])
    assert BL.score_dock(t, p, asm, sc)["label"] == "success"              # 3 cm < 5 cm tolerance
    t, p = _traj(sc.start, (sc.goal[0] + 0.5, sc.goal[1], 0.0))            # drives through the bay wall
    assert BL.score_dock(t, p, asm, sc)["label"] == "collision"
    t, p = _traj(sc.start, sc.goal, hold=5)                                # never 1 s at rest
    assert BL.score_dock(t, p, asm, sc)["label"] == "not_at_rest"
