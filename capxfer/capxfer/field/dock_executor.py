"""ROS 2 sideways-dock executor `capxfer_dock_executor` (docs/DOCKING_PROTOCOL.md §6; DEMO card 10, PROTOCOL_V2 R4).
People start and stop it; Ctrl-C is the e-stop. rclpy is imported lazily: this module imports without ROS.

  python -m capxfer.field.dock_executor --module M1 --ids M1=m1 --arena arena.yaml --episode D03 --dock 1 \
         --log-dir /data/capxfer/dock_logs [--loop closed|open] [--capset omni|CAPSET.json] [--robot m1] \
         [--pose-topic /m1/dock/pose] [--start-offset 0.05]
  python -m capxfer.field.dock_executor --module M1 --dry-run [--arena arena.yaml] [--start-offset 0.05] \
         [--simulate F4x2d|F0]          # no ROS: prints the plan and the 50 Hz command timeline (+ a sim closed loop)

What it does (closed loop = the protocol's stop-and-go):
  1. waits until the wheels are still (wheel_state velocity < 0.02 rad/s on every wheel), averages the ROBOT-VISIBLE
     localisation fix `/<robot>/dock/pose` (PoseWithCovarianceStamped: lidar scan matching against the arena outline,
     arena frame, assembly origin) over 0.5 s. The AprilTag truth `/fleet/truth/*` is EVALUATOR-ONLY: this node never
     subscribes to it (a pose topic containing "truth" is refused).
  2. stops with `done` when the fix is within 0.025 m / 0.04 rad of the goal (plan.dock.EXEC_POS_TOL / EXEC_HEAD_TOL);
  3. plans with capxfer.plan.dock.plan_dock on the arena scene (`arena_scene`) and the run's CapSet (DEMO: `omni` for
     both modules, PROTOCOL_V2 R4: the arms differ only in which module they send); `infeasible` if no plan;
  4. publishes the plan prefix (<= 0.30 m of predicted path or one rotation; plan.dock._prefix) as the exact blended
     twist samples the simulator executes (plan.dock.CommandSchedule, ramp 0.12 s, then 0.6 s of zeros) at 50 Hz on
     absolute deadlines: TwistStamped (frame "assembly") on /fleet/<id>/cmd_vel of every member + the logged copy
     /capxfer/cmd_vel (FieldLog src 7); then back to 1. `timeout` after 90 s of commanded motion or 40 plans.
  Open loop: one plan from the fix (or, without localisation, from the start mark given by --start-offset), executed once.
  Events: JSON on /capxfer/dock/event (run_start, plan, chunk_start, chunk_end, fix, done, infeasible, timeout, abort,
  fix_lost). The executor publishes NO /capxfer/trial markers: dock rows are not probe trials (CSV_SPEC §2.3, C9).

Log (CSV_SPEC §7 "docking executor events, plan segments, pose fixes" until FieldLog has DOCKEV/DOCKFIX/DOCKPLAN
sources): `<log-dir>/dock_<episode>_<dock>_<module>_<utc>.csv`, columns LOG_COLS, one row per published command sample
(CMD), plan segment (PLAN), localisation fix (FIX) and event (EV); plus one `..._plan<k>.json` (capxfer.dock_plan/v1)
per plan. Never inside a FieldLog session folder.

E-stop: SIGINT / SIGTERM -> zero twist 10 times over 0.2 s to every member, `abort` event and log row, exit. The firmware
watchdog (no command for 100 ms) also stops the wheels.

Arena file (`capxfer.dock_arena/v1`, YAML; TEAM measures it): variant (narrow: DEMO D15, goal 0.06 m short of the back
wall; wide: old H4, goal = bay centre), y_face, bay_x (bay centre), bay_width, bay_depth, floor_x (4.0), start_front_gap
(0.9: start-mark front edge to the face). Without --arena the plan.dock.make_scene dimensions are used (bay at x = 2.0 m;
narrow for --module, wide for --assembly).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .. import assembly as A
from ..plan import capability as CAP
from ..plan import dock as DK

PARAMS_PATH = Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml"
ARENA_SCHEMA = "capxfer.dock_arena/v1"
RATE_HZ = 50.0
STILL_RADPS = 0.02
FIX_S = 0.5
STILL_TIMEOUT_S = 5.0
LOG_COLS = ("t_mono_ns", "t_wall_ns", "kind", "run", "episode", "module", "arm_code", "replan_idx", "seg", "vx", "vy",
            "wz", "T", "event", "fix_x_m", "fix_y_m", "fix_yaw_rad", "detail")
STATUSES = ("done", "infeasible", "timeout", "abort", "fix_lost")


# ================================================================================================ arena
def default_arena(asm, bay_x=2.0, variant="narrow"):
    """Arena as plan.dock.make_scene, bay at bay_x. narrow (DEMO, D15 / CHOICE.md §1: sideways entry only): width =
    footprint x-length + 0.12 m, depth = y-length + 0.12 m; wide (old H4): width = depth = footprint diameter + 0.16 m."""
    pts = DK.footprint_points(asm)
    rmax = float(np.hypot(pts[:, 0], pts[:, 1]).max())
    x0, x1, y0, y1 = DK.extents(asm)
    ws, depth = ((x1 - x0) + 0.12, (y1 - y0) + 0.12) if variant == "narrow" else (2 * rmax + 0.16, 2 * rmax + 0.16)
    return {"schema": ARENA_SCHEMA, "variant": variant, "y_face": 2.6, "bay_x": float(bay_x), "bay_width": ws,
            "bay_depth": depth, "floor_x": 4.0, "start_front_gap": 0.9, "source": f"default (make_scene {variant})"}


def load_arena(path, asm=None):
    import yaml
    d = yaml.safe_load(Path(path).read_text())
    if d.get("schema") != ARENA_SCHEMA:
        raise ValueError(f"{path}: not a {ARENA_SCHEMA} file")
    need = ("y_face", "bay_x", "bay_width", "bay_depth")
    miss = [k for k in need if not isinstance(d.get(k), (int, float))]
    if miss:
        raise ValueError(f"{path}: missing numeric {miss}")
    d.setdefault("floor_x", 4.0)
    d.setdefault("start_front_gap", 0.9)
    d.setdefault("variant", "narrow")
    if d["variant"] not in ("narrow", "wide"):
        raise ValueError(f"{path}: variant must be narrow or wide")
    if asm is not None:
        x0, x1, _, _ = DK.extents(asm)
        if d["bay_width"] < (x1 - x0) + 0.02:
            raise ValueError(f"{path}: bay_width {d['bay_width']} narrower than the footprint")
    return d


def arena_scene(asm, arena, start=None):
    """Deterministic plan.dock.Scene of the measured arena (frame: x along the face, y toward the bay, origin A0).
    Goal = bay centre (wide) at heading 0; start default = the centre start mark."""
    res = DK.RES
    ws, depth, y_face, WX = float(arena["bay_width"]), float(arena["bay_depth"]), float(arena["y_face"]), float(arena["floor_x"])
    xs = float(arena["bay_x"])
    x0, x1, y0, y1 = DK.extents(asm)
    H = y_face + depth + 0.2
    occ = np.ones((int(round(WX / res)), int(round(H / res))), bool)
    floor = (0.0, WX, 0.0, y_face)
    bay = (xs - ws / 2, xs + ws / 2, y_face - 1e-6, y_face + depth)

    def carve(r):
        i0, i1 = int(math.ceil(r[0] / res - 1e-9)), int(math.floor(r[1] / res + 1e-9))
        j0, j1 = int(math.ceil(r[2] / res - 1e-9)), int(math.floor(r[3] / res + 1e-9))
        occ[max(i0, 0):i1, max(j0, 0):j1] = False
    carve((0.0 + res, WX - res, 0.0 + res, y_face))
    carve(bay)
    w = DK.World(occ, res, "dock_arena", dict(slot=dict(x=xs, w=ws, depth=depth, y_face=y_face)))
    gy = y_face + depth / 2 if arena.get("variant", "narrow") == "wide" else y_face + depth - 0.06 - y1
    goal = (xs - 0.5 * (x0 + x1), gy, 0.0)
    if start is None:
        start = start_pose(asm, arena, 0.0)
    pts = DK.footprint_points(asm)
    return DK.Scene(w, tuple(float(v) for v in start), goal, floor, bay, arena.get("variant", "narrow"),
                    dict(ws=ws, depth=depth, y_face=y_face, Rmax=float(np.hypot(pts[:, 0], pts[:, 1]).max()),
                         Lx=x1 - x0, Ly=y1 - y0, arena=dict(arena)))


def start_pose(asm, arena, offset_m=0.0):
    """Start mark: x = bay_x + offset (sealed offsets -0.10..+0.10 m), front edge start_front_gap from the face."""
    x0, x1, _, y1 = DK.extents(asm)
    return (float(arena["bay_x"]) + float(offset_m) - 0.5 * (x0 + x1), float(arena["y_face"]) - y1 - float(arena["start_front_gap"]), 0.0)


def capset_from(spec, asm):
    """'omni' (DEMO, PROTOCOL_V2 R4) or a JSON file {lo, hi[, drift]} (14 values each, probes.TARGET_KEYS order)."""
    if spec in (None, "omni"):
        return CAP.omni(asm)
    d = json.loads(Path(spec).read_text())
    return CAP.from_intervals(d["lo"], d["hi"], asm, drift=d.get("drift"), name=d.get("name", "file"))


# ================================================================================================ log
class DockLog:
    def __init__(self, path, meta, wall=time.time_ns, mono=time.monotonic_ns):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta, self.wall, self.mono = meta, wall, mono
        self.fh = open(self.path, "w", newline="")
        self.w = csv.writer(self.fh, lineterminator="\n")
        self.w.writerow(LOG_COLS)
        self.n = 0

    def row(self, kind, **kv):
        base = {"t_mono_ns": self.mono(), "t_wall_ns": self.wall(), "kind": kind, "run": self.meta.get("run", ""),
                "episode": self.meta.get("episode", ""), "module": self.meta.get("module", ""),
                "arm_code": self.meta.get("arm_code", "")}
        base.update(kv)
        self.w.writerow(["" if base.get(c) is None else base.get(c, "") for c in LOG_COLS])
        self.n += 1
        if kind != "CMD":
            self.fh.flush()

    def close(self):
        self.fh.flush()
        self.fh.close()


class NullLog:
    n = 0

    def row(self, kind, **kv):
        self.n += 1

    def close(self):
        pass


# ================================================================================================ executor core
def _circ_mean(a):
    return float(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


class DockExecutor:
    """ROS-agnostic stop-and-go executor. `io` implements twist(xi), event(dict), fix(duration_s) -> (x, y, yaw) | None,
    still() -> bool, and optionally sleep(dt) (spins callbacks)."""

    def __init__(self, io, asm, scene, capset, loop="closed", log=None, clock=time.monotonic, sleep=None,
                 rate_hz=RATE_HZ, max_iters=DK.MAX_ITERS, t_max=DK.T_MAX, fix_s=FIX_S, still_timeout=STILL_TIMEOUT_S,
                 plan_dir=None, plan_tag="plan", budget=12000):
        if loop not in ("closed", "open"):
            raise ValueError(loop)
        self.io, self.asm, self.scene, self.capset, self.loop = io, asm, scene, capset, loop
        self.log = log or NullLog()
        self.clock = clock
        self.sleep = sleep or getattr(io, "sleep", time.sleep)
        self.rate, self.max_iters, self.t_max, self.fix_s, self.still_timeout = rate_hz, max_iters, t_max, fix_s, still_timeout
        self.plan_dir, self.plan_tag, self.budget = plan_dir, plan_tag, budget
        self.stop = False
        self.replans = 0
        self.t_cmd = 0.0
        self.chunks = 0

    def request_stop(self, *_):
        self.stop = True

    def _event(self, name, **kv):
        d = dict(event=name, **kv)
        self.io.event(d)
        self.log.row("EV", event=name, replan_idx=self.replans, detail=json.dumps(kv, default=float) if kv else "")

    def estop(self):
        for _ in range(10):
            self.io.twist((0.0, 0.0, 0.0))
            self.sleep(0.02)
        self._event("abort", t_cmd=self.t_cmd)

    def _wait_still(self):
        t0 = self.clock()
        while not self.io.still():
            if self.stop:
                return False
            if self.clock() - t0 > self.still_timeout:
                self._event("still_timeout", s=self.still_timeout)
                return True
            self.io.twist((0.0, 0.0, 0.0))
            self.sleep(1.0 / self.rate)
        return True

    def _fix(self):
        f = self.io.fix(self.fix_s)
        if f is not None:
            self.log.row("FIX", replan_idx=self.replans, fix_x_m=f"{f[0]:.5f}", fix_y_m=f"{f[1]:.5f}", fix_yaw_rad=f"{f[2]:.5f}")
        return f

    def _publish(self, cmds):
        """Timer-driven 50 Hz publication of the blended samples of `cmds` (+ the rest). False if stopped."""
        sched = DK.CommandSchedule(cmds)
        n = int(math.ceil(sched.T * self.rate))
        tw = sched.twist(np.arange(n) / self.rate)
        t0 = self.clock()
        for k in range(n):
            while True:
                if self.stop:
                    return False
                dt = t0 + k / self.rate - self.clock()
                if dt <= 0:
                    break
                self.sleep(min(dt, 1.0 / self.rate))
            xi = tuple(float(v) for v in tw[k])
            self.io.twist(xi)
            self.log.row("CMD", replan_idx=self.replans, seg=k, vx=f"{xi[0]:.5f}", vy=f"{xi[1]:.5f}", wz=f"{xi[2]:.5f}")
        self.io.twist((0.0, 0.0, 0.0))
        self.t_cmd += sched.T
        return True

    def _save_plan(self, plan, start):
        if self.plan_dir is None:
            return
        sc = dataclasses.replace(self.scene, start=tuple(float(v) for v in start))
        p = Path(self.plan_dir) / f"{self.plan_tag}{self.replans}.json"
        p.write_text(json.dumps(DK.export_plan(plan, self.capset, sc, self.asm, rate_hz=int(self.rate))))

    def run(self, start=None, meta=None):
        """-> dict(status, replans, chunks, t_cmd, last_fix, routes). `start`: pose used when no fix is available
        (open loop without localisation only)."""
        routes, last = [], None
        self._event("run_start", **(meta or {}), loop=self.loop, capset=self.capset.name,
                    goal=list(self.scene.goal))
        status = "timeout"
        try:
            while True:
                if self.stop:
                    status = "abort"
                    break
                if not self._wait_still():
                    status = "abort"
                    break
                f = self._fix()
                if f is None:
                    if self.loop == "open" and self.replans == 0 and start is not None:
                        f = tuple(start)
                        self._event("fix_missing_use_start_mark", start=list(f))
                    elif self.loop == "open" and self.replans > 0:
                        status = "done"
                        break
                    else:
                        status = "fix_lost"
                        break
                last = f
                g = self.scene.goal
                if self.chunks:
                    self._event("chunk_end", fix=list(f))
                if (math.hypot(f[0] - g[0], f[1] - g[1]) <= DK.EXEC_POS_TOL
                        and abs(DK.wrap(f[2] - g[2])) <= DK.EXEC_HEAD_TOL):
                    status = "done"
                    break
                if self.loop == "open" and self.replans > 0:
                    status = "done"
                    break
                if self.replans >= self.max_iters or self.t_cmd >= self.t_max:
                    status = "timeout"
                    break
                plan = DK.plan_dock(self.capset, self.asm, self.scene, start=tuple(f), budget=self.budget)
                self.replans += 1
                ok = plan.status == "ok"
                self._event("plan", status=plan.status, route=plan.route() if ok else "none", n_segments=len(plan.cmd),
                            reason=plan.reason)
                for j, (c, T) in enumerate(plan.cmd):
                    self.log.row("PLAN", replan_idx=self.replans, seg=j, vx=f"{c[0]:.5f}", vy=f"{c[1]:.5f}",
                                 wz=f"{c[2]:.5f}", T=f"{T:.4f}")
                self._save_plan(plan, f)
                if not ok:
                    status = "infeasible"
                    break
                if not plan.cmd:
                    status = "done"
                    break
                routes.append(plan.route())
                cmds = plan.cmd if self.loop == "open" else DK._prefix(plan)
                self._event("chunk_start", chunk=self.chunks, n_segments=len(cmds),
                            duration_s=float(DK.CommandSchedule(cmds).T))
                if not self._publish(cmds):
                    status = "abort"
                    break
                self.chunks += 1
        except KeyboardInterrupt:
            status = "abort"
        if status == "abort":
            self.estop()
        else:
            self.io.twist((0.0, 0.0, 0.0))
            self._event(status, t_cmd=self.t_cmd, replans=self.replans)
        return dict(status=status, replans=self.replans, chunks=self.chunks, t_cmd=self.t_cmd,
                    last_fix=None if last is None else list(last), route=routes[0] if routes else "none")


# ================================================================================================ sim IO (tests, dry run)
class SampledSchedule:
    """Schedule duck type for sim.simulate: zero-order hold of the published 50 Hz samples (what the firmware sees)."""

    def __init__(self, samples, rate=RATE_HZ, rest=0.3):
        self.s = np.asarray(samples, float).reshape(-1, 3)
        self.rate = rate
        self.T = len(self.s) / rate + rest
        self.segments = []

    def twist(self, t):
        t = np.asarray(t, float)
        k = np.floor(t * self.rate + 1e-9).astype(int)
        out = np.zeros(t.shape + (3,))
        m = (k >= 0) & (k < len(self.s))
        out[m] = self.s[k[m]]
        return out

    def windows(self):
        return []


class SimIO:
    """Executor IO backed by capxfer.sim: published samples are simulated when the executor next asks for stillness or
    a fix; the fix is the true pose + plan.dock.LOC_SIGMA noise. Records contact with the arena walls (truth)."""

    def __init__(self, asm, params, scene, dom, mode, g, seed=0, device="cpu"):
        self.asm, self.params, self.scene, self.dom, self.mode, self.g = asm, params, scene, dom, mode, g
        self.seed, self.device = seed, device
        self.rng = np.random.default_rng([seed, 99])
        self.true = np.array(scene.start, float)
        self.buf, self.events, self.collided, self.n_twist = [], [], False, 0
        self.pts = DK.footprint_points(asm, 0.02)
        self.t = 0.0

    def twist(self, xi):
        self.n_twist += 1
        self.buf.append(xi)

    def event(self, d):
        self.events.append(d)

    def sleep(self, dt):
        self.t += dt

    def clock(self):
        return self.t

    def _flush(self):
        if not any(abs(v) > 1e-9 for xi in self.buf for v in xi):
            self.buf = []
            return
        from .. import sim
        sched = SampledSchedule(self.buf)
        d = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in self.dom.items()}
        R = sim.simulate(self.asm, self.params, d, self.mode, self.g, schedule=sched, device=self.device, seed=self.seed)
        poses = DK.compose(self.true, R["pose"][0].cpu().numpy().astype(float))
        if DK.collides(self.scene, self.pts, poses):
            self.collided = True
        self.true = poses[-1]
        self.buf = []

    def still(self):
        self._flush()
        return True

    def fix(self, duration):
        self._flush()
        self.t += duration
        return tuple(self.true + self.rng.normal(0, np.asarray(DK.LOC_SIGMA)))

    def outcome(self):
        g = np.array(self.scene.goal)
        pe = float(math.hypot(*(self.true[:2] - g[:2])))
        he = float(abs(DK.wrap(self.true[2] - g[2])))
        return dict(pos_err=pe, head_err=he, collision=self.collided,
                    success=bool(not self.collided and pe <= DK.POS_TOL and he <= DK.HEAD_TOL))


def simulate_dock(asm, params, arena, fault="F4x2d", offset_m=0.0, seed=0, loop="closed", capset="omni", log=None):
    """One sim dock through the SAME DockExecutor (SimIO). fault: plan.dock.fault_spec kind. -> (result, outcome)."""
    rng = np.random.default_rng(seed)
    scene = arena_scene(asm, arena, start_pose(asm, arena, offset_m))
    mode, g = DK.fault_spec(asm, fault, rng=rng)
    dom = DK.nominal_domain()
    io = SimIO(asm, params, scene, dom, mode, g, seed=seed)
    ex = DockExecutor(io, asm, scene, capset_from(capset, asm), loop=loop, log=log, clock=io.clock, sleep=io.sleep)
    res = ex.run(start=scene.start)
    res["events"] = [e["event"] for e in io.events]
    return res, io.outcome()


# ================================================================================================ ROS IO
def check_pose_topic(topic):
    """The closed loop must never see the evaluator-only AprilTag truth."""
    if "truth" in topic.lower():
        raise SystemExit(f"refusing pose topic {topic}: the AprilTag truth (/fleet/truth/*) is evaluator-only")
    return topic


class RosIO:
    """rclpy publishers / subscribers (lazy import). Never subscribes to /fleet/truth."""

    def __init__(self, node, member_ids, pose_topic, frames):
        check_pose_topic(pose_topic)
        from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        import rclpy
        self.rclpy, self.node, self.TwistStamped, self.String = rclpy, node, TwistStamped, String
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.p_cmd = [node.create_publisher(TwistStamped, f"/fleet/{i}/cmd_vel", 10) for i in member_ids.values()]
        self.p_log = node.create_publisher(TwistStamped, "/capxfer/cmd_vel", 10)
        self.p_ev = node.create_publisher(String, "/capxfer/dock/event", 10)
        p_asm = {m: node.create_publisher(String, f"/fleet/{i}/assembly", latched) for m, i in member_ids.items()}
        for m, fr in frames.items():
            msg = String()
            msg.data = json.dumps(fr)
            p_asm[m].publish(msg)
        self.poses = []
        self.vel = {}
        self.sub_pose = node.create_subscription(PoseWithCovarianceStamped, pose_topic, self._pose, 10)
        self.sub_w = [node.create_subscription(JointState, f"/fleet/{i}/wheel_state",
                                               lambda msg, i=i: self._wheel(i, msg), qos_profile_sensor_data)
                      for i in member_ids.values()]
        self.ids = list(member_ids.values())

    def _now(self):
        return time.monotonic()

    def _pose(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.poses.append((self._now(), p.x, p.y, 2 * math.atan2(q.z, q.w)))
        if len(self.poses) > 500:
            del self.poses[:250]

    def _wheel(self, fid, msg):
        self.vel[fid] = (self._now(), [abs(v) for v in msg.velocity])

    def sleep(self, dt):
        self.rclpy.spin_once(self.node, timeout_sec=max(0.0, dt))

    def twist(self, xi):
        msg = self.TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "assembly"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = (float(v) for v in xi)
        for p in self.p_cmd:
            p.publish(msg)
        self.p_log.publish(msg)

    def event(self, d):
        msg = self.String()
        msg.data = json.dumps(dict(d, t_ns=int(self.node.get_clock().now().nanoseconds)), default=float)
        self.p_ev.publish(msg)

    def still(self):
        now = self._now()
        for fid in self.ids:
            if fid not in self.vel or now - self.vel[fid][0] > 0.3:
                return False
            if max(self.vel[fid][1] or [0.0]) >= STILL_RADPS:
                return False
        return True

    def fix(self, duration):
        t0 = self._now()
        while self._now() - t0 < duration:
            self.sleep(0.02)
        got = [p for p in self.poses if p[0] >= t0]
        if not got:
            return None
        a = np.array(got)
        return (float(a[:, 1].mean()), float(a[:, 2].mean()), _circ_mean(a[:, 3]))


# ================================================================================================ CLI
def parse_ids(s):
    ids = {m: m.lower() for m in ("M1", "M2", "M3", "M4")}
    for kv in (s or "").split(","):
        if kv:
            k, v = kv.split("=")
            ids[k] = v
    return ids


def build_assembly(module=None, assembly=None, params=None):
    params = params or A.load_params(PARAMS_PATH)
    if module:
        return A.build("single", [(module, 0, 0, 0)], params), params
    return A.physical(assembly, params), params


def timeline_text(asm, scene, capset, loop="closed"):
    """Dry run: the plan from the start mark and the 50 Hz command timeline of the first chunk (closed loop) or of the
    whole plan (open loop)."""
    plan = DK.plan_dock(capset, asm, scene, start=scene.start)
    L = [f"assembly {asm.name} ({', '.join(m for m, *_ in asm.modules)}); capset {capset.name}; loop {loop}",
         f"start {np.round(scene.start, 3).tolist()} -> goal {np.round(scene.goal, 3).tolist()}; bay "
         f"{scene.meta['ws']:.3f} x {scene.meta['depth']:.3f} m at x = {scene.meta['arena']['bay_x']:.3f}",
         f"plan: {plan.status} route {plan.route() if plan.status == 'ok' else 'none'}; {len(plan.cmd)} segments, "
         f"{plan.duration:.2f} s; {plan.expansions} expansions"]
    t = 0.0
    for j, (c, T) in enumerate(plan.cmd):
        L.append(f"  seg {j:2d}  t {t:6.2f}-{t + T:6.2f} s  vx {c[0]:+.3f}  vy {c[1]:+.3f}  wz {c[2]:+.3f}")
        t += T
    if plan.status == "ok" and plan.cmd:
        cmds = plan.cmd if loop == "open" else DK._prefix(plan)
        sch = DK.CommandSchedule(cmds)
        n = int(math.ceil(sch.T * RATE_HZ))
        tw = sch.twist(np.arange(n) / RATE_HZ)
        L.append(f"{'plan' if loop == 'open' else 'first chunk'}: {len(cmds)} segment(s), {n} samples at {RATE_HZ:.0f} Hz "
                 f"({sch.T:.2f} s incl. {DK.RAMP} s ramps and {DK.REST} s rest); then stop, wait still, fix, replan")
        for k in range(0, n, max(1, n // 25)):
            L.append(f"  t {k / RATE_HZ:6.2f} s  cmd ({tw[k, 0]:+.3f}, {tw[k, 1]:+.3f}, {tw[k, 2]:+.3f})")
    L.append("e-stop: Ctrl-C -> zero twist x10 over 0.2 s, 'abort' event")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--module", help="dock one lone module (DEMO)")
    g.add_argument("--assembly", help="dock a physical assembly type (old H4 card 04)")
    ap.add_argument("--ids", default="")
    ap.add_argument("--arena")
    ap.add_argument("--loop", choices=("closed", "open"), default="closed")
    ap.add_argument("--capset", default="omni")
    ap.add_argument("--robot", help="namespace of the localisation fix (default: fleet id of the (first) module)")
    ap.add_argument("--pose-topic")
    ap.add_argument("--start-offset", type=float, default=0.0, help="sealed start offset (m); used for the dry run / "
                                                                   "open loop without localisation")
    ap.add_argument("--episode", default="")
    ap.add_argument("--dock", default="")
    ap.add_argument("--arm-code", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--log-dir")
    ap.add_argument("--params", default=str(PARAMS_PATH))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--simulate", help="dry run: also run the closed loop in capxfer.sim with this fault (F0, F4, F4x2d)")
    a = ap.parse_args(argv)
    asm, params = build_assembly(a.module, a.assembly, A.load_params(a.params))
    arena = load_arena(a.arena, asm) if a.arena else default_arena(asm, variant="narrow" if a.module else "wide")
    scene = arena_scene(asm, arena, start_pose(asm, arena, a.start_offset))
    capset = capset_from(a.capset, asm)
    if a.dry_run:
        print(timeline_text(asm, scene, capset, a.loop))
        if a.simulate:
            res, out = simulate_dock(asm, params, arena, a.simulate, a.start_offset, loop=a.loop, capset=a.capset)
            print(f"sim ({a.simulate}): {res['status']} after {res['replans']} plan(s), {res['t_cmd']:.1f} s commanded; "
                  f"truth: success={out['success']} pos_err={out['pos_err']:.3f} m head_err={math.degrees(out['head_err']):.1f} "
                  f"deg contact={out['collision']}; events {res['events']}")
        return 0
    if not a.log_dir:
        ap.error("--log-dir is required on the robot")
    ids = parse_ids(a.ids)
    members = {m: ids[m] for m, *_ in asm.modules}
    frames = {m: {"x": float(p[0]), "y": float(p[1]), "yaw_deg": 90.0 * int(r)}
              for (m, _, _, r), p in zip(asm.modules, asm.module_pos)}
    robot = a.robot or next(iter(members.values()))
    pose_topic = check_pose_topic(a.pose_topic or f"/{robot}/dock/pose")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"dock_{a.episode or 'X'}_{a.dock or 0}_{a.module or a.assembly}_{stamp}"
    log_dir = Path(a.log_dir)
    if (log_dir / "session.yaml").exists() or any(q.name == "session.yaml" for q in log_dir.parents):
        sys.exit("refusing to log inside a FieldLog session folder")
    meta = dict(run=a.run, episode=a.episode, module=a.module or a.assembly, arm_code=a.arm_code, dock=a.dock)
    log = DockLog(log_dir / f"{tag}.csv", meta)
    try:
        import rclpy
        from rclpy.signals import SignalHandlerOptions
    except ImportError:
        sys.exit("rclpy not found: source ROS 2 first (or use --dry-run)")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("capxfer_dock_executor")
    io = RosIO(node, members, pose_topic, frames)
    ex = DockExecutor(io, asm, scene, capset, loop=a.loop, log=log, plan_dir=log_dir, plan_tag=f"{tag}_plan")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, ex.request_stop)
    try:
        res = ex.run(start=scene.start, meta=meta)
        print(f"{res['status']}  (plans {res['replans']}, commanded {res['t_cmd']:.1f} s, last fix {res['last_fix']}); "
              f"log {log.path}")
    except KeyboardInterrupt:
        ex.estop()
    finally:
        log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
