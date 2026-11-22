"""Probe/fault trial runner (docs/TEAM_T0_HANDOFF.md §2-§3). People start and stop it; the e-stop works at all times.

  python -m capxfer.field.probe_runner SCHEDULE.json --assembly pair [--from-trial N] [--to-trial M] [--ids M1=m1,M2=m2,...]
         [--gap 3.0] [--confirm] [--dry-run] [--no-relay]
  python -m capxfer.field.probe_runner SCHEDULE.json --session S03 [--from-block POS] [--to-block POS] [--from-trial N]
         --ids ... --confirm
  python -m capxfer.field.probe_runner SCHEDULE.json --set-faults N | --clear-faults N --ids ...   (WS-P2 DEMO docks:
         hold / clear the firmware faults of scheduled trial N, no motion)
         (WS-P 2026-11-21: a whole mixed-type field session in run order; one FieldLog session spans the rebuilds)

For each trial of the chosen assembly session (sealed schedule; its sha256 is checked against schedule.sha256):
  t = 0       trial marker {"trial", "event": "start", "asm_type", "t_ns"} on /capxfer/trial; assembly frame JSON
              {"x", "y", "yaw_deg"} on /fleet/<id>/assembly for every member; fault_cmd JSON {"wheel", "mode", "g", "trial"} on
              /fleet/<id>/fault_cmd for every scheduled fault (operator conditions roller_locked / payload / low_mu are logged on
              the same topic; the firmware ignores them)
  0..T        cmd_vel (TwistStamped, assembly frame) at 50 Hz from probes.Schedule(asm.char_length()) to every member module
              (a trial with "probes": [P1, P5, P2] runs only those primitives, 12.7 s; schedule.trial_schedule, WS-P2)
              (the schedule starts with the 0.5 s settle after the fault is set)
  T           fault_cmd "ok" for every faulted wheel, marker "end"
then a rest gap (or, with --confirm, wait for Enter so the operator can re-centre the robot).

Session mode (--session Sxx): the blocks of field session Sxx in run order (schedule.field_sessions / blocks_Sxx.csv). Before
every block it prints the block instruction and waits for Enter; when the assembly type changes it first prints the rebuild
(module layout) and waits. Every start marker then also carries "block" (position in the session), "session" (the number
of Sxx), "imu" and "lidar" (sensor module slot numbers 1-4, 0 = none); the first trial after a rebuild is preceded by an
"assembly" marker and the first trial of a block by a "block" marker (same trial number, before "start"). The firmware
receives the new /fleet/<id>/assembly frames at the rebuild and again at every trial start.
Mixed-type sessions in ONE FieldLog session (WS-P): the runner also publishes the assembly twist on /capxfer/cmd_vel and
relays the IMU (/fleet/<id>/imu/data) and rf2o odometry (/fleet/<id>/odom_rf2o) of the active assembly's sensor module
(schedule.sensor_modules) to /capxfer/active/imu and /capxfer/active/odom_rf2o, which the FieldLog config logs as src 3
and 42. The relay switches before each start marker, so every trial logs its own assembly's sensor module. --no-relay
turns it off (then the FieldLog config must point at the module topics directly).

E-stop: SIGINT/SIGTERM (Ctrl-C) stops the schedule immediately, publishes a zero twist 10 times over 0.2 s to every member,
an "abort" marker and "ok" fault commands, then exits. (The firmware watchdog also zeroes the wheels 0.25 s after the
last command.) --dry-run prints the timeline without ROS. rclpy is imported lazily, so this module imports without ROS.
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .. import assembly as A
from . import schedule as S

CMD_HZ = 50.0
PARAMS_PATH = Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml"
OPERATOR_MODES = ("roller_locked", "payload", "low_mu")


def split_wheel(w):
    """'M2:fl' -> ('M2', 'fl'); 'none' -> (None, 'none')."""
    if ":" in w:
        m, c = w.split(":", 1)
        return m, c
    return None, w


def fault_messages(trial, session, restore=False):
    """[(module_id, json_dict)] fault commands for a trial (restore=True: the 'ok' commands sent at its end)."""
    lead = session["modules"][0][0]
    out = []
    for f in trial["faults"]:
        m, c = split_wheel(f["wheel"])
        m = m or f.get("module") or lead
        d = {"wheel": c, "mode": "ok" if restore else f["mode"], "trial": int(trial["trial"])}
        if not restore:
            for k in ("g", "mass_kg", "region", "mu", "dx_m", "dy_m", "module"):
                if k in f:
                    d[k] = f[k]
        out.append((m, d))
    return out


def trial_plan(trial, session, params):
    """Everything a trial publishes, as data: dict(asm, schedule, T, start=[(kind, module, payload)], end=[...])."""
    asm = A.build(session["assembly"], [tuple(m) for m in session["modules"]], params)
    sch = S.trial_schedule(asm, trial)                   # full library, or the trial's "probes" subset (WS-P2)
    asm_type = session["assembly"].lower()
    start = [("marker", None, {"trial": int(trial["trial"]), "event": "start", "asm_type": asm_type})]
    start += [("assembly", fr["module"], {"x": fr["x"], "y": fr["y"], "yaw_deg": fr["yaw_deg"]}) for fr in session["frames"]]
    start += [("fault", m, d) for m, d in fault_messages(trial, session)]
    end = [("fault", m, d) for m, d in fault_messages(trial, session, restore=True)]
    end += [("marker", None, {"trial": int(trial["trial"]), "event": "end", "asm_type": asm_type})]
    return dict(asm=asm, schedule=sch, T=sch.T, start=start, end=end, members=[m[0] for m in session["modules"]])


def timeline_text(trials, session, params, gap):
    lines, t0 = [], 0.0
    for tr in trials:
        p = trial_plan(tr, session, params)
        lines.append(f"[{t0:8.2f}s] trial {tr['trial']} {tr['fault_id']} faults={tr['faults']}"
                     + (f" probes={'+'.join(tr['probes'])}" if tr.get("probes") else ""))
        for kind, m, d in p["start"]:
            lines.append(f"    +0.00 {kind:8s} {m or '-':4s} {json.dumps(d, sort_keys=True)}")
        for s in p["schedule"].segments:
            lines.append(f"    +{s.t0:5.2f} cmd_vel  {s.pid}{'+' if s.leg > 0 else '-'} hold {np.round(s.twist, 3).tolist()} "
                         f"[{s.hold0:.2f}, {s.hold1:.2f}]")
        for kind, m, d in p["end"]:
            lines.append(f"    +{p['T']:5.2f} {kind:8s} {m or '-':4s} {json.dumps(d, sort_keys=True)}")
        t0 += p["T"] + gap
    lines.append(f"total {len(trials)} trials, ~{t0 / 60:.1f} min")
    return "\n".join(lines)


class Runner:
    """ROS-agnostic trial loop. `pub` implements marker(d), assembly(module, d), fault(module, d), twist(modules, xyz)."""

    def __init__(self, pub, params, clock=time.monotonic, sleep=time.sleep, spin=lambda: None, confirm=None):
        self.pub, self.params, self.clock, self.sleep, self.spin = pub, params, clock, sleep, spin
        self.confirm = confirm
        self.stop = False
        self.members = []
        self.active_trial = None

    def request_stop(self, *_):
        self.stop = True

    def _emit(self, items):
        for kind, m, d in items:
            if kind == "marker":
                self.pub.marker(dict(d))
            elif kind == "assembly":
                self.pub.assembly(m, dict(d))
            else:
                self.pub.fault(m, dict(d))

    def estop(self):
        for _ in range(10):
            self.pub.twist(self.members, (0.0, 0.0, 0.0))
            self.sleep(0.02)
        if self.active_trial is not None:
            tr, sess = self.active_trial
            self.pub.marker({"trial": int(tr["trial"]), "event": "abort", "asm_type": sess["assembly"].lower()})
            self._emit([("fault", m, d) for m, d in fault_messages(tr, sess, restore=True)])
        self.active_trial = None

    def run_trial(self, trial, session, context=None):
        """context (session mode, optional): {"block", "session", "first_of_block", "assembly_changed"}. Without it the
        published markers are exactly the single-assembly ones."""
        p = trial_plan(trial, session, self.params)
        self.members = p["members"]
        self.active_trial = (trial, session)
        imu_m, lidar_m = session.get("imu_module"), session.get("lidar_module")
        if imu_m is None:
            imu_m, lidar_m = S.sensor_modules(p["asm"])
        if hasattr(self.pub, "sensors"):
            self.pub.sensors(imu_m, lidar_m)                                  # relay switches before the start marker
        start = p["start"]
        if context:
            base = {"trial": int(trial["trial"]), "asm_type": session["assembly"].lower(), "block": int(context["block"]),
                    "session": int(context["session"]), "imu": slot_code(imu_m), "lidar": slot_code(lidar_m)}
            pre = []
            if context.get("assembly_changed"):
                pre.append(("marker", None, dict(base, event="assembly")))
            if context.get("first_of_block"):
                pre.append(("marker", None, dict(base, event="block")))
            start = pre + [("marker", None, dict(base, event="start"))] + [x for x in start if x[0] != "marker"]
        self._emit(start)
        t0 = self.clock()
        k = 0
        n = int(np.ceil(p["T"] * CMD_HZ)) + 1
        while k < n:
            if self.stop:
                self.estop()
                return False
            tk = k / CMD_HZ
            now = self.clock() - t0
            if now < tk:
                self.sleep(min(tk - now, 1.0 / CMD_HZ))
                self.spin()
                continue
            xi = p["schedule"].twist(np.array([min(tk, p["T"])]))[0]
            self.pub.twist(self.members, tuple(float(v) for v in xi))
            k += 1
        self.pub.twist(self.members, (0.0, 0.0, 0.0))
        end = p["end"]
        if context:
            end = [x for x in end if x[0] != "marker"] + [("marker", None, dict(trial=int(trial["trial"]), event="end",
                                                                                  asm_type=session["assembly"].lower(),
                                                                                  block=int(context["block"]),
                                                                                  session=int(context["session"])))]
        self._emit(end)
        self.active_trial = None
        return True

    def run(self, trials, session, gap=3.0, contexts=None):
        done = 0
        for i, tr in enumerate(trials):
            if self.stop:
                break
            if self.confirm is not None and not self.confirm(tr):
                break
            if not self.run_trial(tr, session, contexts[i] if contexts else None):
                break
            done += 1
            t_end = self.clock() + (gap if i < len(trials) - 1 else 0.0)
            while self.clock() < t_end and not self.stop:
                self.pub.twist(self.members, (0.0, 0.0, 0.0))
                self.sleep(1.0 / CMD_HZ)
                self.spin()
        if self.stop:
            self.estop()
        return done

    def run_session(self, plan, gap=3.0, confirm_block=None):
        """plan = session_plan(...): run every block in order; confirm_block(block) is asked before each block (the
        operator does the rebuild / physical change first). Returns the number of trials run."""
        done = 0
        for blk in plan:
            if self.stop:
                break
            if confirm_block is not None and not confirm_block(blk):
                break
            if blk["reassemble"] and hasattr(self.pub, "assembly"):
                for fr in blk["session"]["frames"]:                          # new frames as soon as the rebuild is done
                    self.pub.assembly(fr["module"], {"x": fr["x"], "y": fr["y"], "yaw_deg": fr["yaw_deg"]})
            ctx = [dict(block=blk["pos"], session=blk["session_num"], first_of_block=(i == 0),
                        assembly_changed=(i == 0 and blk["reassemble"])) for i in range(len(blk["trials"]))]
            n = self.run(blk["trials"], blk["session"], gap=gap, contexts=ctx)
            done += n
            if n < len(blk["trials"]):
                break
        return done


def slot_code(m):
    """'M3' -> 3, None -> 0 (marker / CSV code of a module slot)."""
    return int(m[1:]) if m else 0


def session_plan(sched, field_session_id, from_block=None, from_trial=None):
    """Blocks of one field session in run order: [{pos, block, assembly, reassemble, session (schedule entry),
    session_num, trials, instruction, fault_id}]. from_block / from_trial resume after an interruption (the first
    resumed block is treated as a rebuild so the frames and the assembly marker are re-sent)."""
    fs = next((f for f in S.field_sessions(sched) if f["id"] == field_session_id), None)
    if fs is None:
        raise SystemExit(f"no field session {field_session_id!r} in the schedule "
                         f"({[f['id'] for f in S.field_sessions(sched)]})")
    byb = {(s["assembly"], b["block"]): b for s in sched["sessions"] for b in s["blocks"]}
    ent = {(s["assembly"], b["block"]): s for s in sched["sessions"] for b in s["blocks"]}   # WS-P2: a type may appear
    #                                                     with several layouts (e.g. M1 alone, M2 alone after a split)
    num = int("".join(c for c in field_session_id if c.isdigit()) or 0)
    out, started = [], from_block is None and from_trial is None
    for r in fs["blocks"]:
        b = byb[(r["assembly"], r["block"])]
        trials = list(b["trials"])
        if not started:
            if from_block is not None and int(r["pos"]) < int(from_block):
                continue
            if from_trial is not None:
                trials = [t for t in trials if int(t["trial"]) >= int(from_trial)]
                if not trials:
                    continue
            started, resumed = True, True
        else:
            resumed = False
        out.append(dict(pos=int(r["pos"]), block=int(r["block"]), assembly=r["assembly"],
                        reassemble=bool(r["reassemble"]) or resumed, session=ent[(r["assembly"], r["block"])], session_num=num,
                        trials=trials, fault_id=r.get("fault_id"),
                        instruction=b.get("condition", {}).get("instruction", "")))
    return out


def layout_text(session):
    """Top view of the assembly, module fronts (+x) up the page, +y to the left."""
    cells = {(int(i), int(j)): m for m, i, j, _ in session["modules"]}
    mi, mj = max(i for i, _ in cells), max(j for _, j in cells)
    rows = [" ".join(f"[{cells[(i, j)]}]" if (i, j) in cells else "    " for j in range(mj, -1, -1)) for i in range(mi, -1, -1)]
    return "\n".join("      " + r for r in rows) + "\n      (all module fronts point up the page = +x; +y = left)"


def session_text(plan, params, gap):
    lines, t, n = [], 0.0, 0
    for blk in plan:
        if blk["reassemble"]:
            lines.append(f"== REBUILD -> {blk['assembly']}  (sensor modules imu/lidar: {blk['session']['imu_module']}/"
                         f"{blk['session']['lidar_module']})")
            lines.append(layout_text(blk["session"]))
        tn = [x["trial"] for x in blk["trials"]]
        lines.append(f"-- block pos {blk['pos']} (schedule block {blk['block']}, {blk['fault_id'] or 'auto'}): trials "
                     f"{min(tn)}-{max(tn)} ({len(tn)})  {blk['instruction']}")
        for tr in blk["trials"]:
            T = trial_plan(tr, blk["session"], params)["T"]
            lines.append(f"   [{t:8.1f}s] trial {tr['trial']} {tr.get('condition', tr['fault_id'])} faults={tr['faults']}")
            t += T + gap
            n += 1
    lines.append(f"total {n} trials in {len(plan)} blocks, {sum(b['reassemble'] for b in plan)} rebuild(s), "
                 f"~{t / 60:.1f} min of probing")
    return "\n".join(lines)


class RosPub:
    """rclpy publishers (lazy import)."""

    def __init__(self, node, ids):
        from geometry_msgs.msg import TwistStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.node, self.ids, self.String, self.TwistStamped = node, ids, String, TwistStamped
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.p_marker = node.create_publisher(String, "/capxfer/trial", 10)
        self.p_fault = {m: node.create_publisher(String, f"/fleet/{i}/fault_cmd", 10) for m, i in ids.items()}
        self.p_asm = {m: node.create_publisher(String, f"/fleet/{i}/assembly", latched) for m, i in ids.items()}
        self.p_cmd = {m: node.create_publisher(TwistStamped, f"/fleet/{i}/cmd_vel", 10) for m, i in ids.items()}
        self.p_cmd_log = node.create_publisher(TwistStamped, "/capxfer/cmd_vel", 10)   # the logged copy (FieldLog src 7)
        self.relay = None

    def _now(self):
        return self.node.get_clock().now()

    def _str(self, pub, d):
        msg = self.String()
        msg.data = json.dumps(d)
        pub.publish(msg)

    def marker(self, d):
        d["t_ns"] = int(self._now().nanoseconds)
        self._str(self.p_marker, d)

    def assembly(self, m, d):
        self._str(self.p_asm[m], d)

    def fault(self, m, d):
        self._str(self.p_fault[m], d)

    def twist(self, modules, xi):
        msg = self.TwistStamped()
        msg.header.stamp = self._now().to_msg()
        msg.header.frame_id = "assembly"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = xi
        for m in modules:
            self.p_cmd[m].publish(msg)
        self.p_cmd_log.publish(msg)

    def sensors(self, imu_m, lidar_m):
        if self.relay is not None:
            self.relay.select(self.ids.get(imu_m) if imu_m else None, self.ids.get(lidar_m) if lidar_m else None)


class SensorRelay:
    """Republishes the active assembly's IMU and rf2o odometry on fixed topics (/capxfer/active/imu, /capxfer/active/
    odom_rf2o) so one FieldLog session can span rebuilds. Own node, own executor thread; messages are forwarded
    unchanged (header stamps kept)."""

    def __init__(self, fleet_ids):
        import threading
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        self.node = rclpy.create_node("capxfer_sensor_relay")
        self.imu_id = self.lidar_id = None
        self.n = {"imu": 0, "odom": 0}
        self.p_imu = self.node.create_publisher(Imu, "/capxfer/active/imu", qos_profile_sensor_data)
        self.p_odom = self.node.create_publisher(Odometry, "/capxfer/active/odom_rf2o", 10)
        self.subs = []
        for fid in sorted(set(fleet_ids)):
            self.subs.append(self.node.create_subscription(Imu, f"/fleet/{fid}/imu/data",
                                                           lambda m, f=fid: self._imu(f, m), qos_profile_sensor_data))
            self.subs.append(self.node.create_subscription(Odometry, f"/fleet/{fid}/odom_rf2o",
                                                           lambda m, f=fid: self._odom(f, m), 10))
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.thread = threading.Thread(target=self.ex.spin, daemon=True)
        self.thread.start()

    def select(self, imu_id, lidar_id):
        self.imu_id, self.lidar_id = imu_id, lidar_id

    def _imu(self, fid, msg):
        if fid == self.imu_id:
            self.p_imu.publish(msg)
            self.n["imu"] += 1

    def _odom(self, fid, msg):
        if fid == self.lidar_id:
            self.p_odom.publish(msg)
            self.n["odom"] += 1

    def close(self):
        self.ex.shutdown()
        self.node.destroy_node()


def parse_ids(s):
    ids = {m: m.lower() for m in ("M1", "M2", "M3", "M4")}
    for kv in (s or "").split(","):
        if kv:
            k, v = kv.split("=")
            ids[k] = v
    return ids


def select_trials(sched, assembly, first=None, last=None):
    sess = [s for s in sched["sessions"] if s["assembly"] == assembly]
    if not sess:
        raise SystemExit(f"assembly {assembly!r} not in the schedule ({[s['assembly'] for s in sched['sessions']]})")
    trials = [t for _, _, t in S.iter_trials(sched, assembly)
              if (first is None or t["trial"] >= first) and (last is None or t["trial"] <= last)]
    return sess[0], trials


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assembly", help="run trials of one assembly type (single-type mode)")
    g.add_argument("--session", help="run field session Sxx in run order (mixed-type mode, rebuild prompts)")
    g.add_argument("--set-faults", type=int, metavar="TRIAL",
                   help="WS-P2 DEMO docks: publish the firmware fault commands of scheduled trial TRIAL and exit (the "
                        "faults stay set while the dock executor drives; TEAM-CONFIRM firmware latching); no motion")
    g.add_argument("--clear-faults", type=int, metavar="TRIAL", help="publish 'ok' for every wheel faulted in TRIAL and exit")
    ap.add_argument("--from-trial", type=int)
    ap.add_argument("--to-trial", type=int)
    ap.add_argument("--from-block", type=int, help="session mode: resume at this block position")
    ap.add_argument("--to-block", type=int, help="session mode: stop after this block position")
    ap.add_argument("--ids", default="", help="module id -> fleet id, e.g. M1=m1,M2=m2")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--confirm", action="store_true", help="wait for Enter before every trial")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-relay", action="store_true", help="do not relay the active IMU / rf2o (see the docstring)")
    ap.add_argument("--params", default=str(PARAMS_PATH))
    a = ap.parse_args(argv)
    sched = S.load(a.schedule)
    params = A.load_params(a.params)
    if a.set_faults is not None or a.clear_faults is not None:
        num = a.set_faults if a.set_faults is not None else a.clear_faults
        hit = [(x, t) for x, _, t in S.iter_trials(sched) if int(t["trial"]) == int(num)]
        if not hit:
            raise SystemExit(f"trial {num} not in the schedule")
        sess, tr = hit[0]
        msgs = fault_messages(tr, sess, restore=a.clear_faults is not None)
        msgs = [(m, d) for m, d in msgs if d["mode"] not in OPERATOR_MODES]
        for m, d in msgs:
            print(f"fault_cmd {m} {json.dumps(d, sort_keys=True)}")
        if a.dry_run or not msgs:
            return 0
        try:
            import rclpy
        except ImportError:
            sys.exit("rclpy not found: source ROS 2 first (or use --dry-run)")
        rclpy.init()
        node = rclpy.create_node("capxfer_fault_hold")
        ids = parse_ids(a.ids)
        pub = RosPub(node, {m: ids[m] for m in {m for m, _ in msgs}})
        for _ in range(3):                                   # latched topic + repeats; the echo confirms
            for m, d in msgs:
                pub.fault(m, dict(d))
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        return 0
    if a.session:
        plan = session_plan(sched, a.session, a.from_block, a.from_trial)
        if a.to_block is not None:
            plan = [b for b in plan if b["pos"] <= a.to_block]
        members = sorted({m[0] for b in plan for m in b["session"]["modules"]})
    else:
        session, trials = select_trials(sched, a.assembly, a.from_trial, a.to_trial)
        members = [m[0] for m in session["modules"]]
    if a.dry_run:
        print(session_text(plan, params, a.gap) if a.session else timeline_text(trials, session, params, a.gap))
        return 0
    try:
        import rclpy
        from rclpy.signals import SignalHandlerOptions
    except ImportError:
        sys.exit("rclpy not found: source ROS 2 first (or use --dry-run)")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("capxfer_probe_runner")
    ids = parse_ids(a.ids)
    pub = RosPub(node, {m: ids[m] for m in members})
    if not a.no_relay:
        pub.relay = SensorRelay([ids[m] for m in members])

    def ask(prompt):
        old = signal.signal(signal.SIGINT, signal.default_int_handler)     # Ctrl-C must interrupt input() at once
        try:
            input(prompt)
            return True
        except (EOFError, KeyboardInterrupt):
            runner.request_stop()
            return False
        finally:
            signal.signal(signal.SIGINT, old)

    def confirm(tr):
        return ask(f"trial {tr['trial']} ({tr.get('condition', tr['fault_id'])} {tr['faults']}): trial tool 's' first, "
                   f"then Enter to run, Ctrl-C to stop ")

    def confirm_block(blk):
        if blk["reassemble"]:
            print(f"\n=== REBUILD to {blk['assembly']} ===\n{layout_text(blk['session'])}")
        tn = [t["trial"] for t in blk["trials"]]
        print(f"--- block pos {blk['pos']} ({blk['fault_id'] or 'auto'}), trials {min(tn)}-{max(tn)}: {blk['instruction']}")
        return ask("physical change done and checked by both people? Enter to continue, Ctrl-C to stop ")
    runner = Runner(pub, params, spin=lambda: rclpy.spin_once(node, timeout_sec=0.0), confirm=confirm if a.confirm else None)
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, runner.request_stop)
    try:
        if a.session:
            n = runner.run_session(plan, gap=a.gap, confirm_block=confirm_block)
            node.get_logger().info(f"ran {n}/{sum(len(b['trials']) for b in plan)} trials of {a.session}")
        else:
            n = runner.run(trials, session, gap=a.gap)
            node.get_logger().info(f"ran {n}/{len(trials)} trials")
        if pub.relay is not None:
            node.get_logger().info(f"relayed {pub.relay.n['imu']} IMU and {pub.relay.n['odom']} rf2o messages")
    except KeyboardInterrupt:
        runner.estop()
    finally:
        if pub.relay is not None:
            pub.relay.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
