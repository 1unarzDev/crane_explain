"""FieldLog v2 layout for the capxfer mecanum study (single source for the config, the loader and the synthetic session).

`build_config()` produces the dict written to ~/crane_ml/Tools/FieldLog/configs/capxfer_mecanum.yaml (`write_config`).
Topic names follow the fleet spec (/fleet/<id>/..., Tools/Fleet/spec/ros2_test_interface_v0.md) with {PARAM}
placeholders the team confirms (TEAM-CONFIRM); `render()` substitutes them into a concrete config for a run.

The official codebook_v2.yaml (FieldLog-owned, not edited here) has no capxfer sources, so the config carries a
`codebook_extension:` block (extra key, ignored by ros2_csvlog_v2.py). `overlay_codebook()` merges it into a copy of the
official codebook, like sim_ground/gnd_codebook.py does, until the FieldLog owner merges it (request in docs/coordination/B.md).
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

FIELDLOG = Path.home() / "crane_ml" / "Tools" / "FieldLog"
OFFICIAL_CODEBOOK = FIELDLOG / "codebook_v2.yaml"
CONFIG_PATH = FIELDLOG / "configs" / "capxfer_mecanum.yaml"

MODULES = ("M1", "M2", "M3", "M4")
CORNERS = ("fl", "fr", "rl", "rr")
PORTION_TOKEN, PORTION_CODE = "capxprobe", 67
SRC = dict(CTL=7, IMU=3, MARK=8, ENTRY=9, TRIAL=40, TRUTH=41, LODOM=42,
           **{f"WHL_{m}": 21 + k for k, m in enumerate(MODULES)},
           **{f"CUR_{m}": 25 + k for k, m in enumerate(MODULES)},
           **{f"FLT_{m}": 31 + k for k, m in enumerate(MODULES)},
           **{f"ASM_{m}": 35 + k for k, m in enumerate(MODULES)})
FAULT_MODE_CODES = {"ok": 0, "weak": 1, "coast": 2, "brake": 3, "roller_locked": 4, "payload": 5, "low_mu": 6}
WHEEL_CODES = {c: k for k, c in enumerate(CORNERS)}
TRIAL_EVENT_CODES = {"start": 1, "end": 2, "abort": 3, "assembly": 4, "block": 5}   # 4/5: session mode (WS-P)
ASM_TYPE_CODES = {"single": 1, "pair": 2, "l3": 3, "square4": 4}
PARAMS = {   # {PARAM} -> default; TEAM-CONFIRM every value
    "M1": "m1", "M2": "m2", "M3": "m3", "M4": "m4",       # fleet ids of the physical modules
    "LEAD": "m1",                                          # unused since WS-P (cmd logged from /capxfer/cmd_vel); kept for old renders
    "IMU_MODULE": "m1", "LIDAR_MODULE": "m1",              # unused since WS-P (the runner relays the active module's IMU/rf2o)
    "TRUTH_ID": "capxfer_asm",                             # AprilTag truth id of the assembly (tag on the assembly origin, else set tag offset in the loader)
}


def mk(m):
    return m.lower()


def wheel_cols(m):
    return [f"cnt_{c}_{mk(m)}" for c in CORNERS] + [f"wspd_{c}_{mk(m)}" for c in CORNERS] + [f"duty_{c}_{mk(m)}" for c in CORNERS]


def cur_cols(m):
    return [f"cur_{c}_{mk(m)}" for c in CORNERS]


def flt_cols(m):
    return [f"fs_{k}_{mk(m)}" for k in ("wheel_code", "mode_code", "g", "trial", "tick")]


def asm_cols(m):
    return [f"asm_{k}_{mk(m)}" for k in ("x_m", "y_m", "yaw_deg")]


TRIAL_COLS = ["trial_num", "trial_event_code", "asm_type_code", "block_pos", "sess_num", "imu_slot", "lidar_slot"]
TRUTH_COLS = ["gt_x_m", "gt_y_m", "gt_qz", "gt_qw"]
LODOM_COLS = ["lo_vx_mps", "lo_vy_mps", "lo_wz_radps"]
CTL_COLS = ["cmd_vx", "cmd_vy", "cmd_wz"]
IMU_COLS = ["wz_radps", "ax_mps2", "ay_mps2"]


def source_table():
    """src -> (name, columns, stamp_sem, min_hz, decimable)."""
    t = {SRC["CTL"]: ("CTL", CTL_COLS, 2, 40, False), SRC["IMU"]: ("IMU", IMU_COLS, 9, 50, False),
         SRC["TRIAL"]: ("TRIAL", TRIAL_COLS, 1, None, False), SRC["TRUTH"]: ("TRUTH", TRUTH_COLS, 9, 20, False),
         SRC["LODOM"]: ("LODOM", LODOM_COLS, 9, 8, False)}
    for m in MODULES:
        t[SRC[f"WHL_{m}"]] = (f"WHL_{m}", wheel_cols(m), 9, None, False)
        t[SRC[f"CUR_{m}"]] = (f"CUR_{m}", cur_cols(m), 9, None, False)
        t[SRC[f"FLT_{m}"]] = (f"FLT_{m}", flt_cols(m), 9, None, False)
        t[SRC[f"ASM_{m}"]] = (f"ASM_{m}", asm_cols(m), 9, None, False)
    return t


def column_specs():
    c = {}
    for m in MODULES:
        for k in CORNERS:
            c[f"cnt_{k}_{mk(m)}"] = dict(type="i64", unit="count", min=None, max=None)
            c[f"wspd_{k}_{mk(m)}"] = dict(type="f32", unit="rad/s", min=-100, max=100)
            c[f"duty_{k}_{mk(m)}"] = dict(type="f32", unit="1", min=-1, max=1)
            c[f"cur_{k}_{mk(m)}"] = dict(type="f32", unit="A", min=-20, max=20)
        c[f"fs_wheel_code_{mk(m)}"] = dict(type="i8", unit="code", min=0, max=99)
        c[f"fs_mode_code_{mk(m)}"] = dict(type="i8", unit="code", min=0, max=99)
        c[f"fs_g_{mk(m)}"] = dict(type="f32", unit="1", min=0, max=1)
        c[f"fs_trial_{mk(m)}"] = dict(type="i32", unit="1", min=0, max=None)
        c[f"fs_tick_{mk(m)}"] = dict(type="i64", unit="tick", min=None, max=None)
        c[f"asm_x_m_{mk(m)}"] = dict(type="f32", unit="m", min=-5, max=5)
        c[f"asm_y_m_{mk(m)}"] = dict(type="f32", unit="m", min=-5, max=5)
        c[f"asm_yaw_deg_{mk(m)}"] = dict(type="f32", unit="deg", min=-360, max=360)
    c.update(block_pos=dict(type="i16", unit="1", min=0, max=None), sess_num=dict(type="i16", unit="1", min=0, max=None),
             imu_slot=dict(type="i8", unit="code", min=0, max=4), lidar_slot=dict(type="i8", unit="code", min=0, max=4),
             trial_num=dict(type="i32", unit="1", min=0, max=None),
             trial_event_code=dict(type="i8", unit="code", min=1, max=99),
             asm_type_code=dict(type="i8", unit="code", min=0, max=99),
             gt_x_m=dict(type="f64", unit="m", min=-100, max=100), gt_y_m=dict(type="f64", unit="m", min=-100, max=100),
             gt_qz=dict(type="f32", unit="1", min=-1, max=1), gt_qw=dict(type="f32", unit="1", min=-1, max=1),
             lo_vx_mps=dict(type="f32", unit="m/s", min=-5, max=5), lo_vy_mps=dict(type="f32", unit="m/s", min=-5, max=5),
             lo_wz_radps=dict(type="f32", unit="rad/s", min=-10, max=10),
             cmd_vx=dict(type="f32", unit="m/s", min=-2, max=2), cmd_vy=dict(type="f32", unit="m/s", min=-2, max=2),
             cmd_wz=dict(type="f32", unit="rad/s", min=-4, max=4))
    return c


def portion_columns():
    cols = []
    for s, (_, cc, *_r) in source_table().items():
        cols += cc
    return list(dict.fromkeys(cols))


def build_config():
    topics = [
        {"topic": "/capxfer/cmd_vel", "type": "geometry_msgs/msg/TwistStamped", "src": SRC["CTL"],
         "columns": {"cmd_vx": "twist.linear.x", "cmd_vy": "twist.linear.y", "cmd_wz": "twist.angular.z"}},
        {"topic": "/capxfer/active/imu", "type": "sensor_msgs/msg/Imu", "qos": "best_effort", "src": SRC["IMU"],
         "columns": {"wz_radps": "angular_velocity.z", "ax_mps2": "linear_acceleration.x", "ay_mps2": "linear_acceleration.y"}},
        {"topic": "/capxfer/active/odom_rf2o", "type": "nav_msgs/msg/Odometry", "src": SRC["LODOM"],
         "columns": {"lo_vx_mps": "twist.twist.linear.x", "lo_vy_mps": "twist.twist.linear.y", "lo_wz_radps": "twist.twist.angular.z"}},
        {"topic": "/capxfer/trial", "type": "std_msgs/msg/String", "src": SRC["TRIAL"], "t_src": "json:t_ns",
         "columns": {"trial_num": {"json": "trial", "path": "data"},
                     "trial_event_code": {"json": "event", "path": "data", "values": dict(TRIAL_EVENT_CODES, **{"*": 99})},
                     "asm_type_code": {"json": "asm_type", "path": "data", "values": dict(ASM_TYPE_CODES, **{"*": 99})},
                     "block_pos": {"json": "block", "path": "data"}, "sess_num": {"json": "session", "path": "data"},
                     "imu_slot": {"json": "imu", "path": "data"}, "lidar_slot": {"json": "lidar", "path": "data"}}},
        {"topic": "/fleet/truth/{TRUTH_ID}/pose", "type": "geometry_msgs/msg/PoseStamped", "src": SRC["TRUTH"],
         "columns": {"gt_x_m": "pose.position.x", "gt_y_m": "pose.position.y", "gt_qz": "pose.orientation.z",
                     "gt_qw": "pose.orientation.w"}},
    ]
    for m in MODULES:
        w = {}
        for i, c in enumerate(CORNERS):
            w[f"cnt_{c}_{mk(m)}"] = f"position[{i}]"
        for i, c in enumerate(CORNERS):
            w[f"wspd_{c}_{mk(m)}"] = f"velocity[{i}]"
        for i, c in enumerate(CORNERS):
            w[f"duty_{c}_{mk(m)}"] = f"effort[{i}]"
        topics.append({"topic": "/fleet/{%s}/wheel_state" % m, "type": "sensor_msgs/msg/JointState", "src": SRC[f"WHL_{m}"],
                       "columns": w})
        topics.append({"topic": "/fleet/{%s}/wheel_current" % m, "type": "std_msgs/msg/Float32MultiArray",
                       "src": SRC[f"CUR_{m}"], "t_src": "none",
                       "columns": {f"cur_{c}_{mk(m)}": f"data[{i}]" for i, c in enumerate(CORNERS)}})
        topics.append({"topic": "/fleet/{%s}/fault_state" % m, "type": "std_msgs/msg/String", "src": SRC[f"FLT_{m}"],
                       "t_src": "none",
                       "columns": {f"fs_wheel_code_{mk(m)}": {"json": "wheel", "path": "data", "values": dict(WHEEL_CODES, **{"*": 9})},
                                   f"fs_mode_code_{mk(m)}": {"json": "mode", "path": "data", "values": dict(FAULT_MODE_CODES, **{"*": 99})},
                                   f"fs_g_{mk(m)}": {"json": "g", "path": "data"},
                                   f"fs_trial_{mk(m)}": {"json": "trial", "path": "data"},
                                   f"fs_tick_{mk(m)}": {"json": "tick", "path": "data"}}})
        topics.append({"topic": "/fleet/{%s}/assembly" % m, "type": "std_msgs/msg/String", "src": SRC[f"ASM_{m}"], "t_src": "none",
                       "columns": {f"asm_x_m_{mk(m)}": {"json": "x", "path": "data"},
                                   f"asm_y_m_{mk(m)}": {"json": "y", "path": "data"},
                                   f"asm_yaw_deg_{mk(m)}": {"json": "yaw_deg", "path": "data"}}})
    srcs = {}
    for s, (name, cols, sem, mhz, dec) in source_table().items():
        e = dict(name=name, columns=list(cols), stamp_sem=sem, batched=False, decimable=dec, clock_domain="ros")
        if mhz:
            e["min_hz"] = mhz
        srcs[s] = e
    all_srcs = sorted(source_table())
    portion = {"code": PORTION_CODE, "split": "CAL", "srcs": all_srcs,
               "required_srcs": [SRC["CTL"], SRC["IMU"], SRC["TRIAL"], SRC["TRUTH"]],
               "columns": portion_columns(), "min_hz": {s: e["min_hz"] for s, e in srcs.items() if "min_hz" in e},
               "open_loop": True}
    return {"schema_version": 2, "platform": "gnd", "params": dict(PARAMS), "topics": topics,
            "codebook_extension": {"platform": "gnd", "sources": srcs, "columns": column_specs(),
                                   "portions": {PORTION_TOKEN: portion}}}


HEADER = """\
# FieldLog v2 config: capxfer capability-transfer study, modular mecanum (docs/TEAM_T0_HANDOFF.md §4 in ~/crane_explain/capxfer).
# DRAFT generated by capxfer/field/fieldlog_config.py (WS-B). TEAM-CONFIRM every topic, message type and field path below.
# Topic names use {PARAM} placeholders per the fleet spec (/fleet/<id>/..., Tools/Fleet/spec/ros2_test_interface_v0.md); defaults in
# `params:`. Render a concrete copy before a run:  python -m capxfer.field.fieldlog_config render --set M1=<id> ... OUT.yaml
# WS-P 2026-11-21 (mixed-type sessions in one FieldLog session): cmd is logged from /capxfer/cmd_vel (the runner's copy of the assembly
# twist), IMU and rf2o from /capxfer/active/{imu,odom_rf2o} (the runner relays the active assembly's sensor module, schedule.sensor_modules);
# trial markers also carry block position, session number and the IMU/lidar slot (1-4, 0 none); events 4 assembly change, 5 block start.
# Streams: cmd_vel (TwistStamped, 50 Hz); per module wheel_state JointState (position[i] = cumulative encoder COUNTS, velocity[i] =
# encoder speed rad/s, effort[i] = PI duty before fault scaling; wheel order fl, fr, rl, rr: TEAM-CONFIRM); wheel_current
# (Float32MultiArray, only if CS is wired); fault_state JSON echo {"wheel","mode","g","trial","tick"} (handoff §2); assembly frame JSON
# {"x","y","yaw_deg"}; IMU and rf2o lidar odometry of the active sensor module (relayed); (assumed at the lidar module centre: TEAM-CONFIRM mount offset); probe-runner
# trial markers /capxfer/trial JSON {"trial","event","asm_type","t_ns"}; AprilTag truth /fleet/truth/<id>/pose (EVALUATOR-ONLY: the
# logger node must be on the truth allow-list; no robot node subscribes to it).
# The official codebook has no capxfer sources: `codebook_extension` (ignored by ros2_csvlog_v2.py) proposes srcs 21-42 and portion
# capxprobe (code 67) for the FieldLog owner to merge into codebook_v2.yaml (TEAM-CONFIRM). Until then capxfer/field/fieldlog_config.py
# builds an overlay codebook from it (pass --codebook to ros2_csvlog_v2.py).
"""


def write_config(path=CONFIG_PATH):
    path = Path(path)
    path.write_text(HEADER + yaml.safe_dump(build_config(), sort_keys=False, width=140, default_flow_style=None))
    return path


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def render(cfg, values=None):
    """Substitute {PARAM}s in topic names: a concrete config dict for ros2_csvlog_v2.py."""
    cfg = copy.deepcopy(cfg)
    vals = dict(cfg.get("params") or PARAMS, **(values or {}))
    for t in cfg["topics"]:
        t["topic"] = t["topic"].format(**vals)
    cfg["params"] = vals
    return cfg


def overlay_codebook(out_dir, cfg=None, official=OFFICIAL_CODEBOOK):
    """Official codebook + the config's codebook_extension -> overlay file in out_dir. Returns its path."""
    cfg = cfg or build_config()
    ext = cfg["codebook_extension"]
    d = yaml.safe_load(Path(official).read_text())
    d = copy.deepcopy(d)
    p = d["platforms"][ext["platform"]]
    p.setdefault("sources", {})
    for s, e in ext["sources"].items():
        if int(s) in p["sources"]:
            if list(p["sources"][int(s)]["columns"]) != list(e["columns"]):
                raise ValueError(f"src {s} already defined in the official codebook with other columns")
            continue                                          # official definition wins (CTL 7, IMU 3)
        p["sources"][int(s)] = copy.deepcopy(e)
    for k, v in ext["columns"].items():
        d["columns"].setdefault(k, dict(v))
    for tok, por in ext["portions"].items():
        if tok in p["portions"] and p["portions"][tok].get("code") != por["code"]:
            raise ValueError(f"portion {tok} clashes with the official codebook")
        if any(int(q.get("code", -1)) == por["code"] for t, q in p["portions"].items() if t != tok):
            raise ValueError(f"portion code {por['code']} already used")
        p["portions"][tok] = copy.deepcopy(por)
    d["codebook_version"] = str(d["codebook_version"]) + "+capxfer_overlay"
    raw = yaml.safe_dump(d, sort_keys=False).encode()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"codebook_v2.capxfer_overlay.{hashlib.sha256(raw).hexdigest()[:12]}.yaml"
    path.write_bytes(raw)
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="(re)write the draft config into crane_ml FieldLog/configs")
    w.add_argument("--out", default=str(CONFIG_PATH))
    r = sub.add_parser("render", help="concrete config with {PARAM}s substituted")
    r.add_argument("out")
    r.add_argument("--config", default=str(CONFIG_PATH))
    r.add_argument("--set", action="append", default=[], help="KEY=VALUE")
    c = sub.add_parser("codebook", help="write the overlay codebook")
    c.add_argument("out_dir")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        print(write_config(a.out))
    elif a.cmd == "render":
        cfg = render(load_config(a.config), dict(kv.split("=", 1) for kv in a.set))
        Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False, width=140))
        print(a.out)
    else:
        print(overlay_codebook(a.out_dir, load_config()))


if __name__ == "__main__":
    main()
