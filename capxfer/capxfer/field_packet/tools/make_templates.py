#!/usr/bin/env python3
"""(Re)write the fillable CSV templates from packet_lib.SCHEMAS and the measurement item list below.

  make_templates.py            writes ../templates/<name>.csv (header only) and ../MEASUREMENTS_SHEET.csv (pre-filled rows)

Run it only when the schemas change (the generated files are part of the packet). The team never runs it on returned data.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402

# (item_id, module, quantity, unit, method, params_key, team_confirm)
_PER_MODULE = [
    ("MASS", "total mass with battery (lifter arm stowed)", "kg", "kitchen/luggage scale, +-10 g; 3 weighings", "module.mass_kg.{m}", 0),
    ("HASLIDAR", "has RPLIDAR + D435 + Orange Pi 5 (1 yes, 0 no)", "1", "look", "module.lidar_modules", 1),
    ("HASLIFTER", "has a lifter arm/deck (1 yes, 0 no)", "1", "look", "module.lifter_modules", 1),
    ("LIFTERMASS", "lifter arm mass (if it can be removed; else empty + NA)", "kg", "scale", "-", 1),
    ("SIDEX", "chassis outer length along the module front axis", "m", "tape or caliper across the outer walls", "module.side_x_m", 0),
    ("SIDEY", "chassis outer width", "m", "tape or caliper", "module.side_y_m", 0),
    ("WHEELBASE", "front-axle to rear-axle distance (wheel centres)", "m", "tape between hub centres, both sides, average", "module.half_wheelbase_m (/2)", 0),
    ("TRACK", "left to right wheel mid-plane distance", "m", "tape between tyre mid-planes, front and rear, average", "module.half_track_m (/2)", 0),
    ("ROLL5", "floor distance for 5 full wheel revolutions, module loaded, driving straight", "m",
     "tape mark on a tyre; push or drive the module 5 turns on floor A; measure the travel; 3 repeats", "wheel.radius_m (= d / (10 pi))", 0),
    ("WHEELOD", "wheel outer diameter, unloaded", "m", "caliper over two opposite rollers", "-", 0),
    ("NROLLERS", "rollers per wheel", "count", "count", "wheel.rollers", 1),
    ("ROLLER_FL", "contact-roller axis of the front-left wheel seen from ABOVE, module front up: +1 = '/', -1 = '\\'", "sign",
     "look at the bottom roller (floor contact) through the chassis or with a mirror", "wheel.roller_pattern.fl", 1),
    ("ROLLER_FR", "contact-roller axis front-right: +1 '/', -1 '\\'", "sign", "as ROLLER_FL", "wheel.roller_pattern.fr", 1),
    ("ROLLER_RL", "contact-roller axis rear-left: +1 '/', -1 '\\'", "sign", "as ROLLER_FL", "wheel.roller_pattern.rl", 1),
    ("ROLLER_RR", "contact-roller axis rear-right: +1 '/', -1 '\\'", "sign", "as ROLLER_FL", "wheel.roller_pattern.rr", 1),
    ("CPR10", "encoder counts for 10 hand turns of the fl wheel (wheel_state position[0] after minus before)", "count",
     "mark the tyre, turn 10 full turns by hand with the driver in coast, read ros2 topic echo", "motor.counts_per_output_rev (/10)", 0),
    ("CGH", "CG height above the floor", "m", "tip test: tilt about one wheel line until it balances, h = (half track) / tan(angle)", "module.cg_height_m", 0),
    ("FRONTLOAD", "fraction of weight on the front axle (lifter stowed)", "1", "two scales under front and rear wheel pairs", "-", 0),
    ("BACKDRIVE", "force at the tyre to turn the fl wheel slowly with the driver in COAST", "N",
     "spring scale tangential at the tyre, ~1 rev in 5 s, 3 repeats", "motor.backdrive_torque_nm (= F r)", 0),
    ("VBATT_FULL", "pack voltage, full, resting", "V", "multimeter at the balance or main lead", "-", 0),
    ("CS_WIRED", "number of wheels whose G2 current-sense (CS) reaches an ADC (0-4)", "count", "wiring check", "rung E3", 1),
]
_GLOBAL = [
    ("FW_VELHZ", "ALL", "wheel velocity-loop rate", "Hz", "firmware source", "firmware.vel_loop_hz", 1),
    ("FW_CTRL", "ALL", "controller type: 1 P, 2 PI, 3 PID", "code", "firmware source", "-", 1),
    ("FW_KP", "ALL", "proportional gain in duty per (rad/s) of wheel speed error", "1/(rad/s)", "firmware source (convert units!)", "firmware.kp_duty_per_radps", 1),
    ("FW_KI", "ALL", "integral gain in duty per rad of integrated wheel speed error", "1/rad", "firmware source (convert units!)", "firmware.ki_duty_per_rad", 1),
    ("FW_AW", "ALL", "anti-windup: 0 none, 1 conditional integration on the unscaled clamp, 2 back-calculation", "code", "firmware source", "-", 1),
    ("FW_PWMHZ", "ALL", "PWM frequency", "Hz", "firmware source", "-", 1),
    ("FW_MAXDUTY", "ALL", "duty clamp", "1", "firmware source", "-", 1),
    ("FW_ENCHZ", "ALL", "wheel_state publish rate", "Hz", "ros2 topic hz /fleet/<id>/wheel_state", "firmware.encoder_rate_hz", 1),
    ("FW_WATCHDOG", "ALL", "command timeout after which the wheels stop", "ms", "firmware source + the ESTOP test", "-", 1),
    ("FW_ASMIK", "ALL", "firmware computes wheel speeds from the ASSEMBLY twist + /assembly frame (1) or module-local only (0)", "code", "firmware source", "-", 1),
    ("FW_ORDER", "ALL", "wheel_state order is fl, fr, rl, rr (1 yes, 0 no: describe in note)", "code", "firmware source + hook test IK check", "-", 1),
    ("FW_POSCOUNTS", "ALL", "wheel_state position[] is cumulative encoder COUNTS (1) or radians (0)", "code", "firmware source", "-", 1),
    ("FW_EFFORT", "ALL", "wheel_state effort[] is the PI duty BEFORE the weak scaling, in [-1, 1] (1 yes)", "code", "firmware source", "-", 1),
    ("FW_CMDLAT", "ALL", "cmd_vel to PWM latency", "s", "optional: scope on PWM + logged cmd time", "firmware.actuation_latency_s", 1),
    ("CS_SYNC", "ALL", "current sampled synchronously with the PWM on-time (1) or averaged (0); empty if not wired", "code", "firmware source", "-", 1),
    ("CS_OFFSET", "ALL", "current-sense zero offset at rest (mean over wheels)", "A", "wheel_current at rest, 10 s", "-", 1),
    ("CS_NOISE", "ALL", "current-sense noise std at constant speed", "A", "wheel_current, wheels lifted, 0.3 m/s command", "sensors.current_noise_a", 1),
    ("LATCH_PLAY", "ALL", "play between two docked modules (push-pull by hand)", "mm", "caliper or feeler gauge", "-", 0),
    ("LATCH_YAW", "ALL", "visible yaw wobble between docked modules (1 yes, 0 no)", "code", "look while pushing a corner", "-", 0),
    ("FLOOR_A_MU", "ARENA", "floor A friction: sled of roller material, tilt angle at slip (5 repeats; value = mean angle)", "deg",
     "sled = small board with >= 3 spare rollers (or roller-cover strips) glued under it, ~1 kg on top; tilt a floor-A sample until it slides", "floor.mu (= tan angle)", 1),
    ("FLOOR_B_MU", "ARENA", "floor B (second floor, card 07) friction, same sled", "deg", "as FLOOR_A_MU", "-", 1),
    ("SHEET1_MU", "ARENA", "low-friction sheet 1 (label it) friction, same sled on the sheet", "deg", "as FLOOR_A_MU; sheet taped flat", "floor.low_mu_sheet", 1),
    ("SHEET2_MU", "ARENA", "low-friction sheet 2 (optional second material)", "deg", "as FLOOR_A_MU", "floor.low_mu_sheet", 1),
    ("SHEET_SIZE", "ARENA", "sheet size (shorter side)", "m", "tape; must be >= 1.2", "-", 0),
    ("SHEET_THICK", "ARENA", "sheet thickness", "mm", "caliper; <= 2", "-", 0),
    ("ARENA_X", "ARENA", "clear arena length", "m", "tape (>= 1.2 per probe, >= 4.0 for docking)", "-", 0),
    ("ARENA_Y", "ARENA", "clear arena width", "m", "tape (>= 1.2 per probe, >= 2.6 for docking)", "-", 0),
    ("TAG_CAMS", "ARENA", "number of overhead cameras", "count", "look", "-", 1),
    ("TAG_RATE", "ARENA", "AprilTag pose rate", "Hz", "ros2 topic hz /fleet/truth/<id>/pose", "truth.apriltag_rate_hz", 1),
    ("TAG_LAT", "ARENA", "AprilTag latency (from truth_check.py motion, median)", "s", "truth_check.py motion on the shakedown session", "truth.apriltag_latency_s", 0),
    ("TAG_NOISE_POS", "ARENA", "static tag position noise (truth_check.py static, mean of std_x/std_y)", "mm", "truth_check.py static", "truth.apriltag_pos_noise_m (/1000)", 0),
    ("TAG_NOISE_YAW", "ARENA", "static tag yaw noise (truth_check.py static)", "mrad", "truth_check.py static", "truth.apriltag_yaw_noise_rad (/1000)", 0),
    ("TAG_SIZE", "ARENA", "robot tag edge length", "mm", "ruler", "-", 0),
    ("PAYLOAD_2", "ARENA", "payload set '2 kg' weighed with tray and straps", "kg", "scale", "-", 0),
    ("PAYLOAD_4", "ARENA", "payload set '4 kg' weighed with tray and straps", "kg", "scale", "-", 0),
    ("PAYLOAD_6", "ARENA", "payload set '6 kg' weighed with tray and straps (must be <= 6.00)", "kg", "scale", "-", 0),
]


def measurement_items():
    items = []
    for m in PL.MODULES:
        for iid, q, u, meth, key, tc in _PER_MODULE:
            items.append((f"{iid}_{m}", m, q, u, meth, key.format(m=m), tc))
    items += _GLOBAL
    return items


def main():
    tdir = PL.PACKET / "templates"
    tdir.mkdir(exist_ok=True)
    for name in PL.SCHEMAS:
        with open(tdir / f"{name}.csv", "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(PL.header(name))
    with open(PL.PACKET / "MEASUREMENTS_SHEET.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PL.header("measurements"))
        for iid, m, q, u, meth, key, tc in measurement_items():
            w.writerow([iid, m, q, "", u, meth, "", "", "", "TODO", "", "", key, tc, ""])
    print(f"wrote {len(PL.SCHEMAS)} templates and {len(measurement_items())} measurement rows")


if __name__ == "__main__":
    main()
